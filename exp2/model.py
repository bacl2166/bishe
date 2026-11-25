"""Teacher-forced encoder/decoder recommendation with complete-vocabulary probabilities."""
from __future__ import annotations

import copy
import json
import os
from numbers import Integral
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PAD, BOS, EOS = 0, 1, 2


def _catalog(data):
    items = [str(row['id']) for row in data['items']]
    if not items or len(items) != len(set(items)):
        raise ValueError('catalog must be nonempty with unique IDs')
    return items


def _tokenize(items, codes):
    validated = {}
    for key, raw_code in codes.items():
        try:
            code = tuple(raw_code)
        except TypeError as error:
            raise ValueError('codes must be nonempty sequences of nonnegative integers') from error
        if not code or any(isinstance(token, (bool, np.bool_)) or
                           not isinstance(token, Integral) or token < 0 for token in code):
            raise ValueError('codes must be nonempty sequences of nonnegative integers (not bool)')
        validated[str(key)] = tuple(int(token) for token in code)
    codes = validated
    if set(items) != set(codes):
        raise ValueError('codes must cover exactly the item catalog')
    raw_tokens = sorted({token for code in codes.values() for token in code})
    mapping = {raw: i + 3 for i, raw in enumerate(raw_tokens)}
    token_codes = {item: tuple(mapping[token] for token in codes[item]) + (EOS,)
                   for item in items}
    if len(set(token_codes.values())) != len(items):
        raise ValueError('item codes must be unique; add collision suffixes before training')
    return mapping, token_codes


def _atomic_write(path, writer):
    """Replace an artifact only after a complete successful write beside its target."""
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_training_samples(data, max_history=50):
    """Expose strict train-only (earlier-history, next-item) pairs for audit/tests."""
    if max_history < 1:
        raise ValueError('max_history must be positive')
    known = set(_catalog(data))
    result = []
    sequences = data.get('train_sequences', list(data.get('histories', {}).values()))
    for sequence in sequences:
        sequence = [str(item) for item in sequence if str(item) in known]
        for position in range(1, len(sequence)):
            result.append((sequence[max(0, position - max_history):position], sequence[position]))
    return result


class RecommendationTransformer(nn.Module):
    def __init__(self, n_items, vocab_size, max_code_len, config):
        super().__init__()
        width = int(config.get('d_model', 32))
        heads = int(config.get('heads', 2))
        self.max_history = int(config.get('max_history', 50))
        self.max_code_len = max_code_len
        dropout = float(config.get('dropout', .1))
        if width < 1 or heads < 1 or width % heads or self.max_history < 1:
            raise ValueError('positive d_model divisible by heads and max_history > 0 required')
        self.history_items = nn.Embedding(n_items + 1, width, padding_idx=PAD)
        self.history_positions = nn.Embedding(self.max_history, width)
        self.output_tokens = nn.Embedding(vocab_size, width, padding_idx=PAD)
        self.output_positions = nn.Embedding(max_code_len, width)
        self.empty_history = nn.Parameter(torch.zeros(1, 1, width))
        self.dropout = nn.Dropout(dropout)
        encoder = nn.TransformerEncoderLayer(width, heads, width * 4, dropout,
                                             batch_first=True, norm_first=True, activation='gelu')
        decoder = nn.TransformerDecoderLayer(width, heads, width * 4, dropout,
                                             batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder, int(config.get('encoder_layers', 1)),
                                             norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder, int(config.get('decoder_layers', 2)),
                                             norm=nn.LayerNorm(width))
        self.output = nn.Linear(width, vocab_size)

    def encode_history(self, history):
        padding = history.eq(PAD)
        hidden = self.history_items(history) + self.history_positions(
            torch.arange(history.shape[1], device=history.device))[None]
        # Empty users receive one learned memory token; never attend to an all-masked row.
        empty = padding.all(dim=1)
        hidden = hidden.clone()
        padding = padding.clone()
        hidden[empty, 0:1] = self.empty_history
        padding[empty, 0] = False
        return self.encoder(self.dropout(hidden), src_key_padding_mask=padding), padding

    def decode(self, decoder_input, memory, memory_padding):
        length = decoder_input.shape[1]
        positions = torch.arange(length, device=decoder_input.device)
        hidden = self.dropout(self.output_tokens(decoder_input) + self.output_positions(positions)[None])
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                       device=decoder_input.device), diagonal=1)
        hidden = self.decoder(hidden, memory, tgt_mask=causal,
                              tgt_key_padding_mask=decoder_input.eq(PAD),
                              memory_key_padding_mask=memory_padding)
        return self.output(hidden)

    def forward(self, history, decoder_input):
        memory, padding = self.encode_history(history)
        return self.decode(decoder_input, memory, padding)


class Generator:
    def __init__(self, data, codes, config, seed):
        self.item_ids = _catalog(data)
        self.item_index = {item: i + 1 for i, item in enumerate(self.item_ids)}
        self.raw_token_mapping, self.token_codes = _tokenize(self.item_ids, codes)
        self.vocab_size = len(self.raw_token_mapping) + 3
        self.max_code_len = max(map(len, self.token_codes.values()))
        self.config, self.seed = dict(config), int(seed)
        self.max_history = int(config.get('max_history', 50))
        torch.set_num_threads(int(config.get('torch_threads', 2)))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = str(config.get('device', 'auto'))
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = RecommendationTransformer(len(self.item_ids), self.vocab_size,
                                                self.max_code_len, config).to(self.device)
        self.metadata = {'seed': seed, 'config': dict(config), 'device': str(self.device),
                         'parameter_count': sum(parameter.numel() for parameter in self.model.parameters()),
                         'vocab_size': self.vocab_size, 'max_code_len': self.max_code_len,
                         'catalog_size': len(self.item_ids), 'architecture': 'encoder_decoder_transformer',
                         'objective': 'teacher_forced_full_vocabulary_cross_entropy_including_eos',
                         'training_source': 'train_sequences', 'empty_history': 'learned_memory_token',
                         'special_tokens': {'PAD': PAD, 'BOS': BOS, 'EOS': EOS},
                         'raw_token_mapping': {str(key): value for key, value in self.raw_token_mapping.items()}}
        self._memory_cache = None

    def encode_item(self, item_id):
        return self.token_codes[str(item_id)]

    def _history_tokens(self, history):
        return [self.item_index[str(item)] for item in history
                if str(item) in self.item_index][-self.max_history:]

    def _batch(self, examples):
        histories = [self._history_tokens(history) for history, _ in examples]
        code_targets = [self.token_codes[target] for _, target in examples]
        history = torch.zeros((len(examples), max(1, max(map(len, histories)))),
                               dtype=torch.long, device=self.device)
        targets = torch.zeros((len(examples), max(map(len, code_targets))),
                               dtype=torch.long, device=self.device)
        decoder_input = torch.zeros_like(targets)
        for row, (past, code) in enumerate(zip(histories, code_targets)):
            if past:
                history[row, :len(past)] = torch.tensor(past, device=self.device)
            targets[row, :len(code)] = torch.tensor(code, device=self.device)
            decoder_input[row, :len(code)] = torch.tensor((BOS,) + code[:-1], device=self.device)
        return history, decoder_input, targets

    def log_probs(self, history: list[str], prefixes: list[tuple[int, ...]]) -> np.ndarray:
        """Full vocabulary probabilities, with no trie masks or length normalization."""
        if not prefixes:
            return np.empty((0, self.vocab_size), dtype=np.float64)
        prefixes = [tuple(prefix) for prefix in prefixes]
        for prefix in prefixes:
            if len(prefix) >= self.max_code_len:
                raise ValueError('prefix exceeds maximum code length before EOS')
            if any(isinstance(token, (bool, np.bool_)) or not isinstance(token, Integral) or
                   token < 3 or token >= self.vocab_size for token in prefix):
                raise ValueError('prefix contains special or out-of-vocabulary token')
        self.model.eval()
        history_tokens = tuple(self._history_tokens(history))
        with torch.no_grad():
            if self._memory_cache is None or self._memory_cache[0] != history_tokens:
                inputs = torch.tensor([history_tokens or (PAD,)], device=self.device)
                memory, padding = self.model.encode_history(inputs)
                self._memory_cache = (history_tokens, memory, padding)
            _, memory, padding = self._memory_cache
            count = len(prefixes)
            tokens = torch.zeros((count, max(map(len, prefixes)) + 1),
                                  dtype=torch.long, device=self.device)
            tokens[:, 0] = BOS
            for row, prefix in enumerate(prefixes):
                if prefix:
                    tokens[row, 1:len(prefix) + 1] = torch.tensor(prefix, device=self.device)
            logits = self.model.decode(tokens, memory.expand(count, -1, -1),
                                       padding.expand(count, -1))
            indices = torch.tensor([len(prefix) for prefix in prefixes], device=self.device)
            selected = logits[torch.arange(count, device=self.device), indices]
            return F.log_softmax(selected, dim=-1).cpu().numpy().astype(np.float64)


def fit_generator(data: dict, codes: dict, config: dict, seed: int, output_dir: Path):
    started = time.perf_counter()
    generator = Generator(data, codes, config, seed)
    if generator.device.type == 'cuda':
        torch.cuda.synchronize(generator.device)
        torch.cuda.reset_peak_memory_stats(generator.device)
    samples = build_training_samples(data, generator.max_history)
    if not samples:
        raise ValueError('generator requires a train sequence with at least two catalog items')
    rng = np.random.default_rng(seed)
    cap = config.get('max_train_samples', 50000)
    if cap is not None and int(cap) > 0 and len(samples) > int(cap):
        samples = [samples[int(i)] for i in rng.choice(len(samples), int(cap), replace=False)]
    validation = [(row['history'], str(row['positive'])) for row in data.get('validation', [])
                  if str(row['positive']) in generator.token_codes]
    epochs, batch_size = int(config.get('epochs', 10)), int(config.get('batch_size', 128))
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    optimizer = torch.optim.AdamW(generator.model.parameters(), lr=float(config.get('lr', .001)),
                                  weight_decay=float(config.get('weight_decay', 0.)))
    best_loss, best_state, stale, records = float('inf'), None, 0, []
    for epoch in range(epochs):
        generator.model.train()
        total_loss, total_tokens = 0., 0
        order = rng.permutation(len(samples))
        for start in range(0, len(samples), batch_size):
            examples = [samples[int(i)] for i in order[start:start + batch_size]]
            history, inputs, targets = generator._batch(examples)
            optimizer.zero_grad(set_to_none=True)
            logits = generator.model(history, inputs)
            loss_sum = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                       ignore_index=PAD, reduction='sum')
            token_count = int(targets.ne(PAD).sum())
            loss = loss_sum / token_count
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite generator training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss_sum.detach())
            total_tokens += token_count
        train_ce = total_loss / total_tokens
        generator.model.eval()
        val_sum, val_tokens = 0., 0
        with torch.no_grad():
            for start in range(0, len(validation), batch_size):
                history, inputs, targets = generator._batch(validation[start:start + batch_size])
                logits = generator.model(history, inputs)
                val_sum += float(F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                                  ignore_index=PAD, reduction='sum'))
                val_tokens += int(targets.ne(PAD).sum())
        val_ce = val_sum / val_tokens if val_tokens else None
        selection_loss = val_ce if val_ce is not None else train_ce
        records.append({'epoch': epoch + 1, 'train_ce': train_ce, 'validation_ce': val_ce})
        val_text = f'{val_ce:.6f}' if val_ce is not None else 'unavailable'
        print(f'[generator seed={seed}] epoch={epoch + 1}/{epochs} '
              f'train_ce={train_ce:.6f} validation_ce={val_text} '
              f'elapsed_seconds={time.perf_counter() - started:.2f}', flush=True)
        if selection_loss < best_loss:
            best_loss, stale = selection_loss, 0
            best_state = copy.deepcopy(generator.model.state_dict())
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= max(1, int(config.get('patience', 3))):
            break
    generator.model.load_state_dict(best_state)
    generator.model.eval()
    if generator.device.type == 'cuda':
        torch.cuda.synchronize(generator.device)
        peak_cuda_memory = int(torch.cuda.max_memory_allocated(generator.device))
    else:
        peak_cuda_memory = None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / 'generator_best.pt'
    generator.metadata.update({'train_samples': len(samples), 'validation_samples': len(validation),
                               'train_seconds': time.perf_counter() - started,
                               'peak_cuda_memory_bytes': peak_cuda_memory,
                               'selection': 'validation_ce' if validation else 'train_ce_fallback',
                               'best_epoch': best_epoch, 'best_ce': best_loss,
                               'epochs_run': len(records), 'loss_history': records,
                               'checkpoint': str(checkpoint)})
    manifest = dict(generator.metadata, item_ids=generator.item_ids,
                    token_codes={item: list(code) for item, code in generator.token_codes.items()})
    state = {'state_dict': {key: value.cpu() for key, value in best_state.items()}, 'manifest': manifest}
    _atomic_write(checkpoint, lambda temporary: torch.save(state, temporary))
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    _atomic_write(output_dir / 'generator_manifest.json',
                  lambda temporary: temporary.write_text(manifest_text, encoding='utf-8'))
    return generator


def load_generator(data: dict, codes: dict, config: dict, seed: int, output_dir: Path):
    """Restore model and token mapping; ``config.device`` may override stored device."""
    checkpoint = Path(output_dir) / 'generator_best.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    manifest = saved['manifest']
    if seed != manifest['seed']:
        raise ValueError('checkpoint seed differs from supplied seed')
    stored_config = dict(manifest['config'])
    for key, value in config.items():
        if key not in ('device', 'torch_threads') and (key not in stored_config or value != stored_config[key]):
            raise ValueError(f'checkpoint config differs for {key}')
    for key in ('device', 'torch_threads'):
        if key in config:
            stored_config[key] = config[key]
    generator = Generator(data, codes, stored_config, int(manifest['seed']))
    if generator.item_ids != manifest['item_ids']:
        raise ValueError('checkpoint catalog order differs from supplied catalog')
    if {key: list(value) for key, value in generator.token_codes.items()} != manifest['token_codes']:
        raise ValueError('checkpoint token codes differ from supplied codes')
    if generator.metadata['raw_token_mapping'] != manifest['raw_token_mapping']:
        raise ValueError('checkpoint raw token mapping differs from supplied codes')
    generator.model.load_state_dict(saved['state_dict'])
    generator.model.eval()
    generator.metadata = {key: value for key, value in manifest.items()
                          if key not in ('item_ids', 'token_codes')}
    generator.metadata.update({'device': str(generator.device), 'checkpoint': str(checkpoint),
                               'loaded_from_checkpoint': True})
    return generator
