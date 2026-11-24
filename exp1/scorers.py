"""Train-only catalog scorers. All returned columns follow ``data['items']``."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def _catalog(data):
    ids = [str(item['id']) for item in data['items']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('items must be a nonempty catalog with unique IDs')
    return ids, {item: i for i, item in enumerate(ids)}


def _sequences(data):
    return data.get('train_sequences', list(data.get('histories', {}).values()))


def _interaction(data):
    ids, index = _catalog(data)
    seqs = _sequences(data)
    rows, cols = [], []
    for row, seq in enumerate(seqs):
        known = {index[str(item)] for item in seq if str(item) in index}
        rows.extend([row] * len(known))
        cols.extend(sorted(known))
    matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                               shape=(len(seqs), len(ids)))
    return matrix


def collaborative_embeddings(data: dict, dim: int = 64, seed: int = 0) -> np.ndarray:
    """Binary train-user/item SVD; no validation/test events or item attributes."""
    if dim < 1:
        raise ValueError('embedding dim must be positive')
    matrix = _interaction(data)
    result = np.zeros((matrix.shape[1], dim), dtype=np.float32)
    if matrix.nnz == 0:
        return result
    if min(matrix.shape) < 2:
        # Exact factor for a one-user or one-item matrix.
        result[:, 0] = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
        return result
    rank = min(dim, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=rank, random_state=seed)
    svd.fit(matrix)
    result[:, :rank] = (svd.components_.T * svd.singular_values_).astype(np.float32)
    return result


class _BaseScorer:
    def __init__(self, data, config, seed, name):
        self.data, self.config, self.seed = data, config, seed
        self.ids, self.index = _catalog(data)
        counts = np.zeros(len(self.ids), dtype=np.float64)
        for seq in _sequences(data):
            for item in seq:
                if str(item) in self.index:
                    counts[self.index[str(item)]] += 1
        self.popularity = counts
        self.popularity_logits = np.log(np.maximum(counts, 1e-12))
        self._embeddings = None
        self.metadata = {'name': name, 'seed': seed, 'catalog_size': len(self.ids),
                         'training_source': 'train_sequences',
                         'empty_history': 'train_popularity', 'config': dict(config)}

    def item_embeddings(self):
        if self._embeddings is None:
            self._embeddings = collaborative_embeddings(
                self.data, int(self.config.get('d_model', 64)), self.seed)
        return self._embeddings.copy()


class MostPopular(_BaseScorer):
    def __init__(self, data, config, seed, name):
        super().__init__(data, config, seed, name)
        self.metadata['score_scale'] = 'log_train_counts_floor_1e-12'

    def scores(self, history: list[str]) -> np.ndarray:
        return self.popularity_logits.copy()


class ItemKNN(_BaseScorer):
    def __init__(self, data, config, seed):
        super().__init__(data, config, seed, 'item_knn')
        matrix = _interaction(data)
        norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
        normalized = matrix @ sparse.diags(1 / np.maximum(norms, 1))
        similarity = (normalized.T @ normalized).tocsr()
        similarity.setdiag(0)
        similarity.eliminate_zeros()
        neighbors = int(config.get('knn_neighbors', 100))
        if neighbors < 1:
            raise ValueError('knn_neighbors must be positive')
        # Retain at most K neighbors per source item, with catalog-order ties.
        rows, cols, values = [], [], []
        for i in range(similarity.shape[0]):
            start, end = similarity.indptr[i:i + 2]
            indices, weights = similarity.indices[start:end], similarity.data[start:end]
            order = np.lexsort((indices, -weights))[:neighbors]
            rows.extend([i] * len(order))
            cols.extend(indices[order])
            values.extend(weights[order])
        self.similarity = sparse.csr_matrix((values, (rows, cols)), shape=similarity.shape)
        self.metadata['knn_neighbors'] = neighbors
        self.metadata['score_scale'] = 'log_similarity_sum_floor_1e-12'

    def scores(self, history: list[str]) -> np.ndarray:
        indices = [self.index[str(item)] for item in history if str(item) in self.index]
        if not indices:
            return self.popularity_logits.copy()
        scores = np.asarray(self.similarity[indices].sum(axis=0)).ravel()
        # A bounded popularity tie breaker; no validation interaction contributes.
        scores += 1e-10 * self.popularity / max(1., self.popularity.max())
        return np.log(np.maximum(scores, 1e-12))


def _torch_model(n_items, config):
    import torch
    from torch import nn

    class SASRecModel(nn.Module):
        def __init__(self):
            super().__init__()
            width = int(config.get('d_model', 64))
            self.max_history = int(config.get('max_history', 50))
            self.items = nn.Embedding(n_items + 1, width, padding_idx=0)
            self.positions = nn.Embedding(self.max_history, width)
            self.dropout = nn.Dropout(float(config.get('dropout', .1)))
            block = nn.TransformerEncoderLayer(
                width, int(config.get('heads', 2)), width * 4,
                float(config.get('dropout', .1)), batch_first=True,
                norm_first=True, activation='gelu')
            self.encoder = nn.TransformerEncoder(
                block, int(config.get('layers', 2)), enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)

        def encode(self, tokens):
            length = tokens.shape[1]
            hidden = self.dropout(self.items(tokens) + self.positions(
                torch.arange(length, device=tokens.device))[None])
            causal = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                           device=tokens.device), diagonal=1)
            return self.norm(self.encoder(hidden, mask=causal,
                                           src_key_padding_mask=tokens.eq(0)))

        def forward(self, tokens):
            hidden = self.encode(tokens)
            last = tokens.ne(0).sum(dim=1) - 1
            state = hidden[torch.arange(len(tokens), device=tokens.device), last]
            # Padding index is excluded from the loss and catalog predictions.
            return state @ self.items.weight[1:].T

    return SASRecModel()


class SASRec(_BaseScorer):
    def __init__(self, data, config, seed, output_dir):
        import torch
        from torch.nn import functional as F

        super().__init__(data, config, seed, 'sasrec')
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(int(config.get('torch_threads', 2)))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = str(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = _torch_model(len(self.ids), config).to(self.device)
        self.max_history = self.model.max_history
        if self.max_history < 1:
            raise ValueError('max_history must be positive')
        rng = np.random.default_rng(seed)
        samples = []
        # Each target sees only the items strictly preceding it in its train sequence.
        for seq in _sequences(data):
            tokens = [self.index[str(item)] + 1 for item in seq if str(item) in self.index]
            samples.extend((tokens[max(0, pos - self.max_history):pos], tokens[pos] - 1)
                           for pos in range(1, len(tokens)))
        cap = config.get('max_train_samples', 5000)
        if cap is not None and int(cap) > 0 and len(samples) > int(cap):
            samples = [samples[int(i)] for i in rng.choice(len(samples), int(cap), replace=False)]
        if not samples:
            raise ValueError('SASRec requires a train sequence with at least two known items')
        validation = []
        for row in data.get('validation', []):
            positive = str(row['positive'])
            history = [self.index[str(item)] + 1 for item in row['history']
                       if str(item) in self.index][-self.max_history:]
            if history and positive in self.index:
                validation.append((history, self.index[positive]))
        batch_size, epochs = int(config.get('batch_size', 128)), int(config.get('epochs', 10))
        if batch_size < 1 or epochs < 1:
            raise ValueError('batch_size and epochs must be positive')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config.get('lr', .001)))
        best, best_state, stale, log = float('inf'), None, 0, []
        patience = int(config.get('patience', 3))
        for epoch in range(epochs):
            self.model.train()
            train_total = 0.
            order = rng.permutation(len(samples))
            for offset in range(0, len(order), batch_size):
                batch = [samples[int(i)] for i in order[offset:offset + batch_size]]
                tokens, targets = self._batch(batch)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.model(tokens), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite SASRec training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                optimizer.step()
                train_total += float(loss.detach()) * len(batch)
            train_loss = train_total / len(samples)
            self.model.eval()
            val_total = 0.
            with torch.no_grad():
                for offset in range(0, len(validation), batch_size):
                    batch = validation[offset:offset + batch_size]
                    tokens, targets = self._batch(batch)
                    val_total += float(F.cross_entropy(self.model(tokens), targets,
                                                      reduction='sum'))
            val_loss = val_total / len(validation) if validation else None
            criterion = val_loss if val_loss is not None else train_loss
            log.append({'epoch': epoch + 1, 'train_ce': train_loss, 'validation_ce': val_loss})
            if criterion < best:
                best, stale = criterion, 0
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1
            else:
                stale += 1
            if stale >= max(1, patience):
                break
        self.model.load_state_dict(best_state)
        self.model.eval()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / 'sasrec_best.pt'
        torch.save({'state_dict': {key: value.cpu() for key, value in best_state.items()},
                    'config': dict(config), 'item_ids': self.ids, 'seed': seed,
                    'best_epoch': best_epoch}, checkpoint)
        self.metadata.update({'device': str(self.device), 'objective': 'full_catalog_cross_entropy',
                              'train_samples': len(samples), 'validation_samples': len(validation),
                              'selection': 'validation_ce' if validation else 'train_ce_fallback',
                              'best_epoch': best_epoch, 'best_ce': best,
                              'epochs_run': len(log), 'loss_history': log,
                              'checkpoint': str(checkpoint)})

    def _batch(self, samples):
        import torch
        length = max(len(history) for history, _ in samples)
        tokens = torch.zeros((len(samples), length), dtype=torch.long, device=self.device)
        for row, (history, _) in enumerate(samples):
            tokens[row, :len(history)] = torch.tensor(history, device=self.device)
        targets = torch.tensor([target for _, target in samples], device=self.device)
        return tokens, targets

    def scores(self, history: list[str]) -> np.ndarray:
        import torch
        tokens = [self.index[str(item)] + 1 for item in history if str(item) in self.index]
        if not tokens:
            return self.popularity_logits.copy()
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor([tokens[-self.max_history:]], device=self.device)
            return self.model(inputs)[0].cpu().numpy().astype(np.float64)

    def item_embeddings(self):
        return self.model.items.weight[1:].detach().cpu().numpy().copy()


def fit_scorer(name: str, data: dict, config: dict, seed: int, output_dir: Path):
    """Fit a reproducible scorer without reading test labels or item encodings."""
    name = name.lower()
    if name == 'most_popular':
        return MostPopular(data, config, seed, name)
    if name == 'item_knn':
        return ItemKNN(data, config, seed)
    if name == 'sasrec':
        return SASRec(data, config, seed, output_dir)
    raise ValueError(f'unknown scorer: {name}')
