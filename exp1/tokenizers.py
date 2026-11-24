"""Reproducible item encoders for experiment 1 (see docs/TOKENIZERS.md).

Paper methods are tokenizer-only controlled adaptations, not claimed full
reproductions of their recommendation systems. No EOS is emitted here.
"""
from __future__ import annotations

import json
import re
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits


TOKENIZERS = (
    "semantic_hierarchical", "collaborative_hierarchical", "rqvae", "rqkmeans",
    "opq", "letter", "cost", "seater", "topotok", "bonsai",
    "random_balanced", "attribute_first",
)
NAMES = TOKENIZERS


def _normalize(x, n, label):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != n or x.shape[1] < 1:
        raise ValueError(f"{label} must have shape (number of items, positive dimension)")
    if not np.isfinite(x).all():
        raise ValueError(f"{label} contains non-finite values")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _kmeans(x, k, seed, iterations=50):
    k = min(k, len(x), len(np.unique(x, axis=0)))
    model = KMeans(n_clusters=max(1, k), random_state=int(seed) % (2**32-1),
                   n_init=3, max_iter=iterations, algorithm="lloyd").fit(x)
    return model.labels_, model.cluster_centers_


def _balanced_labels(x, centers, exact_limit=1500):
    """Capacity-constrained assignment; exact Hungarian for small nodes."""
    n, k = len(x), len(centers)
    capacities = np.full(k, n // k, dtype=int)
    capacities[:n % k] += 1
    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    if n <= exact_limit:
        slots = np.repeat(np.arange(k), capacities)
        rows, columns = linear_sum_assignment(distances[:, slots])
        labels = np.empty(n, dtype=int)
        labels[rows] = slots[columns]
        return labels
    # Largest nearest-vs-second-nearest regret first, stable item-order ties.
    order = np.argsort(distances, axis=1, kind="stable")
    regret = distances[np.arange(n), order[:, min(1, k-1)]] - distances.min(1)
    labels = np.empty(n, dtype=int)
    for i in np.argsort(-regret, kind="stable"):
        choice = next(int(j) for j in order[i] if capacities[j] > 0)
        labels[i] = choice
        capacities[choice] -= 1
    return labels


def _hierarchical(x, branching, depth, seed, balanced=False, exact_limit=1500):
    result = [()] * len(x)
    def visit(indices, prefix):
        if len(prefix) == depth:
            for i in indices:
                result[i] = prefix
            return
        if len(indices) == 1:
            result[indices[0]] = prefix + (0,) * (depth - len(prefix))
            return
        labels, centers = _kmeans(x[indices], branching, seed + len(prefix)*1009 + int(indices[0]))
        if balanced:
            # Equal embeddings still need separate capacity slots in a balanced tree.
            centers = centers[np.arange(min(branching, len(indices))) % len(centers)]
            labels = _balanced_labels(x[indices], centers, exact_limit)
        for label in sorted(set(labels.tolist())):
            visit(indices[labels == label], prefix + (int(label),))
    visit(np.arange(len(x)), ())
    return result


def _residual_kmeans(x, depth, k, seed):
    residual = x.copy()
    columns, books = [], []
    for level in range(depth):
        labels, centers = _kmeans(residual, k, seed + level)
        columns.append(labels)
        books.append(centers)
        residual -= centers[labels]
    return np.stack(columns, axis=1), books


def _opq(x, depth, k, seed, config, output_dir):
    """Alternating product quantization and orthogonal Procrustes rotation."""
    dim = int(math.ceil(x.shape[1] / depth) * depth)
    padded = np.pad(x, ((0, 0), (0, dim-x.shape[1])))
    rotation = np.eye(dim, dtype=np.float32)
    def pq(rotated):
        reconstructed = np.zeros_like(rotated)
        columns, books = [], []
        for level, block in enumerate(np.array_split(np.arange(dim), depth)):
            labels, centers = _kmeans(rotated[:, block], k, seed + level)
            reconstructed[:, block] = centers[labels]
            columns.append(labels)
            books.append(centers)
        return np.stack(columns, axis=1), reconstructed, books
    for _ in range(int(config.get("opq_iterations", 5))):
        _, reconstructed, _ = pq(padded @ rotation)
        u, _, vt = np.linalg.svd(padded.T @ reconstructed, full_matrices=False)
        rotation = u @ vt
    codes, reconstruction, books = pq(padded @ rotation)
    np.savez(output_dir / "opq_model.npz", rotation=rotation,
             **{f"codebook_{i}": b for i, b in enumerate(books)})
    return codes, {"quantization_mse": float(np.mean((padded @ rotation-reconstruction)**2)),
                   "padded_dimension": dim}


def _random_balanced(n, branching, depth, seed):
    result = [()] * n
    permutation = np.random.default_rng(seed).permutation(n)
    def visit(indices, prefix):
        if len(prefix) == depth or len(indices) == 1:
            for i in indices:
                result[i] = prefix + (0,) * (depth-len(prefix))
            return
        for label, group in enumerate(np.array_split(indices, min(branching, len(indices)))):
            visit(group, prefix + (label,))
    visit(permutation, ())
    return result


def _attribute_first(items, config):
    attributes = config.get("attributes", config.get("attribute_order", ["category", "brand", "price"]))
    if isinstance(attributes, str):
        attributes = [attributes]
    if not attributes:
        raise ValueError("attributes must contain at least one field")
    edges = np.asarray(config.get("price_bins", [1000, 5000, 10000, 50000]), dtype=float)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("price_bins must be strictly increasing (same currency units as input)")
    rows = []
    for item in items:
        row = []
        for field in attributes:
            if field == "price":
                value = item.get("price")
                value = "missing" if value is None else f"bin:{np.searchsorted(edges, value, side='right')}"
            elif field == "category":
                value = item.get("category", [])
                level = config.get("category_level")
                if level is not None:
                    value = value[int(level)] if len(value) > int(level) else None
            elif field.startswith("specs.") or field.startswith("spec:"):
                key = field.split(".", 1)[1] if "." in field else field.split(":", 1)[1]
                value = item.get("specs", {}).get(key)
            elif field in ("brand", "specs"):
                value = item.get(field)
            else:
                raise ValueError(f"unsupported attribute {field!r}; use category/brand/price/specs/specs.KEY")
            row.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        rows.append(row)
    vocab, offset = [], 0
    for level in range(len(attributes)):
        values = sorted({r[level] for r in rows})
        vocab.append({v: offset+j for j, v in enumerate(values)})
        offset += len(values)
    return [tuple(vocab[l][v] for l, v in enumerate(r)) for r in rows], {
        "attributes": list(attributes), "attribute_vocabulary": vocab,
        "price_bins": edges.tolist()}


def _bonsai(items, config):
    """Text feature extraction plus recursive greedy minimum set cover.

    This single-path adapter stops only at singletons or exhausted feature sets.
    It never substitutes embedding clustering for textual trie construction.
    """
    features = []
    supplied = config.get("item_terms", {})
    for item in items:
        if str(item["id"]) in supplied:
            terms = set(map(str, supplied[str(item["id"])]))
        else:
            text = " ".join([str(item.get("title", "")), str(item.get("description", "")),
                             " ".join(item.get("category", [])), str(item.get("brand") or ""),
                             json.dumps(item.get("specs", {}), ensure_ascii=False)])
            terms = {t.lower() for t in re.findall(r"[^\W_]+", text, re.UNICODE) if len(t) > 1}
            terms -= {"the", "and", "for", "with", "from", "this", "that", "none", "null"}
        features.append(terms)
    df = Counter(t for terms in features for t in terms)
    limit = int(config.get("max_terms", 16))
    if limit < 1:
        raise ValueError("max_terms must be positive")
    features = [set(sorted(ts, key=lambda t: (df[t], t))[:limit]) for ts in features]
    vocabulary = {t: i for i, t in enumerate(sorted(set().union(*features)))}
    fallback = len(vocabulary)
    result = [()] * len(items)
    def visit(indices, remaining, prefix):
        if len(indices) == 1 or not any(remaining.values()):
            for i in indices:
                result[i] = prefix or (fallback,)
            return
        uncovered = set(indices)
        groups = []
        # Greedy set cover: minimum cardinality is NP-hard. Deterministic ties.
        while uncovered:
            counts = Counter(t for i in sorted(uncovered) for t in sorted(remaining[i]))
            if not counts:
                groups.append((fallback, sorted(uncovered), None))
                break
            term = min(counts, key=lambda t: (-counts[t], t))
            members = sorted(i for i in uncovered if term in remaining[i])
            groups.append((vocabulary[term], members, term))
            uncovered.difference_update(members)
        for token, members, term in groups:
            rest = {i: remaining[i] - ({term} if term else set()) for i in members}
            visit(members, rest, prefix + (token,))
    visit(list(range(len(items))), {i: set(ts) for i, ts in enumerate(features)}, ())
    return result, {"term_vocabulary": vocabulary, "feature_extractor": "metadata_words_df_filter",
                    "set_cover_solver": "greedy", "paths_per_item": 1}


def _neural(name, x, collaborative, depth, k, config, seed, output_dir):
    import torch
    from torch import nn
    from torch.nn import functional as F

    device = torch.device(config.get("device", "cpu"))
    latent = int(config.get("latent_dim", min(32, x.shape[1])))
    hidden = int(config.get("hidden_dim", max(32, latent*2)))
    batch_size = int(config.get("batch_size", 128))
    epochs = int(config.get("epochs", 3))
    temperature = float(config.get("temperature", 0.2))
    if min(latent, hidden, batch_size) < 1 or epochs < 0 or temperature <= 0:
        raise ValueError("invalid neural dimensions/batch_size/epochs/temperature")

    class RQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, latent))
            self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, x.shape[1]))
            self.books = nn.ParameterList([nn.Parameter(torch.empty(k, latent)) for _ in range(depth)])
            self.cf_projection = nn.Linear(collaborative.shape[1], latent, bias=False)

        def forward(self, batch):
            z = self.encoder(batch)
            residual, total = z, torch.zeros_like(z)
            loss = z.new_zeros(())
            codes, cumulative = [], []
            for book in self.books:
                distances = (residual[:, None, :] - book[None, :, :]).square().sum(-1)
                code = distances.argmin(-1)
                chosen = book[code]
                loss = loss + F.mse_loss(chosen, residual.detach())
                loss = loss + float(config.get("commitment", .25))*F.mse_loss(residual, chosen.detach())
                total = total + chosen
                # Forward is hard quantization; training gradients reach encoder and codebooks.
                cumulative.append(total + z-z.detach())
                residual = residual-chosen.detach()
                codes.append(code)
            straight = z + (total-z).detach()
            reconstructed = self.decoder(straight)
            return reconstructed, loss, torch.stack(codes, 1), cumulative

    def distill(teacher, student, mask=None):
        if len(teacher) < 2:
            return student.sum()*0
        td = torch.cdist(teacher, teacher).square()
        sd = torch.cdist(student, student).square()
        valid = torch.ones_like(td, dtype=torch.bool)
        if mask is not None:
            valid = mask & ~torch.eye(len(td), dtype=torch.bool, device=device)
        rows = valid.any(-1)
        if not rows.any():
            return student.sum()*0
        topo_temperature = float(config.get("topology_temperature", 1.0))
        if topo_temperature <= 0:
            raise ValueError("topology_temperature must be positive")
        tp = (-td/topo_temperature).masked_fill(~valid, -1e9)[rows].softmax(-1).detach()
        sp = (-sd/topo_temperature).masked_fill(~valid, -1e9)[rows].log_softmax(-1)
        return F.kl_div(sp, tp, reduction="batchmean")

    # fork_rng restores the caller's torch random state.
    with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device.index or 0]):
        torch.manual_seed(seed)
        model = RQ().to(device)
        data = torch.tensor(x, device=device)
        cf = torch.tensor(collaborative, device=device)
        with torch.no_grad():
            initial = model.encoder(data).cpu().numpy()
            _, books = _residual_kmeans(initial, depth, k, seed)
            for parameter, book in zip(model.books, books):
                expanded = book[np.arange(k) % len(book)]
                parameter.copy_(torch.tensor(expanded, device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
        rng = np.random.default_rng(seed)
        history = []
        for epoch in range(epochs):
            diversity_groups = []
            if name == "letter":
                for level, book in enumerate(model.books):
                    vectors = book.detach().cpu().numpy()
                    _, centers = _kmeans(vectors, max(1, min(k//2, int(config.get("diversity_groups", 4)))), seed+epoch+level)
                    diversity_groups.append(_balanced_labels(vectors, centers))
            sums = defaultdict(float)
            for start in range(0, len(x), batch_size):
                # An epoch permutation is generated once below, not per batch.
                if start == 0:
                    permutation = rng.permutation(len(x))
                ix = permutation[start:start+batch_size]
                batch = data[ix]
                reconstructed, rq_loss, codes, cumulative = model(batch)
                reconstruction_loss = F.mse_loss(reconstructed, batch)
                loss = rq_loss
                components = {"rq": rq_loss, "reconstruction": reconstruction_loss}
                if name == "cost":
                    logits = F.normalize(batch, dim=-1) @ F.normalize(reconstructed, dim=-1).T / temperature
                    cl = F.cross_entropy(logits, torch.arange(len(batch), device=device))
                    loss = loss + float(config.get("contrastive_weight", 1.0))*cl
                    components["contrastive"] = cl
                else:
                    loss = loss + reconstruction_loss
                if name == "letter":
                    projected = model.cf_projection(cf[ix])
                    logits = cumulative[-1] @ projected.T
                    cf_loss = -logits.softmax(-1).diagonal().mean()
                    div_loss = batch.new_zeros(())
                    for level, book in enumerate(model.books):
                        labels = diversity_groups[level]
                        positives, anchors = [], []
                        for code in codes[:, level].detach().cpu().tolist():
                            peers = [j for j in range(k) if labels[j] == labels[code] and j != code]
                            if peers:
                                anchors.append(code)
                                positives.append(peers[int(rng.integers(len(peers)))])
                        if anchors:
                            sim = book[anchors] @ book.T
                            sim = sim.masked_fill(F.one_hot(torch.tensor(anchors, device=device), k).bool(), -1e9)
                            div_loss = div_loss - sim.softmax(-1)[torch.arange(len(anchors), device=device), positives].mean()
                    loss = loss + float(config.get("collaborative_weight", .02))*cf_loss
                    loss = loss + float(config.get("diversity_weight", .001))*div_loss
                    components.update(collaborative=cf_loss, diversity=div_loss)
                if name == "topotok":
                    topo = batch.new_zeros(())
                    first = codes[:, 0]
                    groups = torch.unique(first)
                    teachers = torch.stack([batch[first == g].mean(0) for g in groups])
                    students = model.books[0][groups]
                    topo = topo + distill(teachers, students)
                    for level in range(1, depth):
                        mask = None
                        if level == 1 and depth > 2:
                            previous = codes[:, level-1]
                            mask = previous[:, None] == previous[None, :]
                        topo = topo + distill(batch, cumulative[level], mask)
                    if depth == 1:
                        topo = topo + distill(batch, cumulative[0])
                    loss = loss + float(config.get("topology_weight", 1.0))*topo
                    components["topology"] = topo
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for key, value in dict(components, total=loss).items():
                    sums[key] += float(value.detach())*len(ix)
            history.append({key: value/len(x) for key, value in sums.items()})
        model.eval()
        with torch.no_grad():
            columns = [model(data[i:i+batch_size])[2].cpu().numpy()
                       for i in range(0, len(data), batch_size)]
        torch.save({"state_dict": model.state_dict(), "method": name,
                    "input_dim": x.shape[1], "collaborative_dim": collaborative.shape[1],
                    "latent_dim": latent, "hidden_dim": hidden, "depth": depth,
                    "codebook_size": k, "config": config}, output_dir / f"{name}_model.pt")
    return np.concatenate(columns), {"training_history": history, "device": str(device)}


def _unique(ids, raw):
    rows = [tuple(int(t) for t in row) for row in raw]
    groups = defaultdict(list)
    for item_id, row in zip(ids, rows):
        groups[row].append(item_id)
    next_token = max(t for row in rows for t in row) + 1
    result = {}
    for row, members in sorted(groups.items()):
        for rank, item_id in enumerate(sorted(members)):
            result[item_id] = row if len(members) == 1 else row + (next_token+rank,)
    return result, sum(len(v) for v in groups.values() if len(v) > 1)


def build_codes(name: str, items: list[dict], semantic: np.ndarray,
                collaborative: np.ndarray, config: dict, seed: int,
                output_dir: Path) -> dict[str, tuple[int, ...]]:
    """Fit one tokenizer and save mapping, model (where relevant) and manifest.

    `native_mapping` accepts a dict or JSON file mapping item IDs to integer
    arrays. Set `mode='native'` to require such a mapping (no silent fallback).
    Rows of both embedding matrices must match the input item order.
    """
    name = name.lower()
    if name not in TOKENIZERS:
        raise ValueError(f"unknown tokenizer {name!r}; choose from {TOKENIZERS}")
    if not items:
        raise ValueError("catalog cannot be empty")
    ids = [str(item["id"]) for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("item IDs must be unique (after conversion to string)")
    config = dict(config or {})
    branching, depth, k = (int(config.get(key, default)) for key, default in
                           (("branching", 8), ("depth", 3), ("codebook_size", 16)))
    if branching < 2 or depth < 1 or k < 1:
        raise ValueError("branching >= 2, depth >= 1, codebook_size >= 1 required")
    semantic = _normalize(semantic, len(items), "semantic")
    collaborative = _normalize(collaborative, len(items), "collaborative")
    # Stable item order makes all fit/assignment ties reproducible across input permutations.
    order = np.argsort(np.asarray(ids), kind="stable")
    ids, items = [ids[i] for i in order], [items[i] for i in order]
    semantic, collaborative = semantic[order], collaborative[order]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    native = config.get("native_mapping")
    mode = config.get("mode", "controlled")
    if mode not in ("controlled", "native"):
        raise ValueError("mode must be controlled or native")
    if mode == "native" and native is None:
        raise ValueError(f"{name}: mode=native requires native_mapping exported from the original implementation")
    with threadpool_limits(limits=int(config.get("threads", 1))):
        if native is not None:
            if isinstance(native, (str, Path)):
                native = json.loads(Path(native).read_text(encoding="utf-8"))
            if set(native) != set(ids):
                raise ValueError("native_mapping IDs must exactly match catalog IDs")
            raw = [native[i] for i in ids]
            if any(not isinstance(r, (list, tuple)) or not r or
                   any(not isinstance(t, (int, np.integer)) or isinstance(t, bool) or t < 0 for t in r) for r in raw):
                raise ValueError("native_mapping values must be non-empty sequences of nonnegative integer tokens")
            mode = "native_import"
        elif name in ("semantic_hierarchical", "collaborative_hierarchical", "seater"):
            x = collaborative if name == "collaborative_hierarchical" else semantic
            raw = _hierarchical(x, branching, depth, seed, name == "seater",
                                int(config.get("balance_exact_limit", 1500)))
            if name == "seater":
                metadata["scope"] = "balanced semantic tree only; no joint recommender/contrastive training"
        elif name == "rqkmeans":
            raw, books = _residual_kmeans(semantic, depth, k, seed)
            np.savez(output_dir / "rqkmeans_model.npz", **{f"codebook_{i}": b for i, b in enumerate(books)})
        elif name == "opq":
            raw, metadata = _opq(semantic, depth, k, seed, config, output_dir)
        elif name in ("rqvae", "letter", "cost", "topotok"):
            raw, metadata = _neural(name, semantic, collaborative, depth, k, config, seed, output_dir)
        elif name == "random_balanced":
            raw = _random_balanced(len(items), branching, depth, seed)
        elif name == "attribute_first":
            raw, metadata = _attribute_first(items, config)
        else:
            raw, metadata = _bonsai(items, config)
    mapping, collisions = _unique(ids, raw)
    manifest = {"name": name, "mode": mode, "seed": int(seed), "item_count": len(ids),
                "branching": branching, "depth": depth, "codebook_size": k,
                "collision_items": collisions, "length_min": min(map(len, mapping.values())),
                "length_max": max(map(len, mapping.values())), "eos_added": False,
                "config": config, "details": metadata,
                "reproduction_note": "See docs/TOKENIZERS.md; controlled adaptations are not full paper reproductions."}
    for filename, value in (("mapping.json", mapping), ("tokenizer_manifest.json", manifest)):
        destination = output_dir / filename
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, destination)
    return mapping
