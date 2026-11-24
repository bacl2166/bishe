"""Enumerated path scores and budgeted decoding share the same edge probabilities."""

from collections import defaultdict
from dataclasses import dataclass
from math import log

import numpy as np


EOS = -1


class Tree:
    def __init__(self, codes):
        if not codes or len(set(codes.values())) != len(codes):
            raise ValueError('Codes must cover a nonempty catalog with unique paths.')
        self.ids = list(codes)
        self.paths = {}
        self.children = defaultdict(list)
        self.descendants = defaultdict(list)
        self.leaves = {}
        for index, (item, code) in enumerate(codes.items()):
            if not code or any(int(token) < 0 for token in code):
                raise ValueError('Codes must be nonempty nonnegative integer sequences.')
            path = tuple(map(int, code)) + (EOS,)
            self.paths[item] = path
            self.leaves[path] = item
            for depth in range(len(path) + 1):
                prefix = path[:depth]
                self.descendants[prefix].append(index)
                if depth and prefix not in self.children[prefix[:-1]]:
                    self.children[prefix[:-1]].append(prefix)
        for children in self.children.values():
            children.sort()


@dataclass
class Distribution:
    tree: Tree
    feasible: set
    edges: dict
    cumulative: dict
    log_scores: np.ndarray
    q: np.ndarray
    root_kl: float | None
    root_invalid_mass: float | None

    def top(self, k):
        indices = [i for i, item in enumerate(self.tree.ids) if item in self.feasible]
        indices.sort(key=lambda i: (-self.log_scores[i], self.tree.ids[i]))
        return [self.tree.ids[i] for i in indices[:k]]


def prepare_distribution(tree, q, feasible, mode):
    if mode not in ('preserve', 'local', 'feasible_mass'):
        raise ValueError(f'Unknown probability mode {mode}')
    q = np.asarray(q, dtype=np.float64)
    if len(q) != len(tree.ids) or not np.all(np.isfinite(q)) or np.any(q <= 0):
        raise ValueError('Each catalog item needs a finite positive probability.')
    q = q / q.sum()
    feasible = set(feasible)
    if not feasible <= set(tree.ids):
        raise ValueError('Feasible IDs must belong to the same catalog.')
    is_valid = np.array([item in feasible for item in tree.ids])
    full_mass = {node: float(q[indices].sum()) for node, indices in tree.descendants.items()}
    available_mass = {
        node: float(q[indices][is_valid[indices]].sum()) for node, indices in tree.descendants.items()
    }
    edges = {}
    cumulative = {(): 0.0}
    local_correction = {(): 0.0}
    for parent in sorted(tree.children, key=lambda node: (len(node), node)):
        valid = [node for node in tree.children[parent] if available_mass[node] > 0]
        if not valid:
            continue
        if mode == 'feasible_mass':
            denominator = available_mass[parent]
        elif mode == 'local':
            denominator = sum(full_mass[node] for node in valid)
        else:
            denominator = full_mass[parent]
        correction = 0.0
        if mode == 'local' and len(valid) != len(tree.children[parent]):
            correction = log(full_mass[parent]) - log(denominator)
        for child in valid:
            numerator = available_mass[child] if mode == 'feasible_mass' else full_mass[child]
            edges[child] = log(numerator) - log(denominator)
            local_correction[child] = local_correction[parent] + correction
            if mode == 'feasible_mass':
                cumulative[child] = log(available_mass[child]) - log(available_mass[()])
            elif mode == 'local':
                cumulative[child] = log(full_mass[child]) + local_correction[child]
            else:
                cumulative[child] = log(full_mass[child])
    log_scores = np.array([cumulative.get(tree.paths[item], -np.inf) for item in tree.ids])
    # Telescoping has a closed form in these modes. Using it at leaves keeps
    # mathematically tied items tied even when path depths and sums differ.
    if feasible and mode in ('preserve', 'feasible_mass'):
        correction = log(float(q[is_valid].sum())) if mode == 'feasible_mass' else 0.0
        for index in np.flatnonzero(is_valid):
            value = log(float(q[index])) - correction
            log_scores[index] = value
            cumulative[tree.paths[tree.ids[index]]] = value
    roots = [node for node in tree.children[()] if available_mass[node] > 0]
    root_kl = None
    invalid_mass = None
    if roots:
        ideal = np.array([available_mass[node] for node in roots])
        ideal /= ideal.sum()
        local = np.array([full_mass[node] for node in roots])
        local /= local.sum()
        root_kl = float(np.sum(ideal * (np.log(ideal) - np.log(local))))
        invalid_mass = float(sum(full_mass[node] - available_mass[node] for node in roots))
    return Distribution(tree, feasible, edges, cumulative, log_scores, q, root_kl, invalid_mass)


@dataclass
class SearchResult:
    items: list
    scored_nodes: int
    expanded_nodes: int
    budget_exhausted: bool
    pruned: dict


def search(dist, k, beam_width, node_budget=None, tracked=None):
    """Beam limits unfinished prefixes; completed leaves enter a separate Top-K pool.

    A layer is scored atomically. If it exceeds the remaining node budget,
    expansion stops before that layer, making node_budget a hard ceiling.
    """
    if k < 1 or beam_width < 1 or (node_budget is not None and node_budget < 1):
        raise ValueError('Search counts must be positive.')
    frontier = [()]
    completed = set()
    scored = expanded = 0
    exhausted = False
    pruned = {}
    tracked = set(dist.top(k) if tracked is None else tracked) & dist.feasible
    while frontier:
        children = [
            child for parent in frontier for child in dist.tree.children[parent] if child in dist.edges
        ]
        if node_budget is not None and scored + len(children) > node_budget:
            exhausted = True
            break
        expanded += len(frontier)
        scored += len(children)
        candidates = []
        for child in children:
            if child in dist.tree.leaves:
                completed.add(dist.tree.leaves[child])
            else:
                candidates.append(child)
        candidates.sort(key=lambda node: (-dist.cumulative[node], node))
        frontier = candidates[:beam_width]
        discarded = set(candidates[beam_width:])
        for item in tracked - pruned.keys():
            path = dist.tree.paths[item]
            for depth in range(1, len(path)):
                if path[:depth] in discarded:
                    pruned[item] = depth
                    break
    index = {item: i for i, item in enumerate(dist.tree.ids)}
    ordered = sorted(completed, key=lambda item: (-dist.log_scores[index[item]], item))[:k]
    return SearchResult(ordered, scored, expanded, exhausted, pruned)
