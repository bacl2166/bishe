"""Exact and budgeted decoding of actual neural token probabilities.

Codes contain their terminal EOS already. No probability is reconstructed from
item scores or tree masses: every edge comes from ``generator.log_probs`` over
the full vocabulary. Completed paths use summed log probability without length
normalization and remain outside the beam of unfinished prefixes.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np


EOS = 2


class Tree:
    def __init__(self, token_codes):
        if not token_codes or any(not isinstance(item, str) for item in token_codes):
            raise ValueError('Token codes require a nonempty catalog of string item IDs.')
        self.ids = sorted(token_codes)
        self.paths = {}
        self.children = defaultdict(list)
        self.descendants = defaultdict(list)
        self.leaves = {}
        for index, item in enumerate(self.ids):
            path = tuple(token_codes[item])
            if (not path or path[-1] != EOS or
                    any(not isinstance(token, Integral) or isinstance(token, bool) for token in path) or
                    any(token < 3 for token in path[:-1])):
                raise ValueError('Each path needs code tokens >= 3 followed by exactly one EOS=2.')
            path = tuple(map(int, path))
            if path in self.leaves:
                raise ValueError('Item token paths must be unique.')
            self.paths[item] = path
            self.leaves[path] = item
            for depth in range(len(path) + 1):
                prefix = path[:depth]
                self.descendants[prefix].append(index)
                if depth and prefix not in self.children[prefix[:-1]]:
                    self.children[prefix[:-1]].append(prefix)
        for children in self.children.values():
            children.sort()


def _legal_children(tree, feasible, mode, batch_size):
    if mode not in ('preserve', 'local'):
        raise ValueError(f'Unknown decoding mode: {mode}')
    if not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer.')
    feasible = set(feasible)
    if not feasible <= set(tree.ids):
        raise ValueError('Feasible IDs must belong to the token catalog.')
    legal = defaultdict(set)
    for item in sorted(feasible):
        path = tree.paths[item]
        for depth in range(len(path)):
            legal[path[:depth]].add(path[:depth + 1])
    return feasible, {parent: sorted(children) for parent, children in legal.items()}


def _transitions(generator, history, parents, legal, mode, batch_size):
    """Shared edge computation for exact enumeration and beam search."""
    edges = {}
    for start in range(0, len(parents), batch_size):
        prefixes = parents[start:start + batch_size]
        values = np.asarray(generator.log_probs(history, prefixes), dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(prefixes) or not np.isfinite(values).all():
            raise ValueError('Generator must return finite batch-by-vocabulary log probabilities.')
        for row, prefix in enumerate(prefixes):
            children = legal[prefix]
            tokens = [child[-1] for child in children]
            if max(tokens) >= values.shape[1]:
                raise ValueError('A catalog code token lies outside the generator vocabulary.')
            scores = values[row, tokens]
            if mode == 'local':
                # Subtract max first, avoiding cancellation for very negative logits.
                scores = scores - scores.max()
                scores = scores - np.log(np.exp(scores).sum())
            edges.update((child, float(score)) for child, score in zip(children, scores))
    return edges


def exact_scores(generator, tree, history, feasible, mode='preserve', batch_size=256):
    """Enumerate all feasible paths, querying every unique nonterminal prefix once."""
    feasible, legal = _legal_children(tree, feasible, mode, batch_size)
    cumulative = {(): 0.}
    frontier = [()] if feasible else []
    while frontier:
        edges = _transitions(generator, history, frontier, legal, mode, batch_size)
        next_frontier = []
        for parent in frontier:
            for child in legal[parent]:
                cumulative[child] = cumulative[parent] + edges[child]
                if child not in tree.leaves:
                    next_frontier.append(child)
        frontier = next_frontier
    return {item: cumulative[tree.paths[item]] for item in sorted(feasible)}


@dataclass
class SearchResult:
    items: list
    log_scores: dict
    scored_nodes: int
    expanded_nodes: int
    budget_exhausted: bool
    pruned: dict = field(default_factory=dict)


def search(generator, tree, history, feasible, mode, k, beam_width,
           node_budget=None, batch_size=256):
    """Score layers atomically under a strict candidate-child node budget.

    The budget is checked before neural calls for a layer. EOS completions join
    an independent pool, so a short completed item cannot evict an unfinished
    path or be lost while a longer path is expanded.
    """
    feasible, legal = _legal_children(tree, feasible, mode, batch_size)
    if any(not isinstance(value, Integral) or value < 1 for value in (k, beam_width)):
        raise ValueError('k and beam_width must be positive integers.')
    if node_budget is not None and (not isinstance(node_budget, Integral) or node_budget < 0):
        raise ValueError('node_budget must be a nonnegative integer or None.')
    frontier = [()] if feasible else []
    cumulative = {(): 0.}
    completed, pruned = {}, {}
    scored = expanded = 0
    exhausted = False
    # Equal-score prefixes use their first feasible item ID as a stable tie break.
    first_item = {}
    for item in sorted(feasible):
        path = tree.paths[item]
        for depth in range(len(path)):
            first_item.setdefault(path[:depth], item)
    while frontier:
        layer_size = sum(len(legal[parent]) for parent in frontier)
        if node_budget is not None and scored + layer_size > node_budget:
            exhausted = True
            break
        edges = _transitions(generator, history, frontier, legal, mode, batch_size)
        scored += layer_size
        expanded += len(frontier)
        candidates = []
        for parent in frontier:
            for child in legal[parent]:
                cumulative[child] = cumulative[parent] + edges[child]
                if child in tree.leaves:
                    completed[tree.leaves[child]] = cumulative[child]
                else:
                    candidates.append(child)
        candidates.sort(key=lambda node: (-cumulative[node], first_item[node], node))
        frontier = candidates[:beam_width]
        for discarded in candidates[beam_width:]:
            for index in tree.descendants[discarded]:
                item = tree.ids[index]
                if item in feasible:
                    pruned.setdefault(item, len(discarded))
    items = sorted(completed, key=lambda item: (-completed[item], item))[:k]
    return SearchResult(items, {item: completed[item] for item in items},
                        scored, expanded, exhausted, pruned)
