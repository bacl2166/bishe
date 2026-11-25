"""Recommendation metrics on all requests; exact-search diagnostics on a fixed subset."""

import hashlib

import numpy as np

from exp1.metrics import label_metrics, overlap


def select_diagnostics(requests, count, seed):
    """Select by immutable request ID, independently of labels and model results."""
    if count < 0:
        raise ValueError('diagnostic_requests must be nonnegative; 0 means all requests.')
    ids = [r['id'] for r in requests]
    if len(ids) != len(set(ids)):
        raise ValueError('Request IDs must be unique.')
    ordered = sorted(ids, key=lambda key: (hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest(), key))
    return set(ordered[:count] if count else ordered)


def _normalized_logs(scores, ids):
    values = np.array([scores[item] for item in ids], dtype=np.float64)
    values -= values.max()
    return values - np.log(np.exp(values).sum())


def evaluate(found, positive, feasible, k, original_scores=None, own_scores=None):
    feasible = set(feasible)
    actual = list(found.items)
    if len(actual) != len(set(actual)) or not set(actual) <= feasible:
        raise ValueError('Search must return unique feasible catalog items.')
    hr, ndcg = label_metrics(actual, positive, feasible)
    diagnostics = original_scores is not None and own_scores is not None
    if (original_scores is None) != (own_scores is None):
        raise ValueError('Both original and own exact scores must be supplied together.')
    row = {
        'hr': hr, 'ndcg': ndcg, 'diagnostic': int(diagnostics),
        'positive_feasible': int(positive in feasible),
        'compliance': 1.0 if actual else None, 'returned': len(actual),
        'shortfall': int(len(actual) < min(k, len(feasible))),
        'candidate_shortage': int(len(feasible) < k), 'empty_feasible': int(not feasible),
        'scored_nodes': found.scored_nodes, 'expanded_nodes': found.expanded_nodes,
        'budget_exhausted': int(found.budget_exhausted),
    }
    for field in ('exact_hr', 'exact_ndcg', 'own_exact_hr', 'own_exact_ndcg',
                  'oracle_recall', 'own_oracle_recall', 'objective_topk_overlap',
                  'distribution_kl', 'lost_hit', 'gained_hit', 'own_lost_hit',
                  'objective_lost_hit', 'search_score_loss', 'own_search_score_loss',
                  'positive_original_rank', 'positive_own_rank'):
        row[field] = None
    row.update(reference=None, own_reference=None)
    if not diagnostics:
        return row
    for scores in (original_scores, own_scores):
        if set(scores) != feasible or not all(np.isfinite(v) for v in scores.values()):
            raise ValueError('Exact scores must cover every feasible item with finite log scores.')
    raw = sorted(feasible, key=lambda item: (-original_scores[item], item))
    own = sorted(feasible, key=lambda item: (-own_scores[item], item))
    reference, own_reference = raw[:k], own[:k]
    exact_hr, exact_ndcg = label_metrics(reference, positive, feasible)
    own_hr, own_ndcg = label_metrics(own_reference, positive, feasible)
    row.update(
        reference=reference, own_reference=own_reference,
        exact_hr=exact_hr, exact_ndcg=exact_ndcg, own_exact_hr=own_hr, own_exact_ndcg=own_ndcg,
        oracle_recall=overlap(actual, reference), own_oracle_recall=overlap(actual, own_reference),
        objective_topk_overlap=overlap(own_reference, reference),
        lost_hit=int(exact_hr == 1 and hr == 0) if hr is not None else None,
        gained_hit=int(exact_hr == 0 and hr == 1) if hr is not None else None,
        own_lost_hit=int(own_hr == 1 and hr == 0) if hr is not None else None,
        objective_lost_hit=int(exact_hr == 1 and own_hr == 0) if hr is not None else None,
        positive_original_rank=raw.index(positive) + 1 if positive in feasible else None,
        positive_own_rank=own.index(positive) + 1 if positive in feasible else None,
    )
    if feasible:
        ids = sorted(feasible)
        log_raw, log_own = _normalized_logs(original_scores, ids), _normalized_logs(own_scores, ids)
        row['distribution_kl'] = max(0., float(np.sum(np.exp(log_raw) * (log_raw - log_own))))
    # Compare lists of equal length; report output shortage separately. This does
    # not assign an arbitrary finite score to a missing recommendation.
    n = min(k, len(actual))
    if n:
        row['search_score_loss'] = float(np.mean([original_scores[x] for x in raw[:n]]) -
                                         np.mean([original_scores[x] for x in actual[:n]]))
        row['own_search_score_loss'] = float(np.mean([own_scores[x] for x in own[:n]]) -
                                             np.mean([own_scores[x] for x in actual[:n]]))
    return row
