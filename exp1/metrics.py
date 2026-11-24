"""Item-level metrics with explicit feasibility and reference denominators."""

from collections import defaultdict
from math import log2

import numpy as np


def overlap(results, reference):
    return len(set(results) & set(reference)) / len(reference) if reference else None


def label_metrics(results, positive, feasible):
    if positive not in feasible:
        return None, None
    if positive not in results:
        return 0, 0.0
    return 1, 1.0 / log2(results.index(positive) + 2)


def measure(dist, original_reference, result, positive, k):
    own_reference = dist.top(k)
    hr, ndcg = label_metrics(result.items, positive, dist.feasible)
    exact_hr, exact_ndcg = label_metrics(original_reference, positive, dist.feasible)
    own_hr, own_ndcg = label_metrics(own_reference, positive, dist.feasible)
    index = {item: i for i, item in enumerate(dist.tree.ids)}
    valid_indices = [index[item] for item in dist.feasible]
    kl = None
    if valid_indices:
        q = dist.q[valid_indices]
        q = q / q.sum()
        log_own = dist.log_scores[valid_indices]
        log_own -= np.max(log_own)
        log_own -= np.log(np.exp(log_own).sum())
        kl = float(np.sum(q * (np.log(q) - log_own)))
    return {
        'hr': hr, 'ndcg': ndcg,
        'exact_hr': exact_hr, 'exact_ndcg': exact_ndcg,
        'own_exact_hr': own_hr, 'own_exact_ndcg': own_ndcg,
        'oracle_recall': overlap(result.items, original_reference),
        'own_oracle_recall': overlap(result.items, own_reference),
        'objective_topk_overlap': overlap(own_reference, original_reference),
        'distribution_kl': kl,
        'root_kl': dist.root_kl, 'root_invalid_mass': dist.root_invalid_mass,
        'lost_hit': int(exact_hr == 1 and hr == 0),
        'gained_hit': int(exact_hr == 0 and hr == 1),
        'own_lost_hit': int(own_hr == 1 and hr == 0),
        'objective_lost_hit': int(exact_hr == 1 and own_hr == 0),
        'compliance': (sum(item in dist.feasible for item in result.items) / len(result.items)
                       if result.items else None),
        'returned': len(result.items),
        'shortfall': int(len(result.items) < min(k, len(dist.feasible))),
        'candidate_shortage': int(len(dist.feasible) < k),
        'empty_feasible': int(not dist.feasible),
        'positive_feasible': int(positive in dist.feasible),
        'scored_nodes': result.scored_nodes, 'expanded_nodes': result.expanded_nodes,
        'budget_exhausted': int(result.budget_exhausted),
    }


GROUP_FIELDS = ('scorer', 'scorer_seed', 'encoding', 'encoding_seed', 'family', 'strength', 'variant', 'matched_family',
                'mode', 'beam', 'node_budget')


def summarize(rows, bootstrap_samples=1000, seed=2026):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in GROUP_FIELDS)].append(row)
    outputs = []
    for key, group in groups.items():
        result = dict(zip(GROUP_FIELDS, key))
        result['requests'] = len(group)
        numeric = [name for row in group for name, value in row.items()
                   if isinstance(value, (int, float)) and name not in GROUP_FIELDS]
        numeric += ['hr', 'ndcg', 'exact_hr', 'exact_ndcg', 'own_exact_hr', 'own_exact_ndcg']
        for name in dict.fromkeys(numeric):
            values = [row[name] for row in group if row.get(name) is not None]
            result[name] = float(np.mean(values)) if values else None
        denominator = sum(row['exact_hr'] == 1 for row in group)
        own_denominator = sum(row['own_exact_hr'] == 1 for row in group)
        result['evaluated_positives'] = sum(row['positive_feasible'] for row in group)
        result['exact_hits'] = denominator
        result['lost_hits'] = sum(row['lost_hit'] for row in group)
        result['gained_hits'] = sum(row['gained_hit'] for row in group)
        result['lost_hit_rate'] = result['lost_hits'] / denominator if denominator else None
        result['own_lost_hit_rate'] = (sum(row['own_lost_hit'] for row in group) / own_denominator
                                       if own_denominator else None)
        # Pair the same user's actual NDCG with its original exact NDCG.
        user_differences = defaultdict(list)
        for row in group:
            if row['ndcg'] is not None:
                user_differences[row['user']].append(row['ndcg'] - row['exact_ndcg'])
        per_user = np.array([np.mean(values) for values in user_differences.values()])
        result['user_mean_ndcg_delta'] = float(per_user.mean()) if len(per_user) else None
        if len(per_user) and bootstrap_samples:
            rng = np.random.default_rng(seed)
            samples = per_user[rng.integers(len(per_user), size=(bootstrap_samples, len(per_user)))].mean(axis=1)
            result['ndcg_delta_ci95'] = np.quantile(samples, [.025, .975]).tolist()
        else:
            result['ndcg_delta_ci95'] = None
        outputs.append(result)
    return outputs


def paired_encoding_comparisons(rows, baseline='random_balanced', samples=1000, seed=2026):
    """Pair the same requests and users, with reference construction held fixed."""
    strata = defaultdict(dict)
    fields = ('scorer', 'scorer_seed', 'encoding_seed', 'family', 'strength', 'variant', 'matched_family', 'mode', 'beam', 'node_budget')
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        strata[key].setdefault(row['encoding'], {})[row['request_id']] = row
    outputs = []
    for key, arms in strata.items():
        if baseline not in arms:
            continue
        for encoding, values in arms.items():
            if encoding == baseline:
                continue
            for metric in ('ndcg', 'hr', 'oracle_recall'):
                users = defaultdict(list)
                for request_id in sorted(values.keys() & arms[baseline].keys()):
                    left, right = values[request_id], arms[baseline][request_id]
                    if left.get(metric) is not None and right.get(metric) is not None:
                        users[left['user']].append(left[metric] - right[metric])
                differences = np.array([np.mean(v) for v in users.values()])
                if not len(differences):
                    continue
                rng = np.random.default_rng(seed)
                draws = (differences[rng.integers(len(differences), size=(samples, len(differences)))].mean(axis=1)
                         if samples else [])
                outputs.append({
                    **dict(zip(fields, key)), 'encoding': encoding, 'baseline': baseline,
                    'metric': metric, 'users': len(users), 'user_mean_difference': float(differences.mean()),
                    'ci95': np.quantile(draws, [.025, .975]).tolist() if len(draws) else None,
                })
    return outputs
