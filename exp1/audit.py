"""Audit constraint support without fitting models: python -m exp1.audit --run RUN."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from .constraints import build_requests


def summarize_requests(requests, min_users=100, min_positive_users=30):
    """Count independent users within each rule stratum, keeping empty requests."""
    if min_users < 1 or min_positive_users < 1:
        raise ValueError('Planning thresholds must be positive integers.')
    lookup = {row['id']: row for row in requests}
    if len(lookup) != len(requests):
        raise ValueError('Request IDs must be unique.')
    groups = defaultdict(list)
    for row in requests:
        matched_family = ''
        if row['family'] == 'random_matched':
            source = lookup.get(row.get('matched_request'))
            if source is None or source['family'] == 'random_matched':
                raise ValueError('Random control needs a matched structural request.')
            if source['user'] != row['user'] or len(source['feasible']) != len(row['feasible']):
                raise ValueError('Random matched control must share user and candidate count.')
            matched_family = source['family']
        key = (row['family'], row.get('strength') or '', row.get('variant') or '', matched_family)
        groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        users = {row['user'] for row in rows}
        positive = [row for row in rows if row['positive_feasible']]
        positive_users = {row['user'] for row in positive}
        nonempty_users = {row['user'] for row in rows if row['feasible']}
        counts = [len(row['feasible']) for row in rows]
        if not positive_users:
            support = 'no_feasible_positive'
        elif len(users) < min_users or len(positive_users) < min_positive_users:
            support = 'limited'
        else:
            support = 'meets_planning_count'
        output.append({
            **dict(zip(('family', 'strength', 'variant', 'matched_family'), key)),
            'requests': len(rows), 'unique_users': len(users),
            'positive_in_catalog_users': len({row['user'] for row in rows if row.get('positive_in_catalog')}),
            'positive_feasible_requests': len(positive), 'positive_feasible_users': len(positive_users),
            'nonempty_users': len(nonempty_users),
            'empty_feasible_rate': sum(count == 0 for count in counts) / len(counts),
            'candidate_mean': mean(counts), 'candidate_median': median(counts),
            'candidate_min': min(counts), 'candidate_max': max(counts),
            'mechanism_support': ('meets_planning_count' if len(nonempty_users) >= min_users else 'limited'),
            'recommendation_support': support,
        })
    return output


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(folder, split='validation', min_users=100, min_positive_users=30):
    if split not in ('validation', 'test'):
        raise ValueError('Audit split must be validation or test.')
    folder = Path(folder)
    manifest, data = _read(folder / 'manifest.json'), _read(folder / 'prepared.json')
    config = dict(manifest['config'])
    config['constraints'] = {**config.get('constraints', {}), 'request_split': split}
    # Reconstruct under the frozen catalog/cohort/config, without changing
    # source files or selecting requests by their positive item.
    requests = build_requests(data, config)
    groups = summarize_requests(requests, min_users, min_positive_users)
    constraint_audit = data['audit']['constraints']
    labels = data.get(split, [])
    catalog_ids = {item['id'] for item in data['items']}
    report = {
        'run': str(folder), 'source_fingerprint': manifest['fingerprint'], 'split': split,
        'planning_thresholds': {'min_users': min_users, 'min_positive_users': min_positive_users},
        'interpretation': 'Descriptive planning counts, not statistical power guarantees; never filter users by positives.',
        'catalog_size': len(data['items']), 'evaluation_users': len(data['histories']),
        'labels': len(labels),
        'catalog_coverage': (sum(row['positive'] in catalog_ids for row in labels) / len(labels) if labels else None),
        'requests': len(requests), 'request_users': constraint_audit['unique_users'],
        'positive_feasible_requests': constraint_audit['positive_feasible_requests'],
        'positive_feasible_users': constraint_audit['positive_feasible_users'],
        'skipped_families': constraint_audit['skipped_families'], 'groups': groups,
    }
    target = folder / f'constraint-support-{split}.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, target)
    csv_path = target.with_suffix('.csv')
    temporary = csv_path.with_suffix('.csv.tmp')
    fields = list(groups[0]) if groups else ['family', 'recommendation_support']
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(groups)
    os.replace(temporary, csv_path)
    print(f"{split}: {len(data['items'])} items, {len(labels)} labels, "
          f"{len(requests)} requests, {constraint_audit['unique_users']} request users", flush=True)
    print(f"Positive-feasible independent users: {constraint_audit['positive_feasible_users']}", flush=True)
    for row in groups:
        print(f"{row['family']}/{row['strength']}/{row['variant']}/{row['matched_family']}: "
              f"users={row['unique_users']}, positive_users={row['positive_feasible_users']}, "
              f"empty={row['empty_feasible_rate']:.1%}, {row['recommendation_support']}", flush=True)
    if not groups:
        print('No requests: inspect label availability, budget and skipped families.', flush=True)
    print(f'Saved {target} and {csv_path}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--min-users', type=int, default=100)
    parser.add_argument('--min-positive-users', type=int, default=30)
    args = parser.parse_args()
    run(args.run, args.split, args.min_users, args.min_positive_users)


if __name__ == '__main__':
    main()
