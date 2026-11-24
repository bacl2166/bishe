"""Label-independent constraint generation and one shared predicate implementation.

Rule dimensions combine with AND; a dimension's alternatives combine with OR.
Missing attributes never satisfy a constraint on that attribute. Random controls
use exact-size uniform subsets, independently seeded from structural rules.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from random import Random


def matches(item: dict, rule: dict) -> bool:
    if rule.get('type') == 'ids':
        return item['id'] in rule['ids']
    price = rule.get('price')
    if price is not None:
        value = item.get('price')
        if value is None or value < price.get('min', 0) or value > price.get('max', float('inf')):
            return False
    if 'brand' in rule and item.get('brand') not in rule['brand']:
        return False
    if 'category' in rule and not set(item.get('category', [])).intersection(rule['category']):
        return False
    if 'rating' in rule:
        rating = rule['rating']
        mean = item.get('rating_mean')
        if mean is None or mean < rating['min_mean'] or item.get('rating_count', 0) < rating['min_count']:
            return False
    for field, alternatives in rule.get('spec', {}).items():
        if item.get('specs', {}).get(field) not in alternatives:
            return False
    return True


def feasible_ids(items, rule):
    return [item['id'] for item in items if matches(item, rule)]


# Public aliases are useful to scorers that evaluate an individual catalog entry.
match_rule = matches


def _quantile(values, q):
    values = sorted(values)
    point = (len(values) - 1) * q
    low = int(point)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (point - low)


def _rng(seed, *keys):
    digest = hashlib.sha256(json.dumps([seed, *keys], ensure_ascii=False, sort_keys=True).encode()).digest()
    return Random(int.from_bytes(digest[:16], 'big'))


def _weighted_choices(counter, n, rng):
    # Stable weighted sampling without replacement; never inspect labels.
    remaining = sorted(counter)
    selected = []
    while remaining and len(selected) < n:
        value = rng.choices(remaining, weights=[counter[v] for v in remaining], k=1)[0]
        selected.append(value)
        remaining.remove(value)
    return sorted(selected)


def build_requests(data: dict, config: dict) -> list[dict]:
    cfg = {**config, **config.get('constraints', {})}
    items = data['items']
    by_id = {item['id']: item for item in items}
    item_ids = sorted(by_id)
    seed = cfg.get('request_seed', 17)
    families = cfg.get('constraint_families', ['no_filter', 'price', 'brand', 'category', 'rating', 'spec',
                                            'price_brand', 'price_brand_spec', 'random_matched'])
    families = list(dict.fromkeys(families))
    prices = [i['price'] for i in items if i.get('price') is not None]
    brands, categories = Counter(), Counter()
    spec_counts = {}
    for item in items:
        weight = item.get('train_count', 1)
        if item.get('brand'):
            brands[item['brand']] += weight
        # Use leaf categories to avoid a trivial domain-wide main category rule.
        if item.get('category'):
            categories[item['category'][-1]] += weight
        for field, value in item.get('specs', {}).items():
            if value:
                spec_counts.setdefault(field, Counter())[value] += weight
    requested_field = cfg.get('spec_field')
    spec_field = requested_field or (max(sorted(spec_counts), key=lambda f: sum(1 for i in items if i.get('specs', {}).get(f))) if spec_counts else None)
    specs = spec_counts.get(spec_field, Counter())
    price_quartiles = [int(round(_quantile(prices, q))) for q in (.25, .5, .75)] if prices else []
    rating_means = [i['rating_mean'] for i in items if i.get('rating_mean') is not None]
    rating_quartiles = [_quantile(rating_means, q) for q in (.25, .5, .75)] if rating_means else []
    rating_min_count = int(cfg.get('rating_min_count', 1))
    templates, skipped = [], []
    supported = {'no_filter', 'price', 'price_max', 'price_interval', 'brand', 'category', 'rating',
                 'spec', 'price_brand', 'price_brand_spec', 'random_matched'}
    for family in families:
        if family not in supported:
            skipped.append({'family': family, 'reason': 'unsupported constraint family'})
            continue
        if family == 'random_matched':
            continue
        prerequisites = []
        if family in ('price', 'price_max', 'price_interval', 'price_brand', 'price_brand_spec') and not prices:
            prerequisites.append('no known training-catalog prices')
        if family in ('brand', 'price_brand', 'price_brand_spec') and not brands:
            prerequisites.append('no known training-catalog brands')
        if family == 'category' and not categories:
            prerequisites.append('no known training-catalog categories')
        if family == 'rating' and not rating_means:
            prerequisites.append('no training review ratings')
        if family in ('spec', 'price_brand_spec') and not specs:
            prerequisites.append(f'no known values for spec field {spec_field!r}')
        if prerequisites:
            skipped.append({'family': family, 'reason': '; '.join(prerequisites)})
            continue
        if family == 'no_filter':
            templates.append((family, 'none', 'all'))
        elif family == 'price':
            templates.extend((family, strength, mode) for mode in ('max', 'interval') for strength in ('loose', 'medium', 'tight'))
        elif family in ('brand', 'category', 'spec'):
            templates.extend((family, strength, mode) for strength, mode in (('tight', 'one'), ('loose', 'two')))
        elif family in ('price_max', 'price_interval', 'rating', 'price_brand', 'price_brand_spec'):
            templates.extend((family, s, 'interval' if family == 'price_interval' else 'max') for s in ('loose', 'medium', 'tight'))
    audit = {'request_seed': seed, 'threshold_source': 'training catalog attributes and training review aggregates only',
             'price_quartiles_cents': price_quartiles, 'rating_mean_quartiles': rating_quartiles,
             'rating_min_count': rating_min_count, 'spec_field': spec_field,
             'skipped_families': skipped, 'rule_values_use_positive': False,
             'logical_policy': 'OR within each dimension, AND across dimensions; unknown constrained attributes fail',
             'random_control': 'uniform subset of full catalog, exact matched structural feasible count',
             'empty_feasible_policy': 'retain request and diagnostics',
             'metadata_temporal_limitation': data.get('audit', {}).get('metadata_temporal_limitation'),
             'request_cap_policy': 'total structural requests plus matched controls, round robin by template and user'}
    split = cfg.get('request_split', 'test')
    labels = sorted(data.get(split, []), key=lambda label: hashlib.sha256(
        f"{seed}:{label['user']}".encode()).digest())
    requests = []
    max_requests = int(cfg.get('max_requests', 100))
    if max_requests < 0:
        raise ValueError('max_requests must be nonnegative')
    def emit(label, family, strength, mode, rule, matched=None):
        feasible = feasible_ids(items, rule)
        user = str(label['user'])
        ident = hashlib.sha256(json.dumps([seed, split, user, family, strength, mode, matched], sort_keys=True).encode()).hexdigest()[:20]
        request = {'id': ident, 'user': user, 'history': list(data['histories'].get(user, label.get('history', []))),
                   'positive': label['positive'], 'rule': rule, 'family': family, 'strength': strength,
                   'variant': mode, 'feasible': feasible, 'feasible_count': len(feasible),
                   'feasible_fraction': len(feasible) / len(items) if items else 0,
                   'positive_in_catalog': label['positive'] in by_id,
                   'positive_feasible': label['positive'] in feasible, 'split': split}
        if matched is not None:
            request['matched_request'] = matched
        requests.append(request)
        return request
    # User round robin preserves all requested templates before progressing users.
    for label in labels:
        user = str(label['user'])
        for family, strength, mode in templates:
            control = 'random_matched' in families and family != 'no_filter'
            if len(requests) + 1 + int(control) > max_requests:
                audit['cap_reached'] = True
                break
            rng = _rng(seed, user, family, 'alternatives')
            rule = {'type': 'and'}
            if family == 'no_filter':
                rule = {'type': 'all'}
            if family in ('price', 'price_max', 'price_interval', 'price_brand', 'price_brand_spec'):
                index = {'tight': 0, 'medium': 1, 'loose': 2}[strength]
                if mode == 'interval':
                    # Nested intervals use training quartile boundaries exclusively.
                    low_q, high_q = {'tight': (.25, .5), 'medium': (.25, .75), 'loose': (0, 1)}[strength]
                    rule['price'] = {'min': int(round(_quantile(prices, low_q))),
                                     'max': int(round(_quantile(prices, high_q)))}
                else:
                    rule['price'] = {'max': price_quartiles[index]}
            if family in ('brand', 'price_brand', 'price_brand_spec'):
                n = 2 if strength == 'loose' else 1
                rule['brand'] = _weighted_choices(brands, n, rng)
            if family == 'category':
                rule['category'] = _weighted_choices(categories, 2 if mode == 'two' else 1, rng)
            if family == 'rating':
                rule['rating'] = {'min_mean': rating_quartiles[{'loose': 0, 'medium': 1, 'tight': 2}[strength]],
                                  'min_count': rating_min_count}
            if family in ('spec', 'price_brand_spec'):
                rule['spec'] = {spec_field: _weighted_choices(specs, 2 if mode == 'two' else 1, rng)}
            request = emit(label, family, strength, mode, rule)
            if control:
                random_rule = {'type': 'ids', 'ids': sorted(_rng(seed, user, 'random_matched', family, strength, mode).sample(item_ids, len(request['feasible'])))}
                emit(label, 'random_matched', strength, mode, random_rule, request['id'])
        if audit.get('cap_reached'):
            break
    audit['request_count'] = len(requests)
    audit['empty_feasible_count'] = sum(not r['feasible'] for r in requests)
    audit['family_counts'] = dict(Counter(r['family'] for r in requests))
    audit['candidate_sizes'] = {r['id']: len(r['feasible']) for r in requests}
    audit['unique_users'] = len({r['user'] for r in requests})
    audit['family_unique_users'] = {family: len({r['user'] for r in requests if r['family'] == family})
                                    for family in sorted({r['family'] for r in requests})}
    audit['joint_attribute_coverage'] = {
        'attributes': ['price', 'brand', 'category', 'rating_mean', f'specs.{spec_field}'],
        'known_items': sum(i.get('price') is not None and bool(i.get('brand')) and bool(i.get('category'))
                           and i.get('rating_mean') is not None and bool(i.get('specs', {}).get(spec_field))
                           for i in items),
        'catalog_items': len(items),
    }
    data.setdefault('audit', {})['constraints'] = audit
    return requests
