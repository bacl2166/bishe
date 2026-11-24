"""Streaming Amazon Reviews 2023 ingestion with frozen chronological histories."""
from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path


def read_jsonl(path, prefix_bytes=None):
    """Read complete JSON objects; a requested byte prefix may end mid-record.

    Prefix bytes refer to the decompressed stream, including for gzip input.
    Without an explicit prefix, malformed/truncated input always raises.
    """
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == '.gz' else open
    if prefix_bytes is not None and int(prefix_bytes) <= 0:
        raise ValueError('review_prefix_bytes must be positive')
    with opener(path, 'rb') as stream:
        remaining = None if prefix_bytes is None else int(prefix_bytes)
        line_no = 0
        while remaining is None or remaining > 0:
            line = stream.readline() if remaining is None else stream.readline(remaining)
            if not line:
                break
            line_no += 1
            if remaining is not None:
                remaining -= len(line)
                if remaining == 0 and not line.endswith(b'\n'):
                    # Explicit byte-prefix sampling excludes its incomplete boundary row.
                    break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('expected a JSON object')
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f'{path}: line {line_no}: {exc}') from exc
            yield record


def _timestamp(value):
    if isinstance(value, bool):
        raise ValueError('boolean timestamp')
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError('non-finite timestamp')
        return int(value if abs(value) >= 100_000_000_000 else value * 1000)
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _review(record, path, row):
    user, item = record.get('user_id'), record.get('parent_asin') or record.get('asin')
    if not user or not item or record.get('timestamp') is None:
        raise ValueError(f'{path}: record {row}: missing user_id, item ID, or timestamp')
    try:
        ts = _timestamp(record['timestamp'])
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{path}: record {row}: invalid timestamp') from exc
    rating = record.get('rating')
    if rating is not None:
        try:
            rating = float(rating)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{path}: record {row}: invalid rating') from exc
        if not math.isfinite(rating) or not 1 <= rating <= 5:
            raise ValueError(f'{path}: record {row}: rating must be finite in [1,5]')
    return str(user), str(item), ts, rating


def _text(value):
    if value is None:
        return ''
    if isinstance(value, (list, tuple)):
        return ' '.join(_text(v) for v in value if v is not None)
    return str(value).strip()


def _price(value):
    try:
        amount = Decimal(str(value).replace('$', '').replace(',', '').strip())
        return int((amount * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP)) if amount.is_finite() and amount > 0 else None
    except (ValueError, InvalidOperation, TypeError):
        return None


SPEC_ALIASES = {
    'size': ('size',), 'material': ('material', 'material type'),
    'string_gauge': ('string gauge', 'string gauge size'),
    'compatible_devices': ('compatible devices', 'compatible device'),
    'instrument': ('instrument', 'instrument type', 'musical instrument'),
    'color': ('color', 'color name', 'colour'),
}


def normalize_item(record):
    details = {str(k).lower().replace('_', ' ').strip(): v for k, v in (record.get('details') or {}).items()}
    specs = {}
    for field, aliases in SPEC_ALIASES.items():
        for alias in aliases:
            value = _text(details.get(alias))
            if value:
                specs[field] = value.casefold()
                break
    brand = _text(details.get('brand') or record.get('brand') or record.get('store')) or None
    categories = record.get('categories') or record.get('category') or []
    if isinstance(categories, str):
        categories = [categories]
    def flatten(values):
        for value in values:
            if isinstance(value, list):
                yield from flatten(value)
            elif value is not None and str(value).strip():
                yield str(value).strip()
    return {
        'id': str(record.get('parent_asin') or record.get('asin') or ''),
        'title': _text(record.get('title')),
        'description': _text([record.get('description'), record.get('features')]),
        'category': list(dict.fromkeys(flatten(categories))), 'brand': brand,
        'price': _price(record.get('price')), 'specs': specs,
    }


def prepare(config: dict, output_dir: Path) -> dict:
    cfg = {**config, **config.get('data', {})}
    metadata, reviews = Path(cfg['metadata']), Path(cfg['reviews'])
    prefix = cfg.get('review_prefix_bytes')
    audit = {'metadata': str(metadata), 'reviews': str(reviews), 'review_prefix_bytes': prefix,
             'prefix_boundary_policy': 'exclude incomplete final row only for explicit byte prefix',
             'rating_source': 'training reviews only; metadata average_rating/rating_number ignored',
             'catalog_selection': 'training interaction count descending; item ID ascending ties',
             'history_policy': 'frozen training-only catalog events for both validation and test',
             'metadata_temporal_limitation': 'Amazon 2023 metadata is a static snapshot, not historical attributes'}
    quantiles = cfg.get('split_quantiles')
    if quantiles is None and ('train_quantile' in cfg or 'test_quantile' in cfg):
        quantiles = [cfg.get('train_quantile', .8), cfg.get('test_quantile', .9)]
    if quantiles is not None:
        q_train, q_test = map(float, quantiles)
        if not 0 < q_train < q_test < 1:
            raise ValueError('split_quantiles must satisfy 0 < train < test < 1')
        days = Counter()
        for row, rec in enumerate(read_jsonl(reviews, prefix), 1):
            _, _, ts, _ = _review(rec, reviews, row)
            days[ts // 86400000] += 1
        total, cumulative, cuts = sum(days.values()), 0, []
        for day, count in sorted(days.items()):
            cumulative += count
            while len(cuts) < 2 and cumulative >= total * (q_train, q_test)[len(cuts)]:
                cuts.append((day + 1) * 86400000)
        if len(cuts) != 2 or cuts[0] >= cuts[1]:
            raise ValueError('quantiles yield identical UTC-day boundaries; use explicit date cutoffs')
        train_before, test_from = cuts
        audit['quantile_policy'] = {'quantiles': list(quantiles), 'resolution': 'UTC day; next midnight after quantile day'}
    else:
        train_before = _timestamp(cfg.get('train_before', '2021-01-01'))
        test_from = _timestamp(cfg.get('test_from', '2022-01-01'))
    if train_before >= test_from:
        raise ValueError('train_before must precede test_from')
    audit['train_before_ms'], audit['test_from_ms'] = train_before, test_from
    counts, rating_sums, rating_counts = Counter(), Counter(), Counter()
    rows = split_counts = 0
    for row, rec in enumerate(read_jsonl(reviews, prefix), 1):
        _, item, ts, rating = _review(rec, reviews, row)
        rows += 1
        if ts < train_before:
            split_counts += 1
            counts[item] += 1
            if rating is not None:
                rating_sums[item] += rating
                rating_counts[item] += 1
    max_items = int(cfg.get('max_items', 2000))
    if max_items <= 0:
        raise ValueError('max_items must be positive')
    min_count = int(cfg.get('min_item_interactions', 1))
    heap, catalog, seen = [], {}, set()
    metadata_rows = 0
    for row, rec in enumerate(read_jsonl(metadata), 1):
        metadata_rows += 1
        item = str(rec.get('parent_asin') or rec.get('asin') or '')
        if not item:
            raise ValueError(f'{metadata}: record {row}: missing item ID')
        if item in seen:
            raise ValueError(f'{metadata}: record {row}: duplicate metadata item {item}')
        seen.add(item)
        if counts[item] < min_count or not counts[item]:
            continue
        key = (counts[item], tuple(-ord(c) for c in item), item)
        if len(heap) == max_items and key <= heap[0]:
            continue
        normalized = normalize_item(rec)
        normalized.update(train_count=counts[item], rating_count=rating_counts[item],
                          rating_mean=rating_sums[item] / rating_counts[item] if rating_counts[item] else None)
        catalog[item] = normalized
        heapq.heappush(heap, key)
        if len(heap) > max_items:
            del catalog[heapq.heappop(heap)[2]]
    del seen
    items = sorted(catalog.values(), key=lambda item: (-item['train_count'], item['id']))
    events, validation_events, test_events = defaultdict(list), {}, {}
    max_history = int(cfg.get('max_history', 50))
    if max_history <= 0:
        raise ValueError('max_history must be positive')
    # Per-user bounded heaps retain the most recent training events during the scan.
    for row, rec in enumerate(read_jsonl(reviews, prefix), 1):
        user, item, ts, _ = _review(rec, reviews, row)
        if ts < train_before:
            if item in catalog:
                history = events[user]
                heapq.heappush(history, (ts, row, item))
                if len(history) > max_history:
                    heapq.heappop(history)
        else:
            target = validation_events if ts < test_from else test_events
            event = (ts, row, item)
            if user not in target or event < target[user]:
                target[user] = event
    all_histories = {u: [item for _, _, item in sorted(ev)] for u, ev in sorted(events.items())}
    min_history, max_users = int(cfg.get('min_history', 3)), cfg.get('max_users', 100)
    eligible = [u for u, h in all_histories.items() if len(h) >= min_history and (u in validation_events or u in test_events)]
    eligible.sort(key=lambda user: hashlib.sha256(f"{cfg.get('request_seed', 17)}:{user}".encode()).digest())
    if max_users is not None:
        if int(max_users) <= 0:
            raise ValueError('max_users must be positive or null')
        eligible = eligible[:int(max_users)]
    histories = {u: all_histories[u] for u in eligible}
    def labels(source):
        return [{'user': u, 'history': list(histories[u]), 'positive': source[u][2],
                 'timestamp': source[u][0], 'in_catalog': source[u][2] in catalog}
                for u in eligible if u in source]
    validation, test = labels(validation_events), labels(test_events)
    audit.update(review_rows=rows, train_review_rows=split_counts, metadata_rows=metadata_rows,
                 catalog_size=len(items), catalog_missing_price=sum(i['price'] is None for i in items),
                 train_users=len(all_histories), evaluation_users=len(histories),
                 validation_labels=len(validation), test_labels=len(test),
                 validation_catalog_coverage=sum(x['in_catalog'] for x in validation) / len(validation) if validation else 0,
                 test_catalog_coverage=sum(x['in_catalog'] for x in test) / len(test) if test else 0,
                 min_history=min_history, max_history=max_history, max_users=max_users)
    result = {'items': items, 'histories': histories,
              'train_sequences': list(all_histories.values()), 'validation': validation, 'test': test, 'audit': audit}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (('prepared.json', result), ('audit.json', audit)):
        destination = output_dir / name
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, destination)
    return result
