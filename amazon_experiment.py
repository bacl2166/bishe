"""Evaluate two recommenders on Amazon Reviews 2023 Musical Instruments."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
from itertools import combinations
from math import sqrt
from pathlib import Path
from random import Random
from statistics import mean

from experiment import Code, Metrics, codes_from_groups, constrained_search, evaluate, exact_top_k


HISTORY_LENGTH = 50
MIN_HISTORY_ITEMS = 3
KNN_NEIGHBORS = 100


def read_jsonl(path: Path, prefix_bytes: int | None = None) -> Iterator[dict]:
    open_file = gzip.open if path.suffix == ".gz" else open
    with open_file(path, "rb") as rows:
        if prefix_bytes is None:
            for line in rows:
                if line.strip():
                    yield json.loads(line)
            return

        remaining = prefix_bytes
        while remaining:
            line = rows.readline(remaining)
            if not line:
                raise ValueError(f"{path} is shorter than the requested {prefix_bytes}-byte prefix.")
            remaining -= len(line)
            if line.endswith(b"\n") and line.strip():
                yield json.loads(line)


def load_prices(path: Path) -> dict[str, int]:
    prices = {}
    for record in read_jsonl(path):
        item = record.get("parent_asin")
        try:
            price = Decimal(str(record.get("price")))
        except InvalidOperation:
            continue
        if item and price.is_finite() and price > 0:
            cents = int(round(price * 100))
            if cents > 0:
                prices[item] = cents
    return prices


@dataclass(frozen=True)
class PilotData:
    prices: dict[str, int]
    train_counts: dict[str, int]
    histories: dict[str, tuple[str, ...]]
    test_items: dict[str, str]
    priced_items: int
    review_rows: int


def prepare_data(
    meta: Path, reviews: Path, cutoff: int, max_items: int, review_prefix_bytes: int | None = None
) -> PilotData:
    prices = load_prices(meta)
    counts: Counter[str] = Counter()
    review_rows = 0
    for record in read_jsonl(reviews, review_prefix_bytes):
        review_rows += 1
        item = record.get("parent_asin")
        timestamp = record.get("timestamp")
        if record.get("user_id") and isinstance(timestamp, int) and timestamp < cutoff and item in prices:
            counts[item] += 1

    catalog = sorted(counts, key=lambda item: (-counts[item], item))[:max_items]
    catalog_prices = {item: prices[item] for item in catalog}
    train_events: dict[str, list[tuple[int, str]]] = defaultdict(list)
    first_test_event: dict[str, tuple[int, str]] = {}
    for record in read_jsonl(reviews, review_prefix_bytes):
        user = record.get("user_id")
        item = record.get("parent_asin")
        timestamp = record.get("timestamp")
        if not user or not item or not isinstance(timestamp, int):
            continue
        event = (timestamp, item)
        if timestamp < cutoff:
            if item in catalog_prices:
                train_events[user].append(event)
        elif user not in first_test_event or event < first_test_event[user]:
            first_test_event[user] = event

    histories = {
        user: tuple(item for _, item in sorted(events)[-HISTORY_LENGTH:])
        for user, events in train_events.items()
    }
    test_items = {user: first_test_event[user][1] for user in histories if user in first_test_event}
    return PilotData(
        catalog_prices, {item: counts[item] for item in catalog}, histories, test_items,
        len(prices), review_rows,
    )


def fit_models(train_counts: dict[str, int], histories: dict[str, tuple[str, ...]]):
    total = sum(train_counts.values())
    popularity = {item: count / total for item, count in train_counts.items()}
    users_per_item: Counter[str] = Counter()
    cooccurrences: dict[str, Counter[str]] = defaultdict(Counter)
    for history in histories.values():
        unique_items = sorted(set(history))
        users_per_item.update(unique_items)
        for left, right in combinations(unique_items, 2):
            cooccurrences[left][right] += 1
            cooccurrences[right][left] += 1

    neighbors = {}
    for item, counts in cooccurrences.items():
        similarities = [
            (neighbor, count / sqrt(users_per_item[item] * users_per_item[neighbor]))
            for neighbor, count in counts.items()
        ]
        similarities.sort(key=lambda pair: (-pair[1], pair[0]))
        neighbors[item] = similarities[:KNN_NEIGHBORS]

    def score_popularity(_history: tuple[str, ...]) -> dict[str, float]:
        return popularity

    def score_knn(history: tuple[str, ...]) -> dict[str, float]:
        scores = {item: 1e-6 * weight for item, weight in popularity.items()}
        for seen in dict.fromkeys(history):
            for item, similarity in neighbors.get(seen, ()):
                scores[item] += similarity
        return scores

    return score_popularity, score_knn


def build_pilot_encodings(prices: dict[str, int], groups: int, seed: int) -> dict[str, dict[str, Code]]:
    sorted_items = sorted(prices, key=lambda item: (prices[item], item))
    shuffled_items = sorted(prices)
    Random(seed).shuffle(shuffled_items)
    sizes = [len(prices) // groups + int(index < len(prices) % groups) for index in range(groups)]

    def encode(ordered_items: list[str]) -> dict[str, Code]:
        partitions = []
        start = 0
        for size in sizes:
            partitions.append(ordered_items[start:start + size])
            start += size
        return codes_from_groups(partitions)

    return {"price_ordered": encode(sorted_items), "random_balanced": encode(shuffled_items)}


def mean_defined(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return mean(valid) if valid else None


def run_pilot(data: PilotData, max_users: int, groups: int, seed: int, k: int, widths: list[int]) -> dict:
    users = [user for user in data.test_items if len(set(data.histories[user])) >= MIN_HISTORY_ITEMS]
    users.sort(key=lambda user: hashlib.sha256(f"{seed}:{user}".encode()).digest())
    requests = users[:max_users]
    if not requests or not data.prices:
        raise ValueError("No eligible test users or priced training items. Check the data and cutoff.")

    # Use an observed catalog median (lower middle for even-sized catalogs).
    price_cap = sorted(data.prices.values())[(len(data.prices) - 1) // 2]
    feasible = {item for item, price in data.prices.items() if price <= price_cap}
    encodings = build_pilot_encodings(data.prices, min(groups, len(data.prices)), seed)
    models = zip(("most_popular", "item_knn"), fit_models(data.train_counts, data.histories))
    results = []
    for model, score in models:
        rows: dict[tuple[str, int | None], list[tuple[Metrics, int | None]]] = defaultdict(list)
        for user in requests:
            scores = score(data.histories[user])
            positive = data.test_items[user]
            reference = exact_top_k(scores, feasible, k)
            rows[("exact", None)].append((evaluate(reference, reference, feasible, positive), None))
            for encoding, codes in encodings.items():
                for width in widths:
                    search = constrained_search(codes, scores, feasible, width, k)
                    metrics = evaluate(search.items, reference, feasible, positive)
                    rows[(encoding, width)].append((metrics, search.scored_nodes))

        for (encoding, width), values in rows.items():
            results.append({
                "model": model,
                "encoding": encoding,
                "beam_width": width,
                "hr": mean_defined([metric.hr for metric, _ in values]),
                "ndcg": mean_defined([metric.ndcg for metric, _ in values]),
                "oracle_recall": mean_defined([metric.oracle_recall for metric, _ in values]),
                "scored_nodes_mean": mean_defined([nodes for _, nodes in values]),
            })

    feasible_positives = sum(data.test_items[user] in feasible for user in requests)
    return {
        "price_cap_cents": price_cap,
        "review_rows": data.review_rows,
        "priced_items": data.priced_items,
        "catalog_items": len(data.prices),
        "catalog_train_reviews": sum(data.train_counts.values()),
        "catalog_train_users": len(data.histories),
        "feasible_items": len(feasible),
        "eligible_test_users": len(users),
        "test_requests": len(requests),
        "in_catalog_positives": sum(data.test_items[user] in data.prices for user in requests),
        "feasible_positives": feasible_positives,
        "positive_feasible_rate": feasible_positives / len(requests),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--review-prefix-bytes", type=int, help="Read only complete review rows in this byte prefix.")
    parser.add_argument("--train-before", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--max-items", type=int, default=2000)
    parser.add_argument("--max-test-users", type=int, default=400)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--beam-widths", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.max_items, args.max_test_users, args.groups, args.top_k, *args.beam_widths) < 1:
        parser.error("Item, user, group, Top-K and beam counts must be positive.")
    if args.review_prefix_bytes is not None and args.review_prefix_bytes < 1:
        parser.error("--review-prefix-bytes must be positive.")
    cutoff = int(datetime.combine(args.train_before, time.min, timezone.utc).timestamp() * 1000)
    data = prepare_data(args.metadata, args.reviews, cutoff, args.max_items, args.review_prefix_bytes)
    report = run_pilot(data, args.max_test_users, args.groups, args.seed, args.top_k, args.beam_widths)
    report["config"] = {
        "dataset": "Amazon Reviews 2023 / Musical_Instruments",
        "review_prefix_bytes": args.review_prefix_bytes,
        "train_before_utc": args.train_before.isoformat(),
        "max_items": args.max_items,
        "max_test_users": args.max_test_users,
        "groups": args.groups,
        "beam_widths": args.beam_widths,
        "top_k": args.top_k,
        "seed": args.seed,
        "min_history_items": MIN_HISTORY_ITEMS,
        "max_history_length": HISTORY_LENGTH,
        "knn_neighbors": KNN_NEIGHBORS,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
