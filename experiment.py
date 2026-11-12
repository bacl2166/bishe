"""Compare three item encodings under a single price cap."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from math import isqrt, log2
from random import Random


Code = tuple[int, ...]


@dataclass(frozen=True)
class Product:
    item_id: str
    product_type: str
    price_cents: int


CATALOG = (
    Product("P1", "phosphor_bronze", 2900),
    Product("P2", "phosphor_bronze", 1800),
    Product("P3", "80_20_bronze", 1900),
    Product("P4", "80_20_bronze", 2400),
    Product("P5", "80_20_bronze", 2200),
)
# Synthetic preference weights shared by all encodings. Total weight is 100.
SCORES = {"P1": 50, "P2": 4, "P3": 35, "P4": 9, "P5": 2}
POSITIVE = "P3"
PRICE_BAND_CENTS = 1000


def feasible_items(catalog: tuple[Product, ...], max_price_cents: int) -> set[str]:
    return {item.item_id for item in catalog if item.price_cents <= max_price_cents}


def codes_from_groups(groups: list[list[str]]) -> dict[str, Code]:
    return {
        item_id: (group_index, item_index)
        for group_index, group in enumerate(groups, start=1)
        for item_index, item_id in enumerate(sorted(group), start=1)
    }


def build_encodings(catalog: tuple[Product, ...], seed: int) -> dict[str, dict[str, Code]]:
    by_type: dict[str, list[str]] = defaultdict(list)
    by_price: dict[int, list[str]] = defaultdict(list)
    for item in catalog:
        by_type[item.product_type].append(item.item_id)
        by_price[item.price_cents // PRICE_BAND_CENTS].append(item.item_id)

    shuffled_ids = sorted(item.item_id for item in catalog)
    Random(seed).shuffle(shuffled_ids)
    group_count = max(1, isqrt(len(shuffled_ids)))
    random_groups = [shuffled_ids[index::group_count] for index in range(group_count)]

    return {
        "product_type": codes_from_groups([by_type[key] for key in sorted(by_type)]),
        "price_band": codes_from_groups([by_price[key] for key in sorted(by_price)]),
        "random_balanced": codes_from_groups(random_groups),
    }


def exact_top_k(scores: dict[str, int], items: set[str], k: int) -> tuple[str, ...]:
    return tuple(sorted(items, key=lambda item_id: (-scores[item_id], item_id))[:k])


@dataclass(frozen=True)
class SearchResult:
    items: tuple[str, ...]
    scored_nodes: int


def constrained_search(
    codes: dict[str, Code],
    scores: dict[str, int],
    feasible: set[str],
    beam_width: int,
    k: int,
) -> SearchResult:
    """Score prefixes using the full catalog, then mask paths with no feasible leaf."""
    prefix_scores: dict[Code, int] = defaultdict(int)
    children: dict[Code, set[Code]] = defaultdict(set)
    leaves = {code: item_id for item_id, code in codes.items()}
    for item_id, code in codes.items():
        for depth in range(1, len(code) + 1):
            prefix = code[:depth]
            prefix_scores[prefix] += scores[item_id]
            if item_id in feasible:
                children[prefix[:-1]].add(prefix)

    frontier: list[Code] = [()]
    completed: set[str] = set()
    scored_nodes = 0
    while frontier:
        candidates: list[Code] = []
        for parent in frontier:
            for child in children.get(parent, ()):
                scored_nodes += 1
                if child in leaves:
                    completed.add(leaves[child])
                else:
                    candidates.append(child)
        candidates.sort(key=lambda prefix: (-prefix_scores[prefix], prefix))
        frontier = candidates[:beam_width]

    return SearchResult(exact_top_k(scores, completed, k), scored_nodes)


@dataclass(frozen=True)
class Metrics:
    hr: int | None
    ndcg: float | None
    oracle_recall: float | None


def evaluate(
    results: tuple[str, ...], reference: tuple[str, ...], feasible: set[str], positive: str
) -> Metrics:
    oracle_recall = len(set(results).intersection(reference)) / len(reference) if reference else None
    if positive not in feasible:
        return Metrics(None, None, oracle_recall)
    if positive not in results:
        return Metrics(0, 0.0, oracle_recall)
    rank = results.index(positive) + 1
    return Metrics(1, 1 / log2(rank + 1), oracle_recall)


def format_metric(value: int | float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-price-cents", type=int, default=2000)
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random encoding.")
    args = parser.parse_args()
    if args.max_price_cents < 0:
        parser.error("--max-price-cents must be non-negative")
    if args.top_k < 1 or any(width < 1 for width in args.beam_widths):
        parser.error("--top-k and all --beam-widths must be positive")

    feasible = feasible_items(CATALOG, args.max_price_cents)
    reference = exact_top_k(SCORES, feasible, args.top_k)
    encodings = build_encodings(CATALOG, args.seed)
    print(f"Price cap: {args.max_price_cents} cents; top-k: {args.top_k}; seed: {args.seed}")
    print(f"Feasible: {', '.join(sorted(feasible)) or '-'}")
    print(f"Exact ranking: {', '.join(reference) or '-'}; positive: {POSITIVE}")
    print(f"Positive feasible: {POSITIVE in feasible}\n")
    print("encoding\tbeam\titems\tHR\tNDCG\tOracleRecall\tscored_nodes")
    for width in args.beam_widths:
        for name, codes in encodings.items():
            result = constrained_search(codes, SCORES, feasible, width, args.top_k)
            metrics = evaluate(result.items, reference, feasible, POSITIVE)
            row = [
                name,
                str(width),
                ",".join(result.items) or "-",
                format_metric(metrics.hr),
                format_metric(metrics.ndcg),
                format_metric(metrics.oracle_recall),
                str(result.scored_nodes),
            ]
            print("\t".join(row))


if __name__ == "__main__":
    main()
