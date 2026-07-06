"""Generic interval and boundary comparison utilities."""

from __future__ import annotations

import bisect
import statistics

Block = tuple[int, int]


def block_sizes(blocks: list[Block]) -> list[int]:
    return [end - start for start, end in blocks]


def size_stats(sizes: list[int]) -> dict[str, float | int]:
    if not sizes:
        return {}
    sorted_sizes = sorted(sizes)
    n = len(sorted_sizes)
    return {
        "n": n,
        "mean_kb": round(statistics.mean(sorted_sizes) / 1000, 1),
        "median_kb": round(statistics.median(sorted_sizes) / 1000, 1),
        "p5_kb": round(sorted_sizes[max(0, int(n * 0.05))] / 1000, 1),
        "p95_kb": round(sorted_sizes[min(n - 1, int(n * 0.95))] / 1000, 1),
    }


def boundaries(blocks: list[Block]) -> list[int]:
    """Return sorted unique boundary positions."""
    positions: set[int] = set()
    for start, end in blocks:
        positions.add(start)
        positions.add(end)
    return sorted(positions)


def nearest_offsets(query: list[int], reference: list[int]) -> list[int]:
    """Return each query position's distance to the nearest reference position."""
    if not reference:
        return [10**9] * len(query)

    offsets: list[int] = []
    for position in query:
        index = bisect.bisect_left(reference, position)
        candidates: list[int] = []
        if index < len(reference):
            candidates.append(abs(reference[index] - position))
        if index > 0:
            candidates.append(abs(reference[index - 1] - position))
        offsets.append(min(candidates))
    return offsets


def match_rate(query: list[int], reference: list[int], tolerance: int) -> float:
    """Fraction of query boundaries within tolerance bp of any reference boundary."""
    if not query:
        return float("nan")
    offsets = nearest_offsets(query, reference)
    return sum(1 for offset in offsets if offset <= tolerance) / len(query)


def boundary_jaccard(a: list[int], b: list[int], tolerance: int) -> float:
    """Jaccard index on boundary sets under a bp tolerance."""
    if not a or not b:
        return float("nan")
    intersection = sum(1 for offset in nearest_offsets(a, b) if offset <= tolerance)
    union = len(a) + len(b) - intersection
    return intersection / union if union else float("nan")


def offset_stats(offsets: list[int]) -> tuple[float, float]:
    """Return median and p90 distances in kb."""
    if not offsets:
        return float("nan"), float("nan")
    sorted_offsets = sorted(offsets)
    n = len(sorted_offsets)
    median = statistics.median(sorted_offsets) / 1000
    p90 = sorted_offsets[min(n - 1, int(n * 0.90))] / 1000
    return round(median, 1), round(p90, 1)


def merge_intervals(blocks: list[Block]) -> list[Block]:
    """Return sorted, non-overlapping merged intervals."""
    if not blocks:
        return []

    merged: list[list[int]] = []
    for start, end in sorted(blocks):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def covered_bp(intervals: list[Block]) -> int:
    return sum(end - start for start, end in intervals)


def intersect_intervals(a: list[Block], b: list[Block]) -> list[Block]:
    """Intersection of two sorted, non-overlapping interval lists."""
    result: list[Block] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            result.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def bp_jaccard(ours: list[Block], reference: list[Block]) -> float:
    """Base-pair Jaccard: interval intersection length divided by union length."""
    if not ours or not reference:
        return float("nan")
    ours_merged = merge_intervals(ours)
    reference_merged = merge_intervals(reference)
    intersection = covered_bp(intersect_intervals(ours_merged, reference_merged))
    union = covered_bp(ours_merged) + covered_bp(reference_merged) - intersection
    return round(intersection / union, 4) if union else float("nan")
