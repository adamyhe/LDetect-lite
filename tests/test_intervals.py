"""Tests for ldetect2._util.intervals."""

from __future__ import annotations

from ldetect2._util.intervals import (
    boundary_jaccard,
    bp_jaccard,
    intersect_intervals,
    match_rate,
    merge_intervals,
    nearest_offsets,
)


def test_nearest_offsets_uses_closest_boundary():
    assert nearest_offsets([9, 20, 31], [10, 30]) == [1, 10, 1]


def test_match_rate_counts_boundaries_within_tolerance():
    assert match_rate([9, 20, 31], [10, 30], 1) == 2 / 3


def test_boundary_jaccard_uses_tolerance():
    assert boundary_jaccard([100, 200, 300], [105, 300], 10) == 2 / 3


def test_merge_intervals_merges_overlaps_and_touching_edges():
    assert merge_intervals([(10, 20), (20, 30), (40, 45), (42, 50)]) == [
        (10, 30),
        (40, 50),
    ]


def test_intersect_intervals_returns_pairwise_overlaps():
    assert intersect_intervals([(10, 30), (40, 50)], [(20, 45)]) == [
        (20, 30),
        (40, 45),
    ]


def test_bp_jaccard_uses_covered_base_pairs():
    assert bp_jaccard([(0, 10)], [(5, 15)]) == 0.3333
