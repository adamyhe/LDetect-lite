"""Tests for ldetect_lite.ld_neighborhood."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ldetect_lite.ld_neighborhood import (
    CATEGORY_ORDER,
    category_masks,
    chromosome_separation_samples,
    write_separation_boxplot,
)
from tests._partition_fixtures import make_custom_partitioned_store


def test_category_masks_classifies_left_across_right() -> None:
    lo = np.array([250, 300, 600, 50])
    hi = np.array([300, 600, 700, 150])
    masks = category_masks(lo, hi, boundary=500, left=200, right=800)

    np.testing.assert_array_equal(masks["left"], [True, False, False, False])
    np.testing.assert_array_equal(masks["across"], [False, True, False, False])
    np.testing.assert_array_equal(masks["right"], [False, False, True, False])


def test_category_masks_excludes_pairs_outside_window() -> None:
    lo = np.array([50, 900])
    hi = np.array([150, 950])
    masks = category_masks(lo, hi, boundary=500, left=200, right=800)

    for category in CATEGORY_ORDER:
        assert not masks[category].any()


def test_chromosome_separation_samples_matches_hand_computed_r2(
    tmp_path: Path,
) -> None:
    # Diagonal entries (i == j) at every position used by an off-diagonal pair,
    # plus three off-diagonal pairs, one in each neighborhood around
    # boundary=500 with window_bp=300 (left=200, right=800):
    #   left:   (250, 300) -- both < boundary
    #   across: (300, 600) -- straddles boundary
    #   right:  (600, 700) -- both >= boundary
    # diag values are all 1.0, so r2 == shrink_ld**2 exactly.
    rows = [
        (250, 250, 1.0),
        (300, 300, 1.0),
        (600, 600, 1.0),
        (700, 700, 1.0),
        (250, 300, 0.5),
        (300, 600, 0.4),
        (600, 700, 0.3),
        # outside the [left, right] window entirely -- must not contribute
        (50, 50, 1.0),
        (150, 150, 1.0),
        (50, 150, 0.9),
    ]
    store = make_custom_partitioned_store(tmp_path, {(0, 1000): rows})

    samples = chromosome_separation_samples(
        boundaries=[500], store=store, name="chr1", window_bp=300
    )

    assert len(samples["left"]) == 1
    assert math.isclose(samples["left"][0], 0.25, rel_tol=1e-9)
    assert len(samples["across"]) == 1
    assert math.isclose(samples["across"][0], 0.16, rel_tol=1e-9)
    assert len(samples["right"]) == 1
    assert math.isclose(samples["right"][0], 0.09, rel_tol=1e-9)


def test_chromosome_separation_samples_aggregates_across_boundaries(
    tmp_path: Path,
) -> None:
    rows = [
        (250, 250, 1.0),
        (300, 300, 1.0),
        (250, 300, 0.5),
        (1250, 1250, 1.0),
        (1300, 1300, 1.0),
        (1250, 1300, 0.6),
    ]
    store = make_custom_partitioned_store(tmp_path, {(0, 2000): rows})

    samples = chromosome_separation_samples(
        boundaries=[500, 1500], store=store, name="chr1", window_bp=300
    )

    # Both boundaries contribute one "left" pair each; aggregated together.
    assert sorted(samples["left"]) == sorted([0.25, 0.36])
    assert samples["across"] == []
    assert samples["right"] == []


def test_chromosome_separation_samples_no_boundaries_returns_empty(
    tmp_path: Path,
) -> None:
    rows = [(250, 250, 1.0), (300, 300, 1.0), (250, 300, 0.5)]
    store = make_custom_partitioned_store(tmp_path, {(0, 1000): rows})

    samples = chromosome_separation_samples(boundaries=[], store=store, name="chr1")

    assert samples == {"left": [], "across": [], "right": []}


def test_write_separation_boxplot_writes_file(tmp_path: Path) -> None:
    samples = {"left": [0.2, 0.3], "across": [0.05], "right": [0.25, 0.4, 0.1]}
    path = tmp_path / "neighborhood.svg"

    write_separation_boxplot(path, samples, title="test")

    assert path.exists()
    assert path.stat().st_size > 0


def test_write_separation_boxplot_handles_no_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "empty.svg"

    write_separation_boxplot(
        path, {"left": [], "across": [], "right": []}, title="test"
    )

    assert path.exists()
    assert path.stat().st_size > 0
