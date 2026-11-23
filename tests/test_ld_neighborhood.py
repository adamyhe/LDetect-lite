"""Tests for ldetect_lite.ld_neighborhood."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ldetect_lite.ld_neighborhood import (
    CATEGORY_ORDER,
    BottomKSampler,
    category_masks,
    chromosome_separation_samples,
    window_row_bounds,
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


def test_bottom_k_sampler_keeps_everything_under_limit() -> None:
    sampler = BottomKSampler(10, np.random.default_rng(0))
    sampler.extend(np.array([0.1, 0.2, 0.3]))
    sampler.extend(np.array([]))
    sampler.extend(np.array([0.4]))

    assert sampler.seen == 4
    assert sorted(sampler.sample()) == [0.1, 0.2, 0.3, 0.4]


def test_bottom_k_sampler_caps_at_limit_without_replacement() -> None:
    sampler = BottomKSampler(50, np.random.default_rng(1))
    values = np.arange(1000, dtype=np.float64)
    for chunk in np.array_split(values, 37):
        sampler.extend(chunk)

    sample = sampler.sample()
    assert sampler.seen == 1000
    assert len(sample) == 50
    assert len(set(sample)) == 50
    assert set(sample) <= set(values.tolist())


def test_bottom_k_sampler_is_deterministic_for_a_seed() -> None:
    def run(seed: int) -> list[float]:
        sampler = BottomKSampler(20, np.random.default_rng(seed))
        for chunk in np.array_split(np.arange(500, dtype=np.float64), 9):
            sampler.extend(chunk)
        return sampler.sample()

    assert run(3) == run(3)
    assert run(3) != run(4)


def test_bottom_k_sampler_inclusion_is_uniform() -> None:
    # Each of 200 values should be kept with probability 20/200 = 0.1,
    # regardless of its position or of how the stream is chunked (uneven
    # chunks exercise both the buffered and the post-threshold paths).
    n, limit, trials = 200, 20, 3000
    counts = np.zeros(n)
    rng = np.random.default_rng(7)
    values = np.arange(n, dtype=np.float64)
    splits = [1, 3, 10, 50, 136]
    for _ in range(trials):
        sampler = BottomKSampler(limit, rng)
        for chunk in np.split(values, np.cumsum(splits)[:-1]):
            sampler.extend(chunk)
        counts[np.array(sampler.sample(), dtype=np.int64)] += 1

    freq = counts / trials
    # Binomial sd = sqrt(0.1 * 0.9 / 3000) ~ 0.0055; allow ~5 sd.
    assert np.all(np.abs(freq - 0.1) < 0.028)
    # First and last chunks should not be systematically favoured.
    assert abs(freq[:1].mean() - 0.1) < 0.028
    assert abs(freq[-136:].mean() - 0.1) < 0.006


def test_window_row_bounds_clips_to_window() -> None:
    assert window_row_bounds(100, 500, False, 50, 1000) == (100, 500, False)
    assert window_row_bounds(100, 500, False, 200, 1000) == (200, 500, True)
    assert window_row_bounds(100, 500, True, 0, 300) == (100, 300, True)


def test_chromosome_separation_samples_matches_brute_force_over_partitions(
    tmp_path: Path,
) -> None:
    # Overlapping partitions that each hold every pair with both ends inside
    # them (consistent values on the overlaps), and boundaries whose windows
    # start/end inside partitions, so the clipped reads and the ownership
    # rules are both exercised. With a large sample_limit the result must be
    # exactly every in-window pair, each counted once.
    rng = np.random.default_rng(11)
    positions = np.unique(rng.integers(1, 3000, size=400))
    diag = {int(p): float(rng.uniform(0.5, 1.5)) for p in positions}
    shrink: dict[tuple[int, int], float] = {}
    for i, lo in enumerate(positions):
        for hi in positions[i + 1 : i + 40]:
            shrink[(int(lo), int(hi))] = float(rng.uniform(-0.5, 0.5))

    bounds = [(0, 1100), (900, 2100), (1900, 3000)]
    partitions = {}
    for start, end in bounds:
        rows = [(p, p, v) for p, v in diag.items() if start <= p <= end]
        rows += [
            (lo, hi, v)
            for (lo, hi), v in shrink.items()
            if start <= lo <= end and start <= hi <= end
        ]
        partitions[(start, end)] = rows
    store = make_custom_partitioned_store(tmp_path, partitions)

    boundaries = [700, 1500, 2300]
    window = 450
    samples = chromosome_separation_samples(
        boundaries=boundaries,
        store=store,
        name="chr1",
        window_bp=window,
        sample_limit=10**9,
    )

    # Pairs that some partition holds (both ends inside one partition).
    held = {
        (lo, hi): v
        for (lo, hi), v in shrink.items()
        if any(s <= lo <= e and s <= hi <= e for s, e in bounds)
    }
    expected: dict[str, list[float]] = {c: [] for c in CATEGORY_ORDER}
    for boundary in boundaries:
        left, right = boundary - window, boundary + window
        for (lo, hi), v in held.items():
            if lo < left or hi > right:
                continue
            r2 = v * v / (diag[lo] * diag[hi])
            if hi < boundary:
                expected["left"].append(r2)
            elif lo >= boundary:
                expected["right"].append(r2)
            else:
                expected["across"].append(r2)

    for category in CATEGORY_ORDER:
        assert len(samples[category]) == len(expected[category]) > 0
        np.testing.assert_allclose(
            sorted(samples[category]), sorted(expected[category]), rtol=1e-12
        )
