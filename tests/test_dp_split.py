"""Tests for the DP-optimal LD-block splitter (`dp_split`)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from ldetect_lite._util.covariance_array import ChromosomeCovariance, metric_from_arrays
from ldetect_lite.dp_split import (
    DPPartition,
    generate_filter_candidates,
    optimal_partitions,
)


def _make_cov(
    loci: list[int], r2_by_pair: dict[tuple[int, int], float]
) -> ChromosomeCovariance:
    items = sorted(r2_by_pair.items())
    i_pos = np.array([pair[0] for pair, _ in items], dtype=np.int64)
    j_pos = np.array([pair[1] for pair, _ in items], dtype=np.int64)
    r2 = np.array([value for _, value in items], dtype=np.float64)
    return ChromosomeCovariance(
        loci=np.array(sorted(loci), dtype=np.int64),
        i_pos=i_pos,
        j_pos=j_pos,
        r2=r2,
        partitions=(),
        partition_arrays=(),
    )


def _block_sizes(loci: np.ndarray, bp: list[int]) -> np.ndarray:
    if not bp:
        return np.array([loci.size])
    bp_arr = np.array(sorted(bp), dtype=np.int64)
    block_ids = np.searchsorted(bp_arr, loci, side="left")
    return np.bincount(block_ids, minlength=bp_arr.size + 1)


def _feasible(loci: np.ndarray, bp: list[int], min_size: int, max_size: int) -> bool:
    sizes = _block_sizes(loci, bp)
    return bool(np.all((sizes >= min_size) & (sizes <= max_size)))


def _brute_force_best(
    cov: ChromosomeCovariance,
    candidates: list[int],
    min_size: int,
    max_size: int,
    max_k: int,
) -> dict[int, float]:
    """Exhaustively search every candidate subset and return the best cost per K."""
    best: dict[int, float] = {}
    for k in range(1, min(max_k, len(candidates) + 1) + 1):
        best_cost = None
        for combo in combinations(sorted(candidates), k - 1):
            bp = list(combo)
            if not _feasible(cov.loci, bp, min_size, max_size):
                continue
            cost = metric_from_arrays(cov, bp)["sum"]
            if best_cost is None or cost < best_cost:
                best_cost = cost
        if best_cost is not None:
            best[k] = best_cost
    return best


# A synthetic chromosome with three tightly-correlated groups and weak
# cross-group leakage. Values are asymmetric enough that each K has a unique
# optimum (no cost ties to worry about when comparing to brute force).
_LOCI = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
_R2 = {
    (10, 20): 0.9, (10, 30): 0.7, (20, 30): 0.8,
    (40, 50): 0.6, (40, 60): 0.5, (40, 70): 0.4,
    (50, 60): 0.7, (50, 70): 0.5, (60, 70): 0.6,
    (80, 90): 0.85, (80, 100): 0.6, (90, 100): 0.75,
    (30, 40): 0.05, (70, 80): 0.05, (20, 90): 0.01,
}
_CANDIDATES = [20, 30, 40, 50, 60, 70, 80, 90]


def test_dp_matches_brute_force_unconstrained() -> None:
    cov = _make_cov(_LOCI, _R2)
    results = optimal_partitions(
        cov, region_start=10, region_end=100, candidate_loci=_CANDIDATES,
        min_size=1, max_size=10, max_k=4,
    )
    expected = _brute_force_best(cov, _CANDIDATES, min_size=1, max_size=10, max_k=4)

    assert {r.n_block for r in results} == set(expected)
    for r in results:
        assert r.cost == pytest.approx(expected[r.n_block], abs=1e-9)


def test_dp_matches_brute_force_with_size_constraints() -> None:
    cov = _make_cov(_LOCI, _R2)
    min_size, max_size = 2, 4
    results = optimal_partitions(
        cov, region_start=10, region_end=100, candidate_loci=_CANDIDATES,
        min_size=min_size, max_size=max_size, max_k=5,
    )
    expected = _brute_force_best(
        cov, _CANDIDATES, min_size=min_size, max_size=max_size, max_k=5
    )

    assert {r.n_block for r in results} == set(expected)
    for r in results:
        assert r.cost == pytest.approx(expected[r.n_block], abs=1e-9)


def test_dp_respects_min_max_size_on_output() -> None:
    cov = _make_cov(_LOCI, _R2)
    min_size, max_size = 3, 4
    results = optimal_partitions(
        cov, region_start=10, region_end=100, candidate_loci=_CANDIDATES,
        min_size=min_size, max_size=max_size, max_k=5,
    )
    for r in results:
        assert _feasible(cov.loci, r.loci, min_size, max_size)


def test_dp_cost_matches_metric_from_arrays() -> None:
    cov = _make_cov(_LOCI, _R2)
    results = optimal_partitions(
        cov, region_start=10, region_end=100, candidate_loci=_CANDIDATES,
        min_size=1, max_size=10, max_k=4,
    )
    for r in results:
        assert r.cost == pytest.approx(metric_from_arrays(cov, r.loci)["sum"], abs=1e-9)


def test_dp_no_candidates_returns_single_block() -> None:
    cov = _make_cov(_LOCI, _R2)
    results = optimal_partitions(
        cov, region_start=10, region_end=100, candidate_loci=[],
        min_size=1, max_size=10, max_k=4,
    )
    assert results == [DPPartition(n_block=1, cost=0.0, loci=[])]


def test_dp_rejects_invalid_size_bounds() -> None:
    cov = _make_cov(_LOCI, _R2)
    with pytest.raises(ValueError):
        optimal_partitions(cov, 10, 100, _CANDIDATES, min_size=0, max_size=10, max_k=2)
    with pytest.raises(ValueError):
        optimal_partitions(cov, 10, 100, _CANDIDATES, min_size=5, max_size=1, max_k=2)
    with pytest.raises(ValueError):
        optimal_partitions(cov, 10, 100, _CANDIDATES, min_size=1, max_size=10, max_k=0)
    with pytest.raises(ValueError):
        optimal_partitions(
            cov, 10, 100, _CANDIDATES, min_size=1, max_size=10, max_k=2,
            min_size_bp=-1,
        )
    with pytest.raises(ValueError):
        optimal_partitions(
            cov, 10, 100, _CANDIDATES, min_size=1, max_size=10, max_k=2,
            min_size_bp=50, max_size_bp=10,
        )


def test_dp_max_size_bp_forbids_physically_wide_blocks() -> None:
    # Two tightly-packed clusters of SNPs 1Mb apart; SNP-count alone can't
    # tell them apart from a physically compact block of the same count.
    loci = [0, 10, 20, 30, 1_000_000, 1_000_010, 1_000_020, 1_000_030]
    r2 = {(0, 30): 0.5, (1_000_000, 1_000_030): 0.5, (30, 1_000_000): 0.2}
    cov = _make_cov(loci, r2)
    candidates = [30]

    unconstrained = optimal_partitions(
        cov, 0, 1_000_030, candidates, min_size=1, max_size=10, max_k=2,
    )
    assert {r.n_block for r in unconstrained} == {1, 2}

    constrained = optimal_partitions(
        cov, 0, 1_000_030, candidates, min_size=1, max_size=10, max_k=2,
        max_size_bp=500_000,
    )
    # The whole-region single block spans 1,000,030bp > 500,000bp: infeasible.
    assert {r.n_block for r in constrained} == {2}


def test_dp_min_size_bp_forbids_physically_narrow_blocks() -> None:
    loci = [0, 1, 2, 100, 101, 102]
    r2 = {(0, 2): 0.5, (100, 102): 0.5, (2, 100): 0.1}
    cov = _make_cov(loci, r2)
    candidates = [2]

    unconstrained = optimal_partitions(
        cov, 0, 102, candidates, min_size=1, max_size=10, max_k=2,
    )
    assert {r.n_block for r in unconstrained} == {1, 2}

    constrained = optimal_partitions(
        cov, 0, 102, candidates, min_size=1, max_size=10, max_k=2,
        min_size_bp=50,
    )
    # Both 2-block halves span only 2bp < 50bp: K=2 becomes infeasible.
    assert {r.n_block for r in constrained} == {1}


def test_dp_thr_r2_drops_weak_pairs() -> None:
    # A single candidate at 10; one strong and one weak pair both cross it.
    cov = _make_cov([0, 10, 20], {(0, 20): 0.5, (10, 20): 0.02})

    unfiltered = optimal_partitions(
        cov, 0, 20, [10], min_size=1, max_size=3, max_k=2, thr_r2=0.0,
    )
    filtered = optimal_partitions(
        cov, 0, 20, [10], min_size=1, max_size=3, max_k=2, thr_r2=0.1,
    )

    cost_unfiltered = next(r.cost for r in unfiltered if r.n_block == 2)
    cost_filtered = next(r.cost for r in filtered if r.n_block == 2)
    assert cost_unfiltered == pytest.approx(0.52)
    assert cost_filtered == pytest.approx(0.5)


def test_dp_max_r2_forbids_splitting_strong_pairs() -> None:
    # A single candidate at 10, with a very strong pair spanning it.
    cov = _make_cov([0, 10, 20], {(0, 20): 0.9})

    unconstrained = optimal_partitions(
        cov, 0, 20, [10], min_size=1, max_size=3, max_k=2, max_r2=1.0,
    )
    constrained = optimal_partitions(
        cov, 0, 20, [10], min_size=1, max_size=3, max_k=2, max_r2=0.5,
    )

    assert {r.n_block for r in unconstrained} == {1, 2}
    assert {r.n_block for r in constrained} == {1}


def test_generate_filter_candidates_finds_valleys() -> None:
    arr = np.ones(200) * 5.0
    arr[40:61] = 1.0 + np.abs(np.arange(21) - 10) * 0.2
    arr[140:161] = 1.0 + np.abs(np.arange(21) - 10) * 0.2
    positions = np.arange(200)

    candidates = generate_filter_candidates(arr, positions, width=5)

    assert 50 in candidates
    assert 150 in candidates
