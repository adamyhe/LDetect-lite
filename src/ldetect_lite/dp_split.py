"""Dynamic-programming LD block splitter: exact-optimal partition of shrunk LD.

This is an alternative to the heuristic `find_minima`/`local_search` pipeline
(binary-search a Hann filter width, then greedily hill-climb each candidate).
Instead, it finds the partition into exactly ``K`` blocks that minimizes the
same quantity :mod:`ldetect_lite.metric` already computes -- ``sum(r^2)`` over
pairs of variants that fall in different blocks -- for every ``K`` from 1 up
to a requested maximum in a single dynamic-programming pass, following the
approach of Privé (2022, ``snp_ldsplit()``) and Berisa & Pickrell (2016).

Unlike a literal port of ``snp_ldsplit()``, this implementation consumes
ldetect-lite's existing sparse ``(i_pos, j_pos, r2)`` triplets directly
(:class:`~ldetect_lite._util.covariance_array.ChromosomeCovariance`) and
restricts candidate cut points to a caller-supplied subset of loci (typically
the local minima of a densely-sampled Hann filter -- see
:func:`generate_filter_candidates`) rather than every SNP. Restricting
candidates aggregates the correlation structure into a small number of
"meta-nodes" (the intervals between consecutive candidates), which keeps the
whole computation small regardless of chromosome size: the O(max_K *
n_meta^2) recurrence below operates on ``n_meta`` candidates, not ``m`` raw
SNPs, so it does not need bigsnpr's per-partition chunking to stay within
memory. See ``notes/shrinkage_ld_optimized_dynamic_programming.md``.

This is a genuinely different algorithm from LDetect's heuristic, not a
"local search" refinement of it -- it should be treated as a distinct,
clearly-labeled alternative (the ``dp`` breakpoint subset), not folded into
the ``fourier``/``uniform`` reproduction-testing story.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from numba import njit

from ldetect_lite._util.covariance_array import ChromosomeCovariance
from ldetect_lite.filters import apply_filter_serial, get_minima_loc

# Imported bare, like every other hard dependency in this codebase (numpy,
# h5py, ...) -- unlike filters.py's numba usage, this DP recurrence has no
# independently-maintained fallback to guard for, so there is nothing a
# try/except could usefully do here beyond what a plain ModuleNotFoundError
# already says.
# `nogil=True` matters if a caller ever runs several DP solves concurrently
# from threads; `fastmath=True` is safe here because, unlike the filter
# convolution in filters.py, nothing downstream depends on bit-exact
# reassociation of this loop's sums.
_F = TypeVar("_F", bound=Callable[..., Any])
_numba_decorator = njit(nogil=True, fastmath=True, cache=True)


def _njit(fn: _F) -> _F:
    """Thin typed wrapper: numba ships no type stubs, so without this,
    mypy would see every `@_njit`-decorated function below as untyped."""
    return _numba_decorator(fn)  # type: ignore[no-any-return]


@dataclass(frozen=True)
class DPPartition:
    """One exact-optimal solution from the DP sweep, for a given block count."""

    n_block: int
    cost: float
    loci: list[int]


def generate_filter_candidates(
    vector_vals: Sequence[float] | np.ndarray,
    vector_pos: Sequence[int] | np.ndarray,
    width: int,
    window_mode: str = "symmetric",
) -> list[int]:
    """Return a dense candidate breakpoint set from a small Hann filter width.

    Reuses :func:`ldetect_lite.filters.apply_filter_serial`, the same
    convolution `pipeline.find_breakpoints` uses for the `fourier` subset --
    but at a caller-chosen, typically much smaller, fixed width instead of the
    width the binary search finds for one target breakpoint count. A smaller
    width yields more local minima, giving the DP more candidates to choose
    from than any single target-K filter search would.
    """
    arr = np.asarray(vector_vals, dtype=np.float64)
    result = apply_filter_serial(arr, width, window_mode)
    return get_minima_loc(result, np.asarray(vector_pos))


def _block_starts(cuts: np.ndarray) -> np.ndarray:
    """Real-loci-index start position for each meta-node in `cuts`."""
    starts = np.empty(cuts.shape[0], dtype=np.int64)
    starts[0] = 0
    starts[1:] = cuts[:-1] + 1
    return starts


def _build_weight_matrix(
    i_idx: np.ndarray, j_idx: np.ndarray, r2: np.ndarray, cuts: np.ndarray
) -> np.ndarray:
    """Aggregate pair weights into an (n_meta x n_meta) upper-triangular matrix.

    Pairs whose two endpoints fall in the same meta-node are dropped: no
    candidate cut point separates them, so they never contribute to any
    partition's cost and can be ignored entirely.
    """
    n_meta = cuts.shape[0]
    t1 = np.searchsorted(cuts, i_idx, side="left")
    t2 = np.searchsorted(cuts, j_idx, side="left")
    cross = t1 != t2
    W = np.zeros((n_meta, n_meta), dtype=np.float64)
    if not np.any(cross):
        return W
    a = np.minimum(t1[cross], t2[cross])
    b = np.maximum(t1[cross], t2[cross])
    np.add.at(W, (a, b), r2[cross])
    return W


def _suffix_matrix(W: np.ndarray) -> np.ndarray:
    """L[i, j] = sum_{q=j}^{n_meta-1} W[i, q], with an all-zero sentinel column
    at j == n_meta (a block ending at the last meta-node has nothing beyond it)."""
    n_meta = W.shape[0]
    L = np.zeros((n_meta, n_meta + 1), dtype=np.float64)
    if n_meta > 0:
        L[:, :n_meta] = np.cumsum(W[:, ::-1], axis=1)[:, ::-1]
    return L


def _apply_max_r2_constraint(
    candidate_idx: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    r2: np.ndarray,
    max_r2: float,
) -> np.ndarray:
    """Drop candidates that would split a pair whose r^2 exceeds `max_r2`.

    Mirrors `snp_ldsplit()`'s `max_r2`: such pairs are assumed too strongly
    correlated to ever separate, regardless of what the DP would otherwise
    choose. Pairs this strong are expected to be rare, so a per-pair loop here
    is not a bottleneck relative to the DP itself.
    """
    high = r2 > max_r2
    if not np.any(high):
        return candidate_idx
    keep = np.ones(candidate_idx.shape[0], dtype=bool)
    for lo, hi in zip(i_idx[high], j_idx[high] - 1):
        keep &= ~((candidate_idx >= lo) & (candidate_idx <= hi))
    return candidate_idx[keep]


# ---------------------------------------------------------------------------
# Core recurrence: numba-jitted tight loops.
#
# Both compute the same thing bigsnpr's get_C() does (Privé 2022): for each
# block-end meta-node j, E[i, j] is the cost of a block spanning meta-nodes
# [i, j] (the sum of squared correlation "leaking" past the block's right
# edge), and C[k, j]/best_start[k, j] is the standard partition-DP min-cost
# table and backpointer. n_meta is kept small by candidate restriction, so
# this runs in well under a second at typical scale (n_meta in the hundreds);
# numba is used because it lets the min/max-size early-exit run at native
# speed with no per-element Python/numpy dispatch overhead, not because plain
# numpy vectorization would be intractable here.
# ---------------------------------------------------------------------------


@_njit
def _compute_E(
    L: np.ndarray,
    starts: np.ndarray,
    cuts: np.ndarray,
    min_size: int,
    max_size: int,
    start_bp: np.ndarray,
    end_bp: np.ndarray,
    min_size_bp: int,
    max_size_bp: int,
) -> np.ndarray:
    """`*_bp` constraints are physical-distance analogs of `min_size`/`max_size`
    (SNP counts): `start_bp`/`end_bp` are `starts`/`cuts` looked up in genomic
    bp position rather than loci-index space. Both size and bp-size grow
    monotonically as `i` decreases (the block only gets wider), so a single
    early-exit `break` on either one is still valid -- no need to check both
    before deciding to stop.
    """
    n_meta = L.shape[0]
    E = np.full((n_meta, n_meta), np.inf)
    for j in range(n_meta):
        running = 0.0
        end_real = cuts[j]
        end_bp_j = end_bp[j]
        for i in range(j, -1, -1):
            running += L[i, j + 1]
            size = end_real - starts[i] + 1
            bp_size = end_bp_j - start_bp[i]
            if size > max_size or bp_size > max_size_bp:
                break
            if size >= min_size and bp_size >= min_size_bp:
                E[i, j] = running
    return E


@_njit
def _dp(E: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray]:
    n_meta = E.shape[0]
    C = np.full((max_k + 1, n_meta), np.inf)
    best_start = np.full((max_k + 1, n_meta), -1, dtype=np.int64)
    for j in range(n_meta):
        C[1, j] = E[0, j]
        best_start[1, j] = 0
    for k in range(2, max_k + 1):
        for j in range(k - 1, n_meta):
            best_cost = np.inf
            best_i = -1
            for i in range(k - 2, j):
                prev = C[k - 1, i]
                if not np.isfinite(prev):
                    continue
                e = E[i + 1, j]
                if not np.isfinite(e):
                    continue
                cost = prev + e
                if cost < best_cost:
                    best_cost = cost
                    best_i = i + 1
            if best_i >= 0:
                C[k, j] = best_cost
                best_start[k, j] = best_i
    return C, best_start


def _reconstruct(
    k: int, best_start: np.ndarray, cuts: np.ndarray, loci: np.ndarray
) -> list[int] | None:
    """Return interior breakpoint loci for a k-block solution, or None if
    `best_start` has no valid path (should not happen for a finite cost)."""
    j = cuts.shape[0] - 1
    block_end_meta: list[int] = []
    kk = k
    while kk >= 1:
        block_end_meta.append(j)
        i = int(best_start[kk, j])
        if i < 0:
            return None
        j = i - 1
        kk -= 1
    block_end_meta.reverse()
    return [int(loci[int(cuts[b])]) for b in block_end_meta[:-1]]


_NO_BP_LIMIT = 2_000_000_000  # exceeds any real chromosome's length in bp


def optimal_partitions(
    cov: ChromosomeCovariance,
    region_start: int,
    region_end: int,
    candidate_loci: Sequence[int],
    min_size: int,
    max_size: int,
    max_k: int,
    thr_r2: float = 0.0,
    max_r2: float = 1.0,
    min_size_bp: int | None = None,
    max_size_bp: int | None = None,
) -> list[DPPartition]:
    """Solve the exact-optimal LD-block partition for every achievable K.

    Args:
        cov: Chromosome (or wider-region) covariance triplets, e.g. from
            :func:`ldetect_lite._util.covariance_array.load_chromosome_covariance`.
        region_start: First SNP position in scope (inclusive).
        region_end: Last SNP position in scope (inclusive).
        candidate_loci: Positions eligible to be a breakpoint. Strictly
            interior positions are kept; `region_start`/`region_end` and
            anything outside them are dropped. See
            :func:`generate_filter_candidates` for the default candidate
            source, or pass `cov.loci` for an exact-over-every-SNP search.
        min_size: Minimum SNPs per block.
        max_size: Maximum SNPs per block.
        max_k: Solve for every block count from 1 up to this many (capped at
            the number of available meta-nodes).
        thr_r2: Drop pairs with r^2 below this threshold before optimizing
            (mirrors `snp_ldsplit()`'s `thr_r2`: ignores noise so the cost
            curve isn't inflated by near-zero correlations).
        max_r2: Forbid any candidate cut that would separate a pair whose
            r^2 exceeds this (mirrors `snp_ldsplit()`'s `max_r2`).
        min_size_bp: Minimum physical (bp) width per block. `None` (default)
            means no minimum. A block must satisfy this *and* `min_size`.
        max_size_bp: Maximum physical (bp) width per block. `None` (default)
            means no maximum. A block must satisfy this *and* `max_size`.
            Physical-distance analog of `snp_ldsplit()`'s `pos_scaled`
            (genetic-distance) constraint -- SNP-count bounds alone give
            wildly different physical block sizes across variable-density
            regions.

    Returns:
        One :class:`DPPartition` per achievable K (some K may have no
        feasible solution given the size constraints and are omitted).
    """
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size")
    if max_k < 1:
        raise ValueError("max_k must be >= 1")
    if min_size_bp is not None and min_size_bp < 0:
        raise ValueError("min_size_bp must be >= 0")
    if (
        min_size_bp is not None
        and max_size_bp is not None
        and max_size_bp < min_size_bp
    ):
        raise ValueError("max_size_bp must be >= min_size_bp")

    loci_mask = (cov.loci >= region_start) & (cov.loci <= region_end)
    loci = cov.loci[loci_mask]
    if loci.size == 0:
        raise ValueError(f"No covariance loci within [{region_start}, {region_end}]")
    region_end_idx = loci.size - 1

    pair_mask = (
        (cov.i_pos >= region_start)
        & (cov.i_pos <= region_end)
        & (cov.j_pos >= region_start)
        & (cov.j_pos <= region_end)
    )
    i_pos = cov.i_pos[pair_mask]
    j_pos = cov.j_pos[pair_mask]
    r2 = cov.r2[pair_mask]
    if thr_r2 > 0.0:
        keep = r2 >= thr_r2
        i_pos, j_pos, r2 = i_pos[keep], j_pos[keep], r2[keep]

    i_idx = np.searchsorted(loci, i_pos)
    j_idx = np.searchsorted(loci, j_pos)

    cand_arr = np.array(sorted({int(c) for c in candidate_loci}), dtype=np.int64)
    cand_arr = cand_arr[(cand_arr > region_start) & (cand_arr < region_end)]
    candidate_idx = np.searchsorted(loci, cand_arr)
    in_bounds = candidate_idx < loci.size
    matches = np.zeros_like(in_bounds)
    matches[in_bounds] = loci[candidate_idx[in_bounds]] == cand_arr[in_bounds]
    candidate_idx = np.unique(candidate_idx[matches])

    if max_r2 < 1.0 and candidate_idx.size:
        candidate_idx = _apply_max_r2_constraint(
            candidate_idx, i_idx, j_idx, r2, max_r2
        )

    cuts = np.unique(np.concatenate([candidate_idx, [region_end_idx]]))
    starts = _block_starts(cuts)
    n_meta = cuts.shape[0]
    max_k_eff = min(max_k, n_meta)

    start_bp = loci[starts].astype(np.int64)
    end_bp = loci[cuts].astype(np.int64)
    min_size_bp_eff = 0 if min_size_bp is None else min_size_bp
    max_size_bp_eff = _NO_BP_LIMIT if max_size_bp is None else max_size_bp

    W = _build_weight_matrix(i_idx, j_idx, r2, cuts)
    L = _suffix_matrix(W)
    E = _compute_E(
        L, starts, cuts, min_size, max_size,
        start_bp, end_bp, min_size_bp_eff, max_size_bp_eff,
    )
    C, best_start = _dp(E, max_k_eff)

    results: list[DPPartition] = []
    for k in range(1, max_k_eff + 1):
        cost = C[k, n_meta - 1]
        if not np.isfinite(cost):
            continue
        interior = _reconstruct(k, best_start, cuts, loci)
        if interior is None:
            continue
        results.append(DPPartition(n_block=k, cost=float(cost), loci=interior))
    return results
