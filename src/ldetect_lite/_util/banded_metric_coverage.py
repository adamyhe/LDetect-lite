"""Prototype: exact multi-breakpoint crossing-sum, two storage/query variants.

Follow-up to ``covariance_sidecars.py``'s flat 1D difference-array metric
coverage sidecar, which
``tests/test_covariance_sidecars.py::test_metric_coverage_violates_single_crossing_assumption_when_breakpoints_are_close``
proved is not just theoretically fragile but wrong on real data (chr2 fixture:
91% of surviving pairs span more than the minimum real breakpoint gap). This
module replaces it with an exact decomposition that works for any number of
breakpoints a pair crosses:

    sum_crossing = total_mass - sum_over_blocks(intra_block_mass(block))

where blocks are the position ranges between consecutive breakpoints. A pair
fully contained in one block contributes to exactly one ``intra_block_mass``
term and is correctly excluded; a pair crossing N>=1 breakpoints is never
"intra" to any single block and is correctly included in ``total_mass`` with
no double-count -- unlike the flat difference array, which double-counted a
pair once per breakpoint it crossed.

Two variants of ``intra_block_mass``, matching two different storage/compute
tradeoffs (see the docstrings below and
``tests/test_banded_metric_coverage.py`` for the size/query-time
measurements requested against v2's lo-less compact cache):

- ``sum_crossing_linear_scan``: no extra storage beyond the v2 compact cache
  (``compact_schema_v2.py``) -- one O(n_rows) pass per evaluation, using the
  v2 cache's own ``lo_offsets``/``hi_idx``/``shrink_ld`` arrays directly.
- ``MergeSortRangeSumTree``: a persisted O(n log n)-space structure enabling
  O(log^2 n) queries *without* touching the v2 cache's per-row arrays at
  query time -- at the cost of extra storage for the tree itself.

Both are exact (validated against ``metric_from_arrays`` bit-for-bit /
near-ULP, including the multi-crossing case the flat difference array got
wrong). Neither is wired into ``calc_covariance``, the CLI, or
``covariance_sidecars.py`` -- prototype only, per the same scoping as
``compact_schema_v2.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ldetect_lite.io.covariance_hdf5 import _h5py


def normalize_v2_pairs(
    *,
    positions: np.ndarray,
    lo_rank_values: np.ndarray,
    lo_offsets: np.ndarray,
    hi_idx: np.ndarray,
    shrink_ld: np.ndarray,
    diag_pos: np.ndarray,
    diag_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize v2 rows to off-diagonal ``r2``, without reconstructing ``lo``.

    Mirrors ``covariance_sidecars._normalize_pairs``'s has-diagonal/positive
    checks, operating on v2's native rank-encoded arrays. Filtering can drop
    rows, so the CSR index (``lo_rank_values``/``lo_offsets``) is rebuilt
    over the surviving, still lo-sorted rows -- both crossing-sum variants
    below depend on that index staying valid for the arrays they're given.

    Returns ``(lo_rank_values, lo_offsets, hi_idx, r2)`` for the filtered rows.
    """
    row_counts = np.diff(lo_offsets)
    lo_rank_per_row = np.repeat(lo_rank_values, row_counts)
    lo_val = positions[lo_rank_per_row]
    hi_val = positions[hi_idx]

    last_diag = max(diag_pos.size - 1, 0)
    diag_lo_idx = np.searchsorted(diag_pos, lo_val)
    diag_hi_idx = np.searchsorted(diag_pos, hi_val)
    has_diag = (diag_lo_idx < diag_pos.size) & (diag_hi_idx < diag_pos.size)
    safe_lo = np.minimum(diag_lo_idx, last_diag)
    safe_hi = np.minimum(diag_hi_idx, last_diag)
    has_diag &= (diag_pos[safe_lo] == lo_val) & (diag_pos[safe_hi] == hi_val)

    off_diag = lo_val != hi_val
    valid = has_diag & off_diag
    valid_idx = np.flatnonzero(valid)
    diag_lo_val = diag_val[safe_lo[valid_idx]]
    diag_hi_val = diag_val[safe_hi[valid_idx]]
    positive = (diag_lo_val > 0) & (diag_hi_val > 0)
    valid_idx = valid_idx[positive]

    lo_rank_out = lo_rank_per_row[valid_idx]
    hi_idx_out = hi_idx[valid_idx]
    shrink_out = shrink_ld[valid_idx]
    diag_lo_final = diag_val[safe_lo[valid_idx]]
    diag_hi_final = diag_val[safe_hi[valid_idx]]
    r2_out = shrink_out * shrink_out / (diag_lo_final * diag_hi_final)

    if lo_rank_out.size == 0:
        new_lo_rank_values = np.array([], dtype=lo_rank_values.dtype)
        new_lo_offsets = np.array([0], dtype=np.int64)
    else:
        starts = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.flatnonzero(lo_rank_out[1:] != lo_rank_out[:-1]) + 1,
            )
        )
        new_lo_rank_values = lo_rank_out[starts]
        new_lo_offsets = np.concatenate(
            (starts, np.array([lo_rank_out.size], dtype=np.int64))
        ).astype(np.int64, copy=False)

    return new_lo_rank_values, new_lo_offsets, hi_idx_out, r2_out


def _block_bounds_for_breakpoints(
    positions: np.ndarray, breakpoints: np.ndarray
) -> np.ndarray:
    """Rank boundaries partitioning ``positions`` to match ``metric_from_arrays``.

    ``metric_from_arrays`` assigns block_id(x) = count(breakpoints < x), via
    ``searchsorted(bp, x, side="left")`` -- so a locus *exactly equal* to a
    breakpoint belongs to the block *before* it (block_id counts breakpoints
    strictly less than itself), not the block starting at it. In rank space
    that means a breakpoint's own rank is the *last* rank included in the
    block ending there, i.e. the boundary is ``rank + 1``, not ``rank`` --
    getting this off by one silently misclassifies every pair with an
    endpoint exactly on a breakpoint locus (breakpoints are chosen from real
    loci, so this is the common case, not an edge case).
    """
    if positions.size == 0 or breakpoints.size == 0:
        return np.array([0, 0], dtype=np.int64)
    ranks = np.searchsorted(positions, breakpoints, side="left")
    ranks = np.unique(np.clip(ranks, 0, positions.size - 1))
    return np.concatenate(([0], ranks + 1, [positions.size])).astype(np.int64)


def sum_crossing_linear_scan(
    *,
    positions: np.ndarray,
    lo_rank_values: np.ndarray,
    lo_offsets: np.ndarray,
    hi_idx: np.ndarray,
    r2: np.ndarray,
    breakpoints: np.ndarray,
) -> float:
    """Exact crossing-sum via one grouped pass over the v2 cache's own arrays.

    No structure beyond the v2 compact partition itself is read or built.
    ``positions``/``lo_rank_values``/``lo_offsets``/``hi_idx``/``shrink_ld``
    (this function takes normalized ``r2``, not ``shrink_ld`` -- callers
    normalize the same way ``covariance_sidecars._normalize_pairs`` does)
    come straight from ``compact_schema_v2.read_v2_index_arrays``.
    """
    total_mass = float(np.sum(r2))
    if hi_idx.size == 0 or breakpoints.size == 0:
        return 0.0

    block_bounds = _block_bounds_for_breakpoints(positions, breakpoints)

    # Row-index bounds (into the globally lo-sorted row arrays) for each
    # block's lo-range, found via one searchsorted into lo_rank_values (a
    # per-distinct-lo array, not per-row) then translated through lo_offsets.
    group_bounds = np.searchsorted(lo_rank_values, block_bounds)
    row_bounds = lo_offsets[group_bounds]

    intra = 0.0
    for i in range(block_bounds.size - 1):
        row_start, row_stop = row_bounds[i], row_bounds[i + 1]
        if row_stop <= row_start:
            continue
        block_hi = hi_idx[row_start:row_stop]
        block_r2 = r2[row_start:row_stop]
        intra += float(np.sum(block_r2[block_hi < block_bounds[i + 1]]))

    return total_mass - intra


def _node_query(hi_sorted: np.ndarray, prefix: np.ndarray, threshold: int) -> float:
    idx = int(np.searchsorted(hi_sorted, threshold))
    return float(prefix[idx - 1]) if idx > 0 else 0.0


@dataclass
class MergeSortRangeSumTree:
    """Persisted 2D range-sum: sum of r2 for row-index in [l, r) with hi_rank < t.

    A standard merge-sort/segment tree: leaves are individual rows (in the
    same lo-sorted order the v2 cache already stores them in); each internal
    node holds its range's ``(hi_rank, r2)`` pairs sorted by ``hi_rank`` with
    a running prefix sum, built bottom-up by merging children. O(n log n)
    space (each row appears in O(log n) nodes), O(log^2 n) per range query
    (O(log n) canonical nodes, one binary search each).
    """

    size: int
    n: int
    node_hi: list[np.ndarray]
    node_prefix: list[np.ndarray]

    @classmethod
    def build(cls, hi_rank: np.ndarray, r2: np.ndarray) -> MergeSortRangeSumTree:
        n = hi_rank.size
        size = 1
        while size < max(n, 1):
            size *= 2
        node_hi: list[np.ndarray] = [np.array([], dtype=hi_rank.dtype)] * (2 * size)
        node_prefix: list[np.ndarray] = [np.array([], dtype=np.float64)] * (2 * size)
        for i in range(n):
            node_hi[size + i] = hi_rank[i : i + 1]
            node_prefix[size + i] = r2[i : i + 1].astype(np.float64, copy=True)
        for node in range(size - 1, 0, -1):
            left_hi, right_hi = node_hi[2 * node], node_hi[2 * node + 1]
            if left_hi.size == 0 and right_hi.size == 0:
                continue
            left_r2 = np.diff(node_prefix[2 * node], prepend=0.0)
            right_r2 = np.diff(node_prefix[2 * node + 1], prepend=0.0)
            merged_hi = np.concatenate([left_hi, right_hi])
            merged_r2 = np.concatenate([left_r2, right_r2])
            order = np.argsort(merged_hi, kind="stable")
            node_hi[node] = merged_hi[order]
            node_prefix[node] = np.cumsum(merged_r2[order])
        return cls(size=size, n=n, node_hi=node_hi, node_prefix=node_prefix)

    def query(self, left: int, right: int, threshold: int) -> float:
        """Sum of r2 for leaf index in [left, right) with hi_rank < threshold."""
        left += self.size
        right += self.size
        total = 0.0
        while left < right:
            if left & 1:
                total += _node_query(
                    self.node_hi[left], self.node_prefix[left], threshold
                )
                left += 1
            if right & 1:
                right -= 1
                total += _node_query(
                    self.node_hi[right], self.node_prefix[right], threshold
                )
            left >>= 1
            right >>= 1
        return total

    def to_hdf5(self, path: Path) -> None:
        h5py = _h5py()
        node_offsets = np.zeros(2 * self.size + 1, dtype=np.int64)
        for i, arr in enumerate(self.node_hi):
            node_offsets[i + 1] = node_offsets[i] + arr.size
        flat_hi = np.concatenate(self.node_hi) if self.node_hi else np.array([])
        flat_prefix = (
            np.concatenate(self.node_prefix) if self.node_prefix else np.array([])
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5:
            h5.attrs["size"] = self.size
            h5.attrs["n"] = self.n
            h5.create_dataset("node_offsets", data=node_offsets)
            h5.create_dataset("flat_hi", data=flat_hi)
            h5.create_dataset("flat_prefix", data=flat_prefix)

    @classmethod
    def from_hdf5(cls, path: Path) -> MergeSortRangeSumTree:
        h5py = _h5py()
        with h5py.File(path, "r") as h5:
            size = int(h5.attrs["size"])
            n = int(h5.attrs["n"])
            node_offsets = np.asarray(h5["node_offsets"][:])
            flat_hi = np.asarray(h5["flat_hi"][:])
            flat_prefix = np.asarray(h5["flat_prefix"][:])
        node_hi = [
            flat_hi[node_offsets[i] : node_offsets[i + 1]] for i in range(2 * size)
        ]
        node_prefix = [
            flat_prefix[node_offsets[i] : node_offsets[i + 1]] for i in range(2 * size)
        ]
        return cls(size=size, n=n, node_hi=node_hi, node_prefix=node_prefix)

    def sum_crossing(
        self,
        *,
        positions: np.ndarray,
        lo_rank_values: np.ndarray,
        lo_offsets: np.ndarray,
        total_mass: float,
        breakpoints: np.ndarray,
    ) -> float:
        """Crossing-sum using only this persisted tree (no per-row array read)."""
        if self.n == 0 or breakpoints.size == 0:
            return 0.0
        block_bounds = _block_bounds_for_breakpoints(positions, breakpoints)
        group_bounds = np.searchsorted(lo_rank_values, block_bounds)
        row_bounds = lo_offsets[group_bounds]

        intra = 0.0
        for i in range(block_bounds.size - 1):
            row_start, row_stop = int(row_bounds[i]), int(row_bounds[i + 1])
            if row_stop <= row_start:
                continue
            intra += self.query(row_start, row_stop, int(block_bounds[i + 1]))
        return total_mass - intra
