"""Prototype: fused direct-vector and metric coverage-array sidecars.

Both sidecars are built by *tee-ing* the exact ``CovarianceRowChunk`` stream
that ``write_compact_covariance_partition_hdf5_chunks`` /
``_write_compact_covariance_partition_hdf5_append`` already consume to
persist the HDF5 covariance partition (see ``shrinkage.calc_covariance``).
Chunks are passed through unchanged -- what gets persisted is untouched --
while this module accumulates two additional, smaller artifacts from the
same stream:

- **Vector fragment**: the per-locus correlation-sum vector fragment
  (matrix-to-vector / raw minima input), built by replaying buffered chunks
  through the *existing* ``_accumulate_vector_chunk`` /
  ``_DiagVectorPartitionResult`` machinery in ``vector_array.py``, so the
  result is drop-in compatible with ``_merge_diag_vector_partition_result``
  with no adaptation.
- **Metric coverage fragment**: a sparse position-keyed difference array
  (``diff[lo] += r2``, ``diff[hi] -= r2``) whose prefix sum at any locus
  gives the total r² mass of pairs whose ``[lo, hi)`` interval straddles it
  -- see ``notes/logs/covariance-cache-redesign-plan.md`` for why this is
  exact only under a single-breakpoint-crossing assumption.

Why tee instead of a second kernel pass: an earlier prototype
(``hdf5-experiments-direct-vector-r2-zarr``) computed "direct vector mode"
via an independent second invocation of the pairwise LD kernel, and left an
unresolved chr9/chr14 vector-value residual versus the persisted-cache path.
Tee-ing guarantees the sidecars see bit-identical ``(lo, hi, shrink_ld)``
values, in the same order, as what is persisted -- the only independently
derived quantity is the per-locus diagonal (``shrinkage._diag_values_impl``),
which is a closed-form vectorized formula, not a second nested-loop kernel
pass, and is unit-tested against the persisted diagonal for exact equality.

Ownership across overlapping partitions is intentionally **not** unified
between the two sidecars: the vector fragment reuses the existing
"pair-center falls in this partition's midpoint-bounded span" rule
(``_accumulate_vector_chunk`` / ``_plan_diag_vector_partitions``), while the
metric coverage fragment reuses ``_metric_arrays_from_partitions``'s
existing "lower (``lo``) endpoint falls in ``(lower_min, lower_max]``" rule.
These are two different, already-existing conventions in this codebase; the
prototype matches each sidecar to the convention its own post-hoc reader
already uses rather than inventing a third, unvalidated rule.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from ldetect_lite._util.vector_array import (
    _accumulate_vector_chunk,
    _DiagVectorPartitionResult,
)
from ldetect_lite.io.covariance_hdf5 import CovarianceRowChunk


@dataclass(frozen=True)
class MetricCoverageFragment:
    """Sparse position-keyed difference-array contribution for one partition."""

    positions: np.ndarray
    deltas: np.ndarray


def _normalize_pairs(
    *,
    diag_pos: np.ndarray,
    diag_val: np.ndarray,
    row_lo: np.ndarray,
    row_hi: np.ndarray,
    row_shrink: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize one row chunk to ``r2 = shrink**2 / (diag_lo * diag_hi)``.

    Mirrors the normalization steps in ``_accumulate_vector_chunk``
    (``vector_array.py``) exactly (same has-diagonal + positive-diagonal
    checks), duplicated rather than imported because that function bundles
    normalization together with its own (different) ownership/accumulation
    logic.
    """
    if row_lo.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float64)

    diag_lo_idx = np.searchsorted(diag_pos, row_lo)
    diag_hi_idx = np.searchsorted(diag_pos, row_hi)
    has_diag = (diag_lo_idx < diag_pos.size) & (diag_hi_idx < diag_pos.size)
    safe_lo_idx = np.minimum(diag_lo_idx, diag_pos.size - 1)
    safe_hi_idx = np.minimum(diag_hi_idx, diag_pos.size - 1)
    has_diag &= (diag_pos[safe_lo_idx] == row_lo) & (diag_pos[safe_hi_idx] == row_hi)
    if not np.any(has_diag):
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float64)

    row_lo = row_lo[has_diag]
    row_hi = row_hi[has_diag]
    row_shrink = row_shrink[has_diag]
    diag_lo = diag_val[diag_lo_idx[has_diag]]
    diag_hi = diag_val[diag_hi_idx[has_diag]]

    positive = (diag_lo > 0.0) & (diag_hi > 0.0)
    if not np.any(positive):
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float64)

    row_lo = row_lo[positive]
    row_hi = row_hi[positive]
    row_shrink = row_shrink[positive]
    diag_lo = diag_lo[positive]
    diag_hi = diag_hi[positive]

    r2 = row_shrink * row_shrink / (diag_lo * diag_hi)
    return row_lo, row_hi, r2


@dataclass
class CovarianceSidecarAccumulator:
    """Tees a ``CovarianceRowChunk`` stream and builds sidecar fragments.

    ``wrap()`` yields every chunk unchanged (splicing this into
    ``calc_covariance``'s write path does not change what gets persisted)
    while buffering it. ``finalize_vector`` / ``finalize_metric_coverage``
    replay the buffered chunks -- bounded by one partition's row count, the
    same data ``calc_covariance`` already holds transiently -- to build each
    sidecar fragment, preserving per-chunk grouping so floating-point
    summation order matches what the post-hoc HDF5 reader would produce.
    """

    diag_pos: np.ndarray | None = None
    diag_val: np.ndarray | None = None
    _buffered: list[CovarianceRowChunk] = field(default_factory=list)

    def set_diagonals(self, diag_pos: np.ndarray, diag_val: np.ndarray) -> None:
        """Set the vectorized diagonal precompute (``shrinkage._diag_values_impl``).

        Called by ``calc_covariance`` before the row-chunk stream is wrapped,
        since the diagonal is cheap to derive from ``hap_sums`` alone and
        known before any pairwise row is generated.
        """
        self.diag_pos = diag_pos
        self.diag_val = diag_val

    def wrap(
        self, chunks: Iterator[CovarianceRowChunk]
    ) -> Iterator[CovarianceRowChunk]:
        for chunk in chunks:
            self._buffered.append(chunk)
            yield chunk

    def finalize_vector(
        self,
        *,
        end: int,
        next_start: int | None,
        snp_last: int,
        center_lower_bound: int,
        center_lower_inclusive: bool,
        checkpoint: str = "fused_sidecar_vector",
    ) -> _DiagVectorPartitionResult:
        """Build the vector fragment, drop-in compatible with the existing merge.

        Parameters mirror ``_compute_diag_vector_partition_hdf5`` exactly so
        the result can be passed straight to
        ``_merge_diag_vector_partition_result`` with no adaptation.
        """
        if self.diag_pos is None or self.diag_val is None:
            raise ValueError("set_diagonals() must be called before finalize_vector()")
        diag_pos = self.diag_pos
        diag_val = self.diag_val
        if not self._buffered:
            loci = np.array([], dtype=np.int64)
            return _DiagVectorPartitionResult(
                loci=loci,
                sum_loci=loci,
                sum_values=np.array([], dtype=np.float64),
                end_locus=end,
                write_cutoff=end,
                profile={"checkpoint": checkpoint},
            )

        loci = np.unique(np.concatenate([chunk.lo for chunk in self._buffered]))

        if next_start is not None:
            end_locus = int((end + next_start) / 2)
            write_cutoff = next_start
        else:
            in_requested_range = loci[loci <= snp_last]
            if in_requested_range.size == 0:
                return _DiagVectorPartitionResult(
                    loci=loci,
                    sum_loci=np.array([], dtype=np.int64),
                    sum_values=np.array([], dtype=np.float64),
                    end_locus=end,
                    write_cutoff=end,
                    profile={"checkpoint": checkpoint},
                )
            end_locus = int(in_requested_range[-1])
            write_cutoff = end_locus

        center_hi = min(end_locus, snp_last)
        center_left = int(
            np.searchsorted(
                loci,
                center_lower_bound,
                side="left" if center_lower_inclusive else "right",
            )
        )
        center_right = int(np.searchsorted(loci, center_hi, side="right"))

        partition_sums = np.zeros(loci.size, dtype=np.float64)
        if diag_pos.size and center_left < center_right:
            for chunk in self._buffered:
                _accumulate_vector_chunk(
                    loci=loci,
                    diag_pos=diag_pos,
                    diag_val=diag_val,
                    row_lo=chunk.lo,
                    row_hi=chunk.hi,
                    row_shrink=chunk.shrink_ld,
                    center_left=center_left,
                    center_right=center_right,
                    partition_sums=partition_sums,
                )

        nonzero = partition_sums > 0.0
        return _DiagVectorPartitionResult(
            loci=loci,
            sum_loci=loci[nonzero],
            sum_values=partition_sums[nonzero],
            end_locus=end_locus,
            write_cutoff=write_cutoff,
            profile={"checkpoint": checkpoint},
        )

    def finalize_metric_coverage(
        self,
        *,
        lower_min: int,
        lower_max: int,
        lower_inclusive: bool,
    ) -> MetricCoverageFragment:
        """Build the metric coverage-array difference-array fragment.

        Ownership matches ``_metric_arrays_from_partitions``'s existing rule:
        a pair belongs to this partition iff its ``lo`` endpoint falls in
        ``(lower_min, lower_max]`` (``[lower_min, lower_max]`` for the first
        partition on a chromosome, via ``lower_inclusive``).
        """
        if self.diag_pos is None or self.diag_val is None:
            raise ValueError(
                "set_diagonals() must be called before finalize_metric_coverage()"
            )
        diag_pos = self.diag_pos
        diag_val = self.diag_val
        pos_parts: list[np.ndarray] = []
        delta_parts: list[np.ndarray] = []
        for chunk in self._buffered:
            row_lo, row_hi, r2 = _normalize_pairs(
                diag_pos=diag_pos,
                diag_val=diag_val,
                row_lo=chunk.lo,
                row_hi=chunk.hi,
                row_shrink=chunk.shrink_ld,
            )
            if row_lo.size == 0:
                continue
            # Off-diagonal pairs only; diagonal rows (lo == hi) never cross.
            off_diag = row_lo != row_hi
            row_lo = row_lo[off_diag]
            row_hi = row_hi[off_diag]
            r2 = r2[off_diag]
            if row_lo.size == 0:
                continue

            owned = row_lo >= lower_min if lower_inclusive else row_lo > lower_min
            owned &= row_lo <= lower_max
            if not np.any(owned):
                continue
            row_lo = row_lo[owned]
            row_hi = row_hi[owned]
            r2 = r2[owned]

            pos_parts.append(row_lo)
            delta_parts.append(r2)
            pos_parts.append(row_hi)
            delta_parts.append(-r2)

        if not pos_parts:
            empty_i = np.array([], dtype=np.int64)
            return MetricCoverageFragment(
                positions=empty_i, deltas=np.array([], dtype=np.float64)
            )

        positions = np.concatenate(pos_parts)
        deltas = np.concatenate(delta_parts)
        order = np.argsort(positions, kind="stable")
        positions = positions[order]
        deltas = deltas[order]
        starts = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.flatnonzero(positions[1:] != positions[:-1]) + 1,
            )
        )
        grouped_positions = positions[starts]
        grouped_deltas = np.add.reduceat(deltas, starts)
        return MetricCoverageFragment(
            positions=grouped_positions, deltas=grouped_deltas
        )


def merge_metric_coverage_fragments(
    fragments: list[MetricCoverageFragment],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge per-partition difference-array fragments into one prefix sum.

    Returns ``(positions, coverage)`` where ``coverage[k]`` is the total r²
    mass of pairs whose ``[lo, hi)`` interval straddles ``positions[k]``
    (and every locus between ``positions[k]`` and ``positions[k+1]``).
    """
    nonempty = [f for f in fragments if f.positions.size]
    if not nonempty:
        empty = np.array([], dtype=np.int64)
        return empty, np.array([], dtype=np.float64)

    positions = np.concatenate([f.positions for f in nonempty])
    deltas = np.concatenate([f.deltas for f in nonempty])
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    deltas = deltas[order]
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(positions[1:] != positions[:-1]) + 1,
        )
    )
    grouped_positions = positions[starts]
    grouped_deltas = np.add.reduceat(deltas, starts)
    coverage = np.cumsum(grouped_deltas)
    return grouped_positions, coverage


def metric_coverage_sum_at_breakpoints(
    positions: np.ndarray, coverage: np.ndarray, breakpoints: np.ndarray
) -> float:
    """Look up total crossing ``r2`` mass at a breakpoint set (lookup + add).

    Exact only under the single-breakpoint-crossing assumption (no pair's
    ``[lo, hi)`` interval contains more than one breakpoint) -- see
    ``notes/logs/covariance-cache-redesign-plan.md``.
    """
    if positions.size == 0 or breakpoints.size == 0:
        return 0.0
    idx = np.searchsorted(positions, breakpoints, side="right") - 1
    valid = idx >= 0
    return float(np.sum(coverage[idx[valid]]))
