"""Fused direct-vector accumulator for the ``--fused-vector`` Step 2/3 path.

Built by *tee-ing* the exact ``CovarianceRowChunk`` stream that
``write_compact_covariance_partition_hdf5_append`` already consumes to persist
the HDF5 covariance partition (see ``shrinkage.calc_covariance``). Chunks are
passed through unchanged -- what gets persisted is untouched -- while this
module accumulates the per-locus correlation-sum vector fragment (matrix-to-
vector / raw minima input) from the same stream, by replaying buffered chunks
through the *existing* ``_accumulate_vector_chunk`` / ``_DiagVectorPartitionResult``
machinery in ``vector_array.py``, so the result is drop-in compatible with
``_merge_diag_vector_partition_result`` with no adaptation.

Why tee instead of a second kernel pass: an earlier prototype
(``hdf5-experiments-direct-vector-r2-zarr``) computed "direct vector mode" via
an independent second invocation of the pairwise LD kernel, and left an
unresolved chr9/chr14 vector-value residual versus the persisted-cache path.
Tee-ing guarantees this accumulator sees bit-identical ``(lo, hi, shrink_ld)``
values, in the same order, as what is persisted -- the only independently
derived quantity is the per-locus diagonal (``shrinkage._diag_values_impl``),
a closed-form vectorized formula, not a second nested-loop kernel pass, and is
unit-tested against the persisted diagonal for exact equality.

The fragment is never written to disk: it travels back from the covariance
worker process as an ordinary return value (small -- per-locus sums, not
per-pair covariance) and is merged directly into the vector output in the
parent process. There is no on-disk sidecar file.
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


@dataclass
class CovarianceSidecarAccumulator:
    """Tees a ``CovarianceRowChunk`` stream and builds the vector fragment.

    ``wrap()`` yields every chunk unchanged (splicing this into
    ``calc_covariance``'s write path does not change what gets persisted)
    while buffering it. ``finalize_vector`` replays the buffered chunks --
    bounded by one partition's row count, the same data ``calc_covariance``
    already holds transiently -- to build the fragment, preserving per-chunk
    grouping so floating-point summation order matches what the post-hoc
    HDF5 reader would produce.

    If ``calc_covariance`` falls back off the single-pass write path partway
    through (a rare defensive fallback, not expected on well-formed real
    data), the buffered chunks are an incomplete view of the partition.
    ``mark_invalid()`` flags this so ``finalize_vector`` refuses to produce a
    silently-wrong fragment; the caller treats a missing fragment the same
    way it already treats a skipped/cached partition -- falling back to the
    post-hoc Step 3 read for the whole run.
    """

    diag_pos: np.ndarray | None = None
    diag_val: np.ndarray | None = None
    invalid: bool = False
    _buffered: list[CovarianceRowChunk] = field(default_factory=list)

    def set_diagonals(self, diag_pos: np.ndarray, diag_val: np.ndarray) -> None:
        """Set the vectorized diagonal precompute (``shrinkage._diag_values_impl``).

        Called by ``calc_covariance`` before the row-chunk stream is wrapped,
        since the diagonal is cheap to derive from ``hap_sums`` alone and
        known before any pairwise row is generated.
        """
        self.diag_pos = diag_pos
        self.diag_val = diag_val

    def mark_invalid(self) -> None:
        """Flag that the buffered chunks no longer reflect the full partition."""
        self.invalid = True

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
        if self.invalid:
            raise RuntimeError(
                "CovarianceSidecarAccumulator is invalid: this partition did "
                "not complete the single-pass write path, so its buffered "
                "chunks are incomplete"
            )
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
