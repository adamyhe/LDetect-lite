"""Prototype vectorized pairwise-LD kernels for benchmarking against the
Numba pair-loop kernel in ``ldetect2.shrinkage``.

Research-only: nothing in ``ldetect2`` imports this module. It exists to
measure whether row-vectorized, chunked-matmul, or bit-packed-popcount
kernels beat ``_pairwise_ld_impl`` before committing to any recompute-cache
redesign (see notes/covariance-streaming-cache-implementation-note.md). Most
of this module is plain NumPy; ``_popcount_sum_rows`` additionally uses numba
(already a hard dependency of the package) for its inner word loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

LdKernelResult = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

__all__ = [
    "LdKernelResult",
    "pack_haplotypes",
    "pairwise_ld_row_vectorized",
    "pairwise_ld_chunked_matmul",
    "pairwise_ld_bitpacked_popcount",
]

_F = TypeVar("_F", bound=Callable[..., Any])

try:
    from numba import njit

    _numba_decorator = njit(cache=True)

    def _njit_fallback(fn: _F) -> _F:
        return _numba_decorator(fn)  # type: ignore[no-any-return]
except ImportError:

    def _njit_fallback(fn: _F) -> _F:
        return fn


def _empty_result() -> LdKernelResult:
    return (
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )


def _finalize_row(
    n11: np.ndarray,
    i: int,
    j_idx: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    n_total: float,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Apply the shared f11/d_naive/ds2/cutoff/diagonal-correction formula to
    one row's precomputed n11 vector. Returns None if no partner in this row
    survives the cutoff, else (ii, jj, d_naive, ds2) all length-matched.
    """
    df = gpos_arr[j_idx] - gpos_arr[i]
    ee = np.exp(-4.0 * ne * df / (2.0 * n_ind))
    f11 = n11.astype(np.float64) / n_total
    f1 = hap_sums[i] / n_total
    f2 = hap_sums[j_idx] / n_total
    d_naive = f11 - f1 * f2
    ds2 = (1.0 - theta) ** 2 * d_naive * ee

    mask = np.abs(ds2) >= cutoff
    if not mask.any():
        return None

    ds2_masked = ds2[mask]
    if mask[0]:
        # j_idx[0] == i, so position 0 of the masked row is the diagonal.
        ds2_masked[0] += (theta / 2.0) * (1.0 - theta / 2.0)

    ii_masked = np.full(int(mask.sum()), i, dtype=np.int32)
    return ii_masked, j_idx[mask].astype(np.int32), d_naive[mask], ds2_masked


def pack_haplotypes(hap_mat: np.ndarray) -> np.ndarray:
    """Pack a ``(n_snps, n_haps)`` uint8 0/1 array into ``uint64`` words.

    Haplotype index ``k`` maps to word ``k // 64``, bit ``k % 64``
    (LSB-relative: bit 0 has value 1). Unused high bits in the last word are
    zero by construction (the padding columns come from ``np.zeros`` and are
    never written to), so ``popcount(packed[i] & packed[j])`` over the full
    row of words always equals ``sum(hap_mat[i] * hap_mat[j])``.
    """
    n_snps, n_haps = hap_mat.shape
    n_words = (n_haps + 63) // 64
    padded_n_haps = n_words * 64

    if padded_n_haps == n_haps:
        padded = hap_mat.astype(np.uint64)
    else:
        padded = np.zeros((n_snps, padded_n_haps), dtype=np.uint64)
        padded[:, :n_haps] = hap_mat

    packed = np.zeros((n_snps, n_words), dtype=np.uint64)
    for k in range(64):
        packed |= padded[:, k::64] << np.uint64(k)
    return packed


@_njit_fallback
def _popcount_sum_rows(bits: np.ndarray) -> np.ndarray:
    """Row-wise sum of 64-bit popcounts.

    ``bits`` is ``(n_rows, n_words)`` uint64; returns ``(n_rows,)`` int64 with
    the sum of popcounts per row (i.e. per (i, j) pair, summed across that
    row's packed words). Uses the portable SWAR/Hamming-weight bit trick
    (add/sub/shift/and/mul only) rather than a hardware popcount intrinsic,
    since correctness must not depend on ARM-vs-x86 codegen (measured
    performance will still vary by platform).
    """
    n_rows, n_words = bits.shape
    out = np.empty(n_rows, dtype=np.int64)
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)
    for r in range(n_rows):
        total = np.uint64(0)
        for w in range(n_words):
            x = bits[r, w]
            x = x - ((x >> np.uint64(1)) & m1)
            x = (x & m2) + ((x >> np.uint64(2)) & m2)
            x = (x + (x >> np.uint64(4))) & m4
            total += (x * h01) >> np.uint64(56)
        out[r] = np.int64(total)
    return out


def pairwise_ld_row_vectorized(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
) -> LdKernelResult:
    """Compute pairwise shrinkage LD by vectorizing each row's ``j`` range.

    Reproduces ``_pairwise_ld_impl``'s math and cutoff filter exactly: the
    ``(i, j)`` index set returned is identical to the reference kernel, and
    ``d_naive``/``ds2`` values match within floating-point tolerance. Row
    order is preserved but the caller should not depend on exact output
    ordering across kernel implementations.
    """
    n_snps = hap_mat.shape[0]
    n_total = float(hap_mat.shape[1])
    # n11 can reach n_haps (thousands); uint8 accumulation would overflow.
    hap_mat_i32 = hap_mat.astype(np.int32)

    ii_parts: list[np.ndarray] = []
    jj_parts: list[np.ndarray] = []
    d_naive_parts: list[np.ndarray] = []
    ds2_parts: list[np.ndarray] = []

    for i in range(n_snps):
        j_stop = int(j_stop_by_i[i])
        if j_stop <= i:
            continue

        j_idx = np.arange(i, j_stop, dtype=np.int64)
        n11 = hap_mat_i32[i:j_stop] @ hap_mat_i32[i]

        row_result = _finalize_row(
            n11, i, j_idx, gpos_arr, hap_sums, n_total, ne, n_ind, theta, cutoff
        )
        if row_result is None:
            continue
        ii_parts.append(row_result[0])
        jj_parts.append(row_result[1])
        d_naive_parts.append(row_result[2])
        ds2_parts.append(row_result[3])

    if not ii_parts:
        return _empty_result()

    return (
        np.concatenate(ii_parts),
        np.concatenate(jj_parts),
        np.concatenate(d_naive_parts),
        np.concatenate(ds2_parts),
    )


def pairwise_ld_bitpacked_popcount(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
) -> LdKernelResult:
    """Compute pairwise shrinkage LD via bit-packed AND + SWAR popcount.

    Packs ``hap_mat`` into uint64 words once (cost amortized over all
    pairs), replacing the reference kernel's O(n_haps) elementwise
    multiply-sum n11 computation with O(n_haps / 64) word-level AND +
    popcount. Reproduces ``_pairwise_ld_impl``'s ``(i, j)`` index set
    exactly; ``d_naive``/``ds2`` match within floating-point tolerance. See
    ``pack_haplotypes`` for the bit-index convention. Correctness of the
    popcount step does not depend on ARM-vs-x86 codegen (portable SWAR bit
    trick), though measured speedups will.
    """
    n_snps = hap_mat.shape[0]
    n_total = float(hap_mat.shape[1])
    packed = pack_haplotypes(hap_mat)

    ii_parts: list[np.ndarray] = []
    jj_parts: list[np.ndarray] = []
    d_naive_parts: list[np.ndarray] = []
    ds2_parts: list[np.ndarray] = []

    for i in range(n_snps):
        j_stop = int(j_stop_by_i[i])
        if j_stop <= i:
            continue

        j_idx = np.arange(i, j_stop, dtype=np.int64)
        and_bits = packed[i:j_stop] & packed[i]
        n11 = _popcount_sum_rows(and_bits)

        row_result = _finalize_row(
            n11, i, j_idx, gpos_arr, hap_sums, n_total, ne, n_ind, theta, cutoff
        )
        if row_result is None:
            continue
        ii_parts.append(row_result[0])
        jj_parts.append(row_result[1])
        d_naive_parts.append(row_result[2])
        ds2_parts.append(row_result[3])

    if not ii_parts:
        return _empty_result()

    return (
        np.concatenate(ii_parts),
        np.concatenate(jj_parts),
        np.concatenate(d_naive_parts),
        np.concatenate(ds2_parts),
    )


def pairwise_ld_chunked_matmul(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
    tile_size: int = 1024,
) -> LdKernelResult:
    """Compute pairwise shrinkage LD via tiled dense matrix multiplication.

    Tiles are skipped whenever no row in the row-tile can reach that far
    (bounded by ``j_stop_by_i``), avoiding wasted dense compute on entirely
    out-of-range tiles. Reproduces ``_pairwise_ld_impl``'s ``(i, j)`` index
    set exactly; ``d_naive``/``ds2`` match within floating-point tolerance.
    """
    n_snps = hap_mat.shape[0]
    n_total = float(hap_mat.shape[1])
    # float32 matmul is exact for 0/1 inputs at any realistic n_haps (sums
    # stay far below 2**24); n11 is upcast to float64 before dividing.
    hap_f32 = hap_mat.astype(np.float32)

    ii_parts: list[np.ndarray] = []
    jj_parts: list[np.ndarray] = []
    d_naive_parts: list[np.ndarray] = []
    ds2_parts: list[np.ndarray] = []

    for i0 in range(0, n_snps, tile_size):
        i1 = min(i0 + tile_size, n_snps)
        row_tile_j_stop_max = int(j_stop_by_i[i0:i1].max())

        for j0 in range(i0, row_tile_j_stop_max, tile_size):
            j1 = min(j0 + tile_size, n_snps)

            n11_tile = hap_f32[i0:i1] @ hap_f32[j0:j1].T

            i_idx = np.arange(i0, i1)[:, None]
            j_idx = np.arange(j0, j1)[None, :]
            valid = (j_idx >= i_idx) & (j_idx < j_stop_by_i[i0:i1, None])
            if not valid.any():
                continue

            gpos_i = gpos_arr[i0:i1][:, None]
            gpos_j = gpos_arr[j0:j1][None, :]
            df = gpos_j - gpos_i
            f11 = n11_tile.astype(np.float64) / n_total
            f1 = hap_sums[i0:i1][:, None] / n_total
            f2 = hap_sums[j0:j1][None, :] / n_total
            d_naive = f11 - f1 * f2
            # df < 0 in the lower-triangle corner (always masked out below by
            # `valid`) can overflow exp(); those entries are never read.
            with np.errstate(over="ignore", invalid="ignore"):
                ee = np.exp(-4.0 * ne * df / (2.0 * n_ind))
                ds2 = (1.0 - theta) ** 2 * d_naive * ee

            cutoff_mask = np.abs(ds2) >= cutoff
            keep = valid & cutoff_mask
            if not keep.any():
                continue

            diag_mask = (i_idx == j_idx) & keep
            ds2 = np.where(
                diag_mask, ds2 + (theta / 2.0) * (1.0 - theta / 2.0), ds2
            )

            local_ii, local_jj = np.nonzero(keep)
            ii_parts.append((local_ii + i0).astype(np.int32))
            jj_parts.append((local_jj + j0).astype(np.int32))
            d_naive_parts.append(d_naive[keep])
            ds2_parts.append(ds2[keep])

    if not ii_parts:
        return _empty_result()

    return (
        np.concatenate(ii_parts),
        np.concatenate(jj_parts),
        np.concatenate(d_naive_parts),
        np.concatenate(ds2_parts),
    )
