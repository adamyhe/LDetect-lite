"""Wen/Stephens shrinkage LD estimation and chromosome partitioning."""

from __future__ import annotations

import gzip
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numba import njit

from ldetect_lite._util.reference_panel import (
    ReferencePanel,
    read_genetic_map,
    read_individuals,
    read_reference_panel,
    warn_reference_panel_skips,
    watterson_theta,
)
from ldetect_lite.io.covariance_hdf5 import (
    HDF5_DATASET_CHUNK_ROWS,
    CovarianceRowChunk,
    write_compact_covariance_partition_hdf5_append,
    write_compact_covariance_partition_hdf5_chunks,
    write_covariance_partition_hdf5,
)

COVARIANCE_WRITE_CHUNK_ROWS = 1_000_000

_F = TypeVar("_F", bound=Callable[..., Any])

_numba_cache_decorator = njit(cache=True)
_numba_inline_decorator = njit(cache=True, inline="always")


def _njit_cache(fn: _F) -> _F:
    """JIT-compile a kernel while preserving its type for mypy."""
    return _numba_cache_decorator(fn)  # type: ignore[no-any-return]


def _njit_inline(fn: _F) -> _F:
    """JIT-compile a small helper for inlining into other kernels."""
    return _numba_inline_decorator(fn)  # type: ignore[no-any-return]


@dataclass(frozen=True)
class _CovarianceInputs:
    """Array form of a reference panel ready for compiled covariance kernels."""

    hap_mat: np.ndarray
    gpos_arr: np.ndarray
    hap_sums: np.ndarray
    j_stop_by_i: np.ndarray
    pos_arr: np.ndarray
    assume_sorted_unique_rows: bool


# ---------------------------------------------------------------------------
# Pairwise LD kernel (Numba-accelerated)
# ---------------------------------------------------------------------------


@_njit_inline
def _shrink_ld_values(
    n11: float,
    n1x: float,
    nx1: float,
    gpos_i: float,
    gpos_j: float,
    inv_n_total: float,
    shrink_scale: float,
    decay_scale: float,
) -> tuple[float, float]:
    """Return naive and Wen/Stephens-shrunk covariance for one SNP pair."""
    f11 = n11 * inv_n_total
    f1 = n1x * inv_n_total
    f2 = nx1 * inv_n_total
    d_naive = f11 - f1 * f2
    ds2 = shrink_scale * d_naive * math.exp(-decay_scale * (gpos_j - gpos_i))
    return d_naive, ds2


@_njit_inline
def _popcount64(x: np.uint64) -> np.int64:
    """Portable uint64 popcount for Numba kernels."""
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return np.int64((x * h01) >> np.uint64(56))


@_njit_cache
def _pack_haplotypes_impl(hap_mat: np.ndarray) -> np.ndarray:
    """Pack a 0/1 haplotype matrix into uint64 words.

    Haplotype index ``k`` maps to word ``k // 64`` and bit ``k % 64``.
    Unused high bits in the final word remain zero.
    """
    n_snps = hap_mat.shape[0]
    n_haps = hap_mat.shape[1]
    n_words = (n_haps + 63) // 64
    packed = np.zeros((n_snps, n_words), dtype=np.uint64)
    one = np.uint64(1)
    for i in range(n_snps):
        for k in range(n_haps):
            if hap_mat[i, k] != 0:
                word = k // 64
                bit = k - word * 64
                packed[i, word] |= one << np.uint64(bit)
    return packed


@_njit_cache
def _count_pairwise_ld_impl(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
) -> int:
    """Count emitted uint8-backend pairs before materializing full output arrays."""
    n_snps = hap_mat.shape[0]
    n_haps = hap_mat.shape[1]
    inv_n_total = 1.0 / float(n_haps)
    shrink_scale = (1.0 - theta) * (1.0 - theta)
    decay_scale = (4.0 * ne) / (2.0 * n_ind)

    cnt = 0
    for i in range(n_snps):
        gpos1 = gpos_arr[i]
        j_stop = j_stop_by_i[i]
        n1x = hap_sums[i]
        for j in range(i, j_stop):
            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            nx1 = hap_sums[j]
            _, ds2 = _shrink_ld_values(
                n11,
                n1x,
                nx1,
                gpos1,
                gpos_arr[j],
                inv_n_total,
                shrink_scale,
                decay_scale,
            )

            if math.fabs(ds2) < cutoff:
                continue

            cnt += 1

    return cnt


@_njit_cache
def _pairwise_ld_impl(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise shrinkage LD for all SNP pairs above the cutoff.

    Args:
        hap_mat: Integer haplotype matrix, shape ``(n_snps, n_haps)``, dtype uint8.
        gpos_arr: Genetic positions (cM) per SNP, shape ``(n_snps,)``.
        ne: Effective population size.
        n_ind: Number of diploid individuals (``n_haps / 2``).
        theta: Watterson's theta.
        cutoff: Pairs with ``|ds2| < cutoff`` are excluded from output.

    Returns:
        Four equal-length arrays ``(i_idx, j_idx, d_naive, ds2)``.
    """
    n_snps = hap_mat.shape[0]
    n_haps = hap_mat.shape[1]
    inv_n_total = 1.0 / float(n_haps)
    shrink_scale = (1.0 - theta) * (1.0 - theta)
    decay_scale = (4.0 * ne) / (2.0 * n_ind)
    diag_adjust = (theta / 2.0) * (1.0 - theta / 2.0)
    n_pairs = _count_pairwise_ld_impl(
        hap_mat, gpos_arr, hap_sums, j_stop_by_i, ne, n_ind, theta, cutoff
    )

    ii = np.empty(n_pairs, dtype=np.int32)
    jj = np.empty(n_pairs, dtype=np.int32)
    d_naive_arr = np.empty(n_pairs, dtype=np.float64)
    ds2_arr = np.empty(n_pairs, dtype=np.float64)

    cnt = 0
    for i in range(n_snps):
        gpos1 = gpos_arr[i]
        j_stop = j_stop_by_i[i]
        n1x = hap_sums[i]
        for j in range(i, j_stop):
            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            nx1 = hap_sums[j]
            d_naive, ds2 = _shrink_ld_values(
                n11,
                n1x,
                nx1,
                gpos1,
                gpos_arr[j],
                inv_n_total,
                shrink_scale,
                decay_scale,
            )

            if math.fabs(ds2) < cutoff:
                continue

            if i == j:
                ds2 += diag_adjust

            ii[cnt] = i
            jj[cnt] = j
            d_naive_arr[cnt] = d_naive
            ds2_arr[cnt] = ds2
            cnt += 1

    return ii, jj, d_naive_arr, ds2_arr


@_njit_cache
def _count_pairwise_ld_by_i_impl(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
) -> np.ndarray:
    """Count compact covariance rows owned by each left-hand SNP index."""
    n_snps = hap_mat.shape[0]
    n_haps = hap_mat.shape[1]
    inv_n_total = 1.0 / float(n_haps)
    shrink_scale = (1.0 - theta) * (1.0 - theta)
    decay_scale = (4.0 * ne) / (2.0 * n_ind)
    counts = np.zeros(n_snps, dtype=np.int64)

    for i in range(n_snps):
        gpos1 = gpos_arr[i]
        j_stop = j_stop_by_i[i]
        n1x = hap_sums[i]
        row_count = 0
        for j in range(i, j_stop):
            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            nx1 = hap_sums[j]
            _, ds2 = _shrink_ld_values(
                n11,
                n1x,
                nx1,
                gpos1,
                gpos_arr[j],
                inv_n_total,
                shrink_scale,
                decay_scale,
            )

            if math.fabs(ds2) < cutoff:
                continue

            row_count += 1
        counts[i] = row_count

    return counts


@_njit_cache
def _pairwise_ld_compact_range_impl(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
    i_start: int,
    i_stop: int,
    n_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize compact uint8-backend pairs for a contiguous i-index range."""
    n_haps = hap_mat.shape[1]
    inv_n_total = 1.0 / float(n_haps)
    shrink_scale = (1.0 - theta) * (1.0 - theta)
    decay_scale = (4.0 * ne) / (2.0 * n_ind)
    diag_adjust = (theta / 2.0) * (1.0 - theta / 2.0)

    ii = np.empty(n_pairs, dtype=np.int32)
    jj = np.empty(n_pairs, dtype=np.int32)
    ds2_arr = np.empty(n_pairs, dtype=np.float64)

    cnt = 0
    for i in range(i_start, i_stop):
        gpos1 = gpos_arr[i]
        j_stop = j_stop_by_i[i]
        n1x = hap_sums[i]
        for j in range(i, j_stop):
            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            nx1 = hap_sums[j]
            _, ds2 = _shrink_ld_values(
                n11,
                n1x,
                nx1,
                gpos1,
                gpos_arr[j],
                inv_n_total,
                shrink_scale,
                decay_scale,
            )

            if math.fabs(ds2) < cutoff:
                continue

            if i == j:
                ds2 += diag_adjust

            ii[cnt] = i
            jj[cnt] = j
            ds2_arr[cnt] = ds2
            cnt += 1

    return ii, jj, ds2_arr


@_njit_cache
def _genetic_stop_bounds_impl(
    gpos_arr: np.ndarray,
    ne: float,
    n_ind: float,
    cutoff: float,
) -> np.ndarray:
    """Find the exclusive right bound for each SNP after genetic-distance pruning."""
    n_snps = gpos_arr.shape[0]
    stops = np.empty(n_snps, dtype=np.int32)
    if cutoff <= 0.0:
        stops.fill(n_snps)
        return stops
    decay_scale = (4.0 * ne) / (2.0 * n_ind)
    max_gdist = -math.log(cutoff) / decay_scale
    stop = 0
    for i in range(n_snps):
        if stop < i:
            stop = i
        gpos1 = gpos_arr[i]
        while stop < n_snps:
            df = gpos_arr[stop] - gpos1
            if df > max_gdist:
                break
            stop += 1
        stops[i] = stop
    return stops


@_njit_cache
def _pairwise_ld_compact_chunk_impl(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
    i_start: int,
    target_rows: int,
    capacity: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one bounded compact-output chunk with the uint8 backend."""
    n_haps = hap_mat.shape[1]
    inv_n_total = 1.0 / float(n_haps)
    n_snps = hap_mat.shape[0]
    shrink_scale = (1.0 - theta) * (1.0 - theta)
    decay_scale = (4.0 * ne) / (2.0 * n_ind)
    diag_adjust = (theta / 2.0) * (1.0 - theta / 2.0)

    ii = np.empty(capacity, dtype=np.int32)
    jj = np.empty(capacity, dtype=np.int32)
    ds2_arr = np.empty(capacity, dtype=np.float64)

    cnt = 0
    i = i_start
    while i < n_snps:
        gpos1 = gpos_arr[i]
        j_stop = j_stop_by_i[i]
        n1x = hap_sums[i]
        for j in range(i, j_stop):
            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            nx1 = hap_sums[j]
            _, ds2 = _shrink_ld_values(
                n11,
                n1x,
                nx1,
                gpos1,
                gpos_arr[j],
                inv_n_total,
                shrink_scale,
                decay_scale,
            )

            if math.fabs(ds2) < cutoff:
                continue

            if i == j:
                ds2 += diag_adjust

            ii[cnt] = i
            jj[cnt] = j
            ds2_arr[cnt] = ds2
            cnt += 1

        i += 1
        if cnt >= target_rows:
            break

    return i, ii[:cnt], jj[:cnt], ds2_arr[:cnt]


@_njit_cache
def _pairwise_ld_compact_chunk_bitpacked_impl(
    packed_hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    n_haps: int,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
    i_start: int,
    target_rows: int,
    capacity: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one bounded compact-output chunk with packed haplotypes."""
    inv_n_total = 1.0 / float(n_haps)
    n_snps = packed_hap_mat.shape[0]
    n_words = packed_hap_mat.shape[1]
    shrink_scale = (1.0 - theta) * (1.0 - theta)
    decay_scale = (4.0 * ne) / (2.0 * n_ind)
    diag_adjust = (theta / 2.0) * (1.0 - theta / 2.0)

    ii = np.empty(capacity, dtype=np.int32)
    jj = np.empty(capacity, dtype=np.int32)
    ds2_arr = np.empty(capacity, dtype=np.float64)

    cnt = 0
    i = i_start
    while i < n_snps:
        gpos1 = gpos_arr[i]
        j_stop = j_stop_by_i[i]
        n1x = hap_sums[i]
        for j in range(i, j_stop):
            if i == j:
                n11 = n1x
            else:
                n11_count = np.int64(0)
                for word in range(n_words):
                    n11_count += _popcount64(
                        packed_hap_mat[i, word] & packed_hap_mat[j, word]
                    )
                n11 = float(n11_count)
            nx1 = hap_sums[j]
            _, ds2 = _shrink_ld_values(
                n11,
                n1x,
                nx1,
                gpos1,
                gpos_arr[j],
                inv_n_total,
                shrink_scale,
                decay_scale,
            )

            if math.fabs(ds2) < cutoff:
                continue

            if i == j:
                ds2 += diag_adjust

            ii[cnt] = i
            jj[cnt] = j
            ds2_arr[cnt] = ds2
            cnt += 1

        i += 1
        if cnt >= target_rows:
            break

    return i, ii[:cnt], jj[:cnt], ds2_arr[:cnt]


def _compact_pair_chunks(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    pos_arr: np.ndarray,
    row_counts: np.ndarray,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
    chunk_rows: int,
) -> Iterator[CovarianceRowChunk]:
    """Yield compact rows after a separate per-SNP count pass.

    This is the memory-bounded fallback for compact output. The prior count pass
    lets each yielded chunk contain complete left-hand SNP rows while respecting
    the requested approximate chunk size.
    """
    max_row_count = int(row_counts.max(initial=0))
    target_rows = max(int(chunk_rows), max_row_count, 1)
    i_start = 0
    n_snps = row_counts.size
    while i_start < n_snps:
        while i_start < n_snps and row_counts[i_start] == 0:
            i_start += 1
        if i_start >= n_snps:
            break

        n_pairs = 0
        i_stop = i_start
        while i_stop < n_snps:
            row_count = int(row_counts[i_stop])
            if n_pairs > 0 and n_pairs + row_count > target_rows:
                break
            n_pairs += row_count
            i_stop += 1
            if n_pairs >= target_rows:
                break

        ii, jj, shrink_ld = _pairwise_ld_compact_range_impl(
            hap_mat,
            gpos_arr,
            hap_sums,
            j_stop_by_i,
            ne,
            n_ind,
            theta,
            cutoff,
            i_start,
            i_stop,
            n_pairs,
        )
        yield CovarianceRowChunk(
            lo=pos_arr[ii],
            hi=pos_arr[jj],
            shrink_ld=shrink_ld,
        )
        i_start = i_stop


def _compact_pair_chunks_single_pass(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    pos_arr: np.ndarray,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
    chunk_rows: int,
) -> Iterator[CovarianceRowChunk]:
    """Stream compact rows without pre-counting pairs.

    The uint8 backend emits complete left-hand SNP rows until the target chunk
    size is reached. This avoids the historical two-pass compact-output path.
    """
    target_rows = max(int(chunk_rows), 1)
    capacity = target_rows + hap_mat.shape[0]
    i_start = 0
    n_snps = hap_mat.shape[0]
    while i_start < n_snps:
        i_stop, ii, jj, shrink_ld = _pairwise_ld_compact_chunk_impl(
            hap_mat,
            gpos_arr,
            hap_sums,
            j_stop_by_i,
            ne,
            n_ind,
            theta,
            cutoff,
            i_start,
            target_rows,
            capacity,
        )
        if ii.size:
            yield CovarianceRowChunk(
                lo=pos_arr[ii],
                hi=pos_arr[jj],
                shrink_ld=shrink_ld,
            )
        if i_stop <= i_start:
            raise RuntimeError("compact covariance chunk generation did not advance")
        i_start = i_stop


def _compact_pair_chunks_single_pass_bitpacked(
    packed_hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    pos_arr: np.ndarray,
    n_haps: int,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
    chunk_rows: int,
) -> Iterator[CovarianceRowChunk]:
    """Stream compact rows from a bit-packed haplotype matrix.

    This mirrors ``_compact_pair_chunks_single_pass`` but computes pairwise
    intersections with popcounts over packed ``uint64`` words.
    """
    target_rows = max(int(chunk_rows), 1)
    capacity = target_rows + packed_hap_mat.shape[0]
    i_start = 0
    n_snps = packed_hap_mat.shape[0]
    while i_start < n_snps:
        i_stop, ii, jj, shrink_ld = _pairwise_ld_compact_chunk_bitpacked_impl(
            packed_hap_mat,
            gpos_arr,
            hap_sums,
            j_stop_by_i,
            n_haps,
            ne,
            n_ind,
            theta,
            cutoff,
            i_start,
            target_rows,
            capacity,
        )
        if ii.size:
            yield CovarianceRowChunk(
                lo=pos_arr[ii],
                hi=pos_arr[jj],
                shrink_ld=shrink_ld,
            )
        if i_stop <= i_start:
            raise RuntimeError("bitpacked compact chunk generation did not advance")
        i_start = i_stop


def _profile_next_chunks(
    row_chunks: Iterator[CovarianceRowChunk],
    profile: dict[str, float] | None,
) -> Iterator[CovarianceRowChunk]:
    """Measure chunk-generation time separately from HDF5 append time.

    The append writer consumes a row-chunk iterator, so wrapping ``next()``
    calls is the narrowest way to profile the streamed covariance kernel
    without changing the writer API.
    """
    if profile is None:
        yield from row_chunks
        return

    iterator = iter(row_chunks)
    while True:
        start = time.perf_counter()
        try:
            chunk = next(iterator)
        except StopIteration:
            profile["chunk_seconds"] = (
                profile.get("chunk_seconds", 0.0) + time.perf_counter() - start
            )
            return
        profile["chunk_seconds"] = (
            profile.get("chunk_seconds", 0.0) + time.perf_counter() - start
        )
        profile["n_chunks"] = profile.get("n_chunks", 0.0) + 1.0
        yield chunk


def _build_covariance_inputs(
    panel: ReferencePanel,
    pos2gpos: dict[int, float],
    ne: float,
    n_ind: int,
    cutoff: float,
) -> _CovarianceInputs:
    """Convert parsed haplotypes into arrays shared by all pair kernels."""
    hap_mat = np.array(panel.haplotypes, dtype=np.uint8)  # (n_snps, n_haps)
    gpos_arr = np.array(
        [pos2gpos[p] for p in panel.positions], dtype=np.float64
    )  # (n_snps,)
    hap_sums = np.asarray(hap_mat.sum(axis=1), dtype=np.float64)
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, ne, float(n_ind), cutoff)
    pos_arr = np.array(panel.positions, dtype=np.int32)
    assume_sorted_unique_rows = bool(
        pos_arr.size <= 1 or np.all(pos_arr[1:] > pos_arr[:-1])
    )
    return _CovarianceInputs(
        hap_mat=hap_mat,
        gpos_arr=gpos_arr,
        hap_sums=hap_sums,
        j_stop_by_i=j_stop_by_i,
        pos_arr=pos_arr,
        assume_sorted_unique_rows=assume_sorted_unique_rows,
    )


# ---------------------------------------------------------------------------
# Chromosome partitioning  (was P00_00_partition_chromosome.py)
# ---------------------------------------------------------------------------


def partition_chromosome(
    genetic_map_path: Path,
    n_individuals: int,
    output_path: Path,
    window_size: int = 5000,
    ne: float = 11418.0,
    cutoff: float = 1.5e-8,
) -> None:
    """Split a chromosome into overlapping windows using a genetic map.

    Each window spans approximately *window_size* SNPs.  The right boundary is
    extended until the recombination fraction between the last SNP in the
    window and the next falls below *cutoff*, ensuring that consecutive windows
    overlap regions of low recombination.

    Args:
        genetic_map_path: Gzipped genetic map file.  Expected columns:
            ``<chr>  <position>  <genetic_position_cM>``.
        n_individuals: Number of individuals in the reference panel.
        output_path: Output text file; each line is ``<start_pos> <end_pos>``.
        window_size: Target number of SNPs per partition window.
        ne: Effective population size.
        cutoff: Minimum recombination fraction threshold for window extension.
    """
    pos2gpos: dict[int, float] = {}
    positions: list[int] = []

    with gzip.open(genetic_map_path, "rt") as f:
        for raw in f:
            parts = raw.strip().split()
            pos = int(parts[1])
            gpos = float(parts[2])
            pos2gpos[pos] = gpos
            positions.append(pos)

    n_snp = len(positions)
    n_chunks = int(math.floor(n_snp / window_size))

    lines: list[str] = []
    for i in range(n_chunks):
        start = i * window_size
        end = i * window_size + window_size

        if i == n_chunks - 1:
            lines.append(f"{positions[start]} {positions[n_snp - 1]}")
            continue

        end_pos = positions[end - 1]
        end_gpos = pos2gpos[end_pos]
        test = end + 1

        while test < n_snp:
            test_gpos = pos2gpos[positions[test]]
            df = test_gpos - end_gpos
            rho = math.exp(-4.0 * ne * df / (2.0 * n_individuals))
            if rho < cutoff:
                break
            test += 1

        end_idx = min(test, n_snp - 1)
        lines.append(f"{positions[start]} {positions[end_idx]}")

    output_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Covariance / LD calculation  (was P00_01_calc_covariance.py)
# ---------------------------------------------------------------------------


def calc_covariance(
    vcf_path: Path,
    region: str | None,
    genetic_map_path: Path,
    individuals_path: Path,
    output_path: Path,
    ne: float = 11418.0,
    cutoff: float = 1e-7,
    compact_output: bool = True,
    compact_chunk_rows: int = COVARIANCE_WRITE_CHUNK_ROWS,
    compression: str | None = "zstd",
    ld_kernel: str = "bitpacked",
    profile: dict[str, float] | None = None,
) -> None:
    """Calculate the Wen/Stephens shrinkage LD estimate from a VCF/BCF file.

    Reads phased genotypes via ``cyvcf2``. Only bi-allelic, phased genotypes
    are supported. Rows with missing (``./.`` or ``.|.``) or unphased (``/``)
    genotypes are skipped with a warning.

    The pairwise LD kernel is JIT-compiled with Numba.

    Args:
        vcf_path: Path to a ``.vcf``, ``.vcf.gz``, or ``.bcf`` file, or the
            literal string ``"-"`` to read from stdin. When *region* is set,
            *vcf_path* must be indexed (``.tbi`` for ``.vcf.gz``, ``.csi`` for
            ``.bcf``).
        region: A ``chrom:start-end`` region string (1-based, inclusive) to
            restrict reading to via an indexed fetch, or ``None`` to read the
            whole file/stream sequentially from the start.
        genetic_map_path: Gzipped genetic map (columns: chr, position, cM).
        individuals_path: Plain-text file; one individual ID per line.
        output_path: HDF5 covariance partition output. When ``compact_output``
            is true, only canonical positions, shrinkage values, and indexes
            are written.
        ne: Effective population size.
        cutoff: Pairs whose ``|Ds2| < cutoff`` are not written.
        compact_output: Write the compact restartable cache schema used by
            ``ldetect run``.
        compact_chunk_rows: Approximate maximum compact HDF5 rows to hold while
            filling one bounded output chunk.
        compression: HDF5 compression codec for the covariance partition
            (``"zstd"`` or ``"lzf"``). See
            ``ldetect_lite.io.covariance_hdf5._dataset_compression_kwargs``.
        ld_kernel: Pair-count backend for compact covariance output.
            ``"bitpacked"`` packs haplotypes into ``uint64`` words and uses
            popcounts for pairwise intersections. ``"uint8"`` keeps the older
            array-sum backend available for reference and diagnostics.
        profile: Optional mutable timing dictionary populated with coarse
            stage timings for benchmark/profiling callers.

    Raises:
        ValueError: If any individual in *individuals_path* is not present in
            the VCF/BCF header.
    """
    from ldetect_lite._util.logging import log_debug
    from ldetect_lite._util.memory import log_memory_checkpoint

    total_start = time.perf_counter()
    log_memory_checkpoint("calc_covariance_start", debug=True)

    if ld_kernel not in {"uint8", "bitpacked"}:
        raise ValueError("ld_kernel must be one of: uint8, bitpacked")
    if ld_kernel == "bitpacked" and not compact_output:
        raise ValueError("ld_kernel='bitpacked' currently requires compact_output=True")

    prep_start = time.perf_counter()
    individuals = read_individuals(individuals_path)
    n_ind = len(individuals)
    n_haps = 2 * n_ind
    theta = watterson_theta(n_haps)
    pos2gpos = read_genetic_map(genetic_map_path)
    if profile is not None:
        profile["prepare_seconds"] = time.perf_counter() - prep_start

    vcf_start = time.perf_counter()
    panel = read_reference_panel(vcf_path, region, individuals, pos2gpos, n_haps)
    if profile is not None:
        profile["vcf_seconds"] = time.perf_counter() - vcf_start
        profile["n_snps"] = float(len(panel.positions))
        profile["n_haps"] = float(n_haps)

    warn_reference_panel_skips(panel)

    if not panel.positions:
        log_debug(
            "calc_covariance empty_partition "
            f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
        )
        return

    array_start = time.perf_counter()
    inputs = _build_covariance_inputs(panel, pos2gpos, ne, n_ind, cutoff)
    log_debug(
        "calc_covariance arrays_built "
        f"n_snps={inputs.hap_mat.shape[0]} n_haps={inputs.hap_mat.shape[1]} "
        f"seconds={time.perf_counter() - array_start:.3f}"
    )
    if profile is not None:
        profile["array_seconds"] = time.perf_counter() - array_start
    log_memory_checkpoint("calc_covariance_arrays_built", debug=True)

    hap_mat = inputs.hap_mat
    gpos_arr = inputs.gpos_arr
    hap_sums = inputs.hap_sums
    j_stop_by_i = inputs.j_stop_by_i
    pos_arr = inputs.pos_arr
    assume_sorted_unique_rows = inputs.assume_sorted_unique_rows

    if compact_output and assume_sorted_unique_rows:
        write_start = time.perf_counter()
        try:
            if ld_kernel == "bitpacked":
                pack_start = time.perf_counter()
                packed_hap_mat = _pack_haplotypes_impl(hap_mat)
                if profile is not None:
                    profile["pack_seconds"] = time.perf_counter() - pack_start
                log_debug(
                    "calc_covariance haplotypes_bitpacked "
                    f"n_words={packed_hap_mat.shape[1]} "
                    f"seconds={time.perf_counter() - pack_start:.3f}"
                )
                row_chunks = _compact_pair_chunks_single_pass_bitpacked(
                    packed_hap_mat,
                    gpos_arr,
                    hap_sums,
                    j_stop_by_i,
                    pos_arr,
                    n_haps,
                    ne,
                    float(n_ind),
                    theta,
                    cutoff,
                    compact_chunk_rows,
                )
            else:
                row_chunks = _compact_pair_chunks_single_pass(
                    hap_mat,
                    gpos_arr,
                    hap_sums,
                    j_stop_by_i,
                    pos_arr,
                    ne,
                    float(n_ind),
                    theta,
                    cutoff,
                    compact_chunk_rows,
                )
                if profile is not None:
                    profile["pack_seconds"] = 0.0
            row_chunks = _profile_next_chunks(row_chunks, profile)
            n_pairs = write_compact_covariance_partition_hdf5_append(
                output_path,
                positions=pos_arr,
                row_chunks=row_chunks,
                chunk_rows=compact_chunk_rows,
                compression=compression,
            )
            if profile is not None:
                write_total = time.perf_counter() - write_start
                profile["write_total_seconds"] = write_total
                profile["write_io_seconds"] = max(
                    0.0, write_total - profile.get("chunk_seconds", 0.0)
                )
                profile["n_pairs"] = float(n_pairs)
                profile["total_seconds"] = time.perf_counter() - total_start
            dataset_chunk_rows = HDF5_DATASET_CHUNK_ROWS if n_pairs else 0
            log_debug(
                "calc_covariance compact_hdf5_written "
                f"n_pairs={n_pairs} output_bytes={output_path.stat().st_size} "
                f"dataset_chunk_rows={dataset_chunk_rows} "
                f"write_chunk_rows={compact_chunk_rows} "
                "single_pass=true "
                f"ld_kernel={ld_kernel} "
                f"seconds={time.perf_counter() - write_start:.3f} "
                f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
            )
            log_memory_checkpoint("calc_covariance_compact_written", debug=True)
            return
        except Exception:
            output_path.unlink(missing_ok=True)
            log_debug("calc_covariance compact_single_pass_failed using fallback")

        count_start = time.perf_counter()
        row_counts = _count_pairwise_ld_by_i_impl(
            hap_mat,
            gpos_arr,
            hap_sums,
            j_stop_by_i,
            ne,
            float(n_ind),
            theta,
            cutoff,
        )
        n_pairs = int(row_counts.sum())
        max_pairs_per_locus = int(row_counts.max(initial=0))
        log_debug(
            "calc_covariance compact_pair_counts "
            f"n_snps={pos_arr.size} n_pairs={n_pairs} "
            f"max_pairs_per_locus={max_pairs_per_locus} "
            f"seconds={time.perf_counter() - count_start:.3f}"
        )
        log_memory_checkpoint("calc_covariance_pair_counted", debug=True)

        write_start = time.perf_counter()
        dataset_chunk_rows = min(HDF5_DATASET_CHUNK_ROWS, n_pairs) if n_pairs else 0
        write_compact_covariance_partition_hdf5_chunks(
            output_path,
            positions=pos_arr,
            row_counts=row_counts,
            row_chunks=_compact_pair_chunks(
                hap_mat,
                gpos_arr,
                hap_sums,
                j_stop_by_i,
                pos_arr,
                row_counts,
                ne,
                float(n_ind),
                theta,
                cutoff,
                compact_chunk_rows,
            ),
            chunk_rows=compact_chunk_rows,
            compression=compression,
        )
        log_debug(
            "calc_covariance compact_hdf5_written "
            f"n_pairs={n_pairs} output_bytes={output_path.stat().st_size} "
            f"dataset_chunk_rows={dataset_chunk_rows} "
            f"write_chunk_rows={compact_chunk_rows} "
            "single_pass=false "
            f"seconds={time.perf_counter() - write_start:.3f} "
            f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
        )
        log_memory_checkpoint("calc_covariance_compact_written", debug=True)
        return

    # --- compute pairwise LD (Numba-accelerated when available) ---
    pair_start = time.perf_counter()
    ii, jj, d_naive_arr, ds2_arr = _pairwise_ld_impl(
        hap_mat, gpos_arr, hap_sums, j_stop_by_i, ne, float(n_ind), theta, cutoff
    )
    log_debug(
        "calc_covariance pair_arrays_materialized "
        f"n_pairs={ii.size} seconds={time.perf_counter() - pair_start:.3f}"
    )
    log_memory_checkpoint("calc_covariance_pair_arrays_materialized", debug=True)

    # --- write output ---
    map_start = time.perf_counter()
    i_pos = pos_arr[ii]
    j_pos = pos_arr[jj]
    log_debug(
        "calc_covariance positions_mapped "
        f"n_pairs={ii.size} seconds={time.perf_counter() - map_start:.3f}"
    )
    log_memory_checkpoint("calc_covariance_positions_mapped", debug=True)
    if compact_output:
        write_start = time.perf_counter()
        write_covariance_partition_hdf5(
            output_path,
            i_pos=i_pos,
            j_pos=j_pos,
            shrink_ld=ds2_arr,
            assume_canonical_sorted_unique=assume_sorted_unique_rows,
            compression=compression,
        )
        log_debug(
            "calc_covariance compact_hdf5_written_fallback "
            f"n_pairs={ii.size} output_bytes={output_path.stat().st_size} "
            f"seconds={time.perf_counter() - write_start:.3f} "
            f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
        )
        log_memory_checkpoint("calc_covariance_compact_written_fallback", debug=True)
        return

    gpos_flat = np.array([pos2gpos[p] for p in panel.positions], dtype=np.float64)
    rs_arr = np.array(panel.rs_ids)
    write_start = time.perf_counter()
    write_covariance_partition_hdf5(
        output_path,
        i_pos=i_pos,
        j_pos=j_pos,
        shrink_ld=ds2_arr,
        i_gpos=gpos_flat[ii],
        j_gpos=gpos_flat[jj],
        naive_ld=d_naive_arr,
        i_id=rs_arr[ii],
        j_id=rs_arr[jj],
        assume_canonical_sorted_unique=assume_sorted_unique_rows,
        compression=compression,
    )
    log_debug(
        "calc_covariance full_hdf5_written "
        f"n_pairs={ii.size} output_bytes={output_path.stat().st_size} "
        f"seconds={time.perf_counter() - write_start:.3f} "
        f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
    )
    log_memory_checkpoint("calc_covariance_full_written", debug=True)
