"""Wen/Stephens shrinkage LD estimation and chromosome partitioning."""

from __future__ import annotations

import csv
import gzip
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import numpy as np

from ldetect2.io.covariance_hdf5 import (
    HDF5_DATASET_CHUNK_ROWS,
    CovarianceRowChunk,
    _canonical_ordered_rows,
    write_compact_covariance_partition_hdf5_append,
    write_compact_covariance_partition_hdf5_chunks,
    write_covariance_partition_hdf5,
)
from ldetect2.io.r2_zarr import R2RowChunk, write_r2_zarr_partition_append

COVARIANCE_WRITE_CHUNK_ROWS = 1_000_000

# ---------------------------------------------------------------------------
# Pairwise LD kernel (Numba-accelerated when available)
# ---------------------------------------------------------------------------

try:
    from numba import njit

    _njit_fallback = njit(cache=True)
except ImportError:

    def _njit_fallback(fn):  # type: ignore[misc]
        return fn


@_njit_fallback
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
    n_snps = hap_mat.shape[0]
    n_haps = hap_mat.shape[1]
    n_total = float(n_haps)

    cnt = 0
    for i in range(n_snps):
        j_stop = j_stop_by_i[i]
        n1x = hap_sums[i]
        for j in range(i, j_stop):
            gpos1 = gpos_arr[i]
            df = gpos_arr[j] - gpos1
            ee = math.exp(-4.0 * ne * df / (2.0 * n_ind))

            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            nx1 = hap_sums[j]

            f11 = n11 / n_total
            f1 = n1x / n_total
            f2 = nx1 / n_total
            d_naive = f11 - f1 * f2
            ds2 = (1.0 - theta) ** 2 * d_naive * ee

            if math.fabs(ds2) < cutoff:
                continue

            cnt += 1

    return cnt


@_njit_fallback
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
    n_total = float(n_haps)
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
            df = gpos_arr[j] - gpos1
            ee = math.exp(-4.0 * ne * df / (2.0 * n_ind))

            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            nx1 = hap_sums[j]

            f11 = n11 / n_total
            f1 = n1x / n_total
            f2 = nx1 / n_total
            d_naive = f11 - f1 * f2
            ds2 = (1.0 - theta) ** 2 * d_naive * ee

            if math.fabs(ds2) < cutoff:
                continue

            if i == j:
                ds2 += (theta / 2.0) * (1.0 - theta / 2.0)

            ii[cnt] = i
            jj[cnt] = j
            d_naive_arr[cnt] = d_naive
            ds2_arr[cnt] = ds2
            cnt += 1

    return ii, jj, d_naive_arr, ds2_arr


@_njit_fallback
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
    n_snps = hap_mat.shape[0]
    n_haps = hap_mat.shape[1]
    n_total = float(n_haps)
    counts = np.zeros(n_snps, dtype=np.int64)

    for i in range(n_snps):
        gpos1 = gpos_arr[i]
        j_stop = j_stop_by_i[i]
        n1x = hap_sums[i]
        row_count = 0
        for j in range(i, j_stop):
            df = gpos_arr[j] - gpos1
            ee = math.exp(-4.0 * ne * df / (2.0 * n_ind))

            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            nx1 = hap_sums[j]

            f11 = n11 / n_total
            f1 = n1x / n_total
            f2 = nx1 / n_total
            d_naive = f11 - f1 * f2
            ds2 = (1.0 - theta) ** 2 * d_naive * ee

            if math.fabs(ds2) < cutoff:
                continue

            row_count += 1
        counts[i] = row_count

    return counts


@_njit_fallback
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
    n_haps = hap_mat.shape[1]
    n_total = float(n_haps)

    ii = np.empty(n_pairs, dtype=np.int32)
    jj = np.empty(n_pairs, dtype=np.int32)
    ds2_arr = np.empty(n_pairs, dtype=np.float64)

    cnt = 0
    for i in range(i_start, i_stop):
        gpos1 = gpos_arr[i]
        j_stop = j_stop_by_i[i]
        n1x = hap_sums[i]
        for j in range(i, j_stop):
            df = gpos_arr[j] - gpos1
            ee = math.exp(-4.0 * ne * df / (2.0 * n_ind))

            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            nx1 = hap_sums[j]

            f11 = n11 / n_total
            f1 = n1x / n_total
            f2 = nx1 / n_total
            d_naive = f11 - f1 * f2
            ds2 = (1.0 - theta) ** 2 * d_naive * ee

            if math.fabs(ds2) < cutoff:
                continue

            if i == j:
                ds2 += (theta / 2.0) * (1.0 - theta / 2.0)

            ii[cnt] = i
            jj[cnt] = j
            ds2_arr[cnt] = ds2
            cnt += 1

    return ii, jj, ds2_arr


@_njit_fallback
def _genetic_stop_bounds_impl(
    gpos_arr: np.ndarray,
    ne: float,
    n_ind: float,
    cutoff: float,
) -> np.ndarray:
    n_snps = gpos_arr.shape[0]
    stops = np.empty(n_snps, dtype=np.int32)
    stop = 0
    for i in range(n_snps):
        if stop < i:
            stop = i
        gpos1 = gpos_arr[i]
        while stop < n_snps:
            df = gpos_arr[stop] - gpos1
            ee = math.exp(-4.0 * ne * df / (2.0 * n_ind))
            if ee < cutoff:
                break
            stop += 1
        stops[i] = stop
    return stops


@_njit_fallback
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
    n_haps = hap_mat.shape[1]
    n_total = float(n_haps)
    n_snps = hap_mat.shape[0]

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
            df = gpos_arr[j] - gpos1
            ee = math.exp(-4.0 * ne * df / (2.0 * n_ind))

            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            nx1 = hap_sums[j]

            f11 = n11 / n_total
            f1 = n1x / n_total
            f2 = nx1 / n_total
            d_naive = f11 - f1 * f2
            ds2 = (1.0 - theta) ** 2 * d_naive * ee

            if math.fabs(ds2) < cutoff:
                continue

            if i == j:
                ds2 += (theta / 2.0) * (1.0 - theta / 2.0)

            ii[cnt] = i
            jj[cnt] = j
            ds2_arr[cnt] = ds2
            cnt += 1

        i += 1
        if cnt >= target_rows:
            break

    return i, ii[:cnt], jj[:cnt], ds2_arr[:cnt]


@_njit_fallback
def _diag_shrink_values_impl(
    hap_mat: np.ndarray,
    hap_sums: np.ndarray,
    theta: float,
    cutoff: float,
) -> np.ndarray:
    """Return per-SNP diagonal shrinkage values eligible for normalization."""
    n_snps = hap_mat.shape[0]
    n_haps = hap_mat.shape[1]
    n_total = float(n_haps)
    diag = np.zeros(n_snps, dtype=np.float64)
    diagonal_adjustment = (theta / 2.0) * (1.0 - theta / 2.0)

    for i in range(n_snps):
        a = hap_mat[i]
        n11 = np.sum(a * a)
        n1x = hap_sums[i]
        f11 = n11 / n_total
        f1 = n1x / n_total
        d_naive = f11 - f1 * f1
        ds2 = (1.0 - theta) ** 2 * d_naive
        if math.fabs(ds2) >= cutoff:
            diag[i] = ds2 + diagonal_adjustment

    return diag


@_njit_fallback
def _direct_corr_sum_vector_impl(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    diag_shrink: np.ndarray,
    j_stop_by_i: np.ndarray,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
) -> np.ndarray:
    """Compute center-locus r2 sums directly from haplotypes."""
    n_snps = hap_mat.shape[0]
    n_haps = hap_mat.shape[1]
    n_total = float(n_haps)
    sums = np.zeros(n_snps, dtype=np.float64)

    for i in range(n_snps):
        diag_i = diag_shrink[i]
        if diag_i <= 0.0:
            continue

        gpos1 = gpos_arr[i]
        j_stop = j_stop_by_i[i]
        n1x = hap_sums[i]
        for j in range(i, j_stop):
            diag_j = diag_shrink[j]
            if diag_j <= 0.0:
                continue

            idx_delta = j - i
            if idx_delta % 2 != 0 and i == 0:
                continue
            center_idx = (i + j) // 2

            df = gpos_arr[j] - gpos1
            ee = math.exp(-4.0 * ne * df / (2.0 * n_ind))

            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            nx1 = hap_sums[j]

            f11 = n11 / n_total
            f1 = n1x / n_total
            f2 = nx1 / n_total
            d_naive = f11 - f1 * f2
            ds2 = (1.0 - theta) ** 2 * d_naive * ee

            if math.fabs(ds2) < cutoff:
                continue

            if i == j:
                ds2 += (theta / 2.0) * (1.0 - theta / 2.0)

            sums[center_idx] += ds2 * ds2 / (diag_i * diag_j)

    return sums


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
    """Yield compact covariance rows in sorted ``(lo, hi)`` order."""
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
    """Yield compact covariance rows once in sorted ``(lo, hi)`` order."""
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


def _r2_pair_chunks_from_covariance(
    row_chunks: Iterator[CovarianceRowChunk],
    positions: np.ndarray,
    diag_shrink: np.ndarray,
) -> Iterator[R2RowChunk]:
    """Yield normalized r2 rows from sorted compact covariance chunks."""
    for chunk in row_chunks:
        lo = np.asarray(chunk.lo)
        hi = np.asarray(chunk.hi)
        shrink = np.asarray(chunk.shrink_ld, dtype=np.float64)
        if lo.size == 0:
            continue

        lo_idx = np.searchsorted(positions, lo)
        hi_idx = np.searchsorted(positions, hi)
        in_bounds = (lo_idx < positions.size) & (hi_idx < positions.size)
        if not np.all(in_bounds):
            raise ValueError("compact covariance row endpoints must exist in positions")
        if not np.all((positions[lo_idx] == lo) & (positions[hi_idx] == hi)):
            raise ValueError("compact covariance row endpoints must exist in positions")

        diag_lo = diag_shrink[lo_idx]
        diag_hi = diag_shrink[hi_idx]
        positive = (diag_lo > 0.0) & (diag_hi > 0.0)
        if not np.any(positive):
            continue

        out_lo = lo[positive]
        out_hi = hi[positive]
        out_shrink = shrink[positive]
        out_diag_lo = diag_lo[positive]
        out_diag_hi = diag_hi[positive]
        out_r2 = out_shrink * out_shrink / (out_diag_lo * out_diag_hi)
        diag_rows = out_lo == out_hi
        if np.any(diag_rows):
            out_r2[diag_rows] = 1.0
        yield R2RowChunk(lo=out_lo, hi=out_hi, r2=out_r2)


def _r2_pair_chunk_from_canonical_rows(rows: CovarianceRowChunk) -> R2RowChunk:
    """Return normalized r2 rows from canonical physical covariance rows."""
    if rows.lo.size == 0:
        return R2RowChunk(
            lo=np.array([], dtype=np.int32),
            hi=np.array([], dtype=np.int32),
            r2=np.array([], dtype=np.float64),
        )

    diag_mask = rows.lo == rows.hi
    diag_pos = rows.lo[diag_mask].astype(np.int64, copy=False)
    diag_val = np.asarray(rows.shrink_ld[diag_mask], dtype=np.float64)
    if diag_pos.size == 0:
        return R2RowChunk(
            lo=np.array([], dtype=np.int32),
            hi=np.array([], dtype=np.int32),
            r2=np.array([], dtype=np.float64),
        )
    diag_lo_idx = np.searchsorted(diag_pos, rows.lo)
    diag_hi_idx = np.searchsorted(diag_pos, rows.hi)
    has_diag = (diag_lo_idx < diag_pos.size) & (diag_hi_idx < diag_pos.size)
    safe_lo_idx = np.minimum(diag_lo_idx, diag_pos.size - 1)
    safe_hi_idx = np.minimum(diag_hi_idx, diag_pos.size - 1)
    has_diag &= (diag_pos[safe_lo_idx] == rows.lo) & (
        diag_pos[safe_hi_idx] == rows.hi
    )
    if not np.any(has_diag):
        return R2RowChunk(
            lo=np.array([], dtype=np.int32),
            hi=np.array([], dtype=np.int32),
            r2=np.array([], dtype=np.float64),
        )

    lo = rows.lo[has_diag]
    hi = rows.hi[has_diag]
    shrink = np.asarray(rows.shrink_ld[has_diag], dtype=np.float64)
    diag_lo = diag_val[diag_lo_idx[has_diag]]
    diag_hi = diag_val[diag_hi_idx[has_diag]]
    positive = (diag_lo > 0.0) & (diag_hi > 0.0)
    if not np.any(positive):
        return R2RowChunk(
            lo=np.array([], dtype=np.int32),
            hi=np.array([], dtype=np.int32),
            r2=np.array([], dtype=np.float64),
        )

    lo = lo[positive]
    hi = hi[positive]
    r2 = shrink[positive] * shrink[positive] / (
        diag_lo[positive] * diag_hi[positive]
    )
    diag_rows = lo == hi
    if np.any(diag_rows):
        r2[diag_rows] = 1.0
    return R2RowChunk(lo=lo, hi=hi, r2=r2)


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


def _read_individuals(individuals_path: Path) -> list[str]:
    individuals: list[str] = []
    with open(individuals_path) as f:
        for line in f:
            line = line.strip()
            if line:
                individuals.append(line.split()[0])
    return individuals


def _read_genetic_map(genetic_map_path: Path) -> dict[int, float]:
    pos2gpos: dict[int, float] = {}
    with gzip.open(genetic_map_path, "rt") as gf:
        for raw in gf:
            parts = raw.strip().split()
            pos2gpos[int(parts[1])] = float(parts[2])
    return pos2gpos


def _parse_vcf_haplotypes(
    vcf_stream: IO[str],
    individuals: list[str],
    pos2gpos: dict[int, float],
) -> tuple[list[int], list[str], list[list[int]]]:
    all_pos: list[int] = []
    all_rs: list[str] = []
    haps: list[list[int]] = []
    ind2col: dict[str, int] = {}

    skipped_unphased = 0
    for raw in vcf_stream:
        raw = raw.rstrip("\n")
        if raw.startswith("##"):
            continue
        parts = raw.split("\t")
        if raw.startswith("#CHROM"):
            for col_idx in range(9, len(parts)):
                if parts[col_idx] in individuals:
                    ind2col[parts[col_idx]] = col_idx
            continue

        pos = int(parts[1])
        rs = parts[2]

        row_haps: list[int] = []
        skip = False
        for ind in individuals:
            col = ind2col.get(ind)
            if col is None:
                skip = True
                break
            gt_field = parts[col].split(":")[0]
            if "|" not in gt_field:
                skipped_unphased += 1
                skip = True
                break
            alleles = gt_field.split("|")
            if "." in alleles:
                skipped_unphased += 1
                skip = True
                break
            row_haps.append(int(alleles[0]))
            row_haps.append(int(alleles[1]))

        if skip or pos not in pos2gpos:
            continue

        all_pos.append(pos)
        all_rs.append(rs)
        haps.append(row_haps)

    if skipped_unphased:
        print(
            f"Warning: skipped {skipped_unphased} variant(s) with unphased or "
            f"missing genotypes",
            file=sys.stderr,
        )

    duplicate_positions = len(all_pos) - len(set(all_pos))
    if duplicate_positions:
        print(
            f"Warning: retained {duplicate_positions} duplicate-position "
            f"variant(s); physical covariance pairs keep first-row precedence",
            file=sys.stderr,
        )

    return all_pos, all_rs, haps


def _prepare_ld_arrays(
    all_pos: list[int],
    haps: list[list[int]],
    pos2gpos: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hap_mat = np.array(haps, dtype=np.uint8)
    gpos_arr = np.array([pos2gpos[p] for p in all_pos], dtype=np.float64)
    hap_sums = np.asarray(hap_mat.sum(axis=1), dtype=np.float64)
    pos_arr = np.array(all_pos, dtype=np.int32)
    return hap_mat, gpos_arr, hap_sums, pos_arr


def _has_duplicate_positions(pos_arr: np.ndarray) -> bool:
    """Return whether *pos_arr* contains repeated physical positions."""
    return bool(pos_arr.size > 1 and np.unique(pos_arr).size != pos_arr.size)


def _canonical_pair_rows_from_arrays(
    pos_arr: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    shrink_ld: np.ndarray,
) -> CovarianceRowChunk:
    """Return physical-pair canonical rows using the HDF5 writer's policy."""
    lo, hi, shrink, _ = _canonical_ordered_rows(
        pos_arr[ii],
        pos_arr[jj],
        shrink_ld,
    )
    return CovarianceRowChunk(lo=lo, hi=hi, shrink_ld=shrink)


def _r2_vector_from_canonical_rows(
    rows: CovarianceRowChunk,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate the matrix-to-vector signal from canonical physical rows."""
    if rows.lo.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    loci = np.unique(np.concatenate((rows.lo, rows.hi))).astype(np.int64, copy=False)
    diag_mask = rows.lo == rows.hi
    diag_pos = rows.lo[diag_mask].astype(np.int64, copy=False)
    diag_val = np.asarray(rows.shrink_ld[diag_mask], dtype=np.float64)
    sums = np.zeros(loci.size, dtype=np.float64)
    if diag_pos.size == 0:
        return loci.astype(np.int32, copy=False), sums

    diag_lo_idx = np.searchsorted(diag_pos, rows.lo)
    diag_hi_idx = np.searchsorted(diag_pos, rows.hi)
    has_diag = (diag_lo_idx < diag_pos.size) & (diag_hi_idx < diag_pos.size)
    safe_lo_idx = np.minimum(diag_lo_idx, diag_pos.size - 1)
    safe_hi_idx = np.minimum(diag_hi_idx, diag_pos.size - 1)
    has_diag &= (diag_pos[safe_lo_idx] == rows.lo) & (
        diag_pos[safe_hi_idx] == rows.hi
    )
    if not np.any(has_diag):
        return loci.astype(np.int32, copy=False), sums

    row_lo = rows.lo[has_diag]
    row_hi = rows.hi[has_diag]
    row_shrink = np.asarray(rows.shrink_ld[has_diag], dtype=np.float64)
    diag_lo = diag_val[diag_lo_idx[has_diag]]
    diag_hi = diag_val[diag_hi_idx[has_diag]]
    positive = (diag_lo > 0.0) & (diag_hi > 0.0)
    if not np.any(positive):
        return loci.astype(np.int32, copy=False), sums

    row_lo = row_lo[positive]
    row_hi = row_hi[positive]
    row_shrink = row_shrink[positive]
    diag_lo = diag_lo[positive]
    diag_hi = diag_hi[positive]
    lo_idx = np.searchsorted(loci, row_lo)
    hi_idx = np.searchsorted(loci, row_hi)
    idx_delta = hi_idx - lo_idx
    legacy_reachable = (idx_delta % 2 == 0) | (lo_idx > 0)
    if not np.any(legacy_reachable):
        return loci.astype(np.int32, copy=False), sums

    center_idx = (lo_idx[legacy_reachable] + hi_idx[legacy_reachable]) // 2
    r2 = (
        row_shrink[legacy_reachable]
        * row_shrink[legacy_reachable]
        / (diag_lo[legacy_reachable] * diag_hi[legacy_reachable])
    )
    sums += np.bincount(center_idx, weights=r2, minlength=loci.size)
    return loci.astype(np.int32, copy=False), sums


def _duplicate_compatible_pair_rows(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    j_stop_by_i: np.ndarray,
    pos_arr: np.ndarray,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
) -> CovarianceRowChunk:
    """Materialize and canonicalize rows for duplicate-position partitions."""
    ii, jj, _, ds2_arr = _pairwise_ld_impl(
        hap_mat,
        gpos_arr,
        hap_sums,
        j_stop_by_i,
        ne,
        n_ind,
        theta,
        cutoff,
    )
    return _canonical_pair_rows_from_arrays(pos_arr, ii, jj, ds2_arr)


def _write_direct_vector(
    output_path: Path,
    positions: np.ndarray,
    sums: np.ndarray,
    *,
    center_lower_bound: int | None = None,
    center_lower_inclusive: bool = True,
    center_upper_bound: int | None = None,
    center_upper_inclusive: bool = True,
    append: bool = False,
) -> None:
    if positions.size == 0:
        return

    if center_upper_bound is None:
        center_upper_bound = int(positions[-1])
        center_upper_inclusive = False

    mode = "at" if append else "wt"
    with gzip.open(output_path, mode) as f:
        writer = csv.writer(f, delimiter="\t")
        for pos, value in zip(positions, sums):
            pos_i = int(pos)
            if center_lower_bound is not None:
                if center_lower_inclusive:
                    if pos_i < center_lower_bound:
                        continue
                elif pos_i <= center_lower_bound:
                    continue
            if center_upper_inclusive:
                if pos_i > center_upper_bound:
                    continue
            elif pos_i >= center_upper_bound:
                continue
            if value > 0.0:
                writer.writerow([pos_i, float(value)])


def calc_covariance_vector(
    vcf_stream: IO[str],
    genetic_map_path: Path,
    individuals_path: Path,
    output_path: Path,
    ne: float = 11418.0,
    cutoff: float = 1e-7,
    center_lower_bound: int | None = None,
    center_lower_inclusive: bool = True,
    center_upper_bound: int | None = None,
    center_upper_inclusive: bool = True,
    append_output: bool = False,
) -> None:
    """Calculate the correlation-sum vector directly from a VCF stream.

    This skips covariance-row materialization for workflows that only need the
    matrix-to-vector signal. The output is a gzipped TSV with
    ``position<TAB>corr_sum`` rows matching the standalone matrix-to-vector
    semantics for a single partition.
    """
    from ldetect2._util.logging import log_debug
    from ldetect2._util.memory import log_memory_checkpoint

    total_start = time.perf_counter()
    log_memory_checkpoint("calc_covariance_vector_start", debug=True)

    individuals = _read_individuals(individuals_path)
    n_ind = len(individuals)
    n_haps = 2 * n_ind
    harmonic = sum(1.0 / i for i in range(1, n_haps))
    theta = (1.0 / harmonic) / (n_haps + 1.0 / harmonic)

    pos2gpos = _read_genetic_map(genetic_map_path)
    all_pos, _, haps = _parse_vcf_haplotypes(vcf_stream, individuals, pos2gpos)
    if not all_pos:
        if not append_output:
            output_path.unlink(missing_ok=True)
        log_debug(
            "calc_covariance_vector empty_partition "
            f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
        )
        return

    array_start = time.perf_counter()
    hap_mat, gpos_arr, hap_sums, pos_arr = _prepare_ld_arrays(all_pos, haps, pos2gpos)
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, ne, float(n_ind), cutoff)
    diag_shrink = _diag_shrink_values_impl(hap_mat, hap_sums, theta, cutoff)
    has_duplicate_positions = _has_duplicate_positions(pos_arr)
    log_debug(
        "calc_covariance_vector arrays_built "
        f"n_snps={hap_mat.shape[0]} n_haps={hap_mat.shape[1]} "
        f"duplicate_positions={has_duplicate_positions} "
        f"seconds={time.perf_counter() - array_start:.3f}"
    )
    log_memory_checkpoint("calc_covariance_vector_arrays_built", debug=True)

    vector_start = time.perf_counter()
    if has_duplicate_positions:
        rows = _duplicate_compatible_pair_rows(
            hap_mat,
            gpos_arr,
            hap_sums,
            j_stop_by_i,
            pos_arr,
            ne,
            float(n_ind),
            theta,
            cutoff,
        )
        vector_positions, sums = _r2_vector_from_canonical_rows(rows)
    else:
        vector_positions = pos_arr
        sums = _direct_corr_sum_vector_impl(
            hap_mat,
            gpos_arr,
            hap_sums,
            diag_shrink,
            j_stop_by_i,
            ne,
            float(n_ind),
            theta,
            cutoff,
        )
    log_debug(
        "calc_covariance_vector sums_calculated "
        f"nonzero={int(np.count_nonzero(sums))} "
        f"duplicate_fallback={has_duplicate_positions} "
        f"seconds={time.perf_counter() - vector_start:.3f}"
    )

    write_start = time.perf_counter()
    _write_direct_vector(
        output_path,
        vector_positions,
        sums,
        center_lower_bound=center_lower_bound,
        center_lower_inclusive=center_lower_inclusive,
        center_upper_bound=center_upper_bound,
        center_upper_inclusive=center_upper_inclusive,
        append=append_output,
    )
    log_debug(
        "calc_covariance_vector written "
        f"output_bytes={output_path.stat().st_size if output_path.exists() else 0} "
        f"seconds={time.perf_counter() - write_start:.3f} "
        f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
    )
    log_memory_checkpoint("calc_covariance_vector_written", debug=True)


def calc_r2_zarr_partition(
    vcf_stream: IO[str],
    genetic_map_path: Path,
    individuals_path: Path,
    output_root: Path,
    name: str,
    start: int,
    end: int,
    ne: float = 11418.0,
    cutoff: float = 1e-7,
    compact_chunk_rows: int = COVARIANCE_WRITE_CHUNK_ROWS,
    vector_output_path: Path | None = None,
    center_lower_bound: int | None = None,
    center_lower_inclusive: bool = True,
    center_upper_bound: int | None = None,
    center_upper_inclusive: bool = True,
    r2_zarr_compressor: str = "default",
) -> None:
    """Calculate normalized r2 rows and optionally a direct vector fragment."""
    from ldetect2._util.logging import log_debug
    from ldetect2._util.memory import log_memory_checkpoint

    total_start = time.perf_counter()
    log_memory_checkpoint("calc_r2_zarr_start", debug=True)

    individuals = _read_individuals(individuals_path)
    n_ind = len(individuals)
    n_haps = 2 * n_ind
    harmonic = sum(1.0 / i for i in range(1, n_haps))
    theta = (1.0 / harmonic) / (n_haps + 1.0 / harmonic)

    pos2gpos = _read_genetic_map(genetic_map_path)
    all_pos, _, haps = _parse_vcf_haplotypes(vcf_stream, individuals, pos2gpos)
    if not all_pos:
        if vector_output_path is not None:
            vector_output_path.unlink(missing_ok=True)
        write_r2_zarr_partition_append(
            output_root,
            name,
            start,
            end,
            positions=np.array([], dtype=np.int32),
            row_chunks=iter(()),
            ne=ne,
            cutoff=cutoff,
            chunk_rows=compact_chunk_rows,
            compressor=r2_zarr_compressor,
        )
        log_debug(
            "calc_r2_zarr empty_partition "
            f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
        )
        return

    array_start = time.perf_counter()
    hap_mat, gpos_arr, hap_sums, pos_arr = _prepare_ld_arrays(all_pos, haps, pos2gpos)
    has_duplicate_positions = _has_duplicate_positions(pos_arr)
    if (
        not has_duplicate_positions
        and pos_arr.size > 1
        and not np.all(pos_arr[1:] > pos_arr[:-1])
    ):
        raise ValueError("r2 Zarr cache requires sorted VCF positions")
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, ne, float(n_ind), cutoff)
    diag_shrink = _diag_shrink_values_impl(hap_mat, hap_sums, theta, cutoff)
    log_debug(
        "calc_r2_zarr arrays_built "
        f"n_snps={hap_mat.shape[0]} n_haps={hap_mat.shape[1]} "
        f"positive_diagonal={int(np.count_nonzero(diag_shrink > 0.0))} "
        f"duplicate_positions={has_duplicate_positions} "
        f"seconds={time.perf_counter() - array_start:.3f}"
    )
    log_memory_checkpoint("calc_r2_zarr_arrays_built", debug=True)

    duplicate_rows = (
        _duplicate_compatible_pair_rows(
            hap_mat,
            gpos_arr,
            hap_sums,
            j_stop_by_i,
            pos_arr,
            ne,
            float(n_ind),
            theta,
            cutoff,
        )
        if has_duplicate_positions
        else None
    )

    if vector_output_path is not None:
        vector_start = time.perf_counter()
        if duplicate_rows is not None:
            vector_positions, sums = _r2_vector_from_canonical_rows(duplicate_rows)
        else:
            vector_positions = pos_arr
            sums = _direct_corr_sum_vector_impl(
                hap_mat,
                gpos_arr,
                hap_sums,
                diag_shrink,
                j_stop_by_i,
                ne,
                float(n_ind),
                theta,
                cutoff,
            )
        _write_direct_vector(
            vector_output_path,
            vector_positions,
            sums,
            center_lower_bound=center_lower_bound,
            center_lower_inclusive=center_lower_inclusive,
            center_upper_bound=center_upper_bound,
            center_upper_inclusive=center_upper_inclusive,
        )
        output_bytes = (
            vector_output_path.stat().st_size if vector_output_path.exists() else 0
        )
        log_debug(
            "calc_r2_zarr vector_written "
            f"nonzero={int(np.count_nonzero(sums))} "
            f"output_bytes={output_bytes} "
            f"duplicate_fallback={has_duplicate_positions} "
            f"seconds={time.perf_counter() - vector_start:.3f}"
        )

    write_start = time.perf_counter()
    if duplicate_rows is not None:
        zarr_positions = np.unique(
            np.concatenate((duplicate_rows.lo, duplicate_rows.hi))
        ).astype(np.int32, copy=False)
        row_chunks = iter([_r2_pair_chunk_from_canonical_rows(duplicate_rows)])
    else:
        zarr_positions = pos_arr
        row_chunks = _r2_pair_chunks_from_covariance(
            _compact_pair_chunks_single_pass(
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
            ),
            pos_arr,
            diag_shrink,
        )
    n_pairs = write_r2_zarr_partition_append(
        output_root,
        name,
        start,
        end,
        positions=zarr_positions,
        row_chunks=row_chunks,
        ne=ne,
        cutoff=cutoff,
        chunk_rows=compact_chunk_rows,
        compressor=r2_zarr_compressor,
    )
    log_debug(
        "calc_r2_zarr written "
        f"n_pairs={n_pairs} "
        f"seconds={time.perf_counter() - write_start:.3f} "
        f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
    )
    log_memory_checkpoint("calc_r2_zarr_written", debug=True)


def calc_covariance(
    vcf_stream: IO[str],
    genetic_map_path: Path,
    individuals_path: Path,
    output_path: Path,
    ne: float = 11418.0,
    cutoff: float = 1e-7,
    compact_output: bool = False,
    compact_chunk_rows: int = COVARIANCE_WRITE_CHUNK_ROWS,
) -> None:
    """Calculate the Wen/Stephens shrinkage LD estimate from a VCF stream.

    Reads phased VCF data from *vcf_stream* (typically stdin piped from
    ``tabix``).  Only bi-allelic, phased genotypes are supported.  Rows with
    missing (``./. `` or ``.|.``) or unphased (``/``) genotypes are skipped
    with a warning.

    The pairwise LD kernel is JIT-compiled with Numba when available, falling
    back to pure Python otherwise.

    Args:
        vcf_stream: Text stream of VCF lines (``##`` headers, ``#CHROM``, then
            data rows).
        genetic_map_path: Gzipped genetic map (columns: chr, position, cM).
        individuals_path: Plain-text file; one individual ID per line.
        output_path: HDF5 covariance partition output. When ``compact_output``
            is true, only canonical positions, shrinkage values, and indexes
            are written.
        ne: Effective population size.
        cutoff: Pairs whose ``|Ds2| < cutoff`` are not written.
        compact_output: Write the compact restartable cache schema used by
            ``ldetect2 run``.
        compact_chunk_rows: Approximate maximum compact HDF5 rows to hold while
            filling one bounded output chunk.
    """
    from ldetect2._util.logging import log_debug
    from ldetect2._util.memory import log_memory_checkpoint

    total_start = time.perf_counter()
    log_memory_checkpoint("calc_covariance_start", debug=True)

    # --- read individuals ---
    individuals = _read_individuals(individuals_path)
    n_ind = len(individuals)
    n_haps = 2 * n_ind

    # Watterson's theta for shrinkage
    harmonic = sum(1.0 / i for i in range(1, n_haps))
    theta = (1.0 / harmonic) / (n_haps + 1.0 / harmonic)

    # --- read genetic map ---
    pos2gpos = _read_genetic_map(genetic_map_path)

    # --- parse VCF ---
    all_pos, all_rs, haps = _parse_vcf_haplotypes(
        vcf_stream,
        individuals,
        pos2gpos,
    )

    if not all_pos:
        log_debug(
            "calc_covariance empty_partition "
            f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
        )
        return

    # --- build numpy arrays for the JIT kernel ---
    array_start = time.perf_counter()
    hap_mat, gpos_arr, hap_sums, pos_arr = _prepare_ld_arrays(
        all_pos,
        haps,
        pos2gpos,
    )
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, ne, float(n_ind), cutoff)
    log_debug(
        "calc_covariance arrays_built "
        f"n_snps={hap_mat.shape[0]} n_haps={hap_mat.shape[1]} "
        f"seconds={time.perf_counter() - array_start:.3f}"
    )
    log_memory_checkpoint("calc_covariance_arrays_built", debug=True)

    assume_sorted_unique_rows = bool(
        pos_arr.size <= 1 or np.all(pos_arr[1:] > pos_arr[:-1])
    )

    if compact_output and assume_sorted_unique_rows:
        write_start = time.perf_counter()
        try:
            n_pairs = write_compact_covariance_partition_hdf5_append(
                output_path,
                positions=pos_arr,
                row_chunks=_compact_pair_chunks_single_pass(
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
                ),
                chunk_rows=compact_chunk_rows,
            )
            dataset_chunk_rows = HDF5_DATASET_CHUNK_ROWS if n_pairs else 0
            log_debug(
                "calc_covariance compact_hdf5_written "
                f"n_pairs={n_pairs} output_bytes={output_path.stat().st_size} "
                f"dataset_chunk_rows={dataset_chunk_rows} "
                f"write_chunk_rows={compact_chunk_rows} "
                "single_pass=true "
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
        )
        log_debug(
            "calc_covariance compact_hdf5_written_fallback "
            f"n_pairs={ii.size} output_bytes={output_path.stat().st_size} "
            f"seconds={time.perf_counter() - write_start:.3f} "
            f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
        )
        log_memory_checkpoint("calc_covariance_compact_written_fallback", debug=True)
        return

    gpos_flat = np.array([pos2gpos[p] for p in all_pos], dtype=np.float64)
    rs_arr = np.array(all_rs)
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
    )
    log_debug(
        "calc_covariance full_hdf5_written "
        f"n_pairs={ii.size} output_bytes={output_path.stat().st_size} "
        f"seconds={time.perf_counter() - write_start:.3f} "
        f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
    )
    log_memory_checkpoint("calc_covariance_full_written", debug=True)
