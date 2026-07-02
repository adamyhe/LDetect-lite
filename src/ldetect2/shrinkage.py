"""Wen/Stephens shrinkage LD estimation and chromosome partitioning."""

from __future__ import annotations

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
    write_compact_covariance_partition_hdf5_append,
    write_compact_covariance_partition_hdf5_chunks,
    write_covariance_partition_hdf5,
)

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
    individuals: list[str] = []
    with open(individuals_path) as f:
        for line in f:
            line = line.strip()
            if line:
                individuals.append(line.split()[0])

    n_ind = len(individuals)
    n_haps = 2 * n_ind

    # Watterson's theta for shrinkage
    harmonic = sum(1.0 / i for i in range(1, n_haps))
    theta = (1.0 / harmonic) / (n_haps + 1.0 / harmonic)

    # --- read genetic map ---
    pos2gpos: dict[int, float] = {}
    with gzip.open(genetic_map_path, "rt") as gf:
        for raw in gf:
            parts = raw.strip().split()
            pos2gpos[int(parts[1])] = float(parts[2])

    # --- parse VCF ---
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

        # Data row
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

        if skip:
            continue

        if pos not in pos2gpos:
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

    duplicate_positions = 0
    if all_pos:
        seen_positions: set[int] = set()
        unique_pos: list[int] = []
        unique_rs: list[str] = []
        unique_haps: list[list[int]] = []
        for pos, rs, row_haps in zip(all_pos, all_rs, haps, strict=True):
            if pos in seen_positions:
                duplicate_positions += 1
                continue
            seen_positions.add(pos)
            unique_pos.append(pos)
            unique_rs.append(rs)
            unique_haps.append(row_haps)
        all_pos = unique_pos
        all_rs = unique_rs
        haps = unique_haps

    if duplicate_positions:
        print(
            f"Warning: skipped {duplicate_positions} duplicate-position "
            f"variant(s); covariance partitions are keyed by physical position",
            file=sys.stderr,
        )

    if not all_pos:
        log_debug(
            "calc_covariance empty_partition "
            f"elapsed_seconds={time.perf_counter() - total_start:.3f}"
        )
        return

    # --- build numpy arrays for the JIT kernel ---
    array_start = time.perf_counter()
    hap_mat = np.array(haps, dtype=np.uint8)  # (n_snps, n_haps)
    gpos_arr = np.array([pos2gpos[p] for p in all_pos], dtype=np.float64)  # (n_snps,)
    hap_sums = np.asarray(hap_mat.sum(axis=1), dtype=np.float64)
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, ne, float(n_ind), cutoff)
    log_debug(
        "calc_covariance arrays_built "
        f"n_snps={hap_mat.shape[0]} n_haps={hap_mat.shape[1]} "
        f"seconds={time.perf_counter() - array_start:.3f}"
    )
    log_memory_checkpoint("calc_covariance_arrays_built", debug=True)

    pos_arr = np.array(all_pos, dtype=np.int32)
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
