"""Wen/Stephens shrinkage LD estimation and chromosome partitioning."""

from __future__ import annotations

import gzip
import math
import sys
from pathlib import Path
from typing import IO

import numpy as np

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
def _pairwise_ld_impl(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
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
    max_pairs = n_snps * (n_snps + 1) // 2

    ii = np.empty(max_pairs, dtype=np.int64)
    jj = np.empty(max_pairs, dtype=np.int64)
    d_naive_arr = np.empty(max_pairs, dtype=np.float64)
    ds2_arr = np.empty(max_pairs, dtype=np.float64)

    cnt = 0
    for i in range(n_snps):
        gpos1 = gpos_arr[i]
        for j in range(i, n_snps):
            df = gpos_arr[j] - gpos1
            ee = math.exp(-4.0 * ne * df / (2.0 * n_ind))
            if ee < cutoff:
                break

            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            n1x = np.sum(a)
            nx1 = np.sum(b)

            f11 = n11 / n_total
            f1 = n1x / n_total
            f2 = nx1 / n_total
            d_naive = f11 - f1 * f2
            ds2 = (1.0 - theta) ** 2 * d_naive * ee
            if i == j:
                ds2 += (theta / 2.0) * (1.0 - theta / 2.0)

            if math.fabs(ds2) < cutoff:
                continue

            ii[cnt] = i
            jj[cnt] = j
            d_naive_arr[cnt] = d_naive
            ds2_arr[cnt] = ds2
            cnt += 1

    return ii[:cnt], jj[:cnt], d_naive_arr[:cnt], ds2_arr[:cnt]


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

        lines.append(f"{positions[start]} {positions[test]}")

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
        output_path: Gzipped 8-column output file:
            ``i_id  j_id  i_pos  j_pos  i_gpos  j_gpos  naive_ld  shrink_ld``
        ne: Effective population size.
        cutoff: Pairs whose ``|Ds2| < cutoff`` are not written.
    """
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

    if not all_pos:
        return

    # --- build numpy arrays for the JIT kernel ---
    n_snps = len(all_pos)
    hap_mat = np.array(haps, dtype=np.uint8)  # (n_snps, n_haps)
    gpos_arr = np.array([pos2gpos[p] for p in all_pos], dtype=np.float64)  # (n_snps,)

    # --- compute pairwise LD (Numba-accelerated when available) ---
    ii, jj, d_naive_arr, ds2_arr = _pairwise_ld_impl(
        hap_mat, gpos_arr, ne, float(n_ind), theta, cutoff
    )

    # --- write output ---
    with gzip.open(output_path, "wt") as out:
        for idx in range(len(ii)):
            i, j = int(ii[idx]), int(jj[idx])
            out.write(
                f"{all_rs[i]} {all_rs[j]} {all_pos[i]} {all_pos[j]} "
                f"{pos2gpos[all_pos[i]]} {pos2gpos[all_pos[j]]} "
                f"{d_naive_arr[idx]} {ds2_arr[idx]}\n"
            )
