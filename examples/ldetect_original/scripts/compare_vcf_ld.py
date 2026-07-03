#!/usr/bin/env python3
"""Compare phased-haplotype LD and allele frequencies between two VCF releases.

The position-set diagnostic (``compare_vcf_positions.py``) only checks which
SNP positions are called in each release. It cannot see genotype- or
phasing-level differences at *shared* positions, e.g. re-imputation or
re-phasing between 1000 Genomes Phase 1 releases (v1 -> v2 -> v3). Those are
exactly what feeds the Wen & Stephens shrinkage covariance in
``ldetect2.shrinkage``, so this script targets them directly.

For a deterministic, evenly-spaced sample of nearby SNP pairs (positions
present in both VCFs), it computes minor allele frequency and pairwise
haplotype r^2 independently within each VCF, then compares the two sets of
values. r^2 is computed separately per VCF from that VCF's own phased
haplotypes, so no cross-VCF haplotype-index alignment is required: a
per-individual/per-chromosome swap of "haplotype 1" and "haplotype 2" leaves
within-VCF r^2 unchanged, only local switch errors would shift it.

Usage:
    uv run python scripts/compare_vcf_ld.py \
        --population EUR \
        --chromosome 10 \
        --baseline-label v3/all \
        --candidate-label v1/all \
        --baseline-vcf  results/provenance_diagnostics/filtered_vcf/v3/all/EUR/chr10.EUR.all.vcf.gz \
        --candidate-vcf results/provenance_diagnostics/filtered_vcf/v1/all/EUR/chr10.EUR.all.vcf.gz \
        --individuals resources/v3/EUR_inds.txt \
        --output results/provenance_diagnostics/EUR/chr10/ld_sets/v1_all_vs_v3_all.tsv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FIELDNAMES = [
    "population",
    "chrom",
    "baseline_label",
    "candidate_label",
    "n_common_individuals",
    "n_shared_positions",
    "n_pairs",
    "maf_pearson_r",
    "maf_mean_abs_diff",
    "maf_median_abs_diff",
    "r2_pearson_r",
    "r2_mean_abs_diff",
    "r2_median_abs_diff",
    "r2_max_abs_diff",
    "baseline_missing_gt_rate",
    "candidate_missing_gt_rate",
]


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def vcf_samples(path: Path) -> list[str]:
    return [line for line in _run(["bcftools", "query", "-l", str(path)]).splitlines() if line]


def vcf_positions(path: Path) -> set[int]:
    out = _run(["bcftools", "query", "-f", "%POS\n", str(path)])
    return {int(line) for line in out.splitlines() if line}


def pick_pairs(
    shared: list[int], window_bp: int, max_anchors: int, pairs_per_anchor: int
) -> list[tuple[int, int]]:
    """Deterministically pick nearby SNP pairs from a sorted shared-position list.

    Anchors are taken at an even stride across the full position list so the
    sample spans the whole chromosome rather than clustering near one end.
    """
    n = len(shared)
    if n < 2:
        return []
    stride = max(1, n // max_anchors)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for anchor_idx in range(0, n, stride):
        anchor = shared[anchor_idx]
        taken = 0
        for j in range(anchor_idx + 1, n):
            partner = shared[j]
            if partner - anchor > window_bp:
                break
            pair = (anchor, partner)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                taken += 1
            if taken >= pairs_per_anchor:
                break
    return pairs


def write_region_file(positions: set[int], chrom: str, path: Path) -> None:
    with path.open("w") as f:
        for pos in sorted(positions):
            f.write(f"{chrom}\t{pos}\t{pos}\n")


def read_phased_haplotypes(
    vcf: Path, region_file: Path, samples: list[str]
) -> tuple[dict[int, np.ndarray], float]:
    """Return ``{pos: haplotype array}`` (0/1/-1=missing, length ``2 * len(samples)``).

    Column order matches the *given* VCF's own query output and is only ever
    used for within-VCF comparisons (r^2, MAF), so it does not need to match
    the column order used for the other VCF.
    """
    out = _run(
        [
            "bcftools",
            "query",
            "-R",
            str(region_file),
            "-s",
            ",".join(samples),
            "-f",
            "%POS[\t%GT]\n",
            str(vcf),
        ]
    )
    haps: dict[int, np.ndarray] = {}
    total_gt = 0
    missing_gt = 0
    for line in out.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        pos = int(parts[0])
        arr = np.full(2 * len(samples), -1, dtype=np.int8)
        for i, gt in enumerate(parts[1:]):
            total_gt += 1
            if "|" not in gt or "." in gt:
                missing_gt += 1
                continue
            a, b = gt.split("|", 1)
            arr[2 * i] = int(a)
            arr[2 * i + 1] = int(b)
        haps[pos] = arr
    missing_rate = missing_gt / total_gt if total_gt else float("nan")
    return haps, missing_rate


def maf(hap: np.ndarray) -> float | None:
    called = hap[hap >= 0]
    if called.size == 0:
        return None
    p = float(called.mean())
    return min(p, 1.0 - p)


def r_squared(hap_a: np.ndarray, hap_b: np.ndarray) -> float | None:
    mask = (hap_a >= 0) & (hap_b >= 0)
    if mask.sum() < 10:
        return None
    a = hap_a[mask].astype(np.float64)
    b = hap_b[mask].astype(np.float64)
    if a.std() == 0.0 or b.std() == 0.0:
        return None
    r = np.corrcoef(a, b)[0, 1]
    if np.isnan(r):
        return None
    return float(r * r)


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{value:.12g}"


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or np.std(xs) == 0.0 or np.std(ys) == 0.0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def compare(args: argparse.Namespace) -> tuple[dict[str, str], list[dict[str, str]]]:
    chrom = f"chr{args.chromosome.removeprefix('chr')}"
    vcf_chrom = args.chromosome.removeprefix("chr")

    individuals = [
        line.split()[0] for line in args.individuals.read_text().splitlines() if line.strip()
    ]
    baseline_samples = set(vcf_samples(args.baseline_vcf))
    candidate_samples = set(vcf_samples(args.candidate_vcf))
    common_individuals = sorted(
        set(individuals) & baseline_samples & candidate_samples
    )
    if not common_individuals:
        raise SystemExit("No individuals shared by --individuals and both VCFs")

    baseline_positions = vcf_positions(args.baseline_vcf)
    candidate_positions = vcf_positions(args.candidate_vcf)
    shared = sorted(baseline_positions & candidate_positions)

    pairs = pick_pairs(shared, args.window_bp, args.max_anchors, args.pairs_per_anchor)
    positions_needed = {p for pair in pairs for p in pair}

    with tempfile.TemporaryDirectory() as tmpdir:
        region_file = Path(tmpdir) / "regions.bed"
        write_region_file(positions_needed, vcf_chrom, region_file)
        baseline_haps, baseline_missing = read_phased_haplotypes(
            args.baseline_vcf, region_file, common_individuals
        )
        candidate_haps, candidate_missing = read_phased_haplotypes(
            args.candidate_vcf, region_file, common_individuals
        )

    maf_baseline: list[float] = []
    maf_candidate: list[float] = []
    for pos in sorted(positions_needed):
        b_hap = baseline_haps.get(pos)
        c_hap = candidate_haps.get(pos)
        if b_hap is None or c_hap is None:
            continue
        b_maf = maf(b_hap)
        c_maf = maf(c_hap)
        if b_maf is None or c_maf is None:
            continue
        maf_baseline.append(b_maf)
        maf_candidate.append(c_maf)

    pair_rows: list[dict[str, str]] = []
    r2_baseline: list[float] = []
    r2_candidate: list[float] = []
    for pos_a, pos_b in pairs:
        b_a, b_b = baseline_haps.get(pos_a), baseline_haps.get(pos_b)
        c_a, c_b = candidate_haps.get(pos_a), candidate_haps.get(pos_b)
        if b_a is None or b_b is None or c_a is None or c_b is None:
            continue
        b_r2 = r_squared(b_a, b_b)
        c_r2 = r_squared(c_a, c_b)
        if b_r2 is None or c_r2 is None:
            continue
        r2_baseline.append(b_r2)
        r2_candidate.append(c_r2)
        pair_rows.append(
            {
                "population": args.population,
                "chrom": chrom,
                "baseline_label": args.baseline_label,
                "candidate_label": args.candidate_label,
                "pos_a": str(pos_a),
                "pos_b": str(pos_b),
                "distance_bp": str(pos_b - pos_a),
                "baseline_r2": _fmt(b_r2),
                "candidate_r2": _fmt(c_r2),
                "abs_diff": _fmt(abs(b_r2 - c_r2)),
            }
        )

    abs_maf_diffs = [abs(b - c) for b, c in zip(maf_baseline, maf_candidate)]
    abs_r2_diffs = [abs(b - c) for b, c in zip(r2_baseline, r2_candidate)]

    summary = {
        "population": args.population,
        "chrom": chrom,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "n_common_individuals": str(len(common_individuals)),
        "n_shared_positions": str(len(shared)),
        "n_pairs": str(len(pair_rows)),
        "maf_pearson_r": _fmt(_pearson(maf_baseline, maf_candidate)),
        "maf_mean_abs_diff": _fmt(statistics.mean(abs_maf_diffs)) if abs_maf_diffs else "nan",
        "maf_median_abs_diff": _fmt(statistics.median(abs_maf_diffs)) if abs_maf_diffs else "nan",
        "r2_pearson_r": _fmt(_pearson(r2_baseline, r2_candidate)),
        "r2_mean_abs_diff": _fmt(statistics.mean(abs_r2_diffs)) if abs_r2_diffs else "nan",
        "r2_median_abs_diff": _fmt(statistics.median(abs_r2_diffs)) if abs_r2_diffs else "nan",
        "r2_max_abs_diff": _fmt(max(abs_r2_diffs)) if abs_r2_diffs else "nan",
        "baseline_missing_gt_rate": _fmt(baseline_missing),
        "candidate_missing_gt_rate": _fmt(candidate_missing),
    }
    return summary, pair_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", required=True)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-vcf", required=True, type=Path)
    parser.add_argument("--candidate-vcf", required=True, type=Path)
    parser.add_argument("--individuals", required=True, type=Path)
    parser.add_argument(
        "--window-bp",
        type=int,
        default=5000,
        help="Max distance between paired SNPs (default: 5000)",
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=500,
        help="Number of evenly-spaced anchor positions sampled across the chromosome (default: 500)",
    )
    parser.add_argument(
        "--pairs-per-anchor",
        type=int,
        default=5,
        help="Max downstream partners paired with each anchor (default: 5)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pairs-output",
        type=Path,
        help="Optional path to write the full per-pair r^2 comparison table",
    )
    args = parser.parse_args()

    summary, pair_rows = compare(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerow(summary)

    if args.pairs_output:
        args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
        pair_fieldnames = [
            "population",
            "chrom",
            "baseline_label",
            "candidate_label",
            "pos_a",
            "pos_b",
            "distance_bp",
            "baseline_r2",
            "candidate_r2",
            "abs_diff",
        ]
        with args.pairs_output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pair_fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(pair_rows)


if __name__ == "__main__":
    main()
