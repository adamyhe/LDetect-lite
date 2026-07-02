"""Compare our LD block output against Berisa & Pickrell (2016) published BED.

Compares against the published ldetect blocks from:
    https://bitbucket.org/nygcresearch/ldetect-data

Metrics reported (overall and per chromosome):
  - Block count
  - Block size: mean, median, 5th/95th percentile
  - Boundary match rate / Jaccard at --tolerance bp
  - Boundary offset distribution: median and 90th-percentile nearest-ref distance
  - Base-pair interval Jaccard: fraction of covered bp in common
  - Recall curve: match rate at [10k, 25k, 50k, 100k, 250k, 500k] bp tolerance

Usage:
    uv run python scripts/compare_blocks.py \
        --ours    results/EUR_LD_blocks.bed \
        --ref     resources/ldetect_ref/EUR_fourier_ls-all.bed \
        --output  results/compare/EUR_block_comparison.tsv \
        [--tolerance 100000]
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from ldetect2._util.intervals import (
    block_sizes,
    boundaries,
    boundary_jaccard,
    bp_jaccard,
    match_rate,
    nearest_offsets,
    offset_stats,
    size_stats,
)
from ldetect2.io.bed import read_genome_bed

RECALL_TOLERANCES = [10_000, 25_000, 50_000, 100_000, 250_000, 500_000]


# ---------------------------------------------------------------------------
# Per-chromosome comparison
# ---------------------------------------------------------------------------


def compare_chrom(
    chrom: str,
    ours: list[tuple[int, int]],
    ref: list[tuple[int, int]],
    tol: int,
) -> dict:
    our_bounds = boundaries(ours)
    ref_bounds = boundaries(ref)
    our_stats = size_stats(block_sizes(ours))
    ref_stats = size_stats(block_sizes(ref))

    recall = match_rate(our_bounds, ref_bounds, tol)
    precision = match_rate(ref_bounds, our_bounds, tol)
    jac = boundary_jaccard(our_bounds, ref_bounds, tol)

    our_offsets = nearest_offsets(our_bounds, ref_bounds)
    ref_offsets = nearest_offsets(ref_bounds, our_bounds)
    our_med, our_p90 = offset_stats(our_offsets)
    ref_med, ref_p90 = offset_stats(ref_offsets)

    bp_jac = bp_jaccard(ours, ref)

    recall_curve = {
        t: round(match_rate(our_bounds, ref_bounds, t), 4)
        for t in RECALL_TOLERANCES
    }

    return {
        "chrom": chrom,
        "our_n": our_stats.get("n", 0),
        "ref_n": ref_stats.get("n", 0),
        "our_mean_kb": our_stats.get("mean_kb", ""),
        "ref_mean_kb": ref_stats.get("mean_kb", ""),
        "our_median_kb": our_stats.get("median_kb", ""),
        "ref_median_kb": ref_stats.get("median_kb", ""),
        "recall": round(recall, 4) if recall == recall else "nan",
        "precision": round(precision, 4) if precision == precision else "nan",
        "jaccard": round(jac, 4) if jac == jac else "nan",
        "our_median_offset_kb": our_med,
        "our_p90_offset_kb": our_p90,
        "ref_median_offset_kb": ref_med,
        "ref_p90_offset_kb": ref_p90,
        "bp_jaccard": bp_jac,
        "_recall_curve": recall_curve,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--tolerance",
        type=int,
        default=100_000,
        metavar="BP",
        help="Primary boundary match tolerance in bp (default: 100000).",
    )
    args = parser.parse_args()

    our_blocks = read_genome_bed(args.ours)
    ref_blocks = read_genome_bed(args.ref)

    all_chroms = sorted(
        set(our_blocks) | set(ref_blocks),
        key=lambda c: int(c.lstrip("chr")),
    )

    rows: list[dict] = []
    for chrom in all_chroms:
        ours = our_blocks.get(chrom, [])
        ref = ref_blocks.get(chrom, [])
        rows.append(compare_chrom(chrom, ours, ref, args.tolerance))

    # --- Per-chromosome table ---
    tsv_cols = [
        "chrom",
        "our_n",
        "ref_n",
        "our_mean_kb",
        "ref_mean_kb",
        "our_median_kb",
        "ref_median_kb",
        "recall",
        "precision",
        "jaccard",
        "our_median_offset_kb",
        "our_p90_offset_kb",
        "ref_median_offset_kb",
        "ref_p90_offset_kb",
        "bp_jaccard",
    ]

    print(
        "Comparing against Berisa & Pickrell (ldetect), "
        f"tolerance={args.tolerance:,} bp\n"
    )
    print("\t".join(tsv_cols))
    for row in rows:
        print("\t".join(str(row.get(c, "")) for c in tsv_cols))

    # --- Genome-wide summary ---
    valid = [r for r in rows if isinstance(r["recall"], float)]
    total_ours = sum(r["our_n"] for r in rows)
    total_ref = sum(r["ref_n"] for r in rows)
    if valid:
        mean_recall = statistics.mean(r["recall"] for r in valid)
        mean_bp_jac = statistics.mean(
            r["bp_jaccard"] for r in valid if isinstance(r["bp_jaccard"], float)
        )
        mean_our_offset = statistics.mean(
            r["our_median_offset_kb"] for r in valid
            if isinstance(r["our_median_offset_kb"], float)
        )
        print(
            f"\nGenome-wide: {total_ours} blocks (ours) vs "
            f"{total_ref} blocks (ldetect)"
        )
        print(f"Mean recall ({args.tolerance//1000} kb tol): {mean_recall:.3f}")
        print(f"Mean bp-Jaccard:          {mean_bp_jac:.3f}")
        print(f"Mean our median offset:   {mean_our_offset:.0f} kb")

    # --- Recall curve ---
    curve_rows = [r for r in rows if r["our_n"] > 0]
    if curve_rows:
        tol_labels = [f"{t // 1000}kb" for t in RECALL_TOLERANCES]
        print("\nRecall curve (our boundaries → nearest ref boundary):")
        print("chrom\t" + "\t".join(tol_labels))
        for row in curve_rows:
            curve = row["_recall_curve"]
            vals = "\t".join(f"{curve[t]:.3f}" for t in RECALL_TOLERANCES)
            print(f"{row['chrom']}\t{vals}")

    # --- Write TSV ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\t".join(tsv_cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in tsv_cols) + "\n")
    print(f"\nComparison written to {args.output}")


if __name__ == "__main__":
    main()
