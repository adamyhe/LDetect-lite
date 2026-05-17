"""Compare our LD block output against a MacDonald et al. (2022) published BED.

Downloads (or reads cached) a published LD block BED from the
jmacdon/LDblocks_GRCh38 GitHub repository and compares it to our output.

Metrics reported (overall and per chromosome):
  - Block count
  - Block size: mean, median, 5th/95th percentile
  - Boundary match rate: fraction of our boundaries within *--tolerance* bp
    of a MacDonald boundary (and vice versa)
  - Jaccard index on the set of block boundaries (within tolerance)

Usage:
    python scripts/compare_blocks.py \
        --ours    results/EUR_LD_blocks.bed \
        --ref     resources/deCODE_EUR_LD_blocks.bed \
        --output  results/compare/block_comparison.tsv \
        [--tolerance 50000]
"""

from __future__ import annotations

import argparse
import statistics
import urllib.request
from pathlib import Path

from ldetect2._util.intervals import (
    block_sizes,
    boundaries,
    boundary_jaccard,
    match_rate,
    size_stats,
)
from ldetect2.io.bed import read_genome_bed

MACDONALDS_BED_BASE_URL = (
    "https://raw.githubusercontent.com/jmacdon/LDblocks_GRCh38"
    "/master/data"
)


def maybe_download(dest: Path) -> None:
    if dest.exists():
        return
    url = f"{MACDONALDS_BED_BASE_URL}/{dest.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading reference BED from {url} ...")
    urllib.request.urlretrieve(url, dest)


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
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True, type=Path,
                        help="Our combined genome-wide BED.")
    parser.add_argument("--ref", type=Path,
                        default=Path("resources/deCODE_EUR_LD_blocks.bed"),
                        help="MacDonald published BED (downloaded if absent).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output TSV with per-chromosome metrics.")
    parser.add_argument("--tolerance", type=int, default=50_000, metavar="BP",
                        help="Boundary match tolerance in bp (default: 50000).")
    args = parser.parse_args()

    maybe_download(args.ref)

    our_blocks = read_genome_bed(args.ours)
    ref_blocks = read_genome_bed(args.ref)

    all_chroms = sorted(
        set(our_blocks) | set(ref_blocks),
        key=lambda c: int(c.lstrip("chr")),
    )

    cols = [
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
    ]

    rows: list[dict] = []
    for chrom in all_chroms:
        ours = our_blocks.get(chrom, [])
        ref = ref_blocks.get(chrom, [])
        rows.append(compare_chrom(chrom, ours, ref, args.tolerance))

    # Print table
    print(
        f"Comparing against MacDonald et al. ({args.ref.name}), "
        f"tolerance={args.tolerance:,} bp\n"
    )
    print("\t".join(cols))
    for row in rows:
        print("\t".join(str(row.get(c, "")) for c in cols))

    # Genome-wide summary
    total_ours = sum(r["our_n"] for r in rows)
    total_ref = sum(r["ref_n"] for r in rows)
    mean_recall = statistics.mean(
        r["recall"] for r in rows if isinstance(r["recall"], float)
    )
    mean_jaccard = statistics.mean(
        r["jaccard"] for r in rows if isinstance(r["jaccard"], float)
    )
    print(
        f"\nGenome-wide: {total_ours} blocks (ours) vs "
        f"{total_ref} blocks (MacDonald)"
    )
    print(f"Mean recall:  {mean_recall:.3f}  (our boundaries found in MacDonald)")
    print(f"Mean Jaccard: {mean_jaccard:.3f}")

    # Write TSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
    print(f"\nComparison written to {args.output}")


if __name__ == "__main__":
    main()
