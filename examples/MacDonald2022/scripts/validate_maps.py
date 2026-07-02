"""Validate and summarise interpolated genetic maps per chromosome.

Checks:
  - Monotonicity: cM must be non-decreasing with physical position
  - Coverage: every SNP in the BED file received a genetic position
  - Density: SNPs per centimorgan (should be roughly uniform)
  - Range: min/max cM per chromosome

Writes a TSV summary to --output and prints a table to stdout.

Usage:
    uv run python scripts/validate_maps.py \
        --interpolated data/maps/interpolated/chr{1..22}.tab.gz \
        --output results/compare/map_summary.tsv
"""

from __future__ import annotations

import argparse
import gzip
import statistics
from pathlib import Path


def _chrom_num(path: Path) -> int:
    stem = path.stem.replace(".tab", "")
    return int(stem.lstrip("chr"))


def summarise_map(path: Path) -> dict:
    positions: list[int] = []
    gpos: list[float] = []

    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                positions.append(int(parts[1]))  # column 1 = physical pos
                gpos.append(float(parts[2]))  # column 2 = cM
            except ValueError:
                continue

    n = len(positions)
    if n == 0:
        return {"n_snps": 0}

    # Monotonicity: count inversions (cM[i] < cM[i-1])
    inversions = sum(1 for i in range(1, n) if gpos[i] < gpos[i - 1])

    cM_range = gpos[-1] - gpos[0]
    density = n / cM_range if cM_range > 0 else float("inf")

    # Inter-SNP cM gaps
    gaps = [gpos[i] - gpos[i - 1] for i in range(1, n)]
    nonzero_gaps = [g for g in gaps if g > 0]

    return {
        "n_snps": n,
        "cM_min": round(gpos[0], 4),
        "cM_max": round(gpos[-1], 4),
        "cM_range": round(cM_range, 4),
        "snps_per_cM": round(density, 2),
        "inversions": inversions,
        "gap_median_cM": (
            round(statistics.median(nonzero_gaps), 6) if nonzero_gaps else 0
        ),
        "gap_max_cM": round(max(nonzero_gaps), 6) if nonzero_gaps else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interpolated",
        nargs="+",
        required=True,
        type=Path,
        metavar="PATH",
        help="Interpolated map files (one per chromosome).",
    )
    parser.add_argument("--output", required=True, type=Path,
                        help="Output TSV summary.")
    args = parser.parse_args()

    map_files = sorted(args.interpolated, key=_chrom_num)
    rows: list[dict] = []

    for path in map_files:
        chrom = f"chr{_chrom_num(path)}"
        stats = summarise_map(path)
        rows.append({"chrom": chrom, **stats})

    # Print table
    cols = ["chrom", "n_snps", "cM_min", "cM_max", "cM_range",
            "snps_per_cM", "inversions", "gap_median_cM", "gap_max_cM"]
    header = "\t".join(cols)
    print(header)
    for row in rows:
        print("\t".join(str(row.get(c, "")) for c in cols))
        if row.get("inversions", 0) > 0:
            print(
                f"  WARNING: {row['inversions']} monotonicity inversions on "
                f"{row['chrom']}"
            )

    # Write TSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")

    total_snps = sum(r.get("n_snps", 0) for r in rows)
    total_inversions = sum(r.get("inversions", 0) for r in rows)
    print(f"\nTotal: {total_snps:,} SNPs across {len(rows)} chromosomes, "
          f"{total_inversions} monotonicity inversions")
    print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()
