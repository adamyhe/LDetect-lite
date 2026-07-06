"""Compare our BED output against the ldetect reference BED.

Usage:
    uv run python scripts/compare_bed.py \
        --ours   work/chr2-ld-blocks.bed \
        --ref    ref/bed/EUR-chr2-50-39967768-40067768.bed \
        --output results/compare_bed.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ldetect_lite.io.bed import read_single_chrom_bed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours",   required=True, type=Path)
    parser.add_argument("--ref",    required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    _, ours = read_single_chrom_bed(args.ours)
    _, ref = read_single_chrom_bed(args.ref)

    n_ours = len(ours)
    n_ref  = len(ref)

    exact_blocks = sum(1 for o, r in zip(ours, ref) if o == r)
    start_match  = sum(1 for o, r in zip(ours, ref) if o[0] == r[0])
    end_match    = sum(1 for o, r in zip(ours, ref) if o[1] == r[1])

    all_our_bounds = sorted({b for blk in ours for b in blk})
    all_ref_bounds = sorted({b for blk in ref  for b in blk})
    exact_bounds   = len(set(all_our_bounds) & set(all_ref_bounds))

    rows = [
        ("metric", "value"),
        ("n_ours_blocks", n_ours),
        ("n_ref_blocks",  n_ref),
        ("exact_block_matches", exact_blocks),
        ("start_position_matches", start_match),
        ("end_position_matches", end_match),
        ("n_our_boundaries", len(all_our_bounds)),
        ("n_ref_boundaries", len(all_ref_bounds)),
        ("exact_boundary_matches", exact_bounds),
        ("boundary_recall",    round(exact_bounds / len(all_ref_bounds), 4)
                                if all_ref_bounds else "nan"),
        ("boundary_precision", round(exact_bounds / len(all_our_bounds), 4)
                                if all_our_bounds else "nan"),
    ]

    print(f"\nBED comparison ({args.ours.name} vs {args.ref.name})")
    for k, v in rows[1:]:
        print(f"  {k}: {v}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
