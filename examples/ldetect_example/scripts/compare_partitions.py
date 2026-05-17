"""Compare our generated partition file against the ldetect reference.

Usage:
    python scripts/compare_partitions.py \
        --ours   work/generated_partitions/chr2_partitions \
        --ref    ref/cov_matrix/scripts/chr2_partitions \
        --output results/compare_partitions.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_partitions(path: Path) -> list[tuple[int, int]]:
    partitions = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                partitions.append((int(parts[0]), int(parts[1])))
    return partitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours",   required=True, type=Path)
    parser.add_argument("--ref",    required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    ours = read_partitions(args.ours)
    ref  = read_partitions(args.ref)

    n_ours = len(ours)
    n_ref  = len(ref)
    exact = sum(1 for o, r in zip(ours, ref) if o == r)

    rows = [
        ("metric", "value"),
        ("n_ours",   n_ours),
        ("n_ref",    n_ref),
        ("n_exact_match", exact),
        ("match", "yes" if ours == ref else "no"),
    ]

    # Per-partition detail
    detail_cols = ["idx", "our_start", "our_end", "ref_start", "ref_end", "match"]
    detail_rows = []
    for i, (o, r) in enumerate(zip(ours, ref)):
        detail_rows.append({
            "idx": i, "our_start": o[0], "our_end": o[1],
            "ref_start": r[0], "ref_end": r[1],
            "match": "yes" if o == r else "no",
        })

    print(f"\nPartition comparison ({args.ours.name} vs {args.ref.name})")
    for k, v in rows[1:]:
        print(f"  {k}: {v}")
    if ours != ref:
        print("\n  Per-partition detail:")
        print("  " + "\t".join(detail_cols))
        for row in detail_rows:
            print("  " + "\t".join(str(row[c]) for c in detail_cols))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
        if detail_rows:
            writer.writerow([])
            writer.writerow(detail_cols)
            for row in detail_rows:
                writer.writerow([str(row[c]) for c in detail_cols])
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
