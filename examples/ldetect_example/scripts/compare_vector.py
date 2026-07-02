"""Compare our correlation-sum vector against the ldetect reference vector.

Both files are gzipped TSV with columns: position  value

Usage:
    uv run python scripts/compare_vector.py \
        --ours   work/vector-chr2.txt.gz \
        --ref    ref/vector/vector-EUR-chr2-39967768-40067768.txt.gz \
        --output results/compare_vector.tsv
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path


def read_vector(path: Path) -> dict[int, float]:
    opener = gzip.open(path, "rt") if path.suffix in (".gz", ".gzip") else open(path)
    out: dict[int, float] = {}
    with opener as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            out[int(row[0])] = float(row[1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours",   required=True, type=Path)
    parser.add_argument("--ref",    required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    ours = read_vector(args.ours)
    ref  = read_vector(args.ref)

    all_pos = sorted(set(ours) | set(ref))
    only_ours = sum(1 for p in all_pos if p not in ref)
    only_ref  = sum(1 for p in all_pos if p not in ours)
    shared    = [p for p in all_pos if p in ours and p in ref]

    abs_diffs = [abs(ours[p] - ref[p]) for p in shared]
    rel_diffs = [
        abs(ours[p] - ref[p]) / max(abs(ref[p]), 1e-30)
        for p in shared
    ]

    max_abs = max(abs_diffs) if abs_diffs else float("nan")
    mean_abs = sum(abs_diffs) / len(abs_diffs) if abs_diffs else float("nan")
    max_rel = max(rel_diffs) if rel_diffs else float("nan")
    exact_match = sum(1 for d in abs_diffs if d == 0.0)

    rows = [
        ("metric", "value"),
        ("n_ours", len(ours)),
        ("n_ref", len(ref)),
        ("n_shared", len(shared)),
        ("only_in_ours", only_ours),
        ("only_in_ref", only_ref),
        ("exact_matches", exact_match),
        ("max_abs_diff", f"{max_abs:.6e}"),
        ("mean_abs_diff", f"{mean_abs:.6e}"),
        ("max_rel_diff", f"{max_rel:.6e}"),
    ]

    print(f"\nVector comparison ({args.ours.name} vs {args.ref.name})")
    for k, v in rows[1:]:
        print(f"  {k}: {v}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
