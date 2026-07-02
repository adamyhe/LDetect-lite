"""Compare our breakpoints (JSON) against the ldetect reference (pickle).

Compares all four subsets: fourier, fourier_ls, uniform, uniform_ls.

Usage:
    uv run python scripts/compare_bpoints.py \
        --ours   work/breakpoints-chr2.json \
        --ref    ref/minima/minima-EUR-chr2-50-39967768-40067768.pickle \
        --output results/compare_bpoints.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

SUBSETS = ("fourier", "fourier_ls", "uniform", "uniform_ls")


def compare_loci(ours: list[int], ref: list[int]) -> dict:
    n_ours = len(ours)
    n_ref  = len(ref)
    ours_set = set(ours)
    ref_set  = set(ref)
    exact    = len(ours_set & ref_set)
    return {
        "n_ours":   n_ours,
        "n_ref":    n_ref,
        "n_exact":  exact,
        "recall":   round(exact / n_ref,  4) if n_ref  else "nan",
        "precision": round(exact / n_ours, 4) if n_ours else "nan",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours",   required=True, type=Path)
    parser.add_argument("--ref",    required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    ours_data = json.loads(args.ours.read_text())
    with open(args.ref, "rb") as f:
        ref_data = pickle.load(f)

    cols = ["subset", "n_ours", "n_ref", "n_exact", "recall", "precision"]
    rows: list[dict] = []

    for subset in SUBSETS:
        our_loci = ours_data.get(subset, {}).get("loci", [])
        ref_loci = ref_data.get(subset, {}).get("loci", [])
        row = {"subset": subset, **compare_loci(our_loci, ref_loci)}
        rows.append(row)

    print(f"\nBreakpoint comparison ({args.ours.name} vs {args.ref.name})")
    print("  " + "\t".join(cols))
    for row in rows:
        print("  " + "\t".join(str(row[c]) for c in cols))

    # Also compare n_bpoints and found_width
    print(
        f"\n  n_bpoints : ours={ours_data.get('n_bpoints')}  "
        f"ref={ref_data.get('n_bpoints')}"
    )
    print(
        f"  found_width: ours={ours_data.get('found_width')}  "
        f"ref={ref_data.get('found_width')}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

        # Append scalar metadata as extra rows
        writer.writerow(
            {
                "subset": "n_bpoints",
                "n_ours": ours_data.get("n_bpoints"),
                "n_ref": ref_data.get("n_bpoints"),
            }
        )
        writer.writerow(
            {
                "subset": "found_width",
                "n_ours": ours_data.get("found_width"),
                "n_ref": ref_data.get("found_width"),
            }
        )

    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
