"""Diff two raw covariance HDF5 files pair-by-pair.

Built for bisecting the ldetect_original ASN/EUR reproduction divergence:
run `ldetect calc-covariance` for the same region under two different
commits/configs, each writing to its own `--output PATH.h5`, then point
this script at both files. Reports:

  - whether the same set of (lo, hi) SNP pairs is present in both (a
    structural difference -- different pairs included/excluded);
  - for pairs present in both, whether `shrink_ld` (and `naive_ld`, if
    present) values are bit-exact, and if not, the max absolute
    difference and the first N mismatching pairs in full detail.

Usage:
    uv run python scripts/compare_covariance_values.py \
        --old old_chr19_partition.h5 \
        --new new_chr19_partition.h5 \
        [--max-report 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ldetect_lite.io.covariance_hdf5 import open_covariance_reader


def _read_pairs(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    with open_covariance_reader(path, 0, 0) as reader:
        chunk = reader.read_all()
        h5 = reader.h5
        naive_ld = (
            np.asarray(h5["covariance/naive_ld"][:], dtype=np.float64)
            if "covariance/naive_ld" in h5
            else None
        )

    pairs: dict[tuple[int, int], dict[str, float]] = {}
    for idx in range(chunk.lo.shape[0]):
        key = (int(chunk.lo[idx]), int(chunk.hi[idx]))
        row = {"shrink_ld": float(chunk.shrink_ld[idx])}
        if naive_ld is not None:
            row["naive_ld"] = float(naive_ld[idx])
        pairs[key] = row
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--max-report", type=int, default=20)
    args = parser.parse_args()

    old_pairs = _read_pairs(args.old)
    new_pairs = _read_pairs(args.new)

    old_keys = set(old_pairs)
    new_keys = set(new_pairs)
    only_old = sorted(old_keys - new_keys)
    only_new = sorted(new_keys - old_keys)
    shared = sorted(old_keys & new_keys)

    print(
        f"old pairs: {len(old_keys)}  new pairs: {len(new_keys)}  "
        f"shared: {len(shared)}"
    )
    if only_old:
        print(
            f"\n{len(only_old)} pair(s) present in --old but not --new "
            f"(first {args.max_report}):"
        )
        for key in only_old[: args.max_report]:
            print(f"  {key}  old={old_pairs[key]}")
    if only_new:
        print(
            f"\n{len(only_new)} pair(s) present in --new but not --old "
            f"(first {args.max_report}):"
        )
        for key in only_new[: args.max_report]:
            print(f"  {key}  new={new_pairs[key]}")

    Mismatch = tuple[tuple[int, int], dict[str, float], dict[str, float], float]
    mismatches: list[Mismatch] = []
    for key in shared:
        old_row = old_pairs[key]
        new_row = new_pairs[key]
        max_diff = max(
            abs(old_row[field] - new_row[field])
            for field in old_row
            if field in new_row
        )
        if max_diff != 0.0:
            mismatches.append((key, old_row, new_row, max_diff))

    print(f"\nvalue mismatches among shared pairs: {len(mismatches)} / {len(shared)}")
    if mismatches:
        mismatches.sort(key=lambda m: -m[3])
        max_abs_diff = mismatches[0][3]
        print(f"max abs diff: {max_abs_diff:.3e}")
        print(f"\nTop {min(args.max_report, len(mismatches))} mismatches by magnitude:")
        for key, old_row, new_row, max_diff in mismatches[: args.max_report]:
            print(f"  {key}  diff={max_diff:.3e}  old={old_row}  new={new_row}")
    else:
        print("All shared pairs are bit-exact.")


if __name__ == "__main__":
    main()
