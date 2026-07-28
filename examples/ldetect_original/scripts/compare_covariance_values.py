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

All set/diff operations are vectorized over numpy arrays (positions are
combined into a single sortable int64 key) rather than Python-level loops,
so this scales to whole-chromosome-sized files with millions of pairs.

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

_LO_SHIFT = 1 << 32


class _PairArrays:
    def __init__(
        self,
        keys: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        shrink_ld: np.ndarray,
        naive_ld: np.ndarray | None,
    ) -> None:
        self.keys = keys
        self.lo = lo
        self.hi = hi
        self.shrink_ld = shrink_ld
        self.naive_ld = naive_ld


def _read_pairs(path: Path) -> _PairArrays:
    print(f"reading {path} ...", flush=True)
    with open_covariance_reader(path, 0, 0) as reader:
        chunk = reader.read_all()
        h5 = reader.h5
        naive_ld = (
            np.asarray(h5["covariance/naive_ld"][:], dtype=np.float64)
            if "covariance/naive_ld" in h5
            else None
        )
    lo = np.asarray(chunk.lo, dtype=np.int64)
    hi = np.asarray(chunk.hi, dtype=np.int64)
    keys = lo * _LO_SHIFT + hi
    order = np.argsort(keys, kind="stable")
    print(f"  {lo.shape[0]:,} pairs", flush=True)
    return _PairArrays(
        keys=keys[order],
        lo=lo[order],
        hi=hi[order],
        shrink_ld=np.asarray(chunk.shrink_ld, dtype=np.float64)[order],
        naive_ld=naive_ld[order] if naive_ld is not None else None,
    )


def _report_keys_only(
    label: str, keys: np.ndarray, pairs: _PairArrays, max_report: int
) -> None:
    print(f"\n{keys.shape[0]:,} pair(s) present in {label} (first {max_report}):")
    positions = np.searchsorted(pairs.keys, keys)
    for pos in positions[:max_report]:
        row = f"shrink_ld={pairs.shrink_ld[pos]:.6g}"
        if pairs.naive_ld is not None:
            row += f" naive_ld={pairs.naive_ld[pos]:.6g}"
        print(f"  ({pairs.lo[pos]}, {pairs.hi[pos]})  {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--max-report", type=int, default=20)
    args = parser.parse_args()

    old = _read_pairs(args.old)
    new = _read_pairs(args.new)

    shared_keys, old_idx, new_idx = np.intersect1d(
        old.keys, new.keys, assume_unique=True, return_indices=True
    )
    only_old_keys = np.setdiff1d(old.keys, new.keys, assume_unique=True)
    only_new_keys = np.setdiff1d(new.keys, old.keys, assume_unique=True)

    print(
        f"\nold pairs: {old.keys.shape[0]:,}  new pairs: {new.keys.shape[0]:,}  "
        f"shared: {shared_keys.shape[0]:,}"
    )
    if only_old_keys.size:
        _report_keys_only("--old but not --new", only_old_keys, old, args.max_report)
    if only_new_keys.size:
        _report_keys_only("--new but not --old", only_new_keys, new, args.max_report)

    shrink_diff = np.abs(old.shrink_ld[old_idx] - new.shrink_ld[new_idx])
    if old.naive_ld is not None and new.naive_ld is not None:
        naive_diff = np.abs(old.naive_ld[old_idx] - new.naive_ld[new_idx])
        max_diff = np.maximum(shrink_diff, naive_diff)
    else:
        max_diff = shrink_diff

    mismatch_mask = max_diff != 0.0
    n_mismatches = int(mismatch_mask.sum())
    print(
        f"\nvalue mismatches among shared pairs: "
        f"{n_mismatches:,} / {shared_keys.shape[0]:,}"
    )

    if n_mismatches:
        print(f"max abs diff: {max_diff.max():.3e}")
        mismatch_indices = np.nonzero(mismatch_mask)[0]
        worst = mismatch_indices[np.argsort(-max_diff[mismatch_indices])][
            : args.max_report
        ]
        print(f"\nTop {min(args.max_report, n_mismatches)} mismatches by magnitude:")
        for rank in worst:
            oi, ni = old_idx[rank], new_idx[rank]
            lo, hi = int(old.lo[oi]), int(old.hi[oi])
            line = (
                f"  ({lo}, {hi})  diff={max_diff[rank]:.3e}  "
                f"old_shrink_ld={old.shrink_ld[oi]:.10g}  "
                f"new_shrink_ld={new.shrink_ld[ni]:.10g}"
            )
            if old.naive_ld is not None and new.naive_ld is not None:
                line += (
                    f"  old_naive_ld={old.naive_ld[oi]:.10g}  "
                    f"new_naive_ld={new.naive_ld[ni]:.10g}"
                )
            print(line)
    else:
        print("All shared pairs are bit-exact.")


if __name__ == "__main__":
    main()
