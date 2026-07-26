#!/usr/bin/env python3
"""Compare ldetect-lite HDF5 covariance partitions against legacy flat files."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ldetect_lite.io.covariance_hdf5 import open_covariance_reader


def _read_partitions(path: Path) -> list[tuple[int, int]]:
    partitions: list[tuple[int, int]] = []
    with path.open() as f:
        for raw in f:
            parts = raw.split()
            if not parts:
                continue
            partitions.append((int(parts[0]), int(parts[1])))
    return partitions


def _legacy_rows(path: Path) -> Iterator[tuple[int, int, float]]:
    with gzip.open(path, "rt") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            if not row:
                continue
            row = [field for field in row if field]
            if len(row) < 8:
                raise ValueError(f"Malformed legacy covariance row in {path}: {row}")
            i_pos = int(row[2])
            j_pos = int(row[3])
            lo, hi = (i_pos, j_pos) if i_pos <= j_pos else (j_pos, i_pos)
            yield lo, hi, float(row[7])


def _lite_rows(path: Path, start: int, end: int) -> Iterator[tuple[int, int, float]]:
    with open_covariance_reader(path, start, end) as reader:
        rows = reader.read_all()
        for lo, hi, shrink in zip(rows.lo, rows.hi, rows.shrink_ld):
            yield int(lo), int(hi), float(shrink)


def _advance(
    iterator: Iterator[tuple[int, int, float]],
) -> tuple[int, int, float] | None:
    return next(iterator, None)


def _compare_partition(
    *,
    lite_path: Path,
    legacy_path: Path,
    start: int,
    end: int,
    tolerance: float,
) -> dict[str, str | int | float]:
    lite_iter = _lite_rows(lite_path, start, end)
    legacy_iter = _legacy_rows(legacy_path)
    lite_row = _advance(lite_iter)
    legacy_row = _advance(legacy_iter)

    lite_rows = 0
    legacy_rows = 0
    shared_rows = 0
    only_lite = 0
    only_legacy = 0
    value_mismatches = 0
    diag_shared = 0
    diag_mismatches = 0
    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    diffs: list[float] = []
    first_diff_key = ""
    first_lite_value = ""
    first_legacy_value = ""

    last_lite_key: tuple[int, int] | None = None
    last_legacy_key: tuple[int, int] | None = None
    lite_sorted = True
    legacy_sorted = True

    while lite_row is not None or legacy_row is not None:
        if lite_row is None:
            legacy_rows += 1
            only_legacy += 1
            legacy_key = (legacy_row[0], legacy_row[1])
            if last_legacy_key is not None and legacy_key < last_legacy_key:
                legacy_sorted = False
            last_legacy_key = legacy_key
            legacy_row = _advance(legacy_iter)
            continue
        if legacy_row is None:
            lite_rows += 1
            only_lite += 1
            lite_key = (lite_row[0], lite_row[1])
            if last_lite_key is not None and lite_key < last_lite_key:
                lite_sorted = False
            last_lite_key = lite_key
            lite_row = _advance(lite_iter)
            continue

        lite_key = (lite_row[0], lite_row[1])
        legacy_key = (legacy_row[0], legacy_row[1])
        if last_lite_key is not None and lite_key < last_lite_key:
            lite_sorted = False
        if last_legacy_key is not None and legacy_key < last_legacy_key:
            legacy_sorted = False
        last_lite_key = lite_key
        last_legacy_key = legacy_key

        if lite_key == legacy_key:
            lite_rows += 1
            legacy_rows += 1
            shared_rows += 1
            diff = abs(lite_row[2] - legacy_row[2])
            sum_abs_diff += diff
            diffs.append(diff)
            if diff > max_abs_diff:
                max_abs_diff = diff
            if diff > tolerance:
                value_mismatches += 1
                if not first_diff_key:
                    first_diff_key = f"{lite_key[0]}:{lite_key[1]}"
                    first_lite_value = repr(lite_row[2])
                    first_legacy_value = repr(legacy_row[2])
                if lite_key[0] == lite_key[1]:
                    diag_mismatches += 1
            if lite_key[0] == lite_key[1]:
                diag_shared += 1
            lite_row = _advance(lite_iter)
            legacy_row = _advance(legacy_iter)
        elif lite_key < legacy_key:
            lite_rows += 1
            only_lite += 1
            lite_row = _advance(lite_iter)
        else:
            legacy_rows += 1
            only_legacy += 1
            legacy_row = _advance(legacy_iter)

    mean_abs_diff = sum_abs_diff / shared_rows if shared_rows else math.nan
    p99_abs_diff = float(np.quantile(diffs, 0.99)) if diffs else math.nan
    return {
        "partition_start": start,
        "partition_end": end,
        "lite_rows": lite_rows,
        "legacy_rows": legacy_rows,
        "shared_rows": shared_rows,
        "only_lite": only_lite,
        "only_legacy": only_legacy,
        "value_mismatches": value_mismatches,
        "diag_shared": diag_shared,
        "diag_mismatches": diag_mismatches,
        "mean_abs_diff": mean_abs_diff,
        "p99_abs_diff": p99_abs_diff,
        "max_abs_diff": max_abs_diff,
        "lite_sorted": lite_sorted,
        "legacy_sorted": legacy_sorted,
        "first_diff_key": first_diff_key,
        "first_lite_value": first_lite_value,
        "first_legacy_value": first_legacy_value,
        "lite_path": str(lite_path),
        "legacy_path": str(legacy_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--ldetect-lite-root", required=True, type=Path)
    parser.add_argument("--legacy-dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    chrom = args.chromosome
    lite_partitions = _read_partitions(
        args.ldetect_lite_root / f"{chrom}_partitions.txt"
    )
    legacy_partitions = _read_partitions(
        args.legacy_dataset_root / "scripts" / f"{chrom}_partitions"
    )
    if lite_partitions != legacy_partitions:
        raise RuntimeError(
            "ldetect-lite and legacy partition files differ; covariance rows are "
            "not directly comparable partition-by-partition"
        )

    rows = []
    for start, end in lite_partitions:
        rows.append(
            _compare_partition(
                lite_path=args.ldetect_lite_root / chrom / f"{chrom}.{start}.{end}.h5",
                legacy_path=(
                    args.legacy_dataset_root / chrom / f"{chrom}.{start}.{end}.gz"
                ),
                start=start,
                end=end,
                tolerance=args.tolerance,
            )
        )

    fieldnames = [
        "partition_start",
        "partition_end",
        "lite_rows",
        "legacy_rows",
        "shared_rows",
        "only_lite",
        "only_legacy",
        "value_mismatches",
        "diag_shared",
        "diag_mismatches",
        "mean_abs_diff",
        "p99_abs_diff",
        "max_abs_diff",
        "lite_sorted",
        "legacy_sorted",
        "first_diff_key",
        "first_lite_value",
        "first_legacy_value",
        "lite_path",
        "legacy_path",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
