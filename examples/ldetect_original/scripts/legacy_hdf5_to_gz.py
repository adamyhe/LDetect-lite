"""Convert ldetect2 HDF5 covariance partitions to legacy ldetect .gz text."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

from ldetect2.io.covariance_hdf5 import open_covariance_reader


def _parse_bounds(path: Path) -> tuple[int, int]:
    parts = path.name.split(".")
    if len(parts) < 4:
        raise ValueError(
            f"Cannot infer partition bounds from {path}; expected name.start.end.h5"
        )
    return int(parts[-3]), int(parts[-2])


def convert_partition(input_path: Path, output_path: Path) -> int:
    """Write the lean legacy 8-column text shape from an HDF5 partition.

    The copied legacy downstream code reads positions from columns 2 and 3 and
    shrinkage LD from column 7. Placeholder IDs/genetic positions/naive LD are
    sufficient for matrix-to-vector, minima, metric, local search, and BED
    extraction diagnostics.
    """
    start, end = _parse_bounds(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with open_covariance_reader(input_path, start, end) as reader:
        with gzip.open(output_path, "wt", newline="") as f:
            writer = csv.writer(f, delimiter=" ")
            for chunk in reader.iter_rows(start, end, 1_000_000):
                for i_pos, j_pos, shrink_ld in zip(
                    chunk.lo, chunk.hi, chunk.shrink_ld
                ):
                    i_pos_int = int(i_pos)
                    j_pos_int = int(j_pos)
                    shrink = float(shrink_ld)
                    writer.writerow(
                        [
                            str(i_pos_int),
                            str(j_pos_int),
                            i_pos_int,
                            j_pos_int,
                            0.0,
                            0.0,
                            shrink,
                            shrink,
                        ]
                    )
                    rows_written += 1
    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = convert_partition(args.input, args.output)
    print(f"converted {args.input} -> {args.output} ({rows} rows)")


if __name__ == "__main__":
    main()
