"""Convert the reference ldetect covariance fixture to ldetect2 HDF5 format.

Input column order:
    i_id  j_id  i_pos  j_pos  i_gpos  j_gpos  naive_ld  shrink_ld

Usage:
    uv run python scripts/convert_covariance.py \
        --input  ref/cov_matrix/chr2/chr2.39967768.40067768.gz \
        --output work/chr2/chr2.39967768.40067768.h5
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np

from ldetect2.io.covariance_hdf5 import (
    validate_covariance_hdf5,
    write_covariance_partition_hdf5,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    i_id_l, j_id_l = [], []
    i_pos_l, j_pos_l = [], []
    i_gpos_l, j_gpos_l = [], []
    naive_l, shrink_l = [], []

    with gzip.open(args.input, "rt") as f:
        reader = csv.reader(f, delimiter=" ")
        for line_no, row in enumerate(reader, start=1):
            if not row or row[0].startswith("#"):
                continue
            if len(row) != 8:
                raise ValueError(
                    f"{args.input}:{line_no}: expected 8 columns in reference "
                    f"covariance row, found {len(row)}"
                )
            i_id_l.append(row[0])
            j_id_l.append(row[1])
            try:
                i_pos_l.append(int(row[2]))
                j_pos_l.append(int(row[3]))
                i_gpos_l.append(float(row[4]))
                j_gpos_l.append(float(row[5]))
                naive_l.append(float(row[6]))
                shrink_l.append(float(row[7]))
            except ValueError as exc:
                raise ValueError(
                    f"{args.input}:{line_no}: malformed numeric value in reference "
                    "covariance row"
                ) from exc

    if not i_pos_l:
        raise ValueError(f"{args.input}: no covariance rows found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    chrom, start, end = _partition_metadata(args.output)
    write_covariance_partition_hdf5(
        args.output,
        chrom=chrom,
        start=start,
        end=end,
        i_pos=np.array(i_pos_l, dtype=np.int32),
        j_pos=np.array(j_pos_l, dtype=np.int32),
        i_gpos=np.array(i_gpos_l, dtype=np.float64),
        j_gpos=np.array(j_gpos_l, dtype=np.float64),
        naive_ld=np.array(naive_l, dtype=np.float64),
        shrink_ld=np.array(shrink_l, dtype=np.float64),
        i_id=np.array(i_id_l),
        j_id=np.array(j_id_l),
    )
    _validate_output(args.output)
    print(f"Converted {len(i_pos_l):,} pairs → {args.output}")


def _partition_metadata(path: Path) -> tuple[str | None, int | None, int | None]:
    parts = path.stem.split(".")
    if len(parts) >= 3:
        try:
            return parts[0], int(parts[-2]), int(parts[-1])
        except ValueError:
            pass
    return None, None, None


def _validate_output(path: Path) -> None:
    if not validate_covariance_hdf5(path, require_full=True):
        raise ValueError(f"{path}: converted HDF5 covariance partition is invalid")


if __name__ == "__main__":
    main()
