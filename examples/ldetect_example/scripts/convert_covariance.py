"""Convert legacy ldetect covariance gz (8-column TSV) to ldetect2 NPZ format.

Legacy column order:
    i_id  j_id  i_pos  j_pos  i_gpos  j_gpos  naive_ld  shrink_ld

Usage:
    python scripts/convert_covariance.py \
        --input  ref/cov_matrix/chr2/chr2.39967768.40067768.gz \
        --output work/chr2/chr2.39967768.40067768.npz
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np


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
                    f"{args.input}:{line_no}: expected 8 columns in legacy "
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
                    f"{args.input}:{line_no}: malformed numeric value in legacy "
                    "covariance row"
                ) from exc

    if not i_pos_l:
        raise ValueError(f"{args.input}: no covariance rows found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        i_id=np.array(i_id_l),
        j_id=np.array(j_id_l),
        i_pos=np.array(i_pos_l, dtype=np.int32),
        j_pos=np.array(j_pos_l, dtype=np.int32),
        i_gpos=np.array(i_gpos_l, dtype=np.float64),
        j_gpos=np.array(j_gpos_l, dtype=np.float64),
        naive_ld=np.array(naive_l, dtype=np.float64),
        shrink_ld=np.array(shrink_l, dtype=np.float64),
    )
    _validate_output(args.output)
    print(f"Converted {len(i_pos_l):,} pairs → {args.output}")


def _validate_output(path: Path) -> None:
    required = {"i_pos", "j_pos", "shrink_ld"}
    with np.load(path) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"{path}: converted NPZ is missing required field(s): "
                f"{', '.join(sorted(missing))}"
            )
        expected_dtypes = {
            "i_pos": np.dtype("int32"),
            "j_pos": np.dtype("int32"),
            "shrink_ld": np.dtype("float64"),
        }
        for key, dtype in expected_dtypes.items():
            if data[key].dtype != dtype:
                raise ValueError(
                    f"{path}: field {key!r} has dtype {data[key].dtype}, "
                    f"expected {dtype}"
                )


if __name__ == "__main__":
    main()
