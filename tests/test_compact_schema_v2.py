"""Round-trip and size validation for the prototype v2 (lo-less) compact schema.

Uses the real EUR chr2 fixture (chr2:39967768-40067768, ~225K surviving
pairs) already relied on elsewhere in this test suite -- a synthetic
handful-of-SNPs fixture wouldn't show a meaningful size delta once HDF5's
per-dataset overhead dominates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ldetect_lite._util.compact_schema_v2 import (
    convert_v1_to_v2,
    read_v2_diagonal,
    read_v2_partition,
)
from ldetect_lite.io.covariance_hdf5 import open_covariance_reader
from ldetect_lite.io.partitions import CovarianceStore

_CHR2_START = 39967768
_CHR2_END = 40067768


@pytest.fixture()
def v1_v2_pair(
    example_data_dir: Path, example_store: CovarianceStore, tmp_path: Path
) -> tuple[Path, Path]:
    v1_path = example_store.partition_path("chr2", _CHR2_START, _CHR2_END)
    v2_path = tmp_path / "chr2.v2.h5"
    convert_v1_to_v2(v1_path, v2_path, start=_CHR2_START, end=_CHR2_END, chrom="chr2")
    return v1_path, v2_path


def test_v2_roundtrip_matches_v1_exactly(v1_v2_pair: tuple[Path, Path]) -> None:
    v1_path, v2_path = v1_v2_pair

    with open_covariance_reader(v1_path, _CHR2_START, _CHR2_END) as reader:
        v1_rows = reader.read_all()
        v1_diag_pos, v1_diag_val = reader.read_diagonal()

    v2_rows = read_v2_partition(v2_path)
    v2_diag_pos, v2_diag_val = read_v2_diagonal(v2_path)

    np.testing.assert_array_equal(v2_rows.lo, v1_rows.lo)
    np.testing.assert_array_equal(v2_rows.hi, v1_rows.hi)
    np.testing.assert_array_equal(v2_rows.shrink_ld, v1_rows.shrink_ld)
    np.testing.assert_array_equal(v2_diag_pos, v1_diag_pos)
    np.testing.assert_array_equal(v2_diag_val, v1_diag_val)


def test_v2_file_size_vs_v1(v1_v2_pair: tuple[Path, Path]) -> None:
    v1_path, v2_path = v1_v2_pair
    v1_size = v1_path.stat().st_size
    v2_size = v2_path.stat().st_size

    with open_covariance_reader(v1_path, _CHR2_START, _CHR2_END) as reader:
        n_rows = reader.row_count

    print(
        f"\n[v2 schema size] n_rows={n_rows} v1_bytes={v1_size} v2_bytes={v2_size} "
        f"ratio={v2_size / v1_size:.3f} "
        f"v1_bytes_per_row={v1_size / n_rows:.2f} "
        f"v2_bytes_per_row={v2_size / n_rows:.2f}"
    )
    assert v2_size < v1_size, (
        f"expected the lo-less v2 schema to be smaller: v1={v1_size} v2={v2_size}"
    )
