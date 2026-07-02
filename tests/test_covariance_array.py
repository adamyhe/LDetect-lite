"""Tests for array-backed covariance loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ldetect2._util.covariance_array import (
    _load_partition_arrays,
    load_covariance_partitions,
)
from ldetect2.io.covariance_hdf5 import write_covariance_partition_hdf5
from ldetect2.io.partitions import CovarianceStore


def test_load_partition_arrays_reports_invalid_hdf5_schema(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "chr1.100.200.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "ldetect2-covariance-h5"
        h5.attrs["version"] = 1
        h5.create_group("covariance").create_dataset(
            "lo", data=np.array([100], dtype=np.int32)
        )

    with pytest.raises(KeyError):
        _load_partition_arrays(path)


def test_load_covariance_partitions_slices_to_requested_range(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cov"
    chrom_dir = root / "chr1"
    chrom_dir.mkdir(parents=True)
    (root / "chr1_partitions.txt").write_text("100 500\n")
    write_covariance_partition_hdf5(
        chrom_dir / "chr1.100.500.h5",
        i_pos=np.array([100, 200, 300, 400], dtype=np.int32),
        j_pos=np.array([100, 300, 500, 400], dtype=np.int32),
        shrink_ld=np.ones(4),
    )

    (partition,) = load_covariance_partitions(
        "chr1",
        CovarianceStore(root=root),
        [(100, 500)],
        snp_first=200,
        snp_last=400,
    )

    assert partition.i_pos.tolist() == [200, 400]
    assert partition.j_pos.tolist() == [300, 400]
    assert partition.i_pos.dtype == np.dtype("int32")


def test_load_partition_arrays_downcasts_int64_positions_when_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chr1.100.300.h5"
    write_covariance_partition_hdf5(
        path,
        i_pos=np.array([100, 200], dtype=np.int64),
        j_pos=np.array([200, 300], dtype=np.int64),
        shrink_ld=np.ones(2),
    )

    i_pos, j_pos, _ = _load_partition_arrays(path)

    assert i_pos.dtype == np.dtype("int32")
    assert j_pos.dtype == np.dtype("int32")


def test_load_partition_arrays_keeps_int64_positions_when_needed(
    tmp_path: Path,
) -> None:
    too_large = np.iinfo(np.int32).max + 1
    path = tmp_path / f"chr1.100.{too_large}.h5"
    write_covariance_partition_hdf5(
        path,
        i_pos=np.array([100, too_large], dtype=np.int64),
        j_pos=np.array([200, too_large], dtype=np.int64),
        shrink_ld=np.ones(2),
    )

    i_pos, j_pos, _ = _load_partition_arrays(path)

    assert i_pos.dtype == np.dtype("int64")
    assert j_pos.dtype == np.dtype("int64")
