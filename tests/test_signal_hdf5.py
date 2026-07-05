"""Tests for the HDF5 signal-sidecar storage helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ldetect2.io.signal_hdf5 import (
    read_signal_partition_hdf5,
    validate_signal_hdf5,
    write_signal_partition_hdf5,
)


def test_write_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "chr1.100.300.signal.h5"
    write_signal_partition_hdf5(
        path,
        loci=np.array([100, 200, 300], dtype=np.int32),
        sum_r2=np.array([0.0, 1.5, 2.5], dtype=np.float64),
        chrom="chr1",
        start=100,
        end=300,
    )

    loci, sum_r2 = read_signal_partition_hdf5(path)

    np.testing.assert_array_equal(loci, np.array([100, 200, 300]))
    np.testing.assert_allclose(sum_r2, np.array([0.0, 1.5, 2.5]))


def test_write_empty_sidecar_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "chr1.100.300.signal.h5"
    write_signal_partition_hdf5(
        path,
        loci=np.array([], dtype=np.int32),
        sum_r2=np.array([], dtype=np.float64),
    )

    loci, sum_r2 = read_signal_partition_hdf5(path)

    assert loci.size == 0
    assert sum_r2.size == 0
    assert validate_signal_hdf5(path)


def test_write_rejects_mismatched_shapes(tmp_path: Path) -> None:
    path = tmp_path / "chr1.100.300.signal.h5"
    with pytest.raises(ValueError, match="identical shapes"):
        write_signal_partition_hdf5(
            path,
            loci=np.array([100, 200], dtype=np.int32),
            sum_r2=np.array([1.0], dtype=np.float64),
        )


def test_write_rejects_unsorted_loci(tmp_path: Path) -> None:
    path = tmp_path / "chr1.100.300.signal.h5"
    with pytest.raises(ValueError, match="sorted ascending"):
        write_signal_partition_hdf5(
            path,
            loci=np.array([200, 100], dtype=np.int32),
            sum_r2=np.array([1.0, 2.0], dtype=np.float64),
        )


def test_write_rejects_duplicate_loci(tmp_path: Path) -> None:
    path = tmp_path / "chr1.100.300.signal.h5"
    with pytest.raises(ValueError, match="sorted ascending"):
        write_signal_partition_hdf5(
            path,
            loci=np.array([100, 100], dtype=np.int32),
            sum_r2=np.array([1.0, 2.0], dtype=np.float64),
        )


def test_validate_rejects_missing_file(tmp_path: Path) -> None:
    assert not validate_signal_hdf5(tmp_path / "missing.signal.h5")


def test_validate_rejects_wrong_format(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "wrong.signal.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "not-a-signal-file"
        h5.attrs["version"] = 1
        h5.create_group("signal").create_dataset("loci", data=np.array([100]))

    assert not validate_signal_hdf5(path)


def test_validate_rejects_missing_dataset(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "partial.signal.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "ldetect2-signal-h5"
        h5.attrs["version"] = 1
        h5.create_group("signal").create_dataset("loci", data=np.array([100]))

    assert not validate_signal_hdf5(path)


def test_read_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_signal_partition_hdf5(tmp_path / "missing.signal.h5")


def test_read_missing_dataset_raises(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "partial.signal.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "ldetect2-signal-h5"
        h5.attrs["version"] = 1
        h5.create_group("signal").create_dataset("loci", data=np.array([100]))

    with pytest.raises(ValueError, match="missing required dataset"):
        read_signal_partition_hdf5(path)
