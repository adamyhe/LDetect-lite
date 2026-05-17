"""Tests for array-backed covariance loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ldetect2._util.covariance_array import (
    _load_partition_arrays,
    load_covariance_partitions,
)
from ldetect2.io.partitions import CovarianceStore


def test_load_partition_arrays_reports_invalid_npz_schema(tmp_path: Path) -> None:
    path = tmp_path / "chr1.100.200.npz"
    np.savez_compressed(
        path,
        i_pos=np.array([100], dtype=np.int32),
        j_pos=np.array([100], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="Missing key\\(s\\): shrink_ld"):
        _load_partition_arrays(path)


def test_load_covariance_partitions_slices_to_requested_range(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cov"
    chrom_dir = root / "chr1"
    chrom_dir.mkdir(parents=True)
    (root / "chr1_partitions.txt").write_text("100 500\n")
    np.savez_compressed(
        chrom_dir / "chr1.100.500.npz",
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
