"""Tests for the run subcommand helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ldetect2._cli.cmd_run import (
    _COMPACT_COVARIANCE_KEYS,
    _FULL_COVARIANCE_KEYS,
    _is_valid_covariance_partition,
)


def test_full_covariance_partition_validates_against_full_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "valid.npz"
    np.savez_compressed(
        path,
        i_pos=np.array([100], dtype=np.int32),
        j_pos=np.array([100], dtype=np.int32),
        i_gpos=np.array([0.1]),
        j_gpos=np.array([0.1]),
        naive_ld=np.array([0.1]),
        shrink_ld=np.array([0.1]),
        i_id=np.array(["rs1"]),
        j_id=np.array(["rs1"]),
    )

    assert _is_valid_covariance_partition(path, _FULL_COVARIANCE_KEYS)
    assert _is_valid_covariance_partition(path, _COMPACT_COVARIANCE_KEYS)


def test_compact_covariance_partition_validates_against_compact_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact.npz"
    np.savez_compressed(
        path,
        i_pos=np.array([100], dtype=np.int32),
        j_pos=np.array([100], dtype=np.int32),
        shrink_ld=np.array([0.1]),
    )

    assert _is_valid_covariance_partition(path, _COMPACT_COVARIANCE_KEYS)
    assert not _is_valid_covariance_partition(path, _FULL_COVARIANCE_KEYS)


def test_invalid_covariance_partition_missing_shrink_ld(tmp_path: Path) -> None:
    path = tmp_path / "invalid.npz"
    np.savez_compressed(
        path,
        i_pos=np.array([100], dtype=np.int32),
        j_pos=np.array([100], dtype=np.int32),
    )

    assert not _is_valid_covariance_partition(path, _COMPACT_COVARIANCE_KEYS)
