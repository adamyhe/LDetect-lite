"""Tests for the run subcommand helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ldetect2._cli.cmd_run import (
    _breakpoint_subsets_for_run,
    _is_valid_covariance_partition,
    register,
)
from ldetect2.io.covariance_hdf5 import write_covariance_partition_hdf5


def test_full_covariance_partition_validates_against_full_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "valid.h5"
    write_covariance_partition_hdf5(
        path,
        i_pos=np.array([100], dtype=np.int32),
        j_pos=np.array([100], dtype=np.int32),
        shrink_ld=np.array([0.1]),
        i_gpos=np.array([0.1]),
        j_gpos=np.array([0.1]),
        naive_ld=np.array([0.1]),
        i_id=np.array(["rs1"]),
        j_id=np.array(["rs1"]),
    )

    assert _is_valid_covariance_partition(path, require_full=True)
    assert _is_valid_covariance_partition(path, require_full=False)


def test_compact_covariance_partition_validates_against_compact_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact.h5"
    write_covariance_partition_hdf5(
        path,
        i_pos=np.array([100], dtype=np.int32),
        j_pos=np.array([100], dtype=np.int32),
        shrink_ld=np.array([0.1]),
    )

    assert _is_valid_covariance_partition(path, require_full=False)
    assert not _is_valid_covariance_partition(path, require_full=True)


def test_invalid_covariance_partition_missing_shrink_ld(tmp_path: Path) -> None:
    path = tmp_path / "invalid.h5"
    path.write_text("not hdf5")

    assert not _is_valid_covariance_partition(path, require_full=False)


def test_run_subset_requests_only_final_breakpoint_subset() -> None:
    assert _breakpoint_subsets_for_run("fourier_ls", False) == {"fourier_ls"}


def test_run_all_breakpoint_subsets_preserves_full_output() -> None:
    assert _breakpoint_subsets_for_run("fourier_ls", True) is None


def _parse_run_args(extra: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    register(subparsers)
    return parser.parse_args(
        [
            "run",
            "--genetic-map",
            "map.gz",
            "--reference-panel",
            "panel.vcf.gz",
            "--individuals",
            "inds.txt",
            "--chromosome",
            "chr1",
            "--output-dir",
            "out",
            *extra,
        ]
    )


def test_covariance_compression_defaults_to_zstd() -> None:
    assert _parse_run_args([]).covariance_compression == "zstd"


def test_covariance_compression_accepts_lzf() -> None:
    args = _parse_run_args(["--covariance-compression", "lzf"])
    assert args.covariance_compression == "lzf"
