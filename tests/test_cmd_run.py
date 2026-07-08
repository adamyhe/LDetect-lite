"""Tests for the run subcommand helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ldetect_lite._cli.cmd_run import (
    _breakpoint_subsets_for_run,
    _delete_covariance_cache,
    _is_valid_covariance_partition,
    _resolve_workers,
    register,
)
from ldetect_lite.io.covariance_hdf5 import write_covariance_partition_hdf5


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


def test_stage_workers_default_to_none_and_inherit_shared_workers() -> None:
    args = _parse_run_args(["--workers", "4"])
    assert args.matrix_workers is None
    assert args.local_search_workers is None
    assert args.metric_workers is None
    assert _resolve_workers(args.matrix_workers, args.workers) == 4
    assert _resolve_workers(args.local_search_workers, args.workers) == 4
    assert _resolve_workers(args.metric_workers, args.workers) == 4


def test_stage_worker_override_takes_precedence_over_shared_workers() -> None:
    args = _parse_run_args(["--workers", "4", "--local-search-workers", "1"])
    assert _resolve_workers(args.matrix_workers, args.workers) == 4
    assert _resolve_workers(args.local_search_workers, args.workers) == 1


def test_delete_covariance_cache_defaults_to_false() -> None:
    assert _parse_run_args([]).delete_covariance_cache is False


def test_delete_covariance_cache_flag_sets_true() -> None:
    args = _parse_run_args(["--delete-covariance-cache"])
    assert args.delete_covariance_cache is True


def test_delete_covariance_cache_removes_directory(tmp_path: Path) -> None:
    cov_dir = tmp_path / "22"
    cov_dir.mkdir()
    (cov_dir / "22.100.200.h5").write_bytes(b"")
    (cov_dir / "22.200.300.h5").write_bytes(b"")

    _delete_covariance_cache(cov_dir)

    assert not cov_dir.exists()


def test_delete_covariance_cache_noop_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    _delete_covariance_cache(missing)  # must not raise
    assert not missing.exists()
