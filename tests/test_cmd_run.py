"""Tests for the run subcommand helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ldetect_lite._cli.cmd_run import (
    _breakpoint_subsets_for_run,
    _fused_vector_ready,
    _is_valid_covariance_partition,
    _resolve_workers,
    _validate_local_search_source,
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


def test_fused_vector_and_local_search_source_default_off() -> None:
    args = _parse_run_args([])
    assert args.fused_vector is False
    assert args.local_search_source == "cache"


def test_local_search_source_accepts_vcf_recompute() -> None:
    args = _parse_run_args(["--local-search-source", "vcf-recompute"])
    assert args.local_search_source == "vcf-recompute"


_PARTITIONS = [(100, 200), (200, 300)]


def test_fused_vector_ready_when_flag_set_and_all_partitions_fresh() -> None:
    assert _fused_vector_ready(
        True, _PARTITIONS, _PARTITIONS, {(100, 200): object(), (200, 300): object()}
    )


def test_fused_vector_not_ready_when_flag_unset() -> None:
    assert not _fused_vector_ready(
        False, _PARTITIONS, _PARTITIONS, {(100, 200): object(), (200, 300): object()}
    )


def test_fused_vector_not_ready_when_some_partitions_skipped() -> None:
    # Only (200, 300) was pending (freshly computed); (100, 200) was skipped
    # as already-valid from a prior run, so it has no fragment.
    pending = [(200, 300)]
    assert not _fused_vector_ready(True, pending, _PARTITIONS, {(200, 300): object()})


def test_fused_vector_not_ready_when_fragment_missing_despite_full_pending() -> None:
    # Defensive: pending matches partitions, but a fragment is somehow
    # missing (e.g. a partition genuinely produced no vector contribution).
    assert not _fused_vector_ready(
        True, _PARTITIONS, _PARTITIONS, {(100, 200): object()}
    )


def test_validate_local_search_source_allows_cache_with_any_workers() -> None:
    assert _validate_local_search_source("cache", 8, True) is None


def test_validate_local_search_source_allows_vcf_recompute_single_worker() -> None:
    assert _validate_local_search_source("vcf-recompute", 1, False) is None


def test_validate_local_search_source_rejects_multiple_workers() -> None:
    error = _validate_local_search_source("vcf-recompute", 4, False)
    assert error is not None
    assert "local-search-workers" in error


def test_validate_local_search_source_rejects_high_precision() -> None:
    error = _validate_local_search_source("vcf-recompute", 1, True)
    assert error is not None
    assert "high-precision" in error
