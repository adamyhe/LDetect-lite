"""Tests for the run subcommand helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ldetect2._cli.cmd_run import (
    _breakpoint_subsets_for_run,
    _is_valid_covariance_partition,
    _regenerate_signal_sidecar,
    register,
)
from ldetect2._util.vector_array import compute_partition_signal
from ldetect2.io.covariance_hdf5 import (
    open_covariance_reader,
    write_covariance_partition_hdf5,
)
from ldetect2.io.partitions import CovarianceStore
from ldetect2.io.signal_hdf5 import read_signal_partition_hdf5, validate_signal_hdf5


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


def test_signal_cache_defaults_to_off() -> None:
    assert _parse_run_args([]).signal_cache == "off"


def test_signal_cache_accepts_auto() -> None:
    assert _parse_run_args(["--signal-cache", "auto"]).signal_cache == "auto"


def test_covariance_compression_defaults_to_zstd() -> None:
    assert _parse_run_args([]).covariance_compression == "zstd"


def test_covariance_compression_accepts_lzf() -> None:
    args = _parse_run_args(["--covariance-compression", "lzf"])
    assert args.covariance_compression == "lzf"


def test_regenerate_signal_sidecar_matches_direct_computation(tmp_path: Path) -> None:
    root = tmp_path / "cov"
    chrom_dir = root / "chr1"
    chrom_dir.mkdir(parents=True)
    write_covariance_partition_hdf5(
        chrom_dir / "chr1.100.300.h5",
        i_pos=np.array([100, 100, 200, 200, 300], dtype=np.int32),
        j_pos=np.array([100, 200, 200, 300, 300], dtype=np.int32),
        shrink_ld=np.array([0.5, 0.3, 0.7, 0.1, 0.9], dtype=np.float64),
    )
    store = CovarianceStore(root=root)

    _regenerate_signal_sidecar(store, "chr1", 100, 300)

    signal_path = store.signal_path("chr1", 100, 300)
    assert validate_signal_hdf5(signal_path)

    with open_covariance_reader(
        store.partition_path("chr1", 100, 300), 100, 300
    ) as reader:
        expected_loci, expected_sum_r2 = compute_partition_signal(reader, 100, 300)
    loci, sum_r2 = read_signal_partition_hdf5(signal_path)
    np.testing.assert_array_equal(loci, expected_loci)
    np.testing.assert_allclose(sum_r2, expected_sum_r2)
