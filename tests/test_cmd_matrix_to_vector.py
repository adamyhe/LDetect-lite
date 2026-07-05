"""Tests for the matrix-to-vector subcommand's --prefer-signal-cache wiring."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pytest

from ldetect2._cli.cmd_matrix_to_vector import register
from ldetect2._util.vector_array import compute_partition_signal
from ldetect2.io.covariance_hdf5 import (
    open_covariance_reader,
    write_covariance_partition_hdf5,
)
from ldetect2.io.partitions import CovarianceStore
from ldetect2.io.signal_hdf5 import write_signal_partition_hdf5


def _parse_args(
    dataset_path: Path, output: Path, extra: list[str]
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    register(subparsers)
    return parser.parse_args(
        [
            "matrix-to-vector",
            "--dataset-path",
            str(dataset_path),
            "--name",
            "testchr",
            "--output",
            str(output),
            *extra,
        ]
    )


def _make_store_with_signal(tmp_path: Path) -> CovarianceStore:
    root = tmp_path / "cov"
    chrom_dir = root / "testchr"
    chrom_dir.mkdir(parents=True)
    (root / "testchr_partitions.txt").write_text("100 300\n")
    write_covariance_partition_hdf5(
        chrom_dir / "testchr.100.300.h5",
        i_pos=np.array([100, 100, 200, 200, 300], dtype=np.int32),
        j_pos=np.array([100, 200, 200, 300, 300], dtype=np.int32),
        shrink_ld=np.array([0.5, 0.3, 0.7, 0.1, 0.9], dtype=np.float64),
    )
    store = CovarianceStore(root=root)
    with open_covariance_reader(
        store.partition_path("testchr", 100, 300), 100, 300
    ) as reader:
        loci, sum_r2 = compute_partition_signal(reader, 100, 300)
    write_signal_partition_hdf5(
        store.signal_path("testchr", 100, 300), loci=loci, sum_r2=sum_r2
    )
    return store


def _read_vector(path: Path) -> dict[int, float]:
    data: dict[int, float] = {}
    with gzip.open(path, "rt") as f:
        for raw in f:
            row = raw.strip().split()
            if row:
                data[int(row[0])] = float(row[1])
    return data


def test_prefer_signal_cache_defaults_to_false() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    register(subparsers)
    args = parser.parse_args(
        [
            "matrix-to-vector",
            "--dataset-path",
            "cov",
            "--name",
            "testchr",
            "--output",
            "out.txt.gz",
        ]
    )
    assert args.prefer_signal_cache is False


def test_prefer_signal_cache_matches_default_path(tmp_path: Path) -> None:
    store = _make_store_with_signal(tmp_path)
    default_out = tmp_path / "default.txt.gz"
    signal_out = tmp_path / "signal.txt.gz"

    args_default = _parse_args(store.root, default_out, [])
    args_signal = _parse_args(store.root, signal_out, ["--prefer-signal-cache"])

    assert args_default.func(args_default) == 0
    assert args_signal.func(args_signal) == 0

    assert _read_vector(default_out) == _read_vector(signal_out)


def test_prefer_signal_cache_rejects_generate_heatmap(tmp_path: Path) -> None:
    store = _make_store_with_signal(tmp_path)
    out_path = tmp_path / "out.txt.gz"
    args = _parse_args(
        store.root, out_path, ["--prefer-signal-cache", "--generate-heatmap"]
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        args.func(args)
