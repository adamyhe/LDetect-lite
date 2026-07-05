"""Tests for the calc-covariance subcommand's --signal-output wiring."""

from __future__ import annotations

import argparse
import gzip
import sys
from io import StringIO
from pathlib import Path

import pytest

from ldetect2._cli.cmd_covariance import register
from ldetect2.io.signal_hdf5 import validate_signal_hdf5


def _parse_args(extra: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    register(subparsers)
    return parser.parse_args(
        [
            "calc-covariance",
            "--genetic-map",
            "map.gz",
            "--individuals",
            "inds.txt",
            "--output",
            "out.h5",
            *extra,
        ]
    )


def test_signal_output_defaults_to_none() -> None:
    assert _parse_args([]).signal_output is None


def test_signal_output_accepts_path() -> None:
    args = _parse_args(["--signal-output", "out.signal.h5"])
    assert args.signal_output == Path("out.signal.h5")


def test_run_writes_signal_sidecar_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    map_path = tmp_path / "map.gz"
    with gzip.open(map_path, "wt") as f:
        f.write("1 100 0.000\n")
        f.write("1 200 0.001\n")
        f.write("1 300 0.002\n")
    individuals_path = tmp_path / "inds.txt"
    individuals_path.write_text("sample_a\nsample_b\n")

    vcf_stream = StringIO(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
                "\tsample_a\tsample_b",
                "1\t100\trs_a\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0",
                "1\t200\trs_b\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|1",
                "1\t300\trs_c\tA\tG\t.\tPASS\t.\tGT\t1|1\t0|1",
                "",
            ]
        )
    )
    monkeypatch.setattr(sys, "stdin", vcf_stream)
    args = _parse_args(
        [
            "--genetic-map",
            str(map_path),
            "--individuals",
            str(individuals_path),
            "--output",
            str(tmp_path / "cov.h5"),
            "--signal-output",
            str(tmp_path / "cov.signal.h5"),
        ]
    )
    assert args.func(args) == 0

    assert validate_signal_hdf5(tmp_path / "cov.signal.h5")
