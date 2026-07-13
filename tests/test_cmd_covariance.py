"""Tests for the calc-covariance subcommand's --covariance-compression wiring."""

from __future__ import annotations

import argparse

from ldetect_lite._cli.cmd_covariance import register


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


def test_covariance_compression_defaults_to_zstd() -> None:
    assert _parse_args([]).covariance_compression == "zstd"


def test_covariance_compression_accepts_lzf() -> None:
    args = _parse_args(["--covariance-compression", "lzf"])
    assert args.covariance_compression == "lzf"


def test_ld_kernel_defaults_to_bitpacked() -> None:
    assert _parse_args([]).ld_kernel == "bitpacked"


def test_ld_kernel_accepts_uint8_reference_backend() -> None:
    args = _parse_args(["--ld-kernel", "uint8"])
    assert args.ld_kernel == "uint8"
