"""Tests for the matrix-to-vector subcommand helpers."""

from __future__ import annotations

import argparse

from ldetect_lite._cli.cmd_matrix_to_vector import register


def _parse_matrix_to_vector_args(extra: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    register(subparsers)
    return parser.parse_args(
        [
            "matrix-to-vector",
            "--dataset-path",
            "dataset",
            "--name",
            "chr1",
            "--output",
            "out.txt.gz",
            *extra,
        ]
    )


def test_workers_defaults_to_one() -> None:
    assert _parse_matrix_to_vector_args([]).workers == 1


def test_workers_accepts_override() -> None:
    assert _parse_matrix_to_vector_args(["--workers", "4"]).workers == 4
