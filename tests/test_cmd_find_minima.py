"""Tests for the find-minima subcommand helpers."""

from __future__ import annotations

import argparse

from ldetect_lite._cli.cmd_find_minima import register
from ldetect_lite._cli.cmd_run import _resolve_workers


def _parse_find_minima_args(extra: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    register(subparsers)
    return parser.parse_args(
        [
            "find-minima",
            "--input",
            "vector.txt.gz",
            "--chr-name",
            "chr1",
            "--dataset-path",
            "dataset",
            "--n-snps-bw-bpoints",
            "10000",
            "--output",
            "out.json",
            *extra,
        ]
    )


def test_metric_workers_defaults_to_none_and_inherits_workers() -> None:
    args = _parse_find_minima_args(["--workers", "4"])
    assert args.metric_workers is None
    assert _resolve_workers(args.metric_workers, args.workers) == 4


def test_metric_workers_override_takes_precedence() -> None:
    args = _parse_find_minima_args(["--workers", "4", "--metric-workers", "1"])
    assert _resolve_workers(args.metric_workers, args.workers) == 1


def test_filter_window_defaults_to_scipy_periodic_and_accepts_symmetric() -> None:
    assert _parse_find_minima_args([]).filter_window == "scipy-periodic"
    assert (
        _parse_find_minima_args(["--filter-window", "symmetric"]).filter_window
        == "symmetric"
    )


def test_filter_workers_defaults_to_one_and_accepts_override() -> None:
    assert _parse_find_minima_args([]).filter_workers == 1
    assert _parse_find_minima_args(["--filter-workers", "4"]).filter_workers == 4
