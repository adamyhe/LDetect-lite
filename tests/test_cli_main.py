"""Tests for early CLI setup that must happen before subcommand imports."""

from __future__ import annotations

import os

from ldetect_lite._cli.main import _configure_native_thread_caps

_CAPS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def test_configure_native_thread_caps_for_run_overrides_ambient_env(
    monkeypatch,
) -> None:
    for name in _CAPS:
        monkeypatch.setenv(name, "64")

    _configure_native_thread_caps(["run", "--workers", "4"])

    assert {name: os.environ[name] for name in _CAPS} == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "4",
    }


def test_configure_native_thread_caps_allows_filter_worker_numba_threads(
    monkeypatch,
) -> None:
    for name in _CAPS:
        monkeypatch.delenv(name, raising=False)

    _configure_native_thread_caps(["run", "--workers", "2", "--filter-workers", "3"])

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["NUMBA_NUM_THREADS"] == "3"


def test_configure_native_thread_caps_ignores_non_pipeline_commands(
    monkeypatch,
) -> None:
    for name in _CAPS:
        monkeypatch.delenv(name, raising=False)

    _configure_native_thread_caps(["--version"])

    assert all(name not in os.environ for name in _CAPS)
