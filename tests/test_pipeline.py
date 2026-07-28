"""Unit tests for pipeline orchestration helpers."""

from __future__ import annotations

import pytest

from ldetect_lite.pipeline import _adaptive_filter_workers


def test_adaptive_filter_workers_use_worker_budget_by_default() -> None:
    assert _adaptive_filter_workers(workers=8, filter_workers=1) == 8


def test_adaptive_filter_workers_honor_larger_filter_override() -> None:
    assert _adaptive_filter_workers(workers=2, filter_workers=4) == 4


def test_adaptive_filter_workers_reject_nonpositive_filter_workers() -> None:
    with pytest.raises(ValueError, match="filter_workers"):
        _adaptive_filter_workers(workers=8, filter_workers=0)


def test_adaptive_filter_workers_reject_nonpositive_workers() -> None:
    with pytest.raises(ValueError, match="workers"):
        _adaptive_filter_workers(workers=0, filter_workers=1)
