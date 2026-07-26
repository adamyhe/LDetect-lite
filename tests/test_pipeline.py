"""Unit tests for pipeline orchestration helpers."""

from __future__ import annotations

import pytest

from ldetect_lite.pipeline import _filter_width_search_workers


def test_filter_width_search_workers_use_candidate_threads_by_default() -> None:
    assert _filter_width_search_workers(workers=8, filter_workers=1) == 8


def test_filter_width_search_workers_disable_candidate_threads() -> None:
    assert _filter_width_search_workers(workers=8, filter_workers=2) == 1


def test_filter_width_search_workers_reject_nonpositive_filter_workers() -> None:
    with pytest.raises(ValueError, match="filter_workers"):
        _filter_width_search_workers(workers=8, filter_workers=0)
