"""Tests for filter-width search helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ldetect_lite.filters import apply_filter_get_minima
from ldetect_lite.find_minima import (
    FlexibleBoundedAccessor,
    _find_end,
    custom_binary_search_with_trackback,
)


def test_flexible_bounded_accessor_caches_width_counts() -> None:
    calls: list[int] = []

    def count_minima(data: np.ndarray, width: int) -> int:
        calls.append(width)
        return int(data[width])

    data = np.arange(10)
    accessor = FlexibleBoundedAccessor(data, count_minima, 0, 9, invert=True)

    assert accessor[2] == 7
    assert accessor[2] == 7
    assert accessor[3] == 6
    assert calls == [7, 6]


def test_trackback_threaded_matches_sequential() -> None:
    """Threaded search (search_workers>1) must return the identical width.

    The threaded path batches independent width evaluations concurrently but
    applies the same first-match-wins decision rule in the same left-to-right
    order as the sequential path -- this pins that the refactor is a pure
    concurrency change, not a behavior change.
    """
    arr = np.random.default_rng(1).normal(size=3000).cumsum()

    sequential = custom_binary_search_with_trackback(
        arr, apply_filter_get_minima, srch_val=15, search_workers=1
    )
    threaded = custom_binary_search_with_trackback(
        arr, apply_filter_get_minima, srch_val=15, search_workers=4
    )

    assert threaded == sequential


def test_find_end_threaded_matches_sequential() -> None:
    """Threaded exponential search must find the identical smallest x.

    Batches up to max_workers doubling candidates per round but still
    returns the smallest x satisfying f(data, x) < val, matching the
    sequential recursive doubling search exactly.
    """
    arr = np.random.default_rng(2).normal(size=3000).cumsum()

    sequential = _find_end(arr, apply_filter_get_minima, x=10, val=15)
    threaded = _find_end(arr, apply_filter_get_minima, x=10, val=15, max_workers=4)

    assert threaded == sequential


def test_find_end_threaded_raises_at_max_srch_val() -> None:
    """Hitting max_srch_val must raise the same error, threaded or not."""
    arr = np.random.default_rng(3).normal(size=3000).cumsum()

    with pytest.raises(ValueError, match="Max search value"):
        _find_end(arr, apply_filter_get_minima, x=10, val=15, max_srch_val=20)

    with pytest.raises(ValueError, match="Max search value"):
        _find_end(
            arr,
            apply_filter_get_minima,
            x=10,
            val=15,
            max_srch_val=20,
            max_workers=4,
        )
