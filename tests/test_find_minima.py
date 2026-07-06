"""Tests for filter-width search helpers."""

from __future__ import annotations

import numpy as np

from ldetect_lite.find_minima import FlexibleBoundedAccessor


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
