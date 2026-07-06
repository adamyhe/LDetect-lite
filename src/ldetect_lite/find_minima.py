"""Filter-width search: find the Hanning window width that yields exactly N minima."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ldetect_lite._util.binary_search import find_le_ind
from ldetect_lite._util.logging import log_msg


class FlexibleBoundedAccessor:
    """Bounded, optionally-inverted wrapper around an array accessed via a function.

    Args:
        data: The underlying data array.
        f: Accessor function ``f(data, index) -> value``.
        min_ind: Minimum valid index.
        max_ind: Maximum valid index (inclusive).
        invert: If True, index ``i`` maps to ``data[max_ind - i]``.
    """

    def __init__(
        self,
        data: np.ndarray,
        f: Callable[[np.ndarray, int], int],
        min_ind: int,
        max_ind: int,
        invert: bool,
    ) -> None:
        self.data = data
        self.f = f
        self.min_ind = min_ind
        self.max_ind = max_ind
        self._invert = invert
        self._cache: dict[int, int] = {}

    def __getitem__(self, i: int) -> int:
        if i < self.min_ind or i > self.max_ind:
            raise IndexError(f"Index {i} out of range [{self.min_ind}, {self.max_ind}]")
        actual = self.max_ind - i if self._invert else i
        cached = self._cache.get(actual)
        if cached is not None:
            return cached
        value = self.f(self.data, actual)
        self._cache[actual] = value
        return value

    def __len__(self) -> int:
        return self.max_ind - self.min_ind + 1


def _find_end(
    data: np.ndarray,
    f: Callable[[np.ndarray, int], int],
    x: int,
    val: int,
    max_srch_val: float = float("inf"),
) -> int:
    """Exponential search upward for the smallest x where f(data, x) < val."""
    if x <= 0:
        raise ValueError("x must be > 0")
    if x >= max_srch_val:
        raise ValueError(f"Max search value {max_srch_val} exceeded")
    if f(data, x) < val:
        return x
    return _find_end(data, f, x * 2, val, max_srch_val)


def _trackback(
    wrapper: FlexibleBoundedAccessor,
    srch_val: int,
    start_search: int,
    delta_coarse: int,
    step_coarse: int,
    step_fine: int = 1,
) -> int:
    """Refine a binary-search result by scanning nearby positions for the target value.

    First performs a coarse sweep over [start+step_coarse, start+delta_coarse),
    then a fine sweep within the last coarse step.
    """
    log_msg("Starting coarse trackback search")
    found_more = True
    while found_more:
        found_more = False
        for i in range(
            start_search + step_coarse, start_search + delta_coarse, step_coarse
        ):
            if i >= len(wrapper):
                break
            if wrapper[i] == srch_val:
                found_more = True
                start_search = i
                break

    if step_fine > 0:
        if step_fine > step_coarse:
            raise ValueError("step_fine must be <= step_coarse")
        delta_fine = step_coarse
        log_msg("Starting fine trackback search")
        found_more = True
        while found_more:
            found_more = False
            for i in range(
                start_search + step_fine, start_search + delta_fine, step_fine
            ):
                if i >= len(wrapper):
                    break
                if wrapper[i] == srch_val:
                    found_more = True
                    start_search = i
                    break

    return start_search


def custom_binary_search_with_trackback(
    np_init_array: np.ndarray,
    f: Callable[[np.ndarray, int], int],
    srch_val: int,
    trackback_delta: int = 200,
    trackback_step: int = 20,
    init_search_location: int = 1000,
) -> int:
    """Find the filter width that produces exactly *srch_val* minima.

    Uses an exponential search followed by binary search on an inverted
    accessor, then a trackback refinement pass.

    Args:
        np_init_array: The correlation-sum vector.
        f: Function ``f(array, width) -> n_minima``.
        srch_val: Target number of minima.
        trackback_delta: Coarse search range for trackback.
        trackback_step: Coarse step size for trackback.
        init_search_location: Starting width for exponential search.

    Returns:
        Filter width (in samples) that yields *srch_val* minima.
    """
    log_msg("Starting custom_binary_search_with_trackback")

    end_v = _find_end(np_init_array, f, init_search_location, srch_val)
    log_msg(f"Exponential search end: {end_v}")

    wrapper = FlexibleBoundedAccessor(np_init_array, f, 0, end_v, invert=True)
    found_width_raw = find_le_ind(wrapper, srch_val)
    found_width = end_v - found_width_raw
    log_msg(f"Binary search found_width: {found_width}")

    found_width_trackback_raw = _trackback(
        wrapper, srch_val, found_width_raw, trackback_delta, trackback_step
    )
    found_width = end_v - found_width_trackback_raw
    log_msg(f"Trackback found_width: {found_width}")

    return found_width
