"""Filter-width search: find the Hanning window width that yields exactly N minima."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

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
    max_workers: int = 1,
) -> int:
    """Exponential search upward for the smallest x where f(data, x) < val.

    This phase is intentionally sequential even when the surrounding search
    uses workers. Later doubling candidates can be much more expensive than
    earlier ones, so batching them can waste minutes evaluating huge filters
    after an earlier candidate already satisfies the stopping rule.
    """
    if x <= 0:
        raise ValueError("x must be > 0")
    while True:
        if x >= max_srch_val:
            raise ValueError(f"Max search value {max_srch_val} exceeded")
        if f(data, x) < val:
            return x
        x *= 2


def _trackback(
    wrapper: FlexibleBoundedAccessor,
    srch_val: int,
    start_search: int,
    delta_coarse: int,
    step_coarse: int,
    step_fine: int = 1,
    max_workers: int = 1,
) -> int:
    """Refine a binary-search result by scanning nearby positions for the target value.

    First performs a coarse sweep over [start+step_coarse, start+delta_coarse),
    then a fine sweep within the last coarse step.

    Each sweep repeatedly scans a window of candidate positions and jumps to
    the first (nearest) match, restarting from there, until a full window
    scan finds nothing further. With ``max_workers > 1``, candidates within
    one window are evaluated concurrently in chunks of ``max_workers`` (each
    call is an independent, side-effect-free ``wrapper[i]`` lookup, and the
    underlying scipy calls release the GIL, so threads give real speedup);
    the first-match decision is still applied sequentially over the chunk in
    the original left-to-right order, so the final result is identical to
    the ``max_workers=1`` sequential scan -- only the wall-clock changes.
    """

    def _sweep(
        delta: int, step: int, start: int, pool: ThreadPoolExecutor | None
    ) -> int:
        found_more = True
        while found_more:
            found_more = False
            candidates = []
            for i in range(start + step, start + delta, step):
                if i >= len(wrapper):
                    break
                candidates.append(i)
            if pool is None:
                for i in candidates:
                    if wrapper[i] == srch_val:
                        found_more = True
                        start = i
                        break
            else:
                for chunk_start in range(0, len(candidates), max_workers):
                    chunk = candidates[chunk_start : chunk_start + max_workers]
                    values = list(pool.map(lambda i: wrapper[i], chunk))
                    for i, value in zip(chunk, values):
                        if value == srch_val:
                            found_more = True
                            start = i
                            break
                    if found_more:
                        break
        return start

    pool = ThreadPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
    try:
        log_msg("Starting coarse trackback search")
        start_search = _sweep(delta_coarse, step_coarse, start_search, pool)

        if step_fine > 0:
            if step_fine > step_coarse:
                raise ValueError("step_fine must be <= step_coarse")
            delta_fine = step_coarse
            log_msg("Starting fine trackback search")
            start_search = _sweep(delta_fine, step_fine, start_search, pool)
    finally:
        if pool is not None:
            pool.shutdown()

    return start_search


def custom_binary_search_with_trackback(
    np_init_array: np.ndarray,
    f: Callable[[np.ndarray, int], int],
    srch_val: int,
    trackback_delta: int = 200,
    trackback_step: int = 20,
    init_search_location: int = 1000,
    search_workers: int = 1,
    trackback_f: Callable[[np.ndarray, int], int] | None = None,
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
        search_workers: Threads for the trackback refinement pass (default:
            1, sequential). Trackback scans a predictable/boundable set of
            candidates per round and applies the same decision rule to
            concurrently-computed results, so the returned width is identical
            regardless of this value. The exponential and binary-search phases
            stay single-candidate: each step is adaptive on the previous
            comparison, so they cannot be pre-batched the same way.
        trackback_f: Optional alternate function for trackback candidate
            evaluations. This lets callers use within-convolution threading
            for adaptive single-candidate phases while forcing each concurrent
            trackback candidate to use a serial convolution, avoiding nested
            parallelism.

    Returns:
        Filter width (in samples) that yields *srch_val* minima.
    """
    log_msg("Starting custom_binary_search_with_trackback")

    end_v = _find_end(
        np_init_array, f, init_search_location, srch_val, max_workers=search_workers
    )
    log_msg(f"Exponential search end: {end_v}")

    binary_wrapper = FlexibleBoundedAccessor(np_init_array, f, 0, end_v, invert=True)
    found_width_raw = find_le_ind(binary_wrapper, srch_val)
    found_width = end_v - found_width_raw
    log_msg(f"Binary search found_width: {found_width}")

    trackback_wrapper = FlexibleBoundedAccessor(
        np_init_array, trackback_f or f, 0, end_v, invert=True
    )
    found_width_trackback_raw = _trackback(
        trackback_wrapper,
        srch_val,
        found_width_raw,
        trackback_delta,
        trackback_step,
        max_workers=search_workers,
    )
    found_width = end_v - found_width_trackback_raw
    log_msg(f"Trackback found_width: {found_width}")

    return found_width
