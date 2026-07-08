"""Signal processing: Hanning-window convolution and local minima detection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict, TypeVar

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as sig

from ldetect_lite._util.logging import log_msg

_F = TypeVar("_F", bound=Callable[..., Any])

try:
    from numba import njit

    _HAVE_NUMBA = True
    _numba_nogil_decorator = njit(nogil=True, fastmath=True, cache=True)

    def _njit_nogil(fn: _F) -> _F:
        """JIT-compile with the GIL released.

        ``nogil=True`` is required, not just a perf nicety: `find_minima.py`'s
        `_find_end`/`_trackback` run this convolution concurrently from
        multiple Python threads (`ThreadPoolExecutor`), which only overlaps
        real work if the GIL is actually released during the call.

        ``fastmath=True`` is also required for a *speed win at all*: without
        it, LLVM does not auto-vectorize `_convolve1d_reflect`'s reduction
        loop and the compiled kernel is ~2x *slower* than
        `scipy.ndimage.convolve1d` (measured); with it, ~2x *faster*. This
        permits floating-point reassociation within the compiled loop, but
        that reassociation is fixed at compile time and applied identically
        at every output position -- it does not compromise the flat-region
        safety property (verified: a constant input run still produces
        bit-identical output at every position), unlike the FFT-based
        convolution that was tried and reverted (docs/optimizations.md #11).
        """
        return _numba_nogil_decorator(fn)  # type: ignore[no-any-return]
except ImportError:
    _HAVE_NUMBA = False

    def _njit_nogil(fn: _F) -> _F:
        return fn


@_njit_nogil
def _reflect_index(i: int, n: int) -> int:
    """Map an arbitrary (possibly out-of-range) index to scipy.ndimage's
    default `mode='reflect'` boundary (== `numpy.pad(..., mode="symmetric")`:
    edge value repeated, `d c b a | a b c d | d c b a`). Handles `i` outside
    `[0, n)` in either direction, including `abs(i) >= n`, by cycling --
    real filter widths here (thousands) can exceed the vector length.
    """
    period = 2 * n
    m = i % period
    if m < 0:
        m += period
    return m if m < n else period - 1 - m


@_njit_nogil
def _pad_reflect(arr: np.ndarray, width: int) -> np.ndarray:
    """Equivalent to `numpy.pad(arr, width, mode="symmetric")`, numba-jittable
    (`numpy.pad` itself is not supported in numba nopython mode)."""
    n = arr.shape[0]
    out = np.empty(n + 2 * width, dtype=np.float64)
    for k in range(out.shape[0]):
        out[k] = arr[_reflect_index(k - width, n)]
    return out


@_njit_nogil
def _convolve1d_reflect(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Direct convolution matching `scipy.ndimage.convolve1d` with its
    default `mode='reflect'`, `origin=0`. `kernel` must be odd-length; the
    Hanning window used by `apply_filter` is symmetric, so convolution and
    correlation coincide here -- do not reuse this with an asymmetric kernel
    without adding the flip back.

    Same shift-invariant summation structure at every output position as a
    direct convolution (same kernel, same loop, applied uniformly per `i`) --
    this is what makes it flat-region-safe: a constant input stretch produces
    bit-identical output at every position, so `argrelextrema`'s strict `<`
    never fires spuriously there. An FFT-based convolution does not have this
    property (see docs/optimizations.md #11) and was reverted for exactly
    this reason.
    """
    n = arr.shape[0]
    klen = kernel.shape[0]
    width = klen // 2
    padded = _pad_reflect(arr, width)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for k in range(klen):
            s += padded[i + k] * kernel[k]
        out[i] = s
    return out


class FilterResult(TypedDict):
    """Output of :func:`apply_filter`."""

    width: int
    window: np.ndarray
    filtered: np.ndarray
    filtered_minima_ind: np.ndarray
    filtered_minima_vals: list[float]


def apply_filter(np_init_array: np.ndarray, width: int) -> FilterResult:
    """Apply a Hanning-window low-pass filter and find local minima.

    Args:
        np_init_array: 1-D array of correlation sums (the "vector").
        width: Half-width of the Hanning window; full window is 2*width+1.

    Returns:
        Dict with keys: width, window, filtered, filtered_minima_ind,
        filtered_minima_vals.
    """
    window = np.hanning(2 * width + 1)
    kernel = window / window.sum()
    if _HAVE_NUMBA:
        arr = np.ascontiguousarray(np_init_array, dtype=np.float64)
        smoothed = _convolve1d_reflect(arr, kernel)
    else:
        # Numba is a hard dependency (pyproject.toml), so this path should be
        # unreachable in a correctly-installed environment. But falling back
        # to an un-jitted `_convolve1d_reflect` here would be catastrophic
        # (a pure-Python O(N*width) triple-nested loop, ~10^8 iterations at
        # production widths) rather than just slower -- fall back to the
        # original scipy implementation instead, which is merely non-optimal.
        smoothed = ndimage.convolve1d(np_init_array, kernel)

    minima_ind = sig.argrelextrema(smoothed, np.less)[0]
    minima_vals = [smoothed[i] for i in minima_ind]

    log_msg(f"Filter width={2 * width + 1}, minima count={len(minima_ind)}")

    return {
        "width": width,
        "window": window,
        "filtered": smoothed,
        "filtered_minima_ind": minima_ind,
        "filtered_minima_vals": minima_vals,
    }


def apply_filter_get_minima(np_init_array: np.ndarray, width: int) -> int:
    """Return the number of local minima for a given filter width."""
    return len(apply_filter(np_init_array, width)["filtered_minima_ind"])


def apply_filter_get_minima_ind(np_init_array: np.ndarray, width: int) -> np.ndarray:
    """Return the indices of local minima for a given filter width."""
    return apply_filter(np_init_array, width)["filtered_minima_ind"]


def get_minima_loc(g: FilterResult, np_init_array_x: np.ndarray) -> list[int]:
    """Convert minima indices to genomic positions.

    Args:
        g: Output dict from :func:`apply_filter`.
        np_init_array_x: Array of genomic positions parallel to the value array.

    Returns:
        List of genomic positions at the minima.
    """
    return [int(np_init_array_x[idx]) for idx in g["filtered_minima_ind"]]


def apply_filters(
    np_init_array: np.ndarray,
    first: int,
    last: int,
    step: int,
) -> list[FilterResult]:
    """Apply filters at a range of widths and return all results."""
    return [apply_filter(np_init_array, w) for w in range(first, last + 1, step)]
