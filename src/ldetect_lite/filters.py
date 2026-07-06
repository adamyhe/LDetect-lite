"""Signal processing: Hanning-window convolution and local minima detection."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as sig

from ldetect_lite._util.logging import log_msg


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
    smoothed = ndimage.convolve1d(np_init_array, window / window.sum())

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
