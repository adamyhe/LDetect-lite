"""Tests for ldetect_lite.filters."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import scipy.ndimage as ndimage
import scipy.signal as sig

from ldetect_lite.filters import (
    _convolve1d_reflect,
    _pad_reflect,
    apply_filter,
    apply_filter_get_minima,
    apply_filter_get_minima_ind,
    apply_filters,
    get_minima_loc,
)

# ---------------------------------------------------------------------------
# Synthetic signals
# ---------------------------------------------------------------------------

# V-shaped valley at index 50; flat high plateau elsewhere.
# np.hanning uses a symmetric window with zero endpoints, so we need a sharp
# valley (not a smooth cosine) to guarantee a strict local minimum after
# convolution.
_ARR = np.ones(200) * 5.0
_ARR[40:61] = 1.0 + np.abs(np.arange(21) - 10) * 0.2  # valley minimum at index 50

# Two V-shaped valleys at indices 50 and 150 (for bimodal tests)
_ARR2 = np.ones(200) * 5.0
_ARR2[40:61] = 1.0 + np.abs(np.arange(21) - 10) * 0.2
_ARR2[140:161] = 1.0 + np.abs(np.arange(21) - 10) * 0.2


def test_apply_filter_returns_keys():
    result = apply_filter(_ARR, width=5)
    assert "width" in result
    assert "window" in result
    assert "filtered" in result
    assert "filtered_minima_ind" in result
    assert "filtered_minima_vals" in result


def test_apply_filter_width_stored():
    result = apply_filter(_ARR, width=5)
    assert result["width"] == 5


def test_apply_filter_window_size():
    width = 7
    result = apply_filter(_ARR, width=width)
    assert len(result["window"]) == 2 * width + 1


def test_apply_filter_filtered_length():
    result = apply_filter(_ARR, width=5)
    assert len(result["filtered"]) == len(_ARR)


def test_unimodal_single_minimum():
    # V-shape with one interior valley → exactly one minimum after filtering
    minima_ind = apply_filter_get_minima_ind(_ARR, width=5)
    assert len(minima_ind) == 1


def test_unimodal_minimum_near_center():
    minima_ind = apply_filter_get_minima_ind(_ARR, width=5)
    assert 40 <= int(minima_ind[0]) <= 60


def test_apply_filter_get_minima_count():
    n = apply_filter_get_minima(_ARR, width=5)
    assert n == 1


def test_bimodal_two_minima():
    # Two valleys → two minima
    minima_ind = apply_filter_get_minima_ind(_ARR2, width=5)
    assert len(minima_ind) == 2


def test_get_minima_loc():
    x_positions = np.arange(200) * 1000  # positions 0, 1000, ..., 199000
    g = apply_filter(_ARR, width=5)
    locs = get_minima_loc(g, x_positions)
    assert len(locs) == 1
    # Valley at index 50 → position 50000
    assert 40000 <= locs[0] <= 60000


def test_get_minima_loc_integers():
    x_positions = np.arange(200) * 1000
    g = apply_filter(_ARR, width=5)
    locs = get_minima_loc(g, x_positions)
    assert all(isinstance(loc, int) for loc in locs)


def test_apply_filters_range():
    results = apply_filters(_ARR, first=3, last=7, step=2)
    assert len(results) == 3  # widths 3, 5, 7
    for r in results:
        assert "filtered_minima_ind" in r


def test_apply_filter_larger_width_fewer_minima():
    # Wider filter → more smoothing → fewer or equal minima, in the
    # asymptotic (very narrow vs. very wide) sense only -- minima count is
    # *not* monotonic in width for this periodic signal at intermediate
    # widths (e.g. width=50 sits in a locally non-monotonic wiggle, right at
    # a scipy-vs-scipy exact tie with width=2: both give 20). width=95 (kernel
    # width ~2x the array length) gives a decisive, non-fragile margin.
    arr_many = np.array([float(3 + 2 * np.sin(i * np.pi / 5)) for i in range(200)])
    n_narrow = apply_filter_get_minima(arr_many, width=2)
    n_wide = apply_filter_get_minima(arr_many, width=95)
    assert n_wide <= n_narrow


# ---------------------------------------------------------------------------
# numba direct-convolution kernel: equivalence against the pre-numba direct
# convolution (scipy.ndimage.convolve1d). `apply_filter` was migrated from
# that to a hand-written numba kernel for speed, while keeping the same
# O(N*width) shift-invariant summation structure (flat-region-safe by
# construction, unlike the FFT-based convolution that was tried and reverted
# -- see docs/optimizations.md #11). These tests pin that migration to be a
# pure implementation change, not a behavior change.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,width", [(10, 3), (10, 10), (10, 17), (10, 25), (1, 1), (2, 1), (3, 5)]
)
def test_pad_reflect_matches_numpy_symmetric(n, width):
    arr = np.arange(n, dtype=np.float64) + 1.0
    mine = _pad_reflect(arr, width)
    ref = np.pad(arr, width, mode="symmetric")
    np.testing.assert_array_equal(mine, ref)


@pytest.mark.parametrize(
    "n,width", [(500, 5), (500, 50), (500, 500), (500, 1200), (5000, 9000)]
)
def test_convolve1d_reflect_matches_scipy_direct_convolution(n, width):
    arr = np.random.default_rng(0).normal(size=n).cumsum()
    window = np.hanning(2 * width + 1)
    kernel = window / window.sum()
    mine = _convolve1d_reflect(np.ascontiguousarray(arr, dtype=np.float64), kernel)
    ref = ndimage.convolve1d(arr, kernel)
    np.testing.assert_allclose(mine, ref, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("width", [5, 50, 200])
def test_minima_exact_match_scipy_on_flat_plateau_fixtures(width):
    """The exact fixtures that exposed the FFT bug: numba and scipy must
    agree exactly on minima indices, not just approximately."""
    for arr in (_ARR, _ARR2):
        numba_minima = apply_filter_get_minima_ind(arr, width)
        window = np.hanning(2 * width + 1)
        kernel = window / window.sum()
        scipy_smoothed = ndimage.convolve1d(arr, kernel)
        scipy_minima = sig.argrelextrema(scipy_smoothed, np.less)[0]
        np.testing.assert_array_equal(numba_minima, scipy_minima)


def test_minima_exact_match_scipy_on_random_vectors():
    rng = np.random.default_rng(1)
    for _ in range(10):
        n = rng.integers(2000, 20000)
        arr = np.abs(rng.normal(size=n)).cumsum() * rng.uniform(0.01, 1.0)
        width = int(rng.integers(50, min(9000, n // 2)))
        numba_minima = apply_filter_get_minima_ind(arr, width)
        window = np.hanning(2 * width + 1)
        kernel = window / window.sum()
        scipy_smoothed = ndimage.convolve1d(arr, kernel)
        scipy_minima = sig.argrelextrema(scipy_smoothed, np.less)[0]
        np.testing.assert_array_equal(numba_minima, scipy_minima)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_apply_filter_degenerate_width_one_on_tiny_arrays(n):
    arr = np.arange(n, dtype=np.float64) + 1.0
    result = apply_filter(arr, width=1)
    assert len(result["filtered"]) == n


def test_apply_filter_width_zero_is_identity():
    # width=0 -> kernel length 1 -> convolution is a no-op (up to the
    # single-tap Hanning window's own normalization, which is 1.0 for length 1)
    result = apply_filter(_ARR, width=0)
    np.testing.assert_allclose(result["filtered"], _ARR)


def test_convolve1d_reflect_releases_gil():
    """Regression guard: the numba kernel must release the GIL, or the
    existing ThreadPoolExecutor-based search_workers parallelism in
    find_minima.py silently degrades to fully serialized.

    Measured via a background counter thread rather than wall-clock speedup
    (which is flaky on shared/low-core CI): with nogil=True the counter
    advances by orders of magnitude during the call; if nogil were dropped,
    it would advance almost nothing since the GIL-holding call would block
    the counter thread from running at all.
    """
    small = np.ones(10, dtype=np.float64)
    window = np.hanning(3)
    _convolve1d_reflect(small, window / window.sum())  # warm up (compile)

    counter = {"n": 0}
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            counter["n"] += 1

    t = threading.Thread(target=spin)
    t.start()
    time.sleep(0.005)
    baseline = counter["n"]

    big = np.random.default_rng(0).normal(size=200_000).cumsum()
    width = 5000
    window = np.hanning(2 * width + 1)
    kernel = window / window.sum()
    _convolve1d_reflect(np.ascontiguousarray(big, dtype=np.float64), kernel)

    during = counter["n"] - baseline
    stop.set()
    t.join()
    assert during > 1000


def test_apply_filter_falls_back_to_scipy_when_numba_unavailable(monkeypatch):
    """If numba is missing (should be unreachable -- it's a hard dependency,
    pyproject.toml), `apply_filter` must fall back to the original
    `scipy.ndimage.convolve1d` call, not an un-jitted `_convolve1d_reflect`
    (a pure-Python O(N*width) loop -- catastrophically slow at production
    widths, not just non-optimal)."""
    import ldetect_lite.filters as filters_mod

    monkeypatch.setattr(filters_mod, "_HAVE_NUMBA", False)

    width = 5
    window = np.hanning(2 * width + 1)
    kernel = window / window.sum()
    expected = ndimage.convolve1d(_ARR, kernel)

    result = apply_filter(_ARR, width)
    np.testing.assert_array_equal(result["filtered"], expected)
