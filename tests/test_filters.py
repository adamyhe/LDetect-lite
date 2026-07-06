"""Tests for ldetect_lite.filters."""

from __future__ import annotations

import numpy as np

from ldetect_lite.filters import (
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
    # Wider filter → more smoothing → fewer or equal minima
    # Use a signal with many sharp valleys (periodic step-like pattern)
    arr_many = np.array([float(3 + 2 * np.sin(i * np.pi / 5)) for i in range(200)])
    n_narrow = apply_filter_get_minima(arr_many, width=2)
    n_wide = apply_filter_get_minima(arr_many, width=50)
    assert n_wide <= n_narrow
