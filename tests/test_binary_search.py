"""Tests for ldetect2._util.binary_search."""

from __future__ import annotations

import pytest

from ldetect2._util.binary_search import (
    find_ge,
    find_ge_ind,
    find_gt,
    find_gt_ind,
    find_le,
    find_le_ind,
    find_lt,
    find_lt_ind,
    index,
)

A = [10, 20, 30, 40, 50]


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def test_index_found():
    assert index(A, 30) == 2


def test_index_first():
    assert index(A, 10) == 0


def test_index_last():
    assert index(A, 50) == 4


def test_index_not_found():
    with pytest.raises(ValueError):
        index(A, 25)


def test_index_empty():
    with pytest.raises(ValueError):
        index([], 1)


# ---------------------------------------------------------------------------
# find_lt / find_lt_ind
# ---------------------------------------------------------------------------

def test_find_lt_middle():
    assert find_lt(A, 35) == 30
    assert find_lt_ind(A, 35) == 2


def test_find_lt_exact_match():
    # find_lt(A, 30) → rightmost < 30 → 20
    assert find_lt(A, 30) == 20
    assert find_lt_ind(A, 30) == 1


def test_find_lt_last():
    assert find_lt(A, 50) == 40
    assert find_lt_ind(A, 50) == 3


def test_find_lt_no_value():
    with pytest.raises(ValueError):
        find_lt(A, 10)


def test_find_lt_empty():
    with pytest.raises(ValueError):
        find_lt([], 5)


# ---------------------------------------------------------------------------
# find_le / find_le_ind
# ---------------------------------------------------------------------------

def test_find_le_middle():
    assert find_le(A, 35) == 30
    assert find_le_ind(A, 35) == 2


def test_find_le_exact():
    assert find_le(A, 30) == 30
    assert find_le_ind(A, 30) == 2


def test_find_le_last():
    assert find_le(A, 50) == 50
    assert find_le_ind(A, 50) == 4


def test_find_le_beyond():
    assert find_le(A, 100) == 50
    assert find_le_ind(A, 100) == 4


def test_find_le_no_value():
    with pytest.raises(ValueError):
        find_le(A, 5)


# ---------------------------------------------------------------------------
# find_gt / find_gt_ind
# ---------------------------------------------------------------------------

def test_find_gt_middle():
    assert find_gt(A, 25) == 30
    assert find_gt_ind(A, 25) == 2


def test_find_gt_exact_match():
    # find_gt(A, 30) → leftmost > 30 → 40
    assert find_gt(A, 30) == 40
    assert find_gt_ind(A, 30) == 3


def test_find_gt_first():
    assert find_gt(A, 10) == 20
    assert find_gt_ind(A, 10) == 1


def test_find_gt_no_value():
    with pytest.raises(ValueError):
        find_gt(A, 50)


def test_find_gt_empty():
    with pytest.raises(ValueError):
        find_gt([], 5)


# ---------------------------------------------------------------------------
# find_ge / find_ge_ind
# ---------------------------------------------------------------------------

def test_find_ge_middle():
    assert find_ge(A, 25) == 30
    assert find_ge_ind(A, 25) == 2


def test_find_ge_exact():
    assert find_ge(A, 30) == 30
    assert find_ge_ind(A, 30) == 2


def test_find_ge_first():
    assert find_ge(A, 10) == 10
    assert find_ge_ind(A, 10) == 0


def test_find_ge_before():
    assert find_ge(A, 5) == 10
    assert find_ge_ind(A, 5) == 0


def test_find_ge_no_value():
    with pytest.raises(ValueError):
        find_ge(A, 55)


# ---------------------------------------------------------------------------
# Single-element list
# ---------------------------------------------------------------------------

def test_single_element():
    a = [7]
    assert index(a, 7) == 0
    with pytest.raises(ValueError):
        find_lt(a, 7)
    assert find_le(a, 7) == 7
    with pytest.raises(ValueError):
        find_gt(a, 7)
    assert find_ge(a, 7) == 7
