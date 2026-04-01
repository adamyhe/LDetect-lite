"""Binary search utilities operating on sorted lists."""

import bisect
from typing import TypeVar

T = TypeVar("T")


def index(a: list[T], x: T) -> int:
    """Return index of the leftmost value exactly equal to x."""
    i = bisect.bisect_left(a, x)  # type: ignore[arg-type]
    if i != len(a) and a[i] == x:
        return i
    raise ValueError(f"index: {x!r} not found")


def find_lt_ind(a: list[T], x: T) -> int:
    """Return index of the rightmost value less than x."""
    i = bisect.bisect_left(a, x)  # type: ignore[arg-type]
    if i:
        return i - 1
    raise ValueError(f"find_lt: no value less than {x!r}")


def find_lt(a: list[T], x: T) -> T:
    return a[find_lt_ind(a, x)]


def find_le_ind(a: list[T], x: T) -> int:
    """Return index of the rightmost value less than or equal to x."""
    i = bisect.bisect_right(a, x)  # type: ignore[arg-type]
    if i:
        return i - 1
    raise ValueError(f"find_le: no value <= {x!r}")


def find_le(a: list[T], x: T) -> T:
    return a[find_le_ind(a, x)]


def find_gt_ind(a: list[T], x: T) -> int:
    """Return index of the leftmost value greater than x."""
    i = bisect.bisect_right(a, x)  # type: ignore[arg-type]
    if i != len(a):
        return i
    raise ValueError(f"find_gt: no value greater than {x!r}")


def find_gt(a: list[T], x: T) -> T:
    return a[find_gt_ind(a, x)]


def find_ge_ind(a: list[T], x: T) -> int:
    """Return index of the leftmost value greater than or equal to x."""
    i = bisect.bisect_left(a, x)  # type: ignore[arg-type]
    if i != len(a):
        return i
    raise ValueError(f"find_ge: no value >= {x!r}")


def find_ge(a: list[T], x: T) -> T:
    return a[find_ge_ind(a, x)]
