"""Binary search utilities operating on sorted lists."""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from _typeshed import SupportsLenAndGetItem, SupportsRichComparison

T = TypeVar("T", bound="SupportsRichComparison")


def index(a: SupportsLenAndGetItem[T], x: T) -> int:
    """Return index of the leftmost value exactly equal to x."""
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError(f"index: {x!r} not found")


def find_lt_ind(a: SupportsLenAndGetItem[T], x: T) -> int:
    """Return index of the rightmost value less than x."""
    i = bisect.bisect_left(a, x)
    if i:
        return i - 1
    raise ValueError(f"find_lt: no value less than {x!r}")


def find_lt(a: SupportsLenAndGetItem[T], x: T) -> T:
    return a[find_lt_ind(a, x)]


def find_le_ind(a: SupportsLenAndGetItem[T], x: T) -> int:
    """Return index of the rightmost value less than or equal to x."""
    i = bisect.bisect_right(a, x)
    if i:
        return i - 1
    raise ValueError(f"find_le: no value <= {x!r}")


def find_le(a: SupportsLenAndGetItem[T], x: T) -> T:
    return a[find_le_ind(a, x)]


def find_gt_ind(a: SupportsLenAndGetItem[T], x: T) -> int:
    """Return index of the leftmost value greater than x."""
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return i
    raise ValueError(f"find_gt: no value greater than {x!r}")


def find_gt(a: SupportsLenAndGetItem[T], x: T) -> T:
    return a[find_gt_ind(a, x)]


def find_ge_ind(a: SupportsLenAndGetItem[T], x: T) -> int:
    """Return index of the leftmost value greater than or equal to x."""
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError(f"find_ge: no value >= {x!r}")


def find_ge(a: SupportsLenAndGetItem[T], x: T) -> T:
    return a[find_ge_ind(a, x)]
