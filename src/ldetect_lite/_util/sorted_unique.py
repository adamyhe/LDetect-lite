"""Sort-free unique/count helpers for covariance rows already sorted by ``lo``.

Compact covariance partitions store rows sorted by ``(lo, hi)``, so the
distinct ``lo`` values and their counts are run boundaries -- an O(n) pass --
rather than something ``np.unique`` has to sort for (O(n log n), and
measurably the largest per-row cost in the covariance writer). Both helpers
return exactly what the ``np.unique`` call they replace would, and fall back
to ``np.unique`` when the sortedness precondition does not hold (e.g. legacy
full-schema rows), so callers need no schema-specific handling.
"""

from __future__ import annotations

import numpy as np


def _is_non_decreasing(values: np.ndarray) -> bool:
    return values.size <= 1 or bool(np.all(values[1:] >= values[:-1]))


def _run_starts(values: np.ndarray) -> np.ndarray:
    """Start indexes of each run of equal values in a non-decreasing array."""
    keep = np.empty(values.size, dtype=bool)
    keep[0] = True
    np.not_equal(values[1:], values[:-1], out=keep[1:])
    return np.flatnonzero(keep)


def sorted_unique_counts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equivalent to ``np.unique(values, return_counts=True)``.

    O(n) when ``values`` is non-decreasing; falls back to ``np.unique``
    otherwise.
    """
    values = np.asarray(values)
    if values.size == 0 or not _is_non_decreasing(values):
        unique_values, counts = np.unique(values, return_counts=True)
        return unique_values, counts
    starts = _run_starts(values)
    counts = np.diff(starts, append=values.size)
    return values[starts], counts


def unique_loci(lo: np.ndarray, hi: np.ndarray | None = None) -> np.ndarray:
    """Equivalent to ``np.unique(np.concatenate((lo, hi)))`` (or ``np.unique(lo)``).

    When ``lo`` is non-decreasing, its distinct values come from run
    boundaries, and ``hi`` only needs a membership check against them: in
    compact covariance every locus with any row has its own diagonal row, so
    ``hi`` endpoints are normally already present and the union is free.
    Any ``hi`` values that are not present are merged in, so the result is
    exact regardless. Falls back to ``np.unique`` when ``lo`` is unsorted.
    """
    lo = np.asarray(lo)
    if hi is None:
        if lo.size == 0 or not _is_non_decreasing(lo):
            result: np.ndarray = np.unique(lo)
            return result
        runs: np.ndarray = lo[_run_starts(lo)]
        return runs

    hi = np.asarray(hi)
    dtype = np.result_type(lo, hi)
    if lo.size == 0 or not _is_non_decreasing(lo):
        merged: np.ndarray = np.unique(np.concatenate((lo, hi)))
        return merged
    base: np.ndarray = lo[_run_starts(lo)].astype(dtype, copy=False)
    if hi.size == 0:
        return base
    idx = np.searchsorted(base, hi)
    np.minimum(idx, base.size - 1, out=idx)
    missing = base[idx] != hi
    if not np.any(missing):
        return base
    union: np.ndarray = np.union1d(base, hi[missing].astype(dtype, copy=False))
    return union
