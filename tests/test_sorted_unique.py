"""Parity tests for the sort-free unique helpers against ``np.unique``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ldetect_lite._util.sorted_unique import sorted_unique_counts, unique_loci
from ldetect_lite.io.covariance_hdf5 import (
    CovarianceRowChunk,
    write_compact_covariance_partition_hdf5_append,
)


def _assert_same_array(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def _canonical_rows(
    rng: np.random.Generator, n_positions: int, max_window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted (lo, hi) rows like a compact partition, diagonal included."""
    positions = np.sort(rng.choice(10_000_000, size=n_positions, replace=False))
    lo_parts = []
    hi_parts = []
    for i in range(n_positions):
        stop = min(n_positions, i + 1 + int(rng.integers(0, max_window)))
        js = np.arange(i, stop)
        js = js[(js == i) | (rng.random(js.size) < 0.6)]
        lo_parts.append(np.full(js.size, positions[i]))
        hi_parts.append(positions[js])
    return (
        positions.astype(np.int32),
        np.concatenate(lo_parts).astype(np.int32),
        np.concatenate(hi_parts).astype(np.int32),
    )


@pytest.mark.parametrize(
    "values",
    [
        np.array([], dtype=np.int32),
        np.array([7], dtype=np.int64),
        np.array([1, 1, 1], dtype=np.int32),
        np.array([1, 2, 2, 5, 5, 5, 9], dtype=np.int32),
        np.array([5, 1, 5, 3, 1], dtype=np.int64),  # unsorted -> fallback
    ],
)
def test_sorted_unique_counts_matches_np_unique(values: np.ndarray) -> None:
    got_values, got_counts = sorted_unique_counts(values)
    exp_values, exp_counts = np.unique(values, return_counts=True)
    _assert_same_array(got_values, exp_values)
    np.testing.assert_array_equal(got_counts, exp_counts)


@pytest.mark.parametrize(
    ("lo", "hi"),
    [
        (np.array([], dtype=np.int32), np.array([], dtype=np.int32)),
        (np.array([], dtype=np.int32), np.array([4, 2], dtype=np.int32)),
        (np.array([3], dtype=np.int32), np.array([], dtype=np.int32)),
        # every hi already a lo (the normal compact-partition case)
        (
            np.array([1, 1, 2, 3], dtype=np.int32),
            np.array([1, 3, 2, 3], dtype=np.int32),
        ),
        # hi values absent from lo: below, between, and above lo's range
        (np.array([5, 5, 8], dtype=np.int32), np.array([2, 6, 8, 20], dtype=np.int32)),
        # mixed dtypes promote like np.concatenate
        (np.array([1, 2, 2], dtype=np.int32), np.array([2, 9], dtype=np.int64)),
        # unsorted lo -> fallback
        (np.array([4, 1, 4], dtype=np.int64), np.array([4, 7, 1], dtype=np.int64)),
    ],
)
def test_unique_loci_matches_np_unique(lo: np.ndarray, hi: np.ndarray) -> None:
    _assert_same_array(unique_loci(lo, hi), np.unique(np.concatenate((lo, hi))))
    _assert_same_array(unique_loci(lo), np.unique(lo))


def test_unique_loci_matches_np_unique_on_random_partitions() -> None:
    rng = np.random.default_rng(20261104)
    for _ in range(25):
        _, lo, hi = _canonical_rows(rng, int(rng.integers(1, 300)), 40)
        _assert_same_array(unique_loci(lo, hi), np.unique(np.concatenate((lo, hi))))
        # drop the diagonal so some hi endpoints are missing from lo
        offdiag = lo != hi
        lo_od, hi_od = lo[offdiag], hi[offdiag]
        _assert_same_array(
            unique_loci(lo_od, hi_od), np.unique(np.concatenate((lo_od, hi_od)))
        )
        values, counts = sorted_unique_counts(lo)
        exp_values, exp_counts = np.unique(lo, return_counts=True)
        _assert_same_array(values, exp_values)
        np.testing.assert_array_equal(counts, exp_counts)


def test_append_writer_lo_index_unchanged_across_chunk_splits(tmp_path: Path) -> None:
    """lo_values/lo_offsets must match an np.unique-derived index, including
    when one lo value's run is split across streamed chunks."""
    import h5py

    rng = np.random.default_rng(7)
    positions, lo, hi = _canonical_rows(rng, 400, 30)
    shrink = rng.random(lo.size)
    cuts = np.sort(rng.choice(np.arange(1, lo.size), size=12, replace=False))
    bounds = np.concatenate(([0], cuts, [lo.size]))
    chunks = [
        CovarianceRowChunk(lo=lo[a:b], hi=hi[a:b], shrink_ld=shrink[a:b])
        for a, b in zip(bounds[:-1], bounds[1:])
    ]
    path = tmp_path / "split_chunks.h5"
    n_rows = write_compact_covariance_partition_hdf5_append(
        path, positions=positions, row_chunks=iter(chunks), compression="zstd"
    )
    assert n_rows == lo.size

    exp_values, exp_counts = np.unique(lo, return_counts=True)
    exp_offsets = np.concatenate(([0], np.cumsum(exp_counts)))
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(h5["index/lo_values"][:], exp_values)
        np.testing.assert_array_equal(h5["index/lo_offsets"][:], exp_offsets)
        np.testing.assert_array_equal(h5["covariance/lo"][:], lo)
        np.testing.assert_array_equal(h5["covariance/hi"][:], hi)
