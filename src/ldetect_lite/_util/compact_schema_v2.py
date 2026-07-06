"""Prototype: lo-less, rank-encoded compact covariance partition schema (v2).

Priority #1 from ``notes/logs/covariance-cache-redesign-plan.md``: the
existing compact schema (``io/covariance_hdf5.py``) stores a full per-row
``lo`` array that is already redundant given the CSR-style
``index/lo_values``/``index/lo_offsets`` index it also writes, and stores
``hi`` as a full genomic position when a rank into this partition's own
position set reconstructs it exactly. This module is a **prototype-only**
writer/reader pair for a v2 layout that drops ``lo`` entirely and stores
``hi`` as a compact rank index. It is not wired into ``calc_covariance``,
the CLI, or any existing reader -- v1 (``io/covariance_hdf5.py``) remains
the only production schema.

v2 layout::

    index/positions:      int32/int64[n_positions]   sorted unique(lo ∪ hi)
    index/lo_rank_values: uint16/uint32[n_lo_groups]  rank-in-positions of
                                                       each distinct lo with
                                                       >=1 row, ascending
    index/lo_offsets:     int64[n_lo_groups + 1]      CSR row pointer, same
                                                       concept as v1
    covariance/hi_idx:    uint16/uint32[n_rows]       rank-in-positions of
                                                       hi, per row
    covariance/shrink_ld: float64[n_rows]             unchanged
    index/diag_pos:       int32/int64[n_diag]         unchanged
    index/diag_val:       float64[n_diag]             unchanged

Per-row payload drops from ``lo(4B) + hi(4B) + shrink_ld(8B) = 16B`` to
``hi_idx(2B) + shrink_ld(8B) = 10B`` before compression (uint32 fallback if
a partition has more than 65,535 distinct positions). ``positions``,
``lo_rank_values``, and ``lo_offsets`` are all O(n_snps), not O(n_rows), so
they're cheap regardless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ldetect_lite.io.covariance_hdf5 import (
    CovarianceRowChunk,
    _dataset_compression_kwargs,
    _h5py,
    _position_array,
)

_FORMAT = "ldetect-lite-covariance-v2-h5"
_VERSION = 2


def _rank_dtype(n_positions: int) -> np.dtype:
    return np.dtype(np.uint16) if n_positions <= 0xFFFF else np.dtype(np.uint32)


def write_v2_partition(
    path: Path,
    *,
    lo: np.ndarray,
    hi: np.ndarray,
    shrink_ld: np.ndarray,
    diag_pos: np.ndarray,
    diag_val: np.ndarray,
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
    compression: str | None = "zstd",
) -> None:
    """Write a v2 (lo-less, rank-encoded) compact covariance partition.

    ``lo``, ``hi``, ``shrink_ld`` must already be canonical rows sorted by
    ``(lo, hi)`` -- the invariant ``calc_covariance``'s kernel already
    provides and ``write_compact_covariance_partition_hdf5_chunks`` already
    trusts. This prototype does not re-sort/dedup.
    """
    lo = _position_array(lo)
    hi = _position_array(hi)
    shrink = np.asarray(shrink_ld, dtype=np.float64)
    if lo.shape != hi.shape or lo.shape != shrink.shape:
        raise ValueError("lo/hi/shrink_ld must have identical shapes")

    positions = np.union1d(lo, hi) if lo.size else np.array([], dtype=lo.dtype)
    rank_dtype = _rank_dtype(positions.size)
    hi_idx = np.searchsorted(positions, hi).astype(rank_dtype, copy=False)

    if lo.size == 0:
        lo_rank_values = np.array([], dtype=rank_dtype)
        lo_offsets = np.array([0], dtype=np.int64)
    else:
        starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(lo[1:] != lo[:-1]) + 1)
        )
        distinct_lo = lo[starts]
        lo_rank_values = np.searchsorted(positions, distinct_lo).astype(
            rank_dtype, copy=False
        )
        lo_offsets = np.concatenate(
            (starts, np.array([lo.size], dtype=np.int64))
        ).astype(np.int64, copy=False)

    diag_pos = _position_array(diag_pos)
    diag_val = np.asarray(diag_val, dtype=np.float64)

    h5py = _h5py()
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {**_dataset_compression_kwargs(compression), "shuffle": True}
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = _FORMAT
        h5.attrs["version"] = _VERSION
        if chrom is not None:
            h5.attrs["chrom"] = chrom
        if start is not None:
            h5.attrs["start"] = int(start)
        if end is not None:
            h5.attrs["end"] = int(end)
        h5.attrs["position_dtype"] = str(positions.dtype)
        h5.attrs["rank_dtype"] = str(rank_dtype)

        idx = h5.create_group("index")
        idx.create_dataset("positions", data=positions, **kwargs)
        idx.create_dataset("lo_rank_values", data=lo_rank_values, **kwargs)
        idx.create_dataset("lo_offsets", data=lo_offsets, **kwargs)
        idx.create_dataset("diag_pos", data=diag_pos, **kwargs)
        idx.create_dataset("diag_val", data=diag_val, **kwargs)

        cov = h5.create_group("covariance")
        cov.create_dataset("hi_idx", data=hi_idx, **kwargs)
        cov.create_dataset("shrink_ld", data=shrink, **kwargs)


def read_v2_partition(path: Path) -> CovarianceRowChunk:
    """Read a v2 partition, fully reconstructing ``lo``/``hi``/``shrink_ld``."""
    h5py = _h5py()
    with h5py.File(path, "r") as h5:
        positions = np.asarray(h5["index/positions"][:])
        lo_rank_values = np.asarray(h5["index/lo_rank_values"][:], dtype=np.int64)
        lo_offsets = np.asarray(h5["index/lo_offsets"][:])
        hi_idx = np.asarray(h5["covariance/hi_idx"][:], dtype=np.int64)
        shrink_ld = np.asarray(h5["covariance/shrink_ld"][:], dtype=np.float64)

    row_counts = np.diff(lo_offsets)
    lo = np.repeat(positions[lo_rank_values], row_counts)
    hi = positions[hi_idx]
    return CovarianceRowChunk(lo=lo, hi=hi, shrink_ld=shrink_ld)


def read_v2_diagonal(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read canonical diagonal positions and values from a v2 partition."""
    h5py = _h5py()
    with h5py.File(path, "r") as h5:
        diag_pos = _position_array(h5["index/diag_pos"][:])
        diag_val = np.asarray(h5["index/diag_val"][:], dtype=np.float64)
    return diag_pos, diag_val


def read_v2_index_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the raw index/covariance arrays without reconstructing lo/hi.

    Returns ``(positions, lo_rank_values, lo_offsets, hi_idx, shrink_ld)`` --
    the native v2 representation, used directly by
    ``banded_metric_coverage.py`` so it never has to reconstruct a full
    per-row ``lo`` array just to compute a crossing sum.
    """
    h5py = _h5py()
    with h5py.File(path, "r") as h5:
        positions = np.asarray(h5["index/positions"][:])
        lo_rank_values = np.asarray(h5["index/lo_rank_values"][:], dtype=np.int64)
        lo_offsets = np.asarray(h5["index/lo_offsets"][:])
        hi_idx = np.asarray(h5["covariance/hi_idx"][:], dtype=np.int64)
        shrink_ld = np.asarray(h5["covariance/shrink_ld"][:], dtype=np.float64)
    return positions, lo_rank_values, lo_offsets, hi_idx, shrink_ld


def convert_v1_to_v2(
    v1_path: Path,
    v2_path: Path,
    *,
    start: int,
    end: int,
    chrom: str | None = None,
    compression: str | None = "zstd",
) -> None:
    """Re-encode an existing v1 compact partition as v2, for benchmarking."""
    from ldetect_lite.io.covariance_hdf5 import open_covariance_reader

    with open_covariance_reader(v1_path, start, end) as reader:
        rows = reader.read_all()
        diag_pos, diag_val = reader.read_diagonal()
    write_v2_partition(
        v2_path,
        lo=rows.lo,
        hi=rows.hi,
        shrink_ld=rows.shrink_ld,
        diag_pos=diag_pos,
        diag_val=diag_val,
        chrom=chrom,
        start=start,
        end=end,
        compression=compression,
    )
