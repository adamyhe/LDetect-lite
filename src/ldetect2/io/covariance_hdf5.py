"""HDF5 covariance partition storage and chunked readers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_FORMAT = "ldetect2-covariance-h5"
_VERSION = 1
HDF5_DATASET_CHUNK_ROWS = 65_536
_REQUIRED_DATASETS = frozenset(
    {
        "covariance/lo",
        "covariance/hi",
        "covariance/shrink_ld",
        "index/diag_pos",
        "index/diag_val",
        "index/lo_values",
        "index/lo_offsets",
    }
)


@dataclass(frozen=True)
class CovarianceRowChunk:
    """A bounded chunk of canonical covariance rows."""

    lo: np.ndarray
    hi: np.ndarray
    shrink_ld: np.ndarray


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "h5py is required for HDF5 covariance partitions. Install ldetect2 "
            "with its project dependencies before running covariance workflows."
        ) from exc
    return h5py


def _position_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64, copy=False)
    if arr.size == 0:
        return arr.astype(np.int32, copy=False)
    int32_info = np.iinfo(np.int32)
    if int(arr.min()) >= int32_info.min and int(arr.max()) <= int32_info.max:
        return arr.astype(np.int32, copy=False)
    return arr.astype(np.int64, copy=False)


def _canonical_ordered_rows(
    i_pos: np.ndarray,
    j_pos: np.ndarray,
    shrink_ld: np.ndarray,
    *metadata: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Return sorted canonical rows plus metadata with first-pair-wins semantics."""
    i_pos = _position_array(i_pos)
    j_pos = _position_array(j_pos)
    if i_pos.size == 0:
        dtype = np.result_type(i_pos.dtype, j_pos.dtype)
        empty_meta = tuple(np.asarray(values)[:0] for values in metadata)
        return (
            np.array([], dtype=dtype),
            np.array([], dtype=dtype),
            np.array([], dtype=np.float64),
            empty_meta,
        )

    dtype = np.result_type(i_pos.dtype, j_pos.dtype)
    lo = np.minimum(i_pos, j_pos).astype(dtype, copy=False)
    hi = np.maximum(i_pos, j_pos).astype(dtype, copy=False)
    shrink = np.asarray(shrink_ld, dtype=np.float64)
    original_order = np.arange(lo.size, dtype=np.int64)
    order = np.lexsort((original_order, hi, lo))
    lo = lo[order]
    hi = hi[order]
    shrink = shrink[order]
    ordered_meta = tuple(np.asarray(values)[order] for values in metadata)

    keep = np.ones(lo.size, dtype=bool)
    keep[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    return (
        lo[keep],
        hi[keep],
        shrink[keep],
        tuple(values[keep] for values in ordered_meta),
    )


def _lo_index(lo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lo.size == 0:
        return np.array([], dtype=np.int64), np.array([0], dtype=np.int64)
    starts = np.concatenate(
        (np.array([0], dtype=np.int64), np.flatnonzero(lo[1:] != lo[:-1]) + 1)
    )
    values = lo[starts].astype(np.int64, copy=False)
    offsets = np.concatenate((starts, np.array([lo.size], dtype=np.int64)))
    return values, offsets.astype(np.int64, copy=False)


def _validate_canonical_sorted_unique(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    chunk_rows: int = 1_000_000,
) -> None:
    """Raise if rows are not canonical, sorted, and duplicate-free."""
    if lo.shape != hi.shape:
        raise ValueError("covariance position arrays must have identical shapes")
    if lo.size == 0:
        return

    prev_lo = int(lo[0])
    prev_hi = int(hi[0])
    if prev_lo > prev_hi:
        raise ValueError("covariance rows must be canonical with lo <= hi")

    for start in range(1, lo.size, chunk_rows):
        stop = min(start + chunk_rows, lo.size)
        lo_chunk = lo[start:stop]
        hi_chunk = hi[start:stop]
        if np.any(lo_chunk > hi_chunk):
            raise ValueError("covariance rows must be canonical with lo <= hi")
        sorted_after_prev = (lo_chunk[0] > prev_lo) or (
            lo_chunk[0] == prev_lo and hi_chunk[0] > prev_hi
        )
        if not sorted_after_prev:
            raise ValueError(
                "covariance rows must be sorted by (lo, hi) with no duplicates"
            )
        if lo_chunk.size > 1:
            lo_prev = lo_chunk[:-1]
            lo_next = lo_chunk[1:]
            hi_prev = hi_chunk[:-1]
            hi_next = hi_chunk[1:]
            sorted_unique = (lo_next > lo_prev) | (
                (lo_next == lo_prev) & (hi_next > hi_prev)
            )
            if not np.all(sorted_unique):
                raise ValueError(
                    "covariance rows must be sorted by (lo, hi) with no duplicates"
                )
        prev_lo = int(lo_chunk[-1])
        prev_hi = int(hi_chunk[-1])


def write_covariance_partition_hdf5(
    path: Path,
    *,
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
    i_pos: np.ndarray,
    j_pos: np.ndarray,
    shrink_ld: np.ndarray,
    naive_ld: np.ndarray | None = None,
    i_gpos: np.ndarray | None = None,
    j_gpos: np.ndarray | None = None,
    i_id: np.ndarray | None = None,
    j_id: np.ndarray | None = None,
    compression: str | None = "lzf",
    assume_canonical_sorted_unique: bool = False,
) -> None:
    """Write one canonical, indexed HDF5 covariance partition.

    When ``assume_canonical_sorted_unique`` is true, the input rows must
    already be canonical ``lo <= hi`` rows sorted by ``(lo, hi)`` with no
    duplicate pairs. This avoids the defensive sort/dedup allocations used for
    generic callers and is intended for ``calc_covariance()``, whose pairwise
    kernel emits unique rows in sorted SNP-index order.
    """
    meta_inputs = [
        values
        for values in (naive_ld, i_gpos, j_gpos, i_id, j_id)
        if values is not None
    ]
    if assume_canonical_sorted_unique:
        i_pos = _position_array(i_pos)
        j_pos = _position_array(j_pos)
        position_dtype = np.result_type(i_pos.dtype, j_pos.dtype)
        lo = i_pos.astype(position_dtype, copy=False)
        hi = j_pos.astype(position_dtype, copy=False)
        shrink = np.asarray(shrink_ld, dtype=np.float64)
        ordered_meta = tuple(np.asarray(values) for values in meta_inputs)
    else:
        lo, hi, shrink, ordered_meta = _canonical_ordered_rows(
            i_pos, j_pos, shrink_ld, *meta_inputs
        )
    if lo.shape != hi.shape or lo.shape != shrink.shape:
        raise ValueError("covariance row arrays must have identical shapes")
    for values in ordered_meta:
        if values.shape != lo.shape:
            raise ValueError("metadata arrays must match covariance row shape")
    _validate_canonical_sorted_unique(lo, hi)
    meta_iter = iter(ordered_meta)
    naive = next(meta_iter) if naive_ld is not None else None
    ig = next(meta_iter) if i_gpos is not None else None
    jg = next(meta_iter) if j_gpos is not None else None
    iid = next(meta_iter) if i_id is not None else None
    jid = next(meta_iter) if j_id is not None else None

    diag_mask = lo == hi
    diag_pos = lo[diag_mask]
    diag_val = shrink[diag_mask]
    lo_values, lo_offsets = _lo_index(lo)

    h5py = _h5py()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = _FORMAT
        h5.attrs["version"] = _VERSION
        if chrom is not None:
            h5.attrs["chrom"] = chrom
        if start is not None:
            h5.attrs["start"] = int(start)
        if end is not None:
            h5.attrs["end"] = int(end)
        h5.attrs["position_dtype"] = str(lo.dtype)
        h5.attrs["sorted_by"] = "lo_hi"
        h5.attrs["deduplicated"] = True
        h5.attrs["compact"] = naive is None and ig is None and iid is None

        cov = h5.create_group("covariance")
        idx = h5.create_group("index")
        kwargs = {"compression": compression, "shuffle": True}
        cov.create_dataset("lo", data=lo, **kwargs)
        cov.create_dataset("hi", data=hi, **kwargs)
        cov.create_dataset("shrink_ld", data=shrink, **kwargs)
        if naive is not None:
            cov.create_dataset(
                "naive_ld",
                data=np.asarray(naive, dtype=np.float64),
                **kwargs,
            )

        meta = h5.create_group("metadata")
        if ig is not None:
            meta.create_dataset(
                "i_gpos",
                data=np.asarray(ig, dtype=np.float64),
                **kwargs,
            )
        if jg is not None:
            meta.create_dataset(
                "j_gpos",
                data=np.asarray(jg, dtype=np.float64),
                **kwargs,
            )
        if iid is not None:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            meta.create_dataset(
                "i_id",
                data=np.asarray(iid, dtype=object),
                dtype=string_dtype,
                compression=compression,
            )
        if jid is not None:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            meta.create_dataset(
                "j_id",
                data=np.asarray(jid, dtype=object),
                dtype=string_dtype,
                compression=compression,
            )

        idx.create_dataset("diag_pos", data=diag_pos, **kwargs)
        idx.create_dataset("diag_val", data=diag_val, **kwargs)
        idx.create_dataset("lo_values", data=lo_values, **kwargs)
        idx.create_dataset("lo_offsets", data=lo_offsets, **kwargs)


def write_compact_covariance_partition_hdf5_chunks(
    path: Path,
    *,
    positions: np.ndarray,
    row_counts: np.ndarray,
    row_chunks: Iterator[CovarianceRowChunk],
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
    compression: str | None = "lzf",
    chunk_rows: int = 1_000_000,
    dataset_chunk_rows: int = HDF5_DATASET_CHUNK_ROWS,
) -> None:
    """Write compact canonical rows from a bounded sorted chunk iterator.

    This is the compact-cache writer used by ``calc_covariance()``. It avoids
    materializing full-partition ``i_pos``/``j_pos`` arrays while preserving the
    same HDF5 schema and ``lo_offsets`` index as ``write_covariance_partition_hdf5``.
    """
    positions = _position_array(positions)
    row_counts = np.asarray(row_counts, dtype=np.int64)
    if positions.shape != row_counts.shape:
        raise ValueError("positions and row_counts must have identical shapes")

    n_rows = int(row_counts.sum())
    nonzero_lo = row_counts > 0
    lo_values = positions[nonzero_lo].astype(np.int64, copy=False)
    lo_offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(row_counts[nonzero_lo], dtype=np.int64),
        )
    )

    h5py = _h5py()
    path.parent.mkdir(parents=True, exist_ok=True)
    position_dtype = positions.dtype
    h5_chunk_rows = max(1, min(int(dataset_chunk_rows), max(n_rows, 1)))
    kwargs = {"compression": compression, "shuffle": True}

    diag_pos_parts: list[np.ndarray] = []
    diag_val_parts: list[np.ndarray] = []
    offset = 0
    prev_lo: int | None = None
    prev_hi: int | None = None

    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = _FORMAT
        h5.attrs["version"] = _VERSION
        if chrom is not None:
            h5.attrs["chrom"] = chrom
        if start is not None:
            h5.attrs["start"] = int(start)
        if end is not None:
            h5.attrs["end"] = int(end)
        h5.attrs["position_dtype"] = str(position_dtype)
        h5.attrs["sorted_by"] = "lo_hi"
        h5.attrs["deduplicated"] = True
        h5.attrs["compact"] = True
        h5.attrs["dataset_chunk_rows"] = int(h5_chunk_rows) if n_rows else 0
        h5.attrs["write_chunk_rows"] = int(chunk_rows)

        cov = h5.create_group("covariance")
        idx = h5.create_group("index")
        if n_rows == 0:
            cov.create_dataset("lo", data=np.array([], dtype=position_dtype), **kwargs)
            cov.create_dataset("hi", data=np.array([], dtype=position_dtype), **kwargs)
            cov.create_dataset(
                "shrink_ld", data=np.array([], dtype=np.float64), **kwargs
            )
        else:
            dataset_kwargs = {
                **kwargs,
                "shape": (n_rows,),
                "chunks": (h5_chunk_rows,),
            }
            lo_ds = cov.create_dataset("lo", dtype=position_dtype, **dataset_kwargs)
            hi_ds = cov.create_dataset("hi", dtype=position_dtype, **dataset_kwargs)
            shrink_ds = cov.create_dataset(
                "shrink_ld", dtype=np.float64, **dataset_kwargs
            )

            for chunk in row_chunks:
                lo = _position_array(chunk.lo).astype(position_dtype, copy=False)
                hi = _position_array(chunk.hi).astype(position_dtype, copy=False)
                shrink = np.asarray(chunk.shrink_ld, dtype=np.float64)
                if lo.shape != hi.shape or lo.shape != shrink.shape:
                    raise ValueError("covariance row chunks must have matching shapes")
                if lo.size == 0:
                    continue
                _validate_canonical_sorted_unique(lo, hi)
                if prev_lo is not None and prev_hi is not None:
                    sorted_after_prev = (int(lo[0]) > prev_lo) or (
                        int(lo[0]) == prev_lo and int(hi[0]) > prev_hi
                    )
                    if not sorted_after_prev:
                        raise ValueError(
                            "covariance row chunks must be globally sorted by (lo, hi)"
                        )
                stop = offset + lo.size
                if stop > n_rows:
                    raise ValueError("row_chunks yielded more rows than row_counts")
                lo_ds[offset:stop] = lo
                hi_ds[offset:stop] = hi
                shrink_ds[offset:stop] = shrink

                diag_mask = lo == hi
                if np.any(diag_mask):
                    diag_pos_parts.append(lo[diag_mask])
                    diag_val_parts.append(shrink[diag_mask])
                prev_lo = int(lo[-1])
                prev_hi = int(hi[-1])
                offset = stop

            if offset != n_rows:
                raise ValueError("row_chunks yielded fewer rows than row_counts")

        diag_pos = (
            np.concatenate(diag_pos_parts).astype(position_dtype, copy=False)
            if diag_pos_parts
            else np.array([], dtype=position_dtype)
        )
        diag_val = (
            np.concatenate(diag_val_parts).astype(np.float64, copy=False)
            if diag_val_parts
            else np.array([], dtype=np.float64)
        )
        idx.create_dataset("diag_pos", data=diag_pos, **kwargs)
        idx.create_dataset("diag_val", data=diag_val, **kwargs)
        idx.create_dataset("lo_values", data=lo_values, **kwargs)
        idx.create_dataset("lo_offsets", data=lo_offsets, **kwargs)


def write_compact_covariance_partition_hdf5_append(
    path: Path,
    *,
    positions: np.ndarray,
    row_chunks: Iterator[CovarianceRowChunk],
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
    compression: str | None = "lzf",
    chunk_rows: int = 1_000_000,
    dataset_chunk_rows: int = HDF5_DATASET_CHUNK_ROWS,
) -> int:
    """Write compact canonical rows from a bounded stream in one generation pass."""
    positions = _position_array(positions)
    h5py = _h5py()
    path.parent.mkdir(parents=True, exist_ok=True)
    position_dtype = positions.dtype
    h5_chunk_rows = max(1, int(dataset_chunk_rows))
    kwargs = {"compression": compression, "shuffle": True}
    dataset_kwargs = {
        **kwargs,
        "shape": (0,),
        "maxshape": (None,),
        "chunks": (h5_chunk_rows,),
    }

    row_counts = np.zeros(positions.size, dtype=np.int64)
    diag_pos_parts: list[np.ndarray] = []
    diag_val_parts: list[np.ndarray] = []
    offset = 0
    prev_lo: int | None = None
    prev_hi: int | None = None

    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = _FORMAT
        h5.attrs["version"] = _VERSION
        if chrom is not None:
            h5.attrs["chrom"] = chrom
        if start is not None:
            h5.attrs["start"] = int(start)
        if end is not None:
            h5.attrs["end"] = int(end)
        h5.attrs["position_dtype"] = str(position_dtype)
        h5.attrs["sorted_by"] = "lo_hi"
        h5.attrs["deduplicated"] = True
        h5.attrs["compact"] = True
        h5.attrs["dataset_chunk_rows"] = 0
        h5.attrs["write_chunk_rows"] = int(chunk_rows)

        cov = h5.create_group("covariance")
        idx = h5.create_group("index")
        lo_ds = cov.create_dataset("lo", dtype=position_dtype, **dataset_kwargs)
        hi_ds = cov.create_dataset("hi", dtype=position_dtype, **dataset_kwargs)
        shrink_ds = cov.create_dataset(
            "shrink_ld", dtype=np.float64, **dataset_kwargs
        )

        for chunk in row_chunks:
            lo = _position_array(chunk.lo).astype(position_dtype, copy=False)
            hi = _position_array(chunk.hi).astype(position_dtype, copy=False)
            shrink = np.asarray(chunk.shrink_ld, dtype=np.float64)
            if lo.shape != hi.shape or lo.shape != shrink.shape:
                raise ValueError("covariance row chunks must have matching shapes")
            if lo.size == 0:
                continue
            _validate_canonical_sorted_unique(lo, hi)
            if prev_lo is not None and prev_hi is not None:
                sorted_after_prev = (int(lo[0]) > prev_lo) or (
                    int(lo[0]) == prev_lo and int(hi[0]) > prev_hi
                )
                if not sorted_after_prev:
                    raise ValueError(
                        "covariance row chunks must be globally sorted by (lo, hi)"
                    )

            stop = offset + lo.size
            lo_ds.resize((stop,))
            hi_ds.resize((stop,))
            shrink_ds.resize((stop,))
            lo_ds[offset:stop] = lo
            hi_ds[offset:stop] = hi
            shrink_ds[offset:stop] = shrink

            lo_values, lo_counts = np.unique(lo, return_counts=True)
            lo_idx = np.searchsorted(positions, lo_values)
            in_bounds = lo_idx < positions.size
            if not np.all(in_bounds):
                raise ValueError("compact row lower endpoints must exist in positions")
            if not np.all(positions[lo_idx] == lo_values):
                raise ValueError("compact row lower endpoints must exist in positions")
            row_counts[lo_idx] += lo_counts.astype(np.int64, copy=False)

            diag_mask = lo == hi
            if np.any(diag_mask):
                diag_pos_parts.append(lo[diag_mask])
                diag_val_parts.append(shrink[diag_mask])
            prev_lo = int(lo[-1])
            prev_hi = int(hi[-1])
            offset = stop

        h5.attrs["dataset_chunk_rows"] = int(h5_chunk_rows) if offset else 0
        diag_pos = (
            np.concatenate(diag_pos_parts).astype(position_dtype, copy=False)
            if diag_pos_parts
            else np.array([], dtype=position_dtype)
        )
        diag_val = (
            np.concatenate(diag_val_parts).astype(np.float64, copy=False)
            if diag_val_parts
            else np.array([], dtype=np.float64)
        )
        nonzero_lo = row_counts > 0
        lo_values = positions[nonzero_lo].astype(np.int64, copy=False)
        lo_offsets = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.cumsum(row_counts[nonzero_lo], dtype=np.int64),
            )
        )
        idx.create_dataset("diag_pos", data=diag_pos, **kwargs)
        idx.create_dataset("diag_val", data=diag_val, **kwargs)
        idx.create_dataset("lo_values", data=lo_values, **kwargs)
        idx.create_dataset("lo_offsets", data=lo_offsets, **kwargs)

    return offset


def validate_covariance_hdf5(path: Path, require_full: bool = False) -> bool:
    """Return whether *path* is a readable ldetect2 HDF5 covariance partition."""
    if not path.exists():
        return False
    try:
        h5py = _h5py()
        with h5py.File(path, "r") as h5:
            if h5.attrs.get("format") != _FORMAT:
                return False
            if int(h5.attrs.get("version", -1)) != _VERSION:
                return False
            for dataset in _REQUIRED_DATASETS:
                if dataset not in h5:
                    return False
            if require_full:
                for dataset in (
                    "covariance/naive_ld",
                    "metadata/i_gpos",
                    "metadata/j_gpos",
                    "metadata/i_id",
                    "metadata/j_id",
                ):
                    if dataset not in h5:
                        return False
            return True
    except Exception:
        return False


class HDF5CovariancePartitionReader:
    """Chunked reader for one HDF5 covariance partition."""

    def __init__(self, path: Path, start: int, end: int) -> None:
        self.path = path
        self.start = start
        self.end = end
        self._h5: Any | None = None

    def __enter__(self) -> HDF5CovariancePartitionReader:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def open(self) -> None:
        if self._h5 is None:
            h5py = _h5py()
            self._h5 = h5py.File(self.path, "r")

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    @property
    def h5(self) -> Any:
        self.open()
        return self._h5

    @property
    def row_count(self) -> int:
        return int(self.h5["covariance/lo"].shape[0])

    def read_all(self) -> CovarianceRowChunk:
        """Read all canonical rows from this partition."""
        h5 = self.h5
        return CovarianceRowChunk(
            lo=_position_array(h5["covariance/lo"][:]),
            hi=_position_array(h5["covariance/hi"][:]),
            shrink_ld=np.asarray(h5["covariance/shrink_ld"][:], dtype=np.float64),
        )

    def read_diagonal(self) -> tuple[np.ndarray, np.ndarray]:
        """Read canonical diagonal positions and values."""
        h5 = self.h5
        return (
            _position_array(h5["index/diag_pos"][:]),
            np.asarray(h5["index/diag_val"][:], dtype=np.float64),
        )

    def read_loci(self) -> np.ndarray:
        """Read sorted unique ``lo`` loci."""
        return _position_array(self.h5["index/lo_values"][:]).astype(
            np.int64, copy=False
        )

    def iter_rows(
        self,
        lo_min: int,
        lo_max: int,
        chunk_rows: int,
    ) -> Iterator[CovarianceRowChunk]:
        """Yield bounded row chunks whose ``lo`` values are in range."""
        h5 = self.h5
        lo_values = h5["index/lo_values"][:]
        lo_offsets = h5["index/lo_offsets"][:]
        left_value = int(np.searchsorted(lo_values, lo_min, side="left"))
        right_value = int(np.searchsorted(lo_values, lo_max, side="right"))
        if left_value >= right_value:
            return
        left = int(lo_offsets[left_value])
        right = int(lo_offsets[right_value])
        for chunk_start in range(left, right, chunk_rows):
            chunk_stop = min(chunk_start + chunk_rows, right)
            yield CovarianceRowChunk(
                lo=_position_array(h5["covariance/lo"][chunk_start:chunk_stop]),
                hi=_position_array(h5["covariance/hi"][chunk_start:chunk_stop]),
                shrink_ld=np.asarray(
                    h5["covariance/shrink_ld"][chunk_start:chunk_stop],
                    dtype=np.float64,
                ),
            )

    def iter_owned_rows(
        self,
        lower_min: int,
        lower_max: int,
        snp_first: int,
        snp_last: int,
        chunk_rows: int,
        include_lower_min: bool = True,
    ) -> Iterator[CovarianceRowChunk]:
        """Yield chunks filtered to the metric/vector ownership window."""
        for chunk in self.iter_rows(lower_min, lower_max, chunk_rows):
            lower_owned = (
                chunk.lo >= lower_min if include_lower_min else chunk.lo > lower_min
            )
            mask = (
                (chunk.lo >= snp_first)
                & (chunk.lo <= snp_last)
                & (chunk.hi >= snp_first)
                & (chunk.hi <= snp_last)
                & lower_owned
                & (chunk.lo <= lower_max)
            )
            if np.any(mask):
                yield CovarianceRowChunk(
                    lo=chunk.lo[mask],
                    hi=chunk.hi[mask],
                    shrink_ld=chunk.shrink_ld[mask],
                )


def open_covariance_reader(
    path: Path, start: int, end: int
) -> HDF5CovariancePartitionReader:
    """Return an HDF5 covariance reader for one partition path."""
    return HDF5CovariancePartitionReader(path, start, end)
