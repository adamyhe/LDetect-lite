"""Experimental Zarr v2 storage for normalized r2 partition rows."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FORMAT = "ldetect2-r2-zarr"
_VERSION = 1
R2_ZARR_DATASET_CHUNK_ROWS = 65_536
_REQUIRED_ARRAYS = frozenset(
    {
        "positions",
        "lo_values",
        "lo_offsets",
        "hi_idx",
        "r2",
    }
)


@dataclass(frozen=True)
class R2RowChunk:
    """A bounded chunk of canonical, normalized r2 rows."""

    lo: np.ndarray
    hi: np.ndarray
    r2: np.ndarray


def _zarr():
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "zarr<3 is required for experimental r2 partition caches. Install "
            "ldetect2 with its project dependencies before using --pair-cache "
            "r2-zarr."
        ) from exc
    return zarr


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


def r2_zarr_path(root: Path, name: str) -> Path:
    """Return the chromosome-level Zarr directory for *name*."""
    return root / f"{name}.r2.zarr"


def r2_partition_group_name(start: int, end: int) -> str:
    """Return the partition group name for a physical interval."""
    return f"{int(start)}_{int(end)}"


def _hi_idx_dtype(n_positions: int) -> np.dtype:
    return np.dtype(np.uint16 if n_positions <= np.iinfo(np.uint16).max else np.uint32)


def _validate_canonical_sorted_unique(lo: np.ndarray, hi: np.ndarray) -> None:
    if lo.shape != hi.shape:
        raise ValueError("r2 row position arrays must have identical shapes")
    if lo.size == 0:
        return
    if np.any(lo > hi):
        raise ValueError("r2 rows must be canonical with lo <= hi")
    if lo.size == 1:
        return
    sorted_unique = (lo[1:] > lo[:-1]) | ((lo[1:] == lo[:-1]) & (hi[1:] > hi[:-1]))
    if not np.all(sorted_unique):
        raise ValueError("r2 rows must be sorted by (lo, hi) with no duplicates")


def write_r2_zarr_partition_append(
    root: Path,
    name: str,
    start: int,
    end: int,
    *,
    positions: np.ndarray,
    row_chunks: Iterator[R2RowChunk],
    ne: float,
    cutoff: float,
    chunk_rows: int = 1_000_000,
    dataset_chunk_rows: int = R2_ZARR_DATASET_CHUNK_ROWS,
) -> int:
    """Write one normalized r2 partition as an indexed Zarr v2 group.

    Rows are expected to be canonical ``lo <= hi`` physical positions, sorted by
    ``(lo, hi)``, and normalized already. The writer stores ``hi_idx`` relative
    to the partition ``positions`` array; ``lo`` is represented by
    ``lo_values``/``lo_offsets``.
    """
    positions = _position_array(positions)
    if positions.size and not np.all(positions[1:] > positions[:-1]):
        raise ValueError("r2 partition positions must be strictly increasing")

    zarr = _zarr()
    zarr_path = r2_zarr_path(root, name)
    zarr_path.mkdir(parents=True, exist_ok=True)
    zroot = zarr.open_group(str(zarr_path), mode="a")
    zroot.attrs["format"] = _FORMAT
    zroot.attrs["version"] = _VERSION
    zroot.attrs["chrom"] = name

    partitions = zroot.require_group("partitions")
    group_name = r2_partition_group_name(start, end)
    if group_name in partitions:
        del partitions[group_name]
    group = partitions.create_group(group_name)

    position_dtype = positions.dtype
    hi_dtype = _hi_idx_dtype(int(positions.size))
    z_chunk_rows = max(1, int(dataset_chunk_rows))
    hi_ds = group.create_dataset(
        "hi_idx",
        shape=(0,),
        chunks=(z_chunk_rows,),
        dtype=hi_dtype,
    )
    r2_ds = group.create_dataset(
        "r2",
        shape=(0,),
        chunks=(z_chunk_rows,),
        dtype=np.float64,
    )

    row_counts = np.zeros(positions.size, dtype=np.int64)
    offset = 0
    prev_lo: int | None = None
    prev_hi: int | None = None

    for chunk in row_chunks:
        lo = _position_array(chunk.lo).astype(position_dtype, copy=False)
        hi = _position_array(chunk.hi).astype(position_dtype, copy=False)
        r2 = np.asarray(chunk.r2, dtype=np.float64)
        if lo.shape != hi.shape or lo.shape != r2.shape:
            raise ValueError("r2 row chunks must have matching shapes")
        if lo.size == 0:
            continue
        _validate_canonical_sorted_unique(lo, hi)
        if prev_lo is not None:
            sorted_after_prev = (int(lo[0]) > prev_lo) or (
                int(lo[0]) == prev_lo and int(hi[0]) > prev_hi
            )
            if not sorted_after_prev:
                raise ValueError("r2 row chunks must be globally sorted by (lo, hi)")

        lo_idx = np.searchsorted(positions, lo)
        hi_idx = np.searchsorted(positions, hi)
        in_bounds = (lo_idx < positions.size) & (hi_idx < positions.size)
        if not np.all(in_bounds):
            raise ValueError("r2 row endpoints must exist in positions")
        if not np.all((positions[lo_idx] == lo) & (positions[hi_idx] == hi)):
            raise ValueError("r2 row endpoints must exist in positions")

        stop = offset + lo.size
        hi_ds.resize((stop,))
        r2_ds.resize((stop,))
        hi_ds[offset:stop] = hi_idx.astype(hi_dtype, copy=False)
        r2_ds[offset:stop] = r2

        lo_values, lo_counts = np.unique(lo, return_counts=True)
        count_idx = np.searchsorted(positions, lo_values)
        row_counts[count_idx] += lo_counts.astype(np.int64, copy=False)

        prev_lo = int(lo[-1])
        prev_hi = int(hi[-1])
        offset = stop

    nonzero_lo = row_counts > 0
    lo_values = positions[nonzero_lo].astype(np.int64, copy=False)
    lo_offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(row_counts[nonzero_lo], dtype=np.int64),
        )
    )
    group.create_dataset("positions", data=positions)
    group.create_dataset("lo_values", data=lo_values)
    group.create_dataset("lo_offsets", data=lo_offsets)

    group.attrs["format"] = _FORMAT
    group.attrs["version"] = _VERSION
    group.attrs["chrom"] = name
    group.attrs["start"] = int(start)
    group.attrs["end"] = int(end)
    group.attrs["ne"] = float(ne)
    group.attrs["cutoff"] = float(cutoff)
    group.attrs["n_pairs"] = int(offset)
    group.attrs["position_dtype"] = str(position_dtype)
    group.attrs["sorted_by"] = "lo_hi"
    group.attrs["normalized"] = True
    group.attrs["write_chunk_rows"] = int(chunk_rows)
    group.attrs["dataset_chunk_rows"] = int(z_chunk_rows) if offset else 0
    return int(offset)


def validate_r2_zarr_partition(root: Path, name: str, start: int, end: int) -> bool:
    """Return whether the requested r2 Zarr partition is readable."""
    path = r2_zarr_path(root, name)
    if not path.exists():
        return False
    try:
        reader = R2ZarrPartitionReader(root, name, start, end)
        reader.open()
        group = reader.group
        if group.attrs.get("format") != _FORMAT:
            return False
        if int(group.attrs.get("version", -1)) != _VERSION:
            return False
        if str(group.attrs.get("chrom", "")) != name:
            return False
        if int(group.attrs.get("start", -1)) != int(start):
            return False
        if int(group.attrs.get("end", -1)) != int(end):
            return False
        for array in _REQUIRED_ARRAYS:
            if array not in group:
                return False
        if group["hi_idx"].shape != group["r2"].shape:
            return False
        if group["lo_offsets"].shape[0] != group["lo_values"].shape[0] + 1:
            return False
        if int(group["lo_offsets"][-1]) != int(group["r2"].shape[0]):
            return False
        if int(group.attrs.get("n_pairs", -1)) != int(group["r2"].shape[0]):
            return False
        return True
    except Exception:
        return False


class R2ZarrPartitionReader:
    """Chunked reader for one normalized r2 Zarr partition."""

    def __init__(self, root: Path, name: str, start: int, end: int) -> None:
        self.root = root
        self.name = name
        self.start = int(start)
        self.end = int(end)
        self._group = None

    def __enter__(self) -> R2ZarrPartitionReader:
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def open(self) -> None:
        if self._group is None:
            zarr = _zarr()
            root_group = zarr.open_group(
                str(r2_zarr_path(self.root, self.name)),
                mode="r",
            )
            self._group = root_group["partitions"][
                r2_partition_group_name(self.start, self.end)
            ]

    def close(self) -> None:
        self._group = None

    @property
    def group(self):
        self.open()
        return self._group

    @property
    def row_count(self) -> int:
        return int(self.group["r2"].shape[0])

    def read_loci(self) -> np.ndarray:
        """Read sorted unique ``lo`` loci."""
        return _position_array(self.group["lo_values"][:]).astype(np.int64, copy=False)

    def iter_rows(
        self,
        lo_min: int,
        lo_max: int,
        chunk_rows: int,
    ) -> Iterator[R2RowChunk]:
        """Yield bounded row chunks whose ``lo`` values are in range."""
        group = self.group
        positions = _position_array(group["positions"][:])
        lo_values = _position_array(group["lo_values"][:])
        lo_offsets = np.asarray(group["lo_offsets"][:], dtype=np.int64)
        left_value = int(np.searchsorted(lo_values, lo_min, side="left"))
        right_value = int(np.searchsorted(lo_values, lo_max, side="right"))
        if left_value >= right_value:
            return
        left = int(lo_offsets[left_value])
        right = int(lo_offsets[right_value])
        chunk_rows = max(1, int(chunk_rows))
        for chunk_start in range(left, right, chunk_rows):
            chunk_stop = min(chunk_start + chunk_rows, right)
            row_numbers = np.arange(chunk_start, chunk_stop, dtype=np.int64)
            lo_idx = np.searchsorted(lo_offsets, row_numbers, side="right") - 1
            hi_idx = np.asarray(group["hi_idx"][chunk_start:chunk_stop], dtype=np.int64)
            yield R2RowChunk(
                lo=lo_values[lo_idx],
                hi=positions[hi_idx],
                r2=np.asarray(group["r2"][chunk_start:chunk_stop], dtype=np.float64),
            )

    def iter_owned_rows(
        self,
        lower_min: int,
        lower_max: int,
        snp_first: int,
        snp_last: int,
        chunk_rows: int,
        include_lower_min: bool = True,
    ) -> Iterator[R2RowChunk]:
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
                yield R2RowChunk(
                    lo=chunk.lo[mask],
                    hi=chunk.hi[mask],
                    r2=chunk.r2[mask],
                )


def open_r2_zarr_reader(
    root: Path, name: str, start: int, end: int
) -> R2ZarrPartitionReader:
    """Return a normalized r2 Zarr reader for one partition."""
    return R2ZarrPartitionReader(root, name, start, end)
