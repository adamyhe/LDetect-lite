"""Experimental Zarr v2 storage for normalized r2 partition rows."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FORMAT = "ldetect2-r2-zarr"
_VERSION = 2
R2_ZARR_DATASET_CHUNK_ROWS = 65_536
R2_ZARR_COMPRESSORS = ("default", "lz4-bitshuffle", "zstd-bitshuffle")
_REQUIRED_ARRAYS_V2 = frozenset(
    {
        "positions",
        "lo_values",
        "lo_offsets",
        "diag_idx",
        "hi_delta",
        "r2",
    }
)
_OWNED_GROUP = "owned_pairs"


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


def r2_zarr_compressor(name: str):
    """Return an optional Zarr v2 compressor for experimental r2 caches."""
    if name == "default":
        return None
    try:
        from numcodecs import Blosc
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "numcodecs is required for explicit r2 Zarr compression modes."
        ) from exc

    if name == "lz4-bitshuffle":
        return Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    if name == "zstd-bitshuffle":
        return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    raise ValueError(
        "unsupported r2 Zarr compressor "
        f"{name!r}; expected one of {', '.join(R2_ZARR_COMPRESSORS)}"
    )


def _dataset_compressor_kwargs(name: str) -> dict[str, object]:
    compressor = r2_zarr_compressor(name)
    return {} if compressor is None else {"compressor": compressor}


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


def _row_index_dtype(n_positions: int) -> np.dtype:
    return _hi_idx_dtype(n_positions)


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


def _write_r2_zarr_group(
    group,
    *,
    positions: np.ndarray,
    row_chunks: Iterator[R2RowChunk],
    ne: float,
    cutoff: float,
    chunk_rows: int,
    dataset_chunk_rows: int,
    compressor: str,
    attrs: dict[str, object],
) -> int:
    """Write normalized rows into one v2 r2 Zarr group."""
    position_dtype = positions.dtype
    index_dtype = _row_index_dtype(int(positions.size))
    z_chunk_rows = max(1, int(dataset_chunk_rows))
    compressor_kwargs = _dataset_compressor_kwargs(compressor)
    hi_delta_ds = group.create_dataset(
        "hi_delta",
        shape=(0,),
        chunks=(z_chunk_rows,),
        dtype=index_dtype,
        **compressor_kwargs,
    )
    r2_ds = group.create_dataset(
        "r2",
        shape=(0,),
        chunks=(z_chunk_rows,),
        dtype=np.float64,
        **compressor_kwargs,
    )

    row_counts = np.zeros(positions.size, dtype=np.int64)
    diag_seen = np.zeros(positions.size, dtype=bool)
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

        diag = lo_idx == hi_idx
        if np.any(diag):
            if not np.all(r2[diag] == 1.0):
                raise ValueError("implicit r2 diagonal rows must have r2=1.0")
            diag_seen[lo_idx[diag]] = True

        offdiag = ~diag
        if np.any(offdiag):
            hi_delta = hi_idx[offdiag] - lo_idx[offdiag]
            if np.any(hi_delta < 0):
                raise ValueError("r2 rows must be canonical with hi_idx >= lo_idx")
            if hi_delta.size and int(hi_delta.max()) > np.iinfo(index_dtype).max:
                raise ValueError("hi_delta exceeds selected storage dtype")

            stop = offset + int(np.count_nonzero(offdiag))
            hi_delta_ds.resize((stop,))
            r2_ds.resize((stop,))
            hi_delta_ds[offset:stop] = hi_delta.astype(index_dtype, copy=False)
            r2_ds[offset:stop] = r2[offdiag]

            lo_values, lo_counts = np.unique(lo[offdiag], return_counts=True)
            count_idx = np.searchsorted(positions, lo_values)
            row_counts[count_idx] += lo_counts.astype(np.int64, copy=False)
            offset = stop

        prev_lo = int(lo[-1])
        prev_hi = int(hi[-1])

    nonzero_lo = row_counts > 0
    lo_values = positions[nonzero_lo].astype(np.int64, copy=False)
    lo_offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(row_counts[nonzero_lo], dtype=np.int64),
        )
    )
    diag_idx = np.flatnonzero(diag_seen).astype(index_dtype, copy=False)
    group.create_dataset("positions", data=positions, **compressor_kwargs)
    group.create_dataset("lo_values", data=lo_values, **compressor_kwargs)
    group.create_dataset("lo_offsets", data=lo_offsets, **compressor_kwargs)
    group.create_dataset("diag_idx", data=diag_idx, **compressor_kwargs)

    group.attrs["format"] = _FORMAT
    group.attrs["version"] = _VERSION
    group.attrs["ne"] = float(ne)
    group.attrs["cutoff"] = float(cutoff)
    group.attrs["n_pairs"] = int(offset + diag_idx.size)
    group.attrs["n_stored_pairs"] = int(offset)
    group.attrs["n_diagonal"] = int(diag_idx.size)
    group.attrs["position_dtype"] = str(position_dtype)
    group.attrs["sorted_by"] = "lo_hi"
    group.attrs["normalized"] = True
    group.attrs["implicit_diagonal"] = True
    group.attrs["coordinate_encoding"] = "hi_delta"
    group.attrs["write_chunk_rows"] = int(chunk_rows)
    group.attrs["dataset_chunk_rows"] = int(z_chunk_rows) if offset else 0
    group.attrs["compressor"] = compressor
    for key, value in attrs.items():
        group.attrs[key] = value
    return int(offset + diag_idx.size)


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
    compressor: str = "default",
) -> int:
    """Write one normalized r2 partition as an indexed Zarr v2 group.

    Rows are expected to be canonical ``lo <= hi`` physical positions, sorted by
    ``(lo, hi)``, and normalized already. The writer stores diagonal rows
    implicitly in ``diag_idx`` and off-diagonal upper endpoints as ``hi_delta``
    relative to each row's lower-endpoint index.
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
    return _write_r2_zarr_group(
        group,
        positions=positions,
        row_chunks=row_chunks,
        ne=ne,
        cutoff=cutoff,
        chunk_rows=chunk_rows,
        dataset_chunk_rows=dataset_chunk_rows,
        compressor=compressor,
        attrs={
            "chrom": name,
            "start": int(start),
            "end": int(end),
            "scope": "partition",
        },
    )


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
        for array in _REQUIRED_ARRAYS_V2:
            if array not in group:
                return False
        if group["hi_delta"].shape != group["r2"].shape:
            return False
        if group["lo_offsets"].shape[0] != group["lo_values"].shape[0] + 1:
            return False
        if int(group["lo_offsets"][-1]) != int(group["r2"].shape[0]):
            return False
        n_pairs = int(group.attrs.get("n_pairs", -1))
        n_stored = int(group.attrs.get("n_stored_pairs", -1))
        n_diag = int(group.attrs.get("n_diagonal", -1))
        if n_stored != int(group["r2"].shape[0]):
            return False
        if n_diag != int(group["diag_idx"].shape[0]):
            return False
        if n_pairs != n_stored + n_diag:
            return False
        return True
    except Exception:
        return False


def validate_r2_zarr_owned_cache(root: Path, name: str) -> bool:
    """Return whether the chromosome-level owned r2 Zarr cache is readable."""
    path = r2_zarr_path(root, name)
    if not path.exists():
        return False
    try:
        reader = R2ZarrPartitionReader(root, name, owned=True)
        reader.open()
        group = reader.group
        if group.attrs.get("format") != _FORMAT:
            return False
        if int(group.attrs.get("version", -1)) != _VERSION:
            return False
        if str(group.attrs.get("chrom", "")) != name:
            return False
        if str(group.attrs.get("scope", "")) != "chromosome_owned":
            return False
        for array in _REQUIRED_ARRAYS_V2:
            if array not in group:
                return False
        if group["hi_delta"].shape != group["r2"].shape:
            return False
        if group["lo_offsets"].shape[0] != group["lo_values"].shape[0] + 1:
            return False
        if int(group["lo_offsets"][-1]) != int(group["r2"].shape[0]):
            return False
        n_pairs = int(group.attrs.get("n_pairs", -1))
        n_stored = int(group.attrs.get("n_stored_pairs", -1))
        n_diag = int(group.attrs.get("n_diagonal", -1))
        if n_stored != int(group["r2"].shape[0]):
            return False
        if n_diag != int(group["diag_idx"].shape[0]):
            return False
        if n_pairs != n_stored + n_diag:
            return False
        return True
    except Exception:
        return False


def _flush_owned_loci(
    buffered: dict[int, dict[int, float]],
    flush_loci: list[int],
    chunk_rows: int,
) -> Iterator[R2RowChunk]:
    out_lo: list[int] = []
    out_hi: list[int] = []
    out_r2: list[float] = []
    for lo in flush_loci:
        hi_map = buffered.pop(lo)
        for hi in sorted(hi_map):
            out_lo.append(lo)
            out_hi.append(hi)
            out_r2.append(hi_map[hi])
            if len(out_lo) >= chunk_rows:
                yield R2RowChunk(
                    lo=np.asarray(out_lo, dtype=np.int64),
                    hi=np.asarray(out_hi, dtype=np.int64),
                    r2=np.asarray(out_r2, dtype=np.float64),
                )
                out_lo = []
                out_hi = []
                out_r2 = []
    if out_lo:
        yield R2RowChunk(
            lo=np.asarray(out_lo, dtype=np.int64),
            hi=np.asarray(out_hi, dtype=np.int64),
            r2=np.asarray(out_r2, dtype=np.float64),
        )


def write_r2_zarr_owned_cache(
    root: Path,
    name: str,
    partitions: list[tuple[int, int]],
    *,
    snp_first: int,
    snp_last: int,
    ne: float,
    cutoff: float,
    chunk_rows: int = 1_000_000,
    dataset_chunk_rows: int = R2_ZARR_DATASET_CHUNK_ROWS,
    compressor: str = "default",
    delete_partitions: bool = True,
) -> int:
    """Write a chromosome-level first-precedence owned r2 cache.

    Partition caches are streamed in partition order.  Duplicate physical pairs
    keep the first retained value, matching local-search row precedence, and
    loci are flushed once no later partition can contain that lower endpoint.
    """
    zarr = _zarr()
    zarr_path = r2_zarr_path(root, name)
    zroot = zarr.open_group(str(zarr_path), mode="a")
    zroot.attrs["format"] = _FORMAT
    zroot.attrs["version"] = _VERSION
    zroot.attrs["chrom"] = name

    position_parts: list[np.ndarray] = []
    for start, end in partitions:
        with open_r2_zarr_reader(root, name, start, end) as reader:
            position_parts.append(_position_array(reader.group["positions"][:]))
    positions = (
        np.unique(np.concatenate(position_parts)).astype(np.int64, copy=False)
        if position_parts
        else np.array([], dtype=np.int64)
    )
    positions = _position_array(positions)

    if _OWNED_GROUP in zroot:
        del zroot[_OWNED_GROUP]
    group = zroot.create_group(_OWNED_GROUP)

    def owned_chunks() -> Iterator[R2RowChunk]:
        buffered: dict[int, dict[int, float]] = {}
        for p_index, (start, end) in enumerate(partitions):
            next_start = (
                int(partitions[p_index + 1][0])
                if p_index + 1 < len(partitions)
                else None
            )
            with open_r2_zarr_reader(root, name, start, end) as reader:
                for chunk in reader.iter_rows(int(start), int(end), chunk_rows):
                    in_range = (
                        (chunk.lo >= snp_first)
                        & (chunk.lo <= snp_last)
                        & (chunk.hi >= snp_first)
                        & (chunk.hi <= snp_last)
                    )
                    if not np.any(in_range):
                        continue
                    for lo_pos, hi_pos, value in zip(
                        chunk.lo[in_range],
                        chunk.hi[in_range],
                        chunk.r2[in_range],
                        strict=True,
                    ):
                        lo_int = int(lo_pos)
                        hi_int = int(hi_pos)
                        hi_map = buffered.setdefault(lo_int, {})
                        if hi_int not in hi_map:
                            hi_map[hi_int] = float(value)
            if next_start is None:
                flush_loci = sorted(buffered)
            else:
                flush_loci = sorted(lo for lo in buffered if lo < next_start)
            yield from _flush_owned_loci(buffered, flush_loci, chunk_rows)

    n_pairs = _write_r2_zarr_group(
        group,
        positions=positions,
        row_chunks=owned_chunks(),
        ne=ne,
        cutoff=cutoff,
        chunk_rows=chunk_rows,
        dataset_chunk_rows=dataset_chunk_rows,
        compressor=compressor,
        attrs={
            "chrom": name,
            "scope": "chromosome_owned",
            "snp_first": int(snp_first),
            "snp_last": int(snp_last),
            "partition_count": int(len(partitions)),
            "first_pair_precedence": True,
        },
    )

    if delete_partitions and "partitions" in zroot:
        del zroot["partitions"]
    return n_pairs


class R2ZarrPartitionReader:
    """Chunked reader for one normalized r2 Zarr partition."""

    def __init__(
        self,
        root: Path,
        name: str,
        start: int | None = None,
        end: int | None = None,
        *,
        owned: bool = False,
    ) -> None:
        self.root = root
        self.name = name
        self.start = None if start is None else int(start)
        self.end = None if end is None else int(end)
        self.owned = bool(owned)
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
            if self.owned:
                self._group = root_group[_OWNED_GROUP]
            else:
                if self.start is None or self.end is None:
                    raise ValueError("partition reader requires start and end")
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
        return int(self.group.attrs.get("n_pairs", self.group["r2"].shape[0]))

    def read_loci(self) -> np.ndarray:
        """Read sorted unique ``lo`` loci."""
        group = self.group
        lo_values = _position_array(group["lo_values"][:]).astype(np.int64, copy=False)
        diag_idx = np.asarray(group["diag_idx"][:], dtype=np.int64)
        if diag_idx.size == 0:
            return lo_values
        positions = _position_array(group["positions"][:]).astype(np.int64, copy=False)
        return np.union1d(lo_values, positions[diag_idx]).astype(np.int64, copy=False)

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
        diag_idx = np.asarray(group["diag_idx"][:], dtype=np.int64)
        diag_pos = (
            positions[diag_idx]
            if diag_idx.size
            else np.array([], dtype=positions.dtype)
        )
        loci = np.union1d(
            lo_values[
                int(np.searchsorted(lo_values, lo_min, side="left")) : int(
                    np.searchsorted(lo_values, lo_max, side="right")
                )
            ],
            diag_pos[
                int(np.searchsorted(diag_pos, lo_min, side="left")) : int(
                    np.searchsorted(diag_pos, lo_max, side="right")
                )
            ],
        )
        if loci.size == 0:
            return
        chunk_rows = max(1, int(chunk_rows))
        out_lo: list[np.ndarray] = []
        out_hi: list[np.ndarray] = []
        out_r2: list[np.ndarray] = []
        pending = 0

        def flush() -> R2RowChunk | None:
            nonlocal out_lo, out_hi, out_r2, pending
            if pending == 0:
                return None
            chunk = R2RowChunk(
                lo=np.concatenate(out_lo).astype(np.int64, copy=False),
                hi=np.concatenate(out_hi).astype(np.int64, copy=False),
                r2=np.concatenate(out_r2).astype(np.float64, copy=False),
            )
            out_lo = []
            out_hi = []
            out_r2 = []
            pending = 0
            return chunk

        diag_loci = set(int(pos) for pos in diag_pos.tolist())
        for locus in loci:
            locus_int = int(locus)
            if locus_int in diag_loci:
                out_lo.append(np.array([locus_int], dtype=np.int64))
                out_hi.append(np.array([locus_int], dtype=np.int64))
                out_r2.append(np.array([1.0], dtype=np.float64))
                pending += 1
            value_idx = int(np.searchsorted(lo_values, locus_int, side="left"))
            if value_idx < lo_values.size and int(lo_values[value_idx]) == locus_int:
                left = int(lo_offsets[value_idx])
                right = int(lo_offsets[value_idx + 1])
                lo_idx = int(np.searchsorted(positions, locus_int, side="left"))
                for chunk_start in range(left, right, chunk_rows):
                    chunk_stop = min(chunk_start + chunk_rows, right)
                    hi_delta = np.asarray(
                        group["hi_delta"][chunk_start:chunk_stop], dtype=np.int64
                    )
                    row_count = chunk_stop - chunk_start
                    out_lo.append(np.full(row_count, locus_int, dtype=np.int64))
                    out_hi.append(positions[lo_idx + hi_delta].astype(np.int64))
                    out_r2.append(
                        np.asarray(
                            group["r2"][chunk_start:chunk_stop], dtype=np.float64
                        )
                    )
                    pending += row_count
                    if pending >= chunk_rows:
                        chunk = flush()
                        if chunk is not None:
                            yield chunk
            if pending >= chunk_rows:
                chunk = flush()
                if chunk is not None:
                    yield chunk
        chunk = flush()
        if chunk is not None:
            yield chunk

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


def open_r2_zarr_owned_reader(root: Path, name: str) -> R2ZarrPartitionReader:
    """Return a normalized chromosome-level owned r2 Zarr reader."""
    return R2ZarrPartitionReader(root, name, owned=True)
