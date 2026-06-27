"""Array-backed covariance helpers for metric and local search."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ldetect2._util.logging import log_debug
from ldetect2.io.covariance_hdf5 import open_covariance_reader
from ldetect2.io.partitions import CovarianceStore

_DEFAULT_CHUNK_ROWS = 1_000_000


@dataclass(frozen=True)
class CovariancePartition:
    """Raw covariance arrays for one partition."""

    start: int
    end: int
    i_pos: np.ndarray
    j_pos: np.ndarray
    shrink_ld: np.ndarray


@dataclass(frozen=True)
class LocalSearchPartition:
    """Canonical covariance rows and diagonals for array local search.

    Rows use lower/upper endpoints, are sorted by ``(lo, hi)``, and keep the
    first value for duplicate endpoint pairs to match the legacy array path.
    ``loci`` contains the sorted unique row ``lo`` values and is used to build
    active local-search segment boundaries without rebuilding full row arrays.
    """

    start: int
    end: int
    source_row_count: int
    lo: np.ndarray
    hi: np.ndarray
    shrink_ld: np.ndarray
    loci: np.ndarray
    diag_pos: np.ndarray
    diag_val: np.ndarray


@dataclass(frozen=True)
class ChromosomeCovariance:
    """Chromosome-level covariance cache for vector and metric calculations."""

    loci: np.ndarray
    i_pos: np.ndarray
    j_pos: np.ndarray
    r2: np.ndarray
    partitions: tuple[tuple[int, int], ...]
    partition_arrays: tuple[CovariancePartition, ...]


CovarianceArrays = ChromosomeCovariance


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


def _load_partition_arrays(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Covariance partition {path} is missing. Array-backed covariance "
            "operations require HDF5 covariance partitions; regenerate covariance "
            "with `ldetect2 run` or `ldetect2 calc-covariance`."
        )

    start, end = _partition_bounds_from_path(path)
    with open_covariance_reader(path, start, end) as reader:
        rows = reader.read_all()
        return (
            _position_array(rows.lo),
            _position_array(rows.hi),
            np.asarray(rows.shrink_ld, dtype=np.float64),
        )


def _partition_bounds_from_path(path) -> tuple[int, int]:
    try:
        parts = path.stem.split(".")
        return int(parts[-2]), int(parts[-1])
    except Exception as exc:
        raise ValueError(f"Cannot infer partition bounds from {path}") from exc


def local_search_partition(partition: CovariancePartition) -> LocalSearchPartition:
    """Return a sorted, deduplicated view of one partition for local search."""
    lo, hi, shrink = canonical_local_search_rows(
        partition.i_pos,
        partition.j_pos,
        partition.shrink_ld,
    )
    diag_mask = lo == hi
    return LocalSearchPartition(
        start=partition.start,
        end=partition.end,
        source_row_count=int(partition.i_pos.size),
        lo=lo,
        hi=hi,
        shrink_ld=shrink,
        loci=_unique_sorted(lo),
        diag_pos=lo[diag_mask],
        diag_val=shrink[diag_mask],
    )


def _unique_sorted(values: np.ndarray) -> np.ndarray:
    """Return unique values from an already sorted array as int64 positions."""
    if values.size == 0:
        return np.array([], dtype=np.int64)
    if values.size == 1:
        return values.astype(np.int64, copy=False)
    keep = np.ones(values.size, dtype=bool)
    keep[1:] = values[1:] != values[:-1]
    return values[keep].astype(np.int64, copy=False)


def canonical_local_search_rows(
    i_pos: np.ndarray,
    j_pos: np.ndarray,
    shrink_ld: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize covariance rows as sorted unique lower/upper endpoints.

    Duplicate endpoint pairs keep the first value from the input arrays.  The
    returned position dtype follows :func:`_position_array`, preserving
    ``int32`` when every position fits.
    """
    i_pos = _position_array(i_pos)
    j_pos = _position_array(j_pos)
    if i_pos.size == 0:
        dtype = np.result_type(i_pos.dtype, j_pos.dtype)
        return (
            np.array([], dtype=dtype),
            np.array([], dtype=dtype),
            np.array([], dtype=np.float64),
        )

    if np.all(i_pos <= j_pos):
        lo = i_pos.astype(np.result_type(i_pos.dtype, j_pos.dtype), copy=False)
        hi = j_pos.astype(lo.dtype, copy=False)
    elif np.all(j_pos <= i_pos):
        hi = i_pos.astype(np.result_type(i_pos.dtype, j_pos.dtype), copy=False)
        lo = j_pos.astype(hi.dtype, copy=False)
    else:
        dtype = np.result_type(i_pos.dtype, j_pos.dtype)
        lo = np.minimum(i_pos, j_pos).astype(dtype, copy=False)
        hi = np.maximum(i_pos, j_pos).astype(dtype, copy=False)
    shrink = np.asarray(shrink_ld, dtype=np.float64)

    original_order = np.arange(lo.size, dtype=np.int64)
    order = np.lexsort((original_order, hi, lo))
    lo = lo[order]
    hi = hi[order]
    shrink = shrink[order]

    keep = np.ones(lo.size, dtype=bool)
    keep[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    return lo[keep], hi[keep], shrink[keep]


def _slice_arrays_to_range(
    i_pos: np.ndarray,
    j_pos: np.ndarray,
    shrink: np.ndarray,
    snp_first: int | None,
    snp_last: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if snp_first is None and snp_last is None:
        return i_pos, j_pos, shrink

    if np.all(i_pos <= j_pos):
        lo = i_pos
        hi = j_pos
    else:
        lo = np.minimum(i_pos, j_pos)
        hi = np.maximum(i_pos, j_pos)
    mask = np.ones(i_pos.size, dtype=bool)
    if snp_first is not None:
        mask &= lo >= snp_first
    if snp_last is not None:
        mask &= hi <= snp_last
    return i_pos[mask], j_pos[mask], shrink[mask]


def load_covariance_arrays(
    name: str,
    store: CovarianceStore,
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
) -> ChromosomeCovariance:
    """Backward-compatible wrapper for the chromosome covariance cache."""
    return load_chromosome_covariance(name, store, partitions, snp_first, snp_last)


def load_covariance_partitions(
    name: str,
    store: CovarianceStore,
    partitions: list[tuple[int, int]],
    snp_first: int | None = None,
    snp_last: int | None = None,
) -> tuple[CovariancePartition, ...]:
    """Load raw covariance arrays for the requested partitions only."""
    return tuple(
        _load_chromosome_partitions(
            name,
            store,
            partitions,
            snp_first=snp_first,
            snp_last=snp_last,
        )
    )


def load_chromosome_covariance(
    name: str,
    store: CovarianceStore,
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
) -> ChromosomeCovariance:
    """Load relevant covariance rows and precompute r² values.

    The returned arrays include only rows whose endpoints are both inside
    ``[snp_first, snp_last]`` and whose diagonal values are positive.
    Diagonal rows are kept in ``loci`` discovery but excluded from pair arrays.
    """
    partition_arrays = _load_chromosome_partitions(name, store, partitions)
    loci, i_pos, j_pos, r2 = _metric_arrays_from_partitions(
        partition_arrays, partitions, snp_first, snp_last
    )
    return ChromosomeCovariance(
        loci=loci,
        i_pos=i_pos,
        j_pos=j_pos,
        r2=r2,
        partitions=tuple(partitions),
        partition_arrays=tuple(partition_arrays),
    )


def load_metric_covariance(
    name: str,
    store: CovarianceStore,
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
) -> ChromosomeCovariance:
    """Load only the arrays needed for metric calculations.

    Unlike :func:`load_chromosome_covariance`, this does not retain raw
    partition arrays.  It reads partitions in two passes: first to collect
    diagonal values, then to emit normalized pair rows.
    """
    loci, i_pos, j_pos, r2 = _metric_arrays_from_files(
        name, store, partitions, snp_first, snp_last
    )
    return ChromosomeCovariance(
        loci=loci,
        i_pos=i_pos,
        j_pos=j_pos,
        r2=r2,
        partitions=tuple(partitions),
        partition_arrays=(),
    )


def metric_from_files(
    name: str,
    store: CovarianceStore,
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
    breakpoints: list[int],
) -> dict:
    """Compute the LD block metric without materializing chromosome-wide pairs.

    The calculation streams covariance partitions twice: first to collect loci
    and diagonal values, then to normalize eligible pairs and accumulate only
    pairs that cross a requested breakpoint.  This keeps peak memory bounded by
    the largest partition plus temporary masks instead of a full-chromosome
    normalized pair array.
    """
    bp = np.asarray(breakpoints, dtype=np.int64)
    if bp.size == 0:
        return {"sum": 0.0, "N_nonzero": 0, "N_zero": 0.0}

    loci_chunks: list[np.ndarray] = []
    diag_pos_chunks: list[np.ndarray] = []
    diag_val_chunks: list[np.ndarray] = []
    index_read_seconds = 0.0
    row_read_seconds = 0.0
    normalize_seconds = 0.0
    crossing_seconds = 0.0
    rows_read = 0
    pair_rows = 0
    normalized_rows = 0
    crossing_rows = 0

    for p_index, (start, end) in enumerate(partitions):
        path = store.partition_path(name, start, end)
        lower_min, lower_max, include_lower_min = _owned_bounds(
            partitions, p_index, snp_first, snp_last
        )
        with open_covariance_reader(path, start, end) as reader:
            index_start = time.perf_counter()
            diag_pos, diag_val = reader.read_diagonal()
            diag_in_range = (
                (diag_pos >= snp_first)
                & (diag_pos <= snp_last)
                & (diag_pos >= lower_min if include_lower_min else diag_pos > lower_min)
                & (diag_pos <= lower_max)
            )
            if np.any(diag_in_range):
                diag_pos_chunks.append(diag_pos[diag_in_range])
                diag_val_chunks.append(diag_val[diag_in_range])
            index_read_seconds += time.perf_counter() - index_start
            for chunk in reader.iter_owned_rows(
                lower_min,
                lower_max,
                snp_first,
                snp_last,
                _DEFAULT_CHUNK_ROWS,
                include_lower_min=include_lower_min,
            ):
                rows_read += int(chunk.lo.size)
                loci_chunks.append(np.unique(np.concatenate((chunk.lo, chunk.hi))))

    if not loci_chunks:
        return {"sum": 0.0, "N_nonzero": 0, "N_zero": 0.0}

    loci = np.unique(np.concatenate(loci_chunks))
    n_zero = _metric_n_zero(loci, bp)
    if not diag_pos_chunks:
        return {"sum": 0.0, "N_nonzero": 0, "N_zero": n_zero}

    diag_pos = np.concatenate(diag_pos_chunks)
    diag_val = np.concatenate(diag_val_chunks)
    order = np.argsort(diag_pos, kind="stable")
    diag_pos = diag_pos[order]
    diag_val = diag_val[order]
    unique_diag_pos, unique_idx = np.unique(diag_pos, return_index=True)
    unique_diag_val = diag_val[unique_idx]

    total_sum = 0.0
    n_nonzero = 0
    for p_index, (start, end) in enumerate(partitions):
        path = store.partition_path(name, start, end)
        lower_min, lower_max, include_lower_min = _owned_bounds(
            partitions, p_index, snp_first, snp_last
        )
        with open_covariance_reader(path, start, end) as reader:
            chunk_iter = reader.iter_owned_rows(
                lower_min,
                lower_max,
                snp_first,
                snp_last,
                _DEFAULT_CHUNK_ROWS,
                include_lower_min=include_lower_min,
            )
            while True:
                read_start = time.perf_counter()
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                row_read_seconds += time.perf_counter() - read_start
                rows_read += int(chunk.lo.size)
                pair_mask = chunk.lo < chunk.hi
                if not np.any(pair_mask):
                    continue
                pair_i = chunk.lo[pair_mask]
                pair_j = chunk.hi[pair_mask]
                pair_s = chunk.shrink_ld[pair_mask]
                pair_rows += int(pair_i.size)

                normalize_start = time.perf_counter()
                diag_i_idx = np.searchsorted(unique_diag_pos, pair_i)
                diag_j_idx = np.searchsorted(unique_diag_pos, pair_j)
                has_diag = (diag_i_idx < unique_diag_pos.size) & (
                    diag_j_idx < unique_diag_pos.size
                )
                safe_i_idx = np.minimum(diag_i_idx, unique_diag_pos.size - 1)
                safe_j_idx = np.minimum(diag_j_idx, unique_diag_pos.size - 1)
                has_diag &= (unique_diag_pos[safe_i_idx] == pair_i) & (
                    unique_diag_pos[safe_j_idx] == pair_j
                )
                if not np.any(has_diag):
                    normalize_seconds += time.perf_counter() - normalize_start
                    continue

                pair_i = pair_i[has_diag]
                pair_j = pair_j[has_diag]
                pair_s = pair_s[has_diag]
                diag_i = unique_diag_val[diag_i_idx[has_diag]]
                diag_j = unique_diag_val[diag_j_idx[has_diag]]

                positive = (diag_i > 0.0) & (diag_j > 0.0)
                if not np.any(positive):
                    normalize_seconds += time.perf_counter() - normalize_start
                    continue

                pair_i = pair_i[positive]
                pair_j = pair_j[positive]
                pair_s = pair_s[positive]
                diag_i = diag_i[positive]
                diag_j = diag_j[positive]
                normalized_rows += int(pair_i.size)
                normalize_seconds += time.perf_counter() - normalize_start

                crossing_start = time.perf_counter()
                i_blocks = np.searchsorted(bp, pair_i, side="left")
                j_blocks = np.searchsorted(bp, pair_j, side="left")
                crossing = i_blocks != j_blocks
                if not np.any(crossing):
                    crossing_seconds += time.perf_counter() - crossing_start
                    continue

                r2 = (
                    pair_s[crossing]
                    * pair_s[crossing]
                    / (diag_i[crossing] * diag_j[crossing])
                )
                total_sum += float(np.sum(r2))
                n_nonzero += int(np.count_nonzero(crossing))
                crossing_rows += int(np.count_nonzero(crossing))
                crossing_seconds += time.perf_counter() - crossing_start

    log_debug(
        "metric_from_files profile "
        f"partitions={len(partitions)} rows_read={rows_read} pair_rows={pair_rows} "
        f"normalized_rows={normalized_rows} crossing_rows={crossing_rows} "
        f"index_read_seconds={index_read_seconds:.6f} "
        f"row_read_seconds={row_read_seconds:.6f} "
        f"normalize_seconds={normalize_seconds:.6f} "
        f"crossing_seconds={crossing_seconds:.6f}"
    )
    return {"sum": total_sum, "N_nonzero": n_nonzero, "N_zero": n_zero}


def _owned_bounds(
    partitions: list[tuple[int, int]],
    p_index: int,
    snp_first: int,
    snp_last: int,
) -> tuple[int, int, bool]:
    start = partitions[p_index][0]
    lower_min = snp_first if p_index == 0 else start
    lower_max = (
        partitions[p_index + 1][0] if p_index + 1 < len(partitions) else snp_last
    )
    return lower_min, lower_max, p_index == 0


def _deduplicate_metric_pairs(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort and deduplicate metric pairs, preserving first-pair semantics."""
    if pair_i.size <= 1:
        return pair_i, pair_j, pair_s
    original_order = np.arange(pair_i.size, dtype=np.int64)
    order = np.lexsort((original_order, pair_j, pair_i))
    pair_i = pair_i[order]
    pair_j = pair_j[order]
    pair_s = pair_s[order]
    keep = np.ones(pair_i.size, dtype=bool)
    keep[1:] = (pair_i[1:] != pair_i[:-1]) | (pair_j[1:] != pair_j[:-1])
    return pair_i[keep], pair_j[keep], pair_s[keep]


def _metric_n_zero(loci: np.ndarray, breakpoints: np.ndarray) -> float:
    """Return the block-area denominator used by array-backed metrics."""
    metric_loci = loci[loci <= breakpoints[-1]]
    block_ids = np.searchsorted(breakpoints, metric_loci, side="left")
    block_widths = np.bincount(block_ids, minlength=len(breakpoints)).astype(
        np.float64
    )

    if block_widths.size <= 1 or breakpoints.size <= 1:
        return 0.0
    total = float(block_widths.sum())
    return float((total * total - np.sum(block_widths * block_widths)) / 2.0)


def _load_chromosome_partitions(
    name: str,
    store: CovarianceStore,
    partitions: list[tuple[int, int]],
    snp_first: int | None = None,
    snp_last: int | None = None,
) -> list[CovariancePartition]:
    partition_arrays: list[CovariancePartition] = []
    for start, end in partitions:
        path = store.partition_path(name, start, end)
        i_pos, j_pos, shrink = _load_partition_arrays(path)
        i_pos, j_pos, shrink = _slice_arrays_to_range(
            i_pos,
            j_pos,
            shrink,
            snp_first,
            snp_last,
        )
        partition_arrays.append(
            CovariancePartition(
                start=start,
                end=end,
                i_pos=i_pos,
                j_pos=j_pos,
                shrink_ld=shrink,
            )
        )
    return partition_arrays


def _owned_in_range_mask(
    i_pos: np.ndarray,
    j_pos: np.ndarray,
    partitions: list[tuple[int, int]],
    p_index: int,
    snp_first: int,
    snp_last: int,
) -> np.ndarray:
    start = partitions[p_index][0]
    lower_min = snp_first if p_index == 0 else start
    lower_max = (
        partitions[p_index + 1][0] if p_index + 1 < len(partitions) else snp_last
    )
    lower_owned = i_pos >= lower_min if p_index == 0 else i_pos > lower_min
    lower_owned &= i_pos <= lower_max
    return (
        (i_pos >= snp_first)
        & (i_pos <= snp_last)
        & (j_pos >= snp_first)
        & (j_pos <= snp_last)
        & lower_owned
    )


def _metric_arrays_from_files(
    name: str,
    store: CovarianceStore,
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diag_pos_chunks: list[np.ndarray] = []
    diag_val_chunks: list[np.ndarray] = []
    loci_chunks: list[np.ndarray] = []

    for p_index, (start, end) in enumerate(partitions):
        path = store.partition_path(name, start, end)
        i_pos, j_pos, shrink = _load_partition_arrays(path)
        in_range = _owned_in_range_mask(
            i_pos, j_pos, partitions, p_index, snp_first, snp_last
        )
        if not np.any(in_range):
            continue

        i_pos = i_pos[in_range]
        j_pos = j_pos[in_range]
        shrink = shrink[in_range]
        loci_chunks.append(np.unique(np.concatenate((i_pos, j_pos))))

        diag_mask = i_pos == j_pos
        if np.any(diag_mask):
            diag_pos_chunks.append(i_pos[diag_mask])
            diag_val_chunks.append(shrink[diag_mask])

    if not loci_chunks:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return empty_i, empty_i, empty_i, empty_f

    loci = np.unique(np.concatenate(loci_chunks))
    if not diag_pos_chunks:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return loci, empty_i, empty_i, empty_f

    diag_pos = np.concatenate(diag_pos_chunks)
    diag_val = np.concatenate(diag_val_chunks)
    order = np.argsort(diag_pos, kind="stable")
    diag_pos = diag_pos[order]
    diag_val = diag_val[order]
    unique_diag_pos, unique_idx = np.unique(diag_pos, return_index=True)
    unique_diag_val = diag_val[unique_idx]

    i_chunks: list[np.ndarray] = []
    j_chunks: list[np.ndarray] = []
    r2_chunks: list[np.ndarray] = []

    for p_index, (start, end) in enumerate(partitions):
        path = store.partition_path(name, start, end)
        i_pos, j_pos, shrink = _load_partition_arrays(path)
        in_range = _owned_in_range_mask(
            i_pos, j_pos, partitions, p_index, snp_first, snp_last
        )
        pair_mask = in_range & (i_pos < j_pos)
        if not np.any(pair_mask):
            continue

        pair_i = i_pos[pair_mask]
        pair_j = j_pos[pair_mask]
        pair_s = shrink[pair_mask]
        diag_i_idx = np.searchsorted(unique_diag_pos, pair_i)
        diag_j_idx = np.searchsorted(unique_diag_pos, pair_j)
        has_diag = (diag_i_idx < unique_diag_pos.size) & (
            diag_j_idx < unique_diag_pos.size
        )
        safe_i_idx = np.minimum(diag_i_idx, unique_diag_pos.size - 1)
        safe_j_idx = np.minimum(diag_j_idx, unique_diag_pos.size - 1)
        has_diag &= (unique_diag_pos[safe_i_idx] == pair_i) & (
            unique_diag_pos[safe_j_idx] == pair_j
        )
        if not np.any(has_diag):
            continue

        pair_i = pair_i[has_diag]
        pair_j = pair_j[has_diag]
        pair_s = pair_s[has_diag]
        diag_i = unique_diag_val[diag_i_idx[has_diag]]
        diag_j = unique_diag_val[diag_j_idx[has_diag]]

        positive = (diag_i > 0.0) & (diag_j > 0.0)
        if not np.any(positive):
            continue
        i_chunks.append(pair_i[positive])
        j_chunks.append(pair_j[positive])
        r2_chunks.append(
            pair_s[positive]
            * pair_s[positive]
            / (diag_i[positive] * diag_j[positive])
        )

    if not i_chunks:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return loci, empty_i, empty_i, empty_f

    pair_i = np.concatenate(i_chunks)
    pair_j = np.concatenate(j_chunks)
    r2 = np.concatenate(r2_chunks)
    order = np.lexsort((pair_j, pair_i))
    pair_i = pair_i[order]
    pair_j = pair_j[order]
    r2 = r2[order]
    keep = np.ones(pair_i.size, dtype=bool)
    keep[1:] = (pair_i[1:] != pair_i[:-1]) | (pair_j[1:] != pair_j[:-1])
    return loci, pair_i[keep], pair_j[keep], r2[keep]


def _metric_arrays_from_partitions(
    partition_arrays: list[CovariancePartition],
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i_chunks: list[np.ndarray] = []
    j_chunks: list[np.ndarray] = []
    s_chunks: list[np.ndarray] = []
    loci_chunks: list[np.ndarray] = []

    for p_index, partition in enumerate(partition_arrays):
        start = partition.start
        i_pos = partition.i_pos
        j_pos = partition.j_pos
        shrink = partition.shrink_ld
        lower_min = snp_first if p_index == 0 else start
        lower_max = (
            partitions[p_index + 1][0]
            if p_index + 1 < len(partitions)
            else snp_last
        )
        lower_owned = i_pos >= lower_min if p_index == 0 else i_pos > lower_min
        lower_owned &= i_pos <= lower_max
        in_range = (
            (i_pos >= snp_first)
            & (i_pos <= snp_last)
            & (j_pos >= snp_first)
            & (j_pos <= snp_last)
            & lower_owned
        )
        if not np.any(in_range):
            continue

        i_pos = i_pos[in_range]
        j_pos = j_pos[in_range]
        shrink = shrink[in_range]

        loci_chunks.append(i_pos)
        loci_chunks.append(j_pos)
        i_chunks.append(i_pos)
        j_chunks.append(j_pos)
        s_chunks.append(shrink)

    if not i_chunks:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return empty_i, empty_i, empty_i, empty_f

    raw_i = np.concatenate(i_chunks)
    raw_j = np.concatenate(j_chunks)
    raw_s = np.concatenate(s_chunks)
    loci = np.unique(np.concatenate(loci_chunks))

    diag_mask = raw_i == raw_j
    diag_pos = raw_i[diag_mask]
    diag_val = raw_s[diag_mask]
    if diag_pos.size == 0:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return loci, empty_i, empty_i, empty_f

    order = np.argsort(diag_pos, kind="stable")
    diag_pos = diag_pos[order]
    diag_val = diag_val[order]
    unique_diag_pos, unique_idx = np.unique(diag_pos, return_index=True)
    unique_diag_val = diag_val[unique_idx]

    pair_mask = raw_i < raw_j
    pair_i = raw_i[pair_mask]
    pair_j = raw_j[pair_mask]
    pair_s = raw_s[pair_mask]

    diag_i_idx = np.searchsorted(unique_diag_pos, pair_i)
    diag_j_idx = np.searchsorted(unique_diag_pos, pair_j)
    has_diag = (diag_i_idx < unique_diag_pos.size) & (
        diag_j_idx < unique_diag_pos.size
    )
    safe_i_idx = np.minimum(diag_i_idx, unique_diag_pos.size - 1)
    safe_j_idx = np.minimum(diag_j_idx, unique_diag_pos.size - 1)
    has_diag &= (unique_diag_pos[safe_i_idx] == pair_i) & (
        unique_diag_pos[safe_j_idx] == pair_j
    )

    pair_i = pair_i[has_diag]
    pair_j = pair_j[has_diag]
    pair_s = pair_s[has_diag]
    diag_i = unique_diag_val[diag_i_idx[has_diag]]
    diag_j = unique_diag_val[diag_j_idx[has_diag]]

    positive = (diag_i > 0.0) & (diag_j > 0.0)
    pair_i = pair_i[positive]
    pair_j = pair_j[positive]
    pair_s = pair_s[positive]
    diag_i = diag_i[positive]
    diag_j = diag_j[positive]

    if pair_i.size:
        order = np.lexsort((pair_j, pair_i))
        pair_i = pair_i[order]
        pair_j = pair_j[order]
        pair_s = pair_s[order]
        diag_i = diag_i[order]
        diag_j = diag_j[order]
        keep = np.ones(pair_i.size, dtype=bool)
        keep[1:] = (pair_i[1:] != pair_i[:-1]) | (pair_j[1:] != pair_j[:-1])
        pair_i = pair_i[keep]
        pair_j = pair_j[keep]
        pair_s = pair_s[keep]
        diag_i = diag_i[keep]
        diag_j = diag_j[keep]

    r2 = pair_s * pair_s / (diag_i * diag_j)
    return loci, pair_i, pair_j, r2


def metric_from_arrays(
    cov: CovarianceArrays,
    breakpoints: list[int],
) -> dict:
    """Compute the LD block metric from array-backed covariance data."""
    if cov.loci.size == 0:
        return {"sum": 0.0, "N_nonzero": 0, "N_zero": 0.0}

    bp = np.asarray(breakpoints, dtype=np.int64)
    if bp.size == 0:
        return {"sum": 0.0, "N_nonzero": 0, "N_zero": 0.0}

    metric_loci = cov.loci[cov.loci <= bp[-1]]
    block_ids = np.searchsorted(bp, metric_loci, side="left")
    block_widths = np.bincount(block_ids, minlength=len(bp)).astype(np.float64)

    if block_widths.size <= 1 or bp.size <= 1:
        n_zero = 0.0
    else:
        total = float(block_widths.sum())
        n_zero = float((total * total - np.sum(block_widths * block_widths)) / 2.0)

    if cov.i_pos.size == 0:
        return {"sum": 0.0, "N_nonzero": 0, "N_zero": n_zero}

    i_blocks = np.searchsorted(bp, cov.i_pos, side="left")
    j_blocks = np.searchsorted(bp, cov.j_pos, side="left")
    crossing = i_blocks != j_blocks

    return {
        "sum": float(np.sum(cov.r2[crossing])),
        "N_nonzero": int(np.count_nonzero(crossing)),
        "N_zero": n_zero,
    }
