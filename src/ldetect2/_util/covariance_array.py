"""Array-backed covariance helpers for metric and local search."""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ldetect2._util.logging import log_debug
from ldetect2.io.covariance_hdf5 import open_covariance_reader
from ldetect2.io.partitions import CovarianceStore
from ldetect2.io.r2_nocache import (
    R2NoCacheConfig,
    R2NoCacheProfile,
    iter_prepared_r2_nocache_partition_rows,
    prepare_r2_nocache_partition,
)
from ldetect2.io.r2_zarr import (
    open_r2_zarr_owned_reader,
    open_r2_zarr_reader,
    validate_r2_zarr_owned_cache,
)

_DEFAULT_CHUNK_ROWS = 1_000_000
_METRIC_WORKER_BREAKPOINTS: np.ndarray | None = None
_METRIC_WORKER_DIAG_POS: np.ndarray | None = None
_METRIC_WORKER_DIAG_VAL: np.ndarray | None = None

try:
    from numba import njit

    _njit_fallback = njit(cache=True)
except ImportError:

    def _njit_fallback(fn):  # type: ignore[misc]
        return fn


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


@dataclass(frozen=True)
class _MetricPartitionResult:
    p_index: int
    total_sum: float
    n_nonzero: int
    rows_read: int
    pair_rows: int
    normalized_rows: int
    crossing_rows: int
    row_read_seconds: float
    normalize_seconds: float
    crossing_seconds: float


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
    workers: int = 1,
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
    loci_index_seconds = 0.0
    diag_read_seconds = 0.0
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
            loci = reader.read_loci()
            lower_owned = (
                loci >= lower_min if include_lower_min else loci > lower_min
            )
            loci_in_range = (
                (loci >= snp_first)
                & (loci <= snp_last)
                & lower_owned
                & (loci <= lower_max)
            )
            if np.any(loci_in_range):
                loci_chunks.append(loci[loci_in_range])
            loci_index_seconds += time.perf_counter() - index_start

            diag_start = time.perf_counter()
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
            diag_read_seconds += time.perf_counter() - diag_start
            index_read_seconds += time.perf_counter() - index_start

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
    worker_wait_seconds = 0.0
    worker_merge_seconds = 0.0
    metric_workers = max(1, int(workers))
    if metric_workers > 1 and len(partitions) > 1:
        worker_start = time.perf_counter()
        results = _metric_partition_results_parallel(
            name=name,
            store=store,
            partitions=partitions,
            snp_first=snp_first,
            snp_last=snp_last,
            breakpoints=bp,
            unique_diag_pos=unique_diag_pos,
            unique_diag_val=unique_diag_val,
            workers=metric_workers,
        )
        worker_wait_seconds += time.perf_counter() - worker_start
    else:
        results = [
            _metric_partition_result(
                name=name,
                store=store,
                partitions=partitions,
                p_index=p_index,
                start=start,
                end=end,
                snp_first=snp_first,
                snp_last=snp_last,
                breakpoints=bp,
                unique_diag_pos=unique_diag_pos,
                unique_diag_val=unique_diag_val,
            )
            for p_index, (start, end) in enumerate(partitions)
        ]

    for result in results:
        merge_start = time.perf_counter()
        total_sum += result.total_sum
        n_nonzero += result.n_nonzero
        rows_read += result.rows_read
        pair_rows += result.pair_rows
        normalized_rows += result.normalized_rows
        crossing_rows += result.crossing_rows
        row_read_seconds += result.row_read_seconds
        normalize_seconds += result.normalize_seconds
        crossing_seconds += result.crossing_seconds
        worker_merge_seconds += time.perf_counter() - merge_start

    log_debug(
        "metric_from_files profile "
        f"partitions={len(partitions)} rows_read={rows_read} pair_rows={pair_rows} "
        f"normalized_rows={normalized_rows} crossing_rows={crossing_rows} "
        f"metric_workers={metric_workers} "
        f"index_read_seconds={index_read_seconds:.6f} "
        f"loci_index_seconds={loci_index_seconds:.6f} "
        f"diag_read_seconds={diag_read_seconds:.6f} "
        f"worker_wait_seconds={worker_wait_seconds:.6f} "
        f"worker_merge_seconds={worker_merge_seconds:.6f} "
        f"row_read_seconds={row_read_seconds:.6f} "
        f"normalize_seconds={normalize_seconds:.6f} "
        f"crossing_seconds={crossing_seconds:.6f}"
    )
    return {"sum": total_sum, "N_nonzero": n_nonzero, "N_zero": n_zero}


def metric_from_r2_zarr_files(
    name: str,
    store: CovarianceStore,
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
    breakpoints: list[int],
    workers: int = 1,
) -> dict:
    """Compute the LD block metric from normalized r2 Zarr partitions."""
    bp = np.asarray(breakpoints, dtype=np.int64)
    if bp.size == 0:
        return {"sum": 0.0, "N_nonzero": 0, "N_zero": 0.0}

    loci_chunks: list[np.ndarray] = []
    index_read_seconds = 0.0
    row_read_seconds = 0.0
    crossing_seconds = 0.0
    rows_read = 0
    pair_rows = 0
    crossing_rows = 0

    has_owned_cache = validate_r2_zarr_owned_cache(store.root, name)
    if has_owned_cache:
        with open_r2_zarr_owned_reader(store.root, name) as reader:
            index_start = time.perf_counter()
            loci = reader.read_loci()
            loci_in_range = (loci >= snp_first) & (loci <= snp_last)
            if np.any(loci_in_range):
                loci_chunks.append(loci[loci_in_range])
            index_read_seconds += time.perf_counter() - index_start
    else:
        for p_index, (start, end) in enumerate(partitions):
            lower_min, lower_max, include_lower_min = _owned_bounds(
                partitions, p_index, snp_first, snp_last
            )
            with open_r2_zarr_reader(store.root, name, start, end) as reader:
                index_start = time.perf_counter()
                loci = reader.read_loci()
                lower_owned = (
                    loci >= lower_min if include_lower_min else loci > lower_min
                )
                loci_in_range = (
                    (loci >= snp_first)
                    & (loci <= snp_last)
                    & lower_owned
                    & (loci <= lower_max)
                )
                if np.any(loci_in_range):
                    loci_chunks.append(loci[loci_in_range])
                index_read_seconds += time.perf_counter() - index_start

    if not loci_chunks:
        return {"sum": 0.0, "N_nonzero": 0, "N_zero": 0.0}

    loci = np.unique(np.concatenate(loci_chunks))
    n_zero = _metric_n_zero(loci, bp)
    total_sum = 0.0
    n_nonzero = 0

    if workers > 1 and len(partitions) > 1:
        log_debug(
            "metric_from_r2_zarr_files currently streams in one process; "
            f"ignoring workers={workers}"
        )

    reader_contexts = []
    for p_index, (start, end) in enumerate(partitions):
        lower_min, lower_max, include_lower_min = _owned_bounds(
            partitions, p_index, snp_first, snp_last
        )
        if has_owned_cache:
            reader_contexts.append(
                (
                    (lower_min, lower_max, include_lower_min, end),
                    open_r2_zarr_owned_reader(store.root, name),
                )
            )
        else:
            reader_contexts.append(
                (
                    (lower_min, lower_max, include_lower_min, end),
                    open_r2_zarr_reader(store.root, name, start, end),
                )
            )

    for bounds, reader_context in reader_contexts:
        with reader_context as reader:
            lower_min, lower_max, include_lower_min, partition_end = bounds
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
                in_range = (
                    (chunk.lo >= snp_first)
                    & (chunk.lo <= snp_last)
                    & (chunk.hi >= snp_first)
                    & (chunk.hi <= snp_last)
                    & (chunk.hi <= partition_end)
                )
                pair_mask = in_range & (chunk.lo < chunk.hi)
                if not np.any(pair_mask):
                    continue
                pair_i = chunk.lo[pair_mask]
                pair_j = chunk.hi[pair_mask]
                pair_r2 = chunk.r2[pair_mask]
                pair_rows += int(pair_i.size)

                crossing_start = time.perf_counter()
                i_blocks = np.searchsorted(bp, pair_i, side="left")
                j_blocks = np.searchsorted(bp, pair_j, side="left")
                crossing = i_blocks != j_blocks
                crossing_count = int(np.count_nonzero(crossing))
                if crossing_count:
                    total_sum += float(np.sum(pair_r2[crossing]))
                    n_nonzero += crossing_count
                    crossing_rows += crossing_count
                crossing_seconds += time.perf_counter() - crossing_start

    log_debug(
        "metric_from_r2_zarr_files profile "
        f"partitions={len(partitions)} rows_read={rows_read} "
        f"pair_rows={pair_rows} crossing_rows={crossing_rows} "
        f"index_read_seconds={index_read_seconds:.6f} "
        f"row_read_seconds={row_read_seconds:.6f} "
        f"crossing_seconds={crossing_seconds:.6f}"
    )
    return {"sum": total_sum, "N_nonzero": n_nonzero, "N_zero": n_zero}


@_njit_fallback
def _searchsorted_left_int(values: np.ndarray, target: int) -> int:
    lo = 0
    hi = values.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@_njit_fallback
def _r2_nocache_metric_fused_impl(
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    hap_sums: np.ndarray,
    pos_arr: np.ndarray,
    diag_shrink: np.ndarray,
    j_stop_by_i: np.ndarray,
    breakpoints: np.ndarray,
    lower_min: int,
    lower_max: int,
    snp_first: int,
    snp_last: int,
    include_lower_min: bool,
    ne: float,
    n_ind: float,
    theta: float,
    cutoff: float,
) -> tuple[float, int, int, int, int]:
    n_snps = hap_mat.shape[0]
    n_haps = hap_mat.shape[1]
    n_total = float(n_haps)
    total_sum = 0.0
    n_nonzero = 0
    rows_read = 0
    pair_rows = 0
    crossing_rows = 0

    for i in range(n_snps):
        pos_i = int(pos_arr[i])
        if pos_i < snp_first or pos_i > snp_last or pos_i > lower_max:
            continue
        if include_lower_min:
            if pos_i < lower_min:
                continue
        elif pos_i <= lower_min:
            continue

        diag_i = diag_shrink[i]
        if diag_i <= 0.0:
            continue

        gpos1 = gpos_arr[i]
        j_stop = j_stop_by_i[i]
        n1x = hap_sums[i]
        block_i = _searchsorted_left_int(breakpoints, pos_i)
        for j in range(i, j_stop):
            pos_j = int(pos_arr[j])
            if pos_j < snp_first or pos_j > snp_last:
                continue
            diag_j = diag_shrink[j]
            if diag_j <= 0.0:
                continue

            df = gpos_arr[j] - gpos1
            ee = np.exp(-4.0 * ne * df / (2.0 * n_ind))
            a = hap_mat[i]
            b = hap_mat[j]
            n11 = np.sum(a * b)
            nx1 = hap_sums[j]
            f11 = n11 / n_total
            f1 = n1x / n_total
            f2 = nx1 / n_total
            d_naive = f11 - f1 * f2
            ds2 = (1.0 - theta) ** 2 * d_naive * ee
            if np.abs(ds2) < cutoff:
                continue

            rows_read += 1
            if i == j:
                continue

            pair_rows += 1
            block_j = _searchsorted_left_int(breakpoints, pos_j)
            if block_i == block_j:
                continue
            total_sum += ds2 * ds2 / (diag_i * diag_j)
            n_nonzero += 1
            crossing_rows += 1

    return total_sum, n_nonzero, rows_read, pair_rows, crossing_rows


def metric_from_r2_nocache(
    config: R2NoCacheConfig,
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
    breakpoints: list[int],
    workers: int = 1,
) -> dict:
    """Compute the LD block metric by recomputing normalized r2 rows."""
    bp = np.asarray(breakpoints, dtype=np.int64)
    if bp.size == 0:
        return {"sum": 0.0, "N_nonzero": 0, "N_zero": 0.0}

    loci_parts: list[np.ndarray] = []
    row_seconds = 0.0
    crossing_seconds = 0.0
    rows_read = 0
    pair_rows = 0
    crossing_rows = 0
    nocache_profile = R2NoCacheProfile()

    if workers > 1 and len(partitions) > 1:
        log_debug(
            "metric_from_r2_nocache currently streams in one process; "
            f"ignoring workers={workers}"
        )

    total_sum = 0.0
    n_nonzero = 0

    for p_index, (start, end) in enumerate(partitions):
        lower_min, lower_max, include_lower_min = _owned_bounds(
            partitions, p_index, snp_first, snp_last
        )
        prepared = prepare_r2_nocache_partition(config, start, end)

        if not prepared.has_duplicate_positions:
            owned_loci = prepared.pos_arr[
                (prepared.diag_shrink > 0.0)
                & (prepared.pos_arr >= snp_first)
                & (prepared.pos_arr <= snp_last)
                & (prepared.pos_arr <= lower_max)
                & (
                    (prepared.pos_arr >= lower_min)
                    if include_lower_min
                    else (prepared.pos_arr > lower_min)
                )
            ]
            if owned_loci.size:
                loci_parts.append(owned_loci)
            row_start = time.perf_counter()
            (
                partition_sum,
                partition_n,
                partition_rows,
                partition_pairs,
                partition_crossing,
            ) = _r2_nocache_metric_fused_impl(
                prepared.hap_mat,
                prepared.gpos_arr,
                prepared.hap_sums,
                prepared.pos_arr,
                prepared.diag_shrink,
                prepared.j_stop_by_i,
                bp,
                lower_min,
                lower_max,
                snp_first,
                snp_last,
                include_lower_min,
                prepared.config.ne,
                float(prepared.n_ind),
                prepared.theta,
                prepared.config.cutoff,
            )
            row_elapsed = time.perf_counter() - row_start
            row_seconds += row_elapsed
            prepared.profile.row_generation_seconds += row_elapsed
            prepared.profile.ld_compute_seconds += row_elapsed
            prepared.profile.tile_count += 1
            prepared.profile.max_tile_snps = max(
                prepared.profile.max_tile_snps,
                int(prepared.pos_arr.size),
            )
            prepared.profile.pair_candidates += partition_rows
            prepared.profile.pairs_after_cutoff += partition_pairs
            total_sum += partition_sum
            n_nonzero += partition_n
            rows_read += partition_rows
            pair_rows += partition_pairs
            crossing_rows += partition_crossing
            nocache_profile.absorb(prepared.profile)
            continue

        chunk_iter = iter_prepared_r2_nocache_partition_rows(
            prepared,
            chunk_rows=_DEFAULT_CHUNK_ROWS,
        )
        while True:
            read_start = time.perf_counter()
            try:
                chunk = next(chunk_iter)
            except StopIteration:
                break
            row_seconds += time.perf_counter() - read_start
            lower_owned = (
                chunk.lo >= lower_min if include_lower_min else chunk.lo > lower_min
            )
            owned = (
                (chunk.lo >= snp_first)
                & (chunk.lo <= snp_last)
                & (chunk.hi >= snp_first)
                & (chunk.hi <= snp_last)
                & lower_owned
                & (chunk.lo <= lower_max)
            )
            rows_read += int(np.count_nonzero(owned))
            if not np.any(owned):
                continue
            loci_parts.append(chunk.lo[owned])
            pair_mask = owned & (chunk.lo < chunk.hi)
            if not np.any(pair_mask):
                continue
            pair_i = chunk.lo[pair_mask]
            pair_j = chunk.hi[pair_mask]
            pair_r2 = chunk.r2[pair_mask]
            pair_rows += int(pair_i.size)

            crossing_start = time.perf_counter()
            i_blocks = np.searchsorted(bp, pair_i, side="left")
            j_blocks = np.searchsorted(bp, pair_j, side="left")
            crossing = i_blocks != j_blocks
            crossing_count = int(np.count_nonzero(crossing))
            if crossing_count:
                total_sum += float(np.sum(pair_r2[crossing]))
                n_nonzero += crossing_count
                crossing_rows += crossing_count
            crossing_seconds += time.perf_counter() - crossing_start
        nocache_profile.absorb(prepared.profile)

    if not loci_parts:
        return {"sum": 0.0, "N_nonzero": 0, "N_zero": 0.0}

    loci = np.unique(np.concatenate(loci_parts))
    n_zero = _metric_n_zero(loci, bp)
    log_debug(
        "metric_from_r2_nocache profile "
        f"partitions={len(partitions)} rows_read={rows_read} "
        f"pair_rows={pair_rows} crossing_rows={crossing_rows} "
        f"row_seconds={row_seconds:.6f} "
        f"crossing_seconds={crossing_seconds:.6f} "
        f"{nocache_profile.log_fields()}"
    )
    return {"sum": total_sum, "N_nonzero": n_nonzero, "N_zero": n_zero}


def _metric_partition_results_parallel(
    *,
    name: str,
    store: CovarianceStore,
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
    breakpoints: np.ndarray,
    unique_diag_pos: np.ndarray,
    unique_diag_val: np.ndarray,
    workers: int,
) -> list[_MetricPartitionResult]:
    task_args = [
        (
            name,
            store.root,
            p_index,
            start,
            end,
            tuple(partitions),
            snp_first,
            snp_last,
        )
        for p_index, (start, end) in enumerate(partitions)
    ]
    results: list[_MetricPartitionResult | None] = [None] * len(task_args)
    next_submit = 0
    pending = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_metric_worker,
        initargs=(breakpoints, unique_diag_pos, unique_diag_val),
    ) as pool:
        while next_submit < len(task_args) and len(pending) < workers:
            pending[pool.submit(_metric_partition_worker, task_args[next_submit])] = (
                next_submit
            )
            next_submit += 1

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result_index = pending.pop(future)
                results[result_index] = future.result()

            while next_submit < len(task_args) and len(pending) < workers:
                pending[
                    pool.submit(_metric_partition_worker, task_args[next_submit])
                ] = next_submit
                next_submit += 1

    return [result for result in results if result is not None]


def _init_metric_worker(
    breakpoints: np.ndarray,
    unique_diag_pos: np.ndarray,
    unique_diag_val: np.ndarray,
) -> None:
    global _METRIC_WORKER_BREAKPOINTS
    global _METRIC_WORKER_DIAG_POS
    global _METRIC_WORKER_DIAG_VAL
    _METRIC_WORKER_BREAKPOINTS = breakpoints
    _METRIC_WORKER_DIAG_POS = unique_diag_pos
    _METRIC_WORKER_DIAG_VAL = unique_diag_val


def _metric_partition_worker(
    args: tuple[
        str,
        Path,
        int,
        int,
        int,
        tuple[tuple[int, int], ...],
        int,
        int,
    ],
) -> _MetricPartitionResult:
    if (
        _METRIC_WORKER_BREAKPOINTS is None
        or _METRIC_WORKER_DIAG_POS is None
        or _METRIC_WORKER_DIAG_VAL is None
    ):
        raise RuntimeError("metric worker was not initialized")
    name, root, p_index, start, end, partitions, snp_first, snp_last = args
    return _metric_partition_result(
        name=name,
        store=CovarianceStore(root=root),
        partitions=list(partitions),
        p_index=p_index,
        start=start,
        end=end,
        snp_first=snp_first,
        snp_last=snp_last,
        breakpoints=_METRIC_WORKER_BREAKPOINTS,
        unique_diag_pos=_METRIC_WORKER_DIAG_POS,
        unique_diag_val=_METRIC_WORKER_DIAG_VAL,
    )


def _metric_partition_result(
    *,
    name: str,
    store: CovarianceStore,
    partitions: list[tuple[int, int]],
    p_index: int,
    start: int,
    end: int,
    snp_first: int,
    snp_last: int,
    breakpoints: np.ndarray,
    unique_diag_pos: np.ndarray,
    unique_diag_val: np.ndarray,
) -> _MetricPartitionResult:
    lower_min, lower_max, include_lower_min = _owned_bounds(
        partitions, p_index, snp_first, snp_last
    )
    total_sum = 0.0
    n_nonzero = 0
    rows_read = 0
    pair_rows = 0
    normalized_rows = 0
    crossing_rows = 0
    row_read_seconds = 0.0
    normalize_seconds = 0.0
    crossing_seconds = 0.0

    path = store.partition_path(name, start, end)
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
            i_blocks = np.searchsorted(breakpoints, pair_i, side="left")
            j_blocks = np.searchsorted(breakpoints, pair_j, side="left")
            crossing = i_blocks != j_blocks
            crossing_count = int(np.count_nonzero(crossing))
            if crossing_count == 0:
                crossing_seconds += time.perf_counter() - crossing_start
                continue

            r2 = (
                pair_s[crossing]
                * pair_s[crossing]
                / (diag_i[crossing] * diag_j[crossing])
            )
            total_sum += float(np.sum(r2))
            n_nonzero += crossing_count
            crossing_rows += crossing_count
            crossing_seconds += time.perf_counter() - crossing_start

    return _MetricPartitionResult(
        p_index=p_index,
        total_sum=total_sum,
        n_nonzero=n_nonzero,
        rows_read=rows_read,
        pair_rows=pair_rows,
        normalized_rows=normalized_rows,
        crossing_rows=crossing_rows,
        row_read_seconds=row_read_seconds,
        normalize_seconds=normalize_seconds,
        crossing_seconds=crossing_seconds,
    )


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
