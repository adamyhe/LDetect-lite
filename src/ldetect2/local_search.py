"""LocalSearch: greedy local refinement of a single breakpoint position."""

from __future__ import annotations

import decimal
import math
import time
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from ldetect2._util.binary_search import find_ge_ind, find_le_ind
from ldetect2._util.covariance_array import (
    ChromosomeCovariance,
    CovariancePartition,
    LocalSearchPartition,
    canonical_local_search_rows,
    load_covariance_arrays,  # noqa: F401 - kept for monkeypatch compatibility
    load_covariance_partitions,
    local_search_partition,
)
from ldetect2._util.logging import log_debug, log_msg
from ldetect2.io.covariance import (
    delete_loci_smaller_than_leanest,
    read_partition_into_matrix_lean,
)
from ldetect2.io.covariance_hdf5 import open_covariance_reader
from ldetect2.io.partitions import CovarianceStore, get_final_partitions

_PREC = 50
HDF5_LOCAL_SEARCH_CHUNK_ROWS = 1_000_000


@dataclass
class LocalSearchPrecomputeStats:
    """Timing and row-count diagnostics for array local-search precompute."""

    partition_load_seconds: float = 0.0
    canonicalize_seconds: float = 0.0
    append_seconds: float = 0.0
    diagonal_seconds: float = 0.0
    slice_seconds: float = 0.0
    normalize_seconds: float = 0.0
    vertical_seconds: float = 0.0
    horizontal_seconds: float = 0.0
    hdf5_read_seconds: float = 0.0
    chunk_filter_seconds: float = 0.0
    dedup_seconds: float = 0.0
    accumulator_seconds: float = 0.0
    candidate_rows: int = 0
    eligible_rows: int = 0
    normalized_rows: int = 0
    rows_read: int = 0
    rows_after_filter: int = 0
    rows_after_dedup: int = 0
    duplicate_rows_skipped: int = 0
    chunks: int = 0
    segments: int = 0
    active_rows_peak: int = 0
    peak_chunk_rows: int = 0

    def log_fields(self) -> str:
        """Return stable key/value fields for debug profiling logs."""
        return (
            f"partition_load_seconds={self.partition_load_seconds:.6f} "
            f"canonicalize_seconds={self.canonicalize_seconds:.6f} "
            f"append_seconds={self.append_seconds:.6f} "
            f"diagonal_seconds={self.diagonal_seconds:.6f} "
            f"slice_seconds={self.slice_seconds:.6f} "
            f"normalize_seconds={self.normalize_seconds:.6f} "
            f"vertical_seconds={self.vertical_seconds:.6f} "
            f"horizontal_seconds={self.horizontal_seconds:.6f} "
            f"hdf5_read_seconds={self.hdf5_read_seconds:.6f} "
            f"chunk_filter_seconds={self.chunk_filter_seconds:.6f} "
            f"dedup_seconds={self.dedup_seconds:.6f} "
            f"accumulator_seconds={self.accumulator_seconds:.6f} "
            f"candidate_rows={self.candidate_rows} "
            f"eligible_rows={self.eligible_rows} "
            f"normalized_rows={self.normalized_rows} "
            f"rows_read={self.rows_read} "
            f"rows_after_filter={self.rows_after_filter} "
            f"rows_after_dedup={self.rows_after_dedup} "
            f"duplicate_rows_skipped={self.duplicate_rows_skipped} "
            f"chunks={self.chunks} "
            f"segments={self.segments} "
            f"active_rows_peak={self.active_rows_peak} "
            f"peak_chunk_rows={self.peak_chunk_rows}"
        )


@dataclass(frozen=True)
class LocalSearchHDF5Partition:
    """Small HDF5 partition handle metadata used for chunked local search."""

    start: int
    end: int
    path: object
    source_row_count: int
    loci: np.ndarray
    lo_offsets: np.ndarray
    diag_pos: np.ndarray
    diag_val: np.ndarray


def local_search_hdf5_partition(
    name: str,
    store: CovarianceStore,
    start: int,
    end: int,
) -> LocalSearchHDF5Partition:
    """Load only HDF5 index metadata for one local-search partition."""
    path = store.partition_path(name, start, end)
    with open_covariance_reader(path, start, end) as reader:
        loci, lo_offsets = reader.read_lo_index()
        diag_pos, diag_val = reader.read_diagonal()
        return LocalSearchHDF5Partition(
            start=start,
            end=end,
            path=path,
            source_row_count=reader.row_count,
            loci=loci,
            lo_offsets=lo_offsets,
            diag_pos=diag_pos,
            diag_val=diag_val,
        )


def _append_partition(
    active_lo: np.ndarray,
    active_hi: np.ndarray,
    active_shrink: np.ndarray,
    partition: LocalSearchPartition,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Append a canonical partition and keep sorted first-pair semantics."""
    if not active_lo.size:
        lo = partition.lo
        hi = partition.hi
        shrink = partition.shrink_ld
    else:
        lo = np.concatenate((active_lo, partition.lo))
        hi = np.concatenate((active_hi, partition.hi))
        shrink = np.concatenate((active_shrink, partition.shrink_ld))
        lo, hi, shrink = canonical_local_search_rows(lo, hi, shrink)
    return lo, hi, shrink, _unique_sorted(lo)


def _active_loci_from_partitions(
    partitions: list[LocalSearchPartition],
    active_min_lo: int,
) -> np.ndarray:
    """Return sorted unique active ``lo`` loci across canonical partitions."""
    loci_arrays = []
    for partition in partitions:
        left = int(np.searchsorted(partition.loci, active_min_lo, side="left"))
        if left < partition.loci.size:
            loci_arrays.append(partition.loci[left:])
    if not loci_arrays:
        return np.array([], dtype=np.int64)
    loci = np.concatenate(loci_arrays).astype(np.int64, copy=False)
    loci.sort()
    return _unique_sorted(loci)


def _active_row_count_from_partitions(
    partitions: list[LocalSearchPartition],
    active_min_lo: int,
) -> int:
    """Return the non-deduplicated active row count for instrumentation."""
    total = 0
    for partition in partitions:
        left = int(np.searchsorted(partition.lo, active_min_lo, side="left"))
        total += int(partition.lo.size - left)
    return total


def _segment_rows_from_partitions(
    partitions: list[LocalSearchPartition],
    active_min_lo: int,
    lo_min: int,
    lo_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical rows for a segment from active canonical partitions."""
    min_lo = max(active_min_lo, lo_min)
    lo_parts: list[np.ndarray] = []
    hi_parts: list[np.ndarray] = []
    shrink_parts: list[np.ndarray] = []
    for partition in partitions:
        left = int(np.searchsorted(partition.lo, min_lo, side="left"))
        right = int(np.searchsorted(partition.lo, lo_max, side="right"))
        if left >= right:
            continue
        lo_parts.append(partition.lo[left:right])
        hi_parts.append(partition.hi[left:right])
        shrink_parts.append(partition.shrink_ld[left:right])

    if not lo_parts:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        )
    lo = np.concatenate(lo_parts)
    hi = np.concatenate(hi_parts)
    shrink = np.concatenate(shrink_parts)
    return canonical_local_search_rows(lo, hi, shrink)


def _active_diagonal_from_partitions(
    partitions: list[LocalSearchPartition],
    active_min_lo: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return active canonical diagonal rows with partition-order first wins."""
    pos_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for partition in partitions:
        left = int(np.searchsorted(partition.diag_pos, active_min_lo, side="left"))
        if left >= partition.diag_pos.size:
            continue
        pos_parts.append(partition.diag_pos[left:])
        val_parts.append(partition.diag_val[left:])

    if not pos_parts:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
    pos = np.concatenate(pos_parts)
    val = np.concatenate(val_parts)
    diag_pos, _, diag_val = canonical_local_search_rows(pos, pos, val)
    return diag_pos, diag_val


def _active_loci_from_hdf5_partitions(
    partitions: list[LocalSearchHDF5Partition],
    active_min_lo: int,
) -> np.ndarray:
    """Return sorted unique active ``lo`` loci across HDF5 partition indexes."""
    loci_arrays = []
    for partition in partitions:
        left = int(np.searchsorted(partition.loci, active_min_lo, side="left"))
        if left < partition.loci.size:
            loci_arrays.append(partition.loci[left:])
    if not loci_arrays:
        return np.array([], dtype=np.int64)
    loci = np.concatenate(loci_arrays).astype(np.int64, copy=False)
    loci.sort()
    return _unique_sorted(loci)


def _active_row_count_from_hdf5_partitions(
    partitions: list[LocalSearchHDF5Partition],
    active_min_lo: int,
) -> int:
    """Return an approximate active row count from HDF5 indexes."""
    total = 0
    for partition in partitions:
        left = int(np.searchsorted(partition.loci, active_min_lo, side="left"))
        if left < partition.loci.size:
            total += partition.source_row_count
    return total


def _segment_rows_from_hdf5_partitions(
    partitions: list[LocalSearchHDF5Partition],
    active_min_lo: int,
    lo_min: int,
    lo_max: int,
    chunk_rows: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical segment rows by reading bounded HDF5 row chunks."""
    min_lo = max(active_min_lo, lo_min)
    lo_parts: list[np.ndarray] = []
    hi_parts: list[np.ndarray] = []
    shrink_parts: list[np.ndarray] = []
    for partition in partitions:
        with open_covariance_reader(
            partition.path, partition.start, partition.end
        ) as reader:
            for chunk in reader.iter_rows(min_lo, lo_max, chunk_rows):
                lo_parts.append(chunk.lo)
                hi_parts.append(chunk.hi)
                shrink_parts.append(chunk.shrink_ld)

    if not lo_parts:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        )
    lo = np.concatenate(lo_parts)
    hi = np.concatenate(hi_parts)
    shrink = np.concatenate(shrink_parts)
    return canonical_local_search_rows(lo, hi, shrink)


def _active_diagonal_from_hdf5_partitions(
    partitions: list[LocalSearchHDF5Partition],
    active_min_lo: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return active canonical diagonal rows with partition-order first wins."""
    pos_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for partition in partitions:
        left = int(np.searchsorted(partition.diag_pos, active_min_lo, side="left"))
        if left >= partition.diag_pos.size:
            continue
        pos_parts.append(partition.diag_pos[left:])
        val_parts.append(partition.diag_val[left:])

    if not pos_parts:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
    pos = np.concatenate(pos_parts)
    val = np.concatenate(val_parts)
    diag_pos, _, diag_val = canonical_local_search_rows(pos, pos, val)
    return diag_pos, diag_val


def _hdf5_rows_in_lo_range(
    partition: LocalSearchHDF5Partition,
    lo_min: int,
    lo_max: int,
) -> int:
    """Return the indexed row count for one HDF5 partition lo range."""
    left_value = int(np.searchsorted(partition.loci, lo_min, side="left"))
    right_value = int(np.searchsorted(partition.loci, lo_max, side="right"))
    if left_value >= right_value:
        return 0
    return int(partition.lo_offsets[right_value] - partition.lo_offsets[left_value])


def _hdf5_rows_in_active_range(
    partitions: list[LocalSearchHDF5Partition],
    active_min_lo: int,
    lo_min: int,
    lo_max: int,
) -> int:
    """Return the estimated active HDF5 rows for a candidate lo window."""
    min_lo = max(active_min_lo, lo_min)
    return sum(
        _hdf5_rows_in_lo_range(partition, min_lo, lo_max)
        for partition in partitions
    )


def _hdf5_locus_windows(
    segment_loci: np.ndarray,
    partitions: list[LocalSearchHDF5Partition],
    active_min_lo: int,
    target_rows: int,
) -> list[tuple[int, int, np.ndarray]]:
    """Split segment loci into row-budgeted lo windows for bounded assembly."""
    if segment_loci.size == 0:
        return []
    target_rows = max(1, int(target_rows))
    windows: list[tuple[int, int, np.ndarray]] = []
    start_idx = 0
    while start_idx < segment_loci.size:
        lo_min = int(segment_loci[start_idx])
        low = start_idx + 1
        high = segment_loci.size
        best = low
        while low <= high:
            mid = (low + high) // 2
            lo_max = int(segment_loci[mid - 1])
            row_count = _hdf5_rows_in_active_range(
                partitions, active_min_lo, lo_min, lo_max
            )
            if row_count <= target_rows or mid == start_idx + 1:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        window_loci = segment_loci[start_idx:best]
        windows.append((int(window_loci[0]), int(window_loci[-1]), window_loci))
        start_idx = best
    return windows


def _read_hdf5_window_rows(
    partitions: list[LocalSearchHDF5Partition],
    active_min_lo: int,
    lo_min: int,
    lo_max: int,
    snp_first: int,
    snp_last: int,
    snp_top: int,
    include_snp_first: bool,
    stats: LocalSearchPrecomputeStats,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read, early-filter, and canonicalize one bounded HDF5 lo window."""
    min_lo = max(active_min_lo, lo_min)
    lo_parts: list[np.ndarray] = []
    hi_parts: list[np.ndarray] = []
    shrink_parts: list[np.ndarray] = []
    filtered_rows = 0

    for partition in partitions:
        with open_covariance_reader(
            partition.path, partition.start, partition.end
        ) as reader:
            iterator = reader.iter_rows(min_lo, lo_max, chunk_rows)
            while True:
                read_start = time.perf_counter()
                try:
                    chunk = next(iterator)
                except StopIteration:
                    stats.hdf5_read_seconds += time.perf_counter() - read_start
                    break
                stats.hdf5_read_seconds += time.perf_counter() - read_start
                stats.rows_read += int(chunk.lo.size)
                stats.peak_chunk_rows = max(stats.peak_chunk_rows, int(chunk.lo.size))

                filter_start = time.perf_counter()
                eligible = (chunk.lo <= snp_last) & (chunk.hi <= snp_top)
                if include_snp_first:
                    eligible &= chunk.lo >= snp_first
                else:
                    eligible &= chunk.lo > snp_first
                eligible_idx = np.flatnonzero(eligible)
                stats.candidate_rows += int(chunk.lo.size)
                stats.eligible_rows += int(eligible_idx.size)
                stats.rows_after_filter += int(eligible_idx.size)
                stats.chunk_filter_seconds += time.perf_counter() - filter_start
                if eligible_idx.size == 0:
                    continue

                lo_parts.append(chunk.lo[eligible_idx])
                hi_parts.append(chunk.hi[eligible_idx])
                shrink_parts.append(chunk.shrink_ld[eligible_idx])
                filtered_rows += int(eligible_idx.size)

    if not lo_parts:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        )

    dedup_start = time.perf_counter()
    lo = np.concatenate(lo_parts)
    hi = np.concatenate(hi_parts)
    shrink = np.concatenate(shrink_parts)
    lo, hi, shrink = canonical_local_search_rows(lo, hi, shrink)
    stats.rows_after_dedup += int(lo.size)
    stats.duplicate_rows_skipped += int(filtered_rows - lo.size)
    stats.dedup_seconds += time.perf_counter() - dedup_start
    return lo, hi, shrink


def _accumulate_sorted_rows(
    row_lo: np.ndarray,
    row_hi: np.ndarray,
    row_shrink: np.ndarray,
    diag_pos: np.ndarray,
    diag_val: np.ndarray,
    sum_vert_by_locus: dict[int, float],
    sum_horiz_by_locus: dict[int, float],
    chunk_size: int = 2_000_000,
    stats: LocalSearchPrecomputeStats | None = None,
) -> None:
    """Normalize and aggregate already-filtered canonical rows."""
    if row_lo.size == 0 or diag_pos.size == 0:
        return

    for chunk_start in range(0, row_lo.size, chunk_size):
        chunk = slice(chunk_start, chunk_start + chunk_size)
        lo_chunk = row_lo[chunk]
        hi_chunk = row_hi[chunk]
        shrink_chunk = row_shrink[chunk]
        if stats is not None:
            stats.chunks += 1

        normalize_start = time.perf_counter()
        diag_lo_idx = np.searchsorted(diag_pos, lo_chunk)
        diag_hi_idx = np.searchsorted(diag_pos, hi_chunk)
        has_diag = (diag_lo_idx < diag_pos.size) & (diag_hi_idx < diag_pos.size)
        safe_lo_idx = np.minimum(diag_lo_idx, diag_pos.size - 1)
        safe_hi_idx = np.minimum(diag_hi_idx, diag_pos.size - 1)
        has_diag &= (diag_pos[safe_lo_idx] == lo_chunk) & (
            diag_pos[safe_hi_idx] == hi_chunk
        )
        if not np.any(has_diag):
            if stats is not None:
                stats.normalize_seconds += time.perf_counter() - normalize_start
            continue

        lo_chunk = lo_chunk[has_diag]
        hi_chunk = hi_chunk[has_diag]
        shrink_chunk = shrink_chunk[has_diag]
        diag_lo = diag_val[diag_lo_idx[has_diag]]
        diag_hi = diag_val[diag_hi_idx[has_diag]]
        positive = (diag_lo > 0.0) & (diag_hi > 0.0)
        if not np.any(positive):
            if stats is not None:
                stats.normalize_seconds += time.perf_counter() - normalize_start
            continue

        lo_chunk = lo_chunk[positive]
        hi_chunk = hi_chunk[positive]
        r2 = (
            shrink_chunk[positive]
            * shrink_chunk[positive]
            / (diag_lo[positive] * diag_hi[positive])
        )
        if stats is not None:
            stats.normalized_rows += int(r2.size)
            stats.normalize_seconds += time.perf_counter() - normalize_start

        vertical_start = time.perf_counter()
        group_starts = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.flatnonzero(lo_chunk[1:] != lo_chunk[:-1]) + 1,
            )
        )
        vert_loci = lo_chunk[group_starts]
        vert_sums = np.add.reduceat(r2, group_starts)
        for locus, value in zip(vert_loci, vert_sums):
            sum_vert_by_locus[int(locus)] = (
                sum_vert_by_locus.get(int(locus), 0.0) + float(value)
            )
        if stats is not None:
            stats.vertical_seconds += time.perf_counter() - vertical_start

        horizontal_start = time.perf_counter()
        horiz_loci, horiz_inverse = np.unique(hi_chunk, return_inverse=True)
        horiz_sums = np.bincount(horiz_inverse, weights=r2)
        for locus, value in zip(horiz_loci, horiz_sums):
            locus_key = int(locus)
            sum_horiz_by_locus[locus_key] = (
                sum_horiz_by_locus.get(locus_key, 0.0) + float(value)
            )
            sum_vert_by_locus.setdefault(locus_key, 0.0)
        if stats is not None:
            stats.horizontal_seconds += time.perf_counter() - horizontal_start


def _add_hdf5_segment_values(
    segment_loci: np.ndarray,
    partitions: list[LocalSearchHDF5Partition],
    active_min_lo: int,
    diag_pos: np.ndarray,
    diag_val: np.ndarray,
    snp_first: int,
    snp_last: int,
    snp_top: int,
    include_snp_first: bool,
    sum_vert_by_locus: dict[int, float],
    sum_horiz_by_locus: dict[int, float],
    stats: LocalSearchPrecomputeStats,
    chunk_rows: int | None = None,
) -> None:
    """Stream bounded HDF5 row windows into local-search aggregations."""
    if segment_loci.size == 0:
        return
    if chunk_rows is None:
        chunk_rows = HDF5_LOCAL_SEARCH_CHUNK_ROWS
    stats.segments += 1
    for locus in segment_loci:
        locus_key = int(locus)
        sum_vert_by_locus.setdefault(locus_key, 0.0)
        sum_horiz_by_locus.setdefault(locus_key, 0.0)
    if diag_pos.size == 0:
        return

    windows = _hdf5_locus_windows(
        segment_loci, partitions, active_min_lo, target_rows=chunk_rows
    )
    for lo_min, lo_max, _window_loci in windows:
        row_lo, row_hi, row_shrink = _read_hdf5_window_rows(
            partitions,
            active_min_lo,
            lo_min,
            lo_max,
            snp_first,
            snp_last,
            snp_top,
            include_snp_first,
            stats,
            chunk_rows,
        )
        accumulator_start = time.perf_counter()
        _accumulate_sorted_rows(
            row_lo,
            row_hi,
            row_shrink,
            diag_pos,
            diag_val,
            sum_vert_by_locus,
            sum_horiz_by_locus,
            stats=stats,
        )
        stats.accumulator_seconds += time.perf_counter() - accumulator_start


def _unique_sorted(values: np.ndarray) -> np.ndarray:
    """Return unique values from an already sorted array."""
    if values.size <= 1:
        return values.astype(np.int64, copy=False)
    keep = np.ones(values.size, dtype=bool)
    keep[1:] = values[1:] != values[:-1]
    return values[keep].astype(np.int64, copy=False)


def _active_diagonal(
    active_lo: np.ndarray,
    active_hi: np.ndarray,
    active_shrink: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted diagonal positions and values from canonical active rows."""
    diag_mask = active_lo == active_hi
    return active_lo[diag_mask], active_shrink[diag_mask]


def _add_array_locus_values(
    curr_locus: int,
    active_lo: np.ndarray,
    active_hi: np.ndarray,
    active_shrink: np.ndarray,
    diag_lookup: dict[int, float],
    snp_top: int,
    sum_vert_by_locus: dict[int, float],
    sum_horiz_by_locus: dict[int, float],
) -> None:
    diag_curr = diag_lookup.get(curr_locus, 0.0)
    if diag_curr <= 0.0:
        sum_vert_by_locus.setdefault(curr_locus, 0.0)
        sum_horiz_by_locus.setdefault(curr_locus, 0.0)
        return

    row_mask = (active_lo == curr_locus) & (active_hi <= snp_top)
    row_hi = active_hi[row_mask]
    row_shrink = active_shrink[row_mask]
    sum_vert_by_locus.setdefault(curr_locus, 0.0)
    sum_horiz_by_locus.setdefault(curr_locus, 0.0)
    for key, shrink in zip(row_hi, row_shrink):
        key = int(key)
        diag_key = diag_lookup.get(key, 0.0)
        if diag_key <= 0.0:
            continue
        r2 = float(shrink * shrink / (diag_curr * diag_key))
        sum_vert_by_locus[curr_locus] += r2
        sum_horiz_by_locus[key] = sum_horiz_by_locus.get(key, 0.0) + r2
        sum_vert_by_locus.setdefault(key, 0.0)


def _add_array_segment_values(
    segment_loci: np.ndarray,
    active_lo: np.ndarray,
    active_hi: np.ndarray,
    active_shrink: np.ndarray,
    diag_pos: np.ndarray,
    diag_val: np.ndarray,
    snp_first: int,
    snp_last: int,
    snp_top: int,
    include_snp_first: bool,
    sum_vert_by_locus: dict[int, float],
    sum_horiz_by_locus: dict[int, float],
    chunk_size: int = 2_000_000,
    stats: LocalSearchPrecomputeStats | None = None,
) -> None:
    """Aggregate local-search vertical/horizontal r² sums for one locus segment.

    This is the array-backed replacement for calling
    :func:`_add_array_locus_values` once per locus.  It preserves the same
    effective row eligibility rules, but scans the active covariance rows once
    per segment and accumulates per-locus sums in bounded chunks.
    """
    if segment_loci.size == 0:
        return
    if stats is not None:
        stats.segments += 1

    for locus in segment_loci:
        locus_key = int(locus)
        sum_vert_by_locus.setdefault(locus_key, 0.0)
        sum_horiz_by_locus.setdefault(locus_key, 0.0)

    if diag_pos.size == 0:
        return

    lo_min = int(segment_loci[0])
    lo_max = int(segment_loci[-1])
    slice_start = time.perf_counter()
    left = int(np.searchsorted(active_lo, lo_min, side="left"))
    right = int(np.searchsorted(active_lo, lo_max, side="right"))
    if left >= right:
        if stats is not None:
            stats.slice_seconds += time.perf_counter() - slice_start
        return

    candidate_lo = active_lo[left:right]
    candidate_hi = active_hi[left:right]
    candidate_shrink = active_shrink[left:right]
    if stats is not None:
        stats.candidate_rows += int(candidate_lo.size)
    eligible = (candidate_lo <= snp_last) & (candidate_hi <= snp_top)
    if include_snp_first:
        eligible &= candidate_lo >= snp_first
    else:
        eligible &= candidate_lo > snp_first
    eligible_idx = np.flatnonzero(eligible)
    if stats is not None:
        stats.eligible_rows += int(eligible_idx.size)
        stats.slice_seconds += time.perf_counter() - slice_start
    if eligible_idx.size == 0:
        return

    for chunk_start in range(0, eligible_idx.size, chunk_size):
        chunk = eligible_idx[chunk_start : chunk_start + chunk_size]
        row_lo = candidate_lo[chunk]
        row_hi = candidate_hi[chunk]
        row_shrink = candidate_shrink[chunk]
        if stats is not None:
            stats.chunks += 1

        normalize_start = time.perf_counter()
        diag_lo_idx = np.searchsorted(diag_pos, row_lo)
        diag_hi_idx = np.searchsorted(diag_pos, row_hi)
        has_diag = (diag_lo_idx < diag_pos.size) & (diag_hi_idx < diag_pos.size)
        safe_lo_idx = np.minimum(diag_lo_idx, diag_pos.size - 1)
        safe_hi_idx = np.minimum(diag_hi_idx, diag_pos.size - 1)
        has_diag &= (diag_pos[safe_lo_idx] == row_lo) & (
            diag_pos[safe_hi_idx] == row_hi
        )
        if not np.any(has_diag):
            if stats is not None:
                stats.normalize_seconds += time.perf_counter() - normalize_start
            continue

        row_lo = row_lo[has_diag]
        row_hi = row_hi[has_diag]
        row_shrink = row_shrink[has_diag]
        diag_lo = diag_val[diag_lo_idx[has_diag]]
        diag_hi = diag_val[diag_hi_idx[has_diag]]
        positive = (diag_lo > 0.0) & (diag_hi > 0.0)
        if not np.any(positive):
            if stats is not None:
                stats.normalize_seconds += time.perf_counter() - normalize_start
            continue

        row_lo = row_lo[positive]
        row_hi = row_hi[positive]
        r2 = (
            row_shrink[positive]
            * row_shrink[positive]
            / (diag_lo[positive] * diag_hi[positive])
        )
        if stats is not None:
            stats.normalized_rows += int(r2.size)
            stats.normalize_seconds += time.perf_counter() - normalize_start

        vertical_start = time.perf_counter()
        group_starts = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.flatnonzero(row_lo[1:] != row_lo[:-1]) + 1,
            )
        )
        vert_loci = row_lo[group_starts]
        vert_sums = np.add.reduceat(r2, group_starts)
        for locus, value in zip(vert_loci, vert_sums):
            sum_vert_by_locus[int(locus)] = (
                sum_vert_by_locus.get(int(locus), 0.0) + float(value)
            )
        if stats is not None:
            stats.vertical_seconds += time.perf_counter() - vertical_start

        horizontal_start = time.perf_counter()
        horiz_loci, horiz_inverse = np.unique(row_hi, return_inverse=True)
        horiz_sums = np.bincount(horiz_inverse, weights=r2)
        for locus, value in zip(horiz_loci, horiz_sums):
            locus_key = int(locus)
            sum_horiz_by_locus[locus_key] = (
                sum_horiz_by_locus.get(locus_key, 0.0) + float(value)
            )
            sum_vert_by_locus.setdefault(locus_key, 0.0)
        if stats is not None:
            stats.horizontal_seconds += time.perf_counter() - horizontal_start


def _diag_lookup(
    lo: np.ndarray, hi: np.ndarray, shrink: np.ndarray
) -> dict[int, float]:
    diag_mask = lo == hi
    return {
        int(locus): float(value)
        for locus, value in zip(lo[diag_mask], shrink[diag_mask])
    }


class LocalSearch:
    """Precomputes per-locus LD sums and searches for the locally-optimal breakpoint.

    The search evaluates each locus within [start_search, stop_search] as a
    candidate breakpoint and returns the one that minimises
    ``sum(r²) / N_zero``.

    Args:
        use_decimal: When *True*, accumulate sums with 50-digit
            :class:`decimal.Decimal` precision (slower but exact).  When
            *False* (default), use ``float`` arithmetic — sufficient for
            almost all practical inputs.
    """

    def __init__(
        self,
        name: str,
        start_search: int,
        stop_search: int,
        initial_breakpoint_index: int,
        breakpoints: list[int],
        total_sum,
        total_n,
        store: CovarianceStore,
        use_decimal: bool = False,
        covariance_cache: ChromosomeCovariance | None = None,
        local_search_partitions: tuple[LocalSearchPartition, ...] | None = None,
        local_search_hdf5_partitions: tuple[LocalSearchHDF5Partition, ...]
        | None = None,
    ) -> None:
        if use_decimal:
            decimal.getcontext().prec = _PREC

        self.name = name
        self.start_search = start_search
        self.stop_search = stop_search
        self.initial_breakpoint_index = initial_breakpoint_index
        self.breakpoints = breakpoints
        self.use_decimal = use_decimal

        if use_decimal:
            self.total_sum = decimal.Decimal(total_sum)
            self.total_n = decimal.Decimal(total_n)
        else:
            self.total_sum = float(total_sum)
            self.total_n = float(total_n)

        self.store = store
        self.covariance_cache = covariance_cache
        self.local_search_partitions = local_search_partitions
        self.local_search_hdf5_partitions = local_search_hdf5_partitions

        self.matrix: dict = {}
        self.locus_list: list[int] = []
        self.locus_list_deleted: list[int] = []

        self.precomputed: dict = {
            "locus_list": [],
            "data": {},
        }
        self._array_loci: np.ndarray | None = None
        self._array_sum_vert: np.ndarray | None = None
        self._array_sum_horiz: np.ndarray | None = None
        self.loaded_partition_count: int | None = None
        self.loaded_row_count: int | None = None
        self.precompute_stats = LocalSearchPrecomputeStats()

        self.dynamic_delete = True
        self.init_complete = False
        self.search_complete = False

        # --- validation ---
        if start_search >= stop_search:
            raise ValueError(
                f"start_search ({start_search}) >= stop_search ({stop_search})"
            )
        if not (0 <= initial_breakpoint_index < len(breakpoints)):
            raise ValueError("initial_breakpoint_index out of bounds")
        if breakpoints[initial_breakpoint_index] >= stop_search:
            raise ValueError("breakpoint >= stop_search")
        if breakpoints[initial_breakpoint_index] <= start_search:
            raise ValueError("breakpoint <= start_search")

        tmp_partitions = get_final_partitions(store, name, start_search, stop_search)

        if not (tmp_partitions[0][0] <= start_search <= tmp_partitions[-1][1]):
            raise ValueError("start_search is out of partition bounds")
        if not (tmp_partitions[0][0] <= stop_search <= tmp_partitions[-1][1]):
            raise ValueError("stop_search is out of partition bounds")

        if initial_breakpoint_index > 0:
            if start_search < breakpoints[initial_breakpoint_index - 1]:
                raise ValueError(
                    "start_search cannot be further than a neighbouring breakpoint"
                )
        if initial_breakpoint_index < len(breakpoints) - 1:
            if stop_search > breakpoints[initial_breakpoint_index + 1]:
                raise ValueError(
                    "stop_search cannot be further than a neighbouring breakpoint"
                )

        self.snp_first = start_search
        self.snp_last = stop_search

        if initial_breakpoint_index + 1 < len(breakpoints):
            self.snp_top = breakpoints[initial_breakpoint_index + 1]
        else:
            self.snp_top = tmp_partitions[-1][1]

        if initial_breakpoint_index - 1 >= 0:
            self.snp_bottom = breakpoints[initial_breakpoint_index - 1]
        else:
            self.snp_bottom = tmp_partitions[0][0]

        log_debug(
            f"LocalSearch: snp_first={self.snp_first} snp_last={self.snp_last} "
            f"snp_bottom={self.snp_bottom} snp_top={self.snp_top}"
        )

        self.partitions = get_final_partitions(
            store, name, self.snp_bottom, self.snp_top
        )

        self.start_locus = -1
        self.start_locus_index = -1
        self.end_locus = -1
        self.end_locus_index = -1

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def init_search(self) -> None:
        """Precompute per-locus vertical and horizontal LD sums (lean path)."""
        if not self.use_decimal:
            self._init_search_array()
            return

        if self.use_decimal:
            decimal.getcontext().prec = _PREC
        log_debug("Start local search init (lean)")

        last_p_num = -1
        for p_num_init in range(len(self.partitions) - 1):
            if self.snp_bottom >= self.partitions[p_num_init + 1][0]:
                log_debug(f"Pre-reading partition: {self.partitions[p_num_init]}")
                read_partition_into_matrix_lean(
                    self.partitions,
                    p_num_init,
                    self.matrix,
                    self.locus_list,
                    self.name,
                    self.store,
                    self.snp_bottom,
                    self.snp_top,
                )
                last_p_num = p_num_init
            else:
                break

        curr_locus = -1
        start_locus = -1
        start_locus_index = -1
        end_locus = -1
        end_locus_index = -1

        for p_num in range(last_p_num + 1, len(self.partitions)):
            p = self.partitions[p_num]
            log_debug(f"Reading partition: {p}")
            read_partition_into_matrix_lean(
                self.partitions,
                p_num,
                self.matrix,
                self.locus_list,
                self.name,
                self.store,
                self.snp_bottom,
                self.snp_top,
            )

            if curr_locus < 0:
                if not self.locus_list:
                    raise RuntimeError("locus_list is empty")
                for i, locus in enumerate(self.locus_list):
                    if locus >= self.snp_bottom:
                        curr_locus = locus
                        start_locus = locus
                        curr_locus_index = i
                        start_locus_index = i
                        break
            else:
                i = bisect_left(self.locus_list, curr_locus)
                if i < len(self.locus_list) and self.locus_list[i] == curr_locus:
                    curr_locus_index = i
                else:
                    if self.locus_list:
                        curr_locus = self.locus_list[0]
                        curr_locus_index = 0
                    else:
                        raise RuntimeError("locus_list is empty")

            if curr_locus < 0:
                log_debug(
                    f"Warning: curr_locus not found in partition {p} "
                    f"(snp_bottom={self.snp_bottom}); skipping"
                )
                continue

            if p_num + 1 < len(self.partitions):
                end_locus = self.partitions[p_num + 1][0]
                end_locus_index = -1
            else:
                end_locus_found = False
                for i in reversed(range(len(self.locus_list))):
                    if self.locus_list[i] <= self.snp_last:
                        end_locus = self.locus_list[i]
                        end_locus_index = i
                        end_locus_found = True
                        break
                if not end_locus_found:
                    end_locus_index = 0
                    end_locus = self.locus_list[0]

            log_debug(f"Precomputing for partition: {p}")

            _zero = decimal.Decimal(0) if self.use_decimal else 0.0

            while curr_locus <= end_locus:
                self._add_locus(curr_locus)

                in_range = (
                    curr_locus > self.snp_first or self.initial_breakpoint_index == 0
                ) and curr_locus <= self.snp_last
                if in_range:
                    for key in self.matrix.get(curr_locus, {}):
                        if key <= self.snp_top:
                            diag_curr = self.matrix[curr_locus].get(curr_locus, 0.0)
                            diag_key = self.matrix.get(key, {}).get(key, 0.0)
                            if diag_curr > 0 and diag_key > 0:
                                corr = self.matrix[curr_locus][key] / math.sqrt(
                                    diag_curr * diag_key
                                )
                                r2 = corr**2
                                self._add_val(
                                    decimal.Decimal(r2) if self.use_decimal else r2,
                                    curr_locus,
                                    key,
                                )
                else:
                    self._add_val(_zero, curr_locus, curr_locus)

                if curr_locus_index + 1 < len(self.locus_list):
                    curr_locus_index += 1
                    curr_locus = self.locus_list[curr_locus_index]
                else:
                    log_debug("curr_locus_index out of bounds")
                    break

            delete_loci_smaller_than_leanest(end_locus, self.matrix, self.locus_list)

        self.start_locus = start_locus
        self.start_locus_index = start_locus_index
        self.end_locus = end_locus
        self.end_locus_index = end_locus_index
        self.init_complete = True

    def _init_search_array(self) -> None:
        """Precompute local-search deltas with exact legacy locus semantics."""
        log_debug("Start local search init (array)")
        self.precompute_stats = LocalSearchPrecomputeStats()
        if self.local_search_hdf5_partitions is not None:
            self.loaded_partition_count = len(self.local_search_hdf5_partitions)
            self.loaded_row_count = int(
                sum(
                    partition.source_row_count
                    for partition in self.local_search_hdf5_partitions
                )
            )
            loci, sum_vert, sum_horiz = self._precompute_array_from_hdf5_partitions(
                self.local_search_hdf5_partitions
            )
        elif self.local_search_partitions is None:
            load_start = time.perf_counter()
            partitions = self._local_covariance_partitions()
            self.precompute_stats.partition_load_seconds = (
                time.perf_counter() - load_start
            )
            self.loaded_partition_count = len(partitions)
            self.loaded_row_count = int(
                sum(partition.i_pos.size for partition in partitions)
            )
            loci, sum_vert, sum_horiz = self._precompute_array_from_partitions(
                partitions
            )
        else:
            self.loaded_partition_count = len(self.local_search_partitions)
            self.loaded_row_count = int(
                sum(
                    partition.source_row_count
                    for partition in self.local_search_partitions
                )
            )
            loci, sum_vert, sum_horiz = self._precompute_array_from_local_partitions(
                self.local_search_partitions
            )

        self._array_loci = loci
        self._array_sum_vert = sum_vert
        self._array_sum_horiz = sum_horiz
        self.precomputed["locus_list"] = loci.tolist()
        self.init_complete = True

    def _local_covariance_partitions(self) -> tuple[CovariancePartition, ...]:
        if self.covariance_cache is None:
            return load_covariance_partitions(
                self.name,
                self.store,
                self.partitions,
                snp_first=self.snp_bottom,
                snp_last=self.snp_top,
            )

        by_bounds = {
            (partition.start, partition.end): partition
            for partition in self.covariance_cache.partition_arrays
        }
        missing = [
            (start, end)
            for start, end in self.partitions
            if (start, end) not in by_bounds
        ]
        if missing:
            raise ValueError(
                "Chromosome covariance cache is missing local-search "
                f"partition(s): {missing}"
            )
        return tuple(by_bounds[(start, end)] for start, end in self.partitions)

    def _precompute_array_from_partitions(
        self,
        partitions: tuple[CovariancePartition, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        canonicalize_start = time.perf_counter()
        local_partitions = tuple(
            local_search_partition(partition) for partition in partitions
        )
        self.precompute_stats.canonicalize_seconds += (
            time.perf_counter() - canonicalize_start
        )
        return self._precompute_array_from_local_partitions(local_partitions)

    def _precompute_array_from_local_partitions(
        self,
        local_partitions: tuple[LocalSearchPartition, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precompute local-search arrays from canonical partition slices."""
        active_partitions: list[LocalSearchPartition] = []
        active_min_lo = -2**63
        active_loci = np.array([], dtype=np.int64)
        precomputed_loci: list[int] = []
        sum_vert_by_locus: dict[int, float] = {}
        sum_horiz_by_locus: dict[int, float] = {}

        last_p_num = -1
        for p_num_init in range(len(local_partitions) - 1):
            if self.snp_bottom >= local_partitions[p_num_init + 1].start:
                append_start = time.perf_counter()
                active_partitions.append(local_partitions[p_num_init])
                active_loci = _active_loci_from_partitions(
                    active_partitions, active_min_lo
                )
                self.precompute_stats.append_seconds += (
                    time.perf_counter() - append_start
                )
                self.precompute_stats.active_rows_peak = max(
                    self.precompute_stats.active_rows_peak,
                    _active_row_count_from_partitions(
                        active_partitions, active_min_lo
                    ),
                )
                last_p_num = p_num_init
            else:
                break

        curr_locus = -1
        start_locus = -1
        start_locus_index = -1
        end_locus = -1
        end_locus_index = -1

        for p_num in range(last_p_num + 1, len(local_partitions)):
            append_start = time.perf_counter()
            active_partitions.append(local_partitions[p_num])
            active_loci = _active_loci_from_partitions(
                active_partitions, active_min_lo
            )
            self.precompute_stats.append_seconds += time.perf_counter() - append_start
            self.precompute_stats.active_rows_peak = max(
                self.precompute_stats.active_rows_peak,
                _active_row_count_from_partitions(active_partitions, active_min_lo),
            )

            if curr_locus < 0:
                if active_loci.size == 0:
                    raise RuntimeError("locus_list is empty")
                curr_locus_index = int(
                    np.searchsorted(active_loci, self.snp_bottom, side="left")
                )
                if curr_locus_index >= active_loci.size:
                    log_debug(
                        "Warning: curr_locus not found in partition "
                        f"{self.partitions[p_num]} (snp_bottom={self.snp_bottom}); "
                        "skipping"
                    )
                    continue
                curr_locus = int(active_loci[curr_locus_index])
                start_locus = curr_locus
                start_locus_index = curr_locus_index
            else:
                curr_locus_index = int(
                    np.searchsorted(active_loci, curr_locus, side="left")
                )
                if (
                    curr_locus_index >= active_loci.size
                    or int(active_loci[curr_locus_index]) != curr_locus
                ):
                    if active_loci.size == 0:
                        raise RuntimeError("locus_list is empty")
                    curr_locus_index = 0
                    curr_locus = int(active_loci[0])

            if p_num + 1 < len(local_partitions):
                end_locus = local_partitions[p_num + 1].start
                end_locus_index = -1
            else:
                end_idx = int(np.searchsorted(active_loci, self.snp_last, side="right"))
                if end_idx > 0:
                    end_locus_index = end_idx - 1
                    end_locus = int(active_loci[end_locus_index])
                else:
                    end_locus_index = 0
                    end_locus = int(active_loci[0])

            segment_end_idx = int(
                np.searchsorted(active_loci, end_locus, side="right")
            )
            if curr_locus_index < segment_end_idx:
                segment_loci = active_loci[curr_locus_index:segment_end_idx]
                precomputed_loci.extend(int(locus) for locus in segment_loci)
                lo_min = int(segment_loci[0])
                lo_max = int(segment_loci[-1])
                append_start = time.perf_counter()
                active_lo, active_hi, active_shrink = _segment_rows_from_partitions(
                    active_partitions,
                    active_min_lo,
                    lo_min,
                    lo_max,
                )
                self.precompute_stats.append_seconds += (
                    time.perf_counter() - append_start
                )
                diagonal_start = time.perf_counter()
                diag_pos, diag_val = _active_diagonal_from_partitions(
                    active_partitions,
                    active_min_lo,
                )
                self.precompute_stats.diagonal_seconds += (
                    time.perf_counter() - diagonal_start
                )
                _add_array_segment_values(
                    segment_loci,
                    active_lo,
                    active_hi,
                    active_shrink,
                    diag_pos,
                    diag_val,
                    self.snp_first,
                    self.snp_last,
                    self.snp_top,
                    self.initial_breakpoint_index == 0,
                    sum_vert_by_locus,
                    sum_horiz_by_locus,
                    stats=self.precompute_stats,
                )

                if segment_end_idx < active_loci.size:
                    curr_locus_index = segment_end_idx
                    curr_locus = int(active_loci[curr_locus_index])
                else:
                    log_debug("curr_locus_index out of bounds")
                    break

            active_min_lo = end_locus
            active_partitions = [
                partition
                for partition in active_partitions
                if partition.end >= active_min_lo
            ]
            active_loci = _active_loci_from_partitions(
                active_partitions, active_min_lo
            )

        loci = np.asarray(precomputed_loci, dtype=np.int64)
        sum_vert = np.asarray(
            [sum_vert_by_locus.get(int(locus), 0.0) for locus in loci],
            dtype=np.float64,
        )
        sum_horiz = np.asarray(
            [sum_horiz_by_locus.get(int(locus), 0.0) for locus in loci],
            dtype=np.float64,
        )
        self.start_locus = start_locus
        self.start_locus_index = start_locus_index
        self.end_locus = end_locus
        self.end_locus_index = end_locus_index
        return loci, sum_vert, sum_horiz

    def _precompute_array_from_hdf5_partitions(
        self,
        local_partitions: tuple[LocalSearchHDF5Partition, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precompute local-search arrays from HDF5 partition row chunks."""
        active_partitions: list[LocalSearchHDF5Partition] = []
        active_min_lo = -2**63
        active_loci = np.array([], dtype=np.int64)
        precomputed_loci: list[int] = []
        sum_vert_by_locus: dict[int, float] = {}
        sum_horiz_by_locus: dict[int, float] = {}

        last_p_num = -1
        for p_num_init in range(len(local_partitions) - 1):
            if self.snp_bottom >= local_partitions[p_num_init + 1].start:
                append_start = time.perf_counter()
                active_partitions.append(local_partitions[p_num_init])
                active_loci = _active_loci_from_hdf5_partitions(
                    active_partitions, active_min_lo
                )
                self.precompute_stats.append_seconds += (
                    time.perf_counter() - append_start
                )
                self.precompute_stats.active_rows_peak = max(
                    self.precompute_stats.active_rows_peak,
                    _active_row_count_from_hdf5_partitions(
                        active_partitions, active_min_lo
                    ),
                )
                last_p_num = p_num_init
            else:
                break

        curr_locus = -1
        start_locus = -1
        start_locus_index = -1
        end_locus = -1
        end_locus_index = -1

        for p_num in range(last_p_num + 1, len(local_partitions)):
            append_start = time.perf_counter()
            active_partitions.append(local_partitions[p_num])
            active_loci = _active_loci_from_hdf5_partitions(
                active_partitions, active_min_lo
            )
            self.precompute_stats.append_seconds += time.perf_counter() - append_start
            self.precompute_stats.active_rows_peak = max(
                self.precompute_stats.active_rows_peak,
                _active_row_count_from_hdf5_partitions(
                    active_partitions, active_min_lo
                ),
            )

            if curr_locus < 0:
                if active_loci.size == 0:
                    raise RuntimeError("locus_list is empty")
                curr_locus_index = int(
                    np.searchsorted(active_loci, self.snp_bottom, side="left")
                )
                if curr_locus_index >= active_loci.size:
                    log_debug(
                        "Warning: curr_locus not found in partition "
                        f"{self.partitions[p_num]} (snp_bottom={self.snp_bottom}); "
                        "skipping"
                    )
                    continue
                curr_locus = int(active_loci[curr_locus_index])
                start_locus = curr_locus
                start_locus_index = curr_locus_index
            else:
                curr_locus_index = int(
                    np.searchsorted(active_loci, curr_locus, side="left")
                )
                if (
                    curr_locus_index >= active_loci.size
                    or int(active_loci[curr_locus_index]) != curr_locus
                ):
                    if active_loci.size == 0:
                        raise RuntimeError("locus_list is empty")
                    curr_locus_index = 0
                    curr_locus = int(active_loci[0])

            if p_num + 1 < len(local_partitions):
                end_locus = local_partitions[p_num + 1].start
                end_locus_index = -1
            else:
                end_idx = int(np.searchsorted(active_loci, self.snp_last, side="right"))
                if end_idx > 0:
                    end_locus_index = end_idx - 1
                    end_locus = int(active_loci[end_locus_index])
                else:
                    end_locus_index = 0
                    end_locus = int(active_loci[0])

            segment_end_idx = int(
                np.searchsorted(active_loci, end_locus, side="right")
            )
            if curr_locus_index < segment_end_idx:
                segment_loci = active_loci[curr_locus_index:segment_end_idx]
                precomputed_loci.extend(int(locus) for locus in segment_loci)
                diagonal_start = time.perf_counter()
                diag_pos, diag_val = _active_diagonal_from_hdf5_partitions(
                    active_partitions,
                    active_min_lo,
                )
                self.precompute_stats.diagonal_seconds += (
                    time.perf_counter() - diagonal_start
                )
                append_start = time.perf_counter()
                _add_hdf5_segment_values(
                    segment_loci,
                    active_partitions,
                    active_min_lo,
                    diag_pos,
                    diag_val,
                    self.snp_first,
                    self.snp_last,
                    self.snp_top,
                    self.initial_breakpoint_index == 0,
                    sum_vert_by_locus,
                    sum_horiz_by_locus,
                    stats=self.precompute_stats,
                )
                self.precompute_stats.append_seconds += (
                    time.perf_counter() - append_start
                )

                if segment_end_idx < active_loci.size:
                    curr_locus_index = segment_end_idx
                    curr_locus = int(active_loci[curr_locus_index])
                else:
                    log_debug("curr_locus_index out of bounds")
                    break

            active_min_lo = end_locus
            active_partitions = [
                partition
                for partition in active_partitions
                if partition.end >= active_min_lo
            ]
            active_loci = _active_loci_from_hdf5_partitions(
                active_partitions, active_min_lo
            )

        loci = np.asarray(precomputed_loci, dtype=np.int64)
        sum_vert = np.asarray(
            [sum_vert_by_locus.get(int(locus), 0.0) for locus in loci],
            dtype=np.float64,
        )
        sum_horiz = np.asarray(
            [sum_horiz_by_locus.get(int(locus), 0.0) for locus in loci],
            dtype=np.float64,
        )
        self.start_locus = start_locus
        self.start_locus_index = start_locus_index
        self.end_locus = end_locus
        self.end_locus_index = end_locus_index
        return loci, sum_vert, sum_horiz

    def _add_val(self, val, curr_locus: int, key: int) -> None:
        zero = decimal.Decimal(0) if self.use_decimal else 0.0
        for loc in (curr_locus, key):
            if loc not in self.precomputed["data"]:
                self.precomputed["data"][loc] = {
                    "sum_vert": zero,
                    "sum_horiz": zero,
                }
        self.precomputed["data"][curr_locus]["sum_vert"] += val
        self.precomputed["data"][key]["sum_horiz"] += val

    def _add_locus(self, locus: int) -> None:
        self.precomputed["locus_list"].append(locus)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self) -> tuple[int | None, dict | None]:
        """Find the locally-optimal breakpoint position.

        Returns:
            ``(best_position, metric_details)`` where *metric_details* has keys
            ``sum`` and ``N_zero``.  Returns the initial breakpoint unchanged if
            no better position is found.
        """
        if not self.init_complete:
            log_debug("init_search() not called — running automatically")
            self.init_search()

        if not self.use_decimal and self._array_loci is not None:
            return self._search_array()

        log_debug("Starting local search")
        locus_list = self.precomputed["locus_list"]

        try:
            snp_bottom_ind = find_ge_ind(locus_list, self.snp_bottom)
            snp_top_ind = find_le_ind(locus_list, self.snp_top)
        except ValueError as exc:
            log_msg(f"Error finding bounds in precomputed locus_list: {exc}")
            log_msg(f"snp_bottom={self.snp_bottom} snp_top={self.snp_top}")
            log_msg("Returning initial breakpoint unchanged")
            return self.breakpoints[self.initial_breakpoint_index], None

        bp_ind = find_le_ind(
            locus_list, self.breakpoints[self.initial_breakpoint_index]
        )
        init_bp_locus = locus_list[bp_ind]

        curr_sum = self.total_sum
        curr_n = self.total_n

        if self.use_decimal:
            min_metric = decimal.Decimal(self.total_sum) / decimal.Decimal(self.total_n)
        else:
            min_metric = self.total_sum / self.total_n

        min_breakpoint: int | None = None
        min_metric_details: dict = {"sum": self.total_sum, "N_zero": self.total_n}
        min_distance_right = 0

        # Search RIGHT
        log_debug("Searching right...")
        if bp_ind + 1 < len(locus_list):
            curr_loc_ind = bp_ind + 1
            curr_loc = locus_list[curr_loc_ind]

            while curr_loc <= self.snp_last:
                data = self.precomputed["data"]
                curr_sum = (
                    curr_sum - data[curr_loc]["sum_horiz"] + data[curr_loc]["sum_vert"]
                )
                horiz_n = curr_loc_ind - snp_bottom_ind - 1
                vert_n = snp_top_ind - curr_loc_ind
                curr_n = curr_n - horiz_n + vert_n

                if self.use_decimal:
                    curr_metric = decimal.Decimal(curr_sum) / decimal.Decimal(curr_n)
                else:
                    curr_metric = curr_sum / curr_n

                if curr_metric < min_metric:
                    min_metric = curr_metric
                    min_breakpoint = curr_loc
                    min_metric_details = {"sum": curr_sum, "N_zero": curr_n}
                    min_distance_right = curr_loc - init_bp_locus

                if curr_loc_ind + 1 < len(locus_list):
                    curr_loc_ind += 1
                    curr_loc = locus_list[curr_loc_ind]
                else:
                    break
        else:
            log_msg("Warning: no loci to the right of initial breakpoint")

        # Reset for left search
        curr_sum = self.total_sum
        curr_n = self.total_n

        # Search LEFT
        log_debug("Searching left...")
        if bp_ind - 1 >= 0:
            curr_loc_ind = bp_ind - 1
            curr_loc = locus_list[curr_loc_ind]

            while curr_loc > self.snp_first:
                data = self.precomputed["data"]
                curr_sum = (
                    curr_sum + data[curr_loc]["sum_horiz"] - data[curr_loc]["sum_vert"]
                )
                horiz_n = curr_loc_ind - snp_bottom_ind - 1
                vert_n = snp_top_ind - curr_loc_ind
                curr_n = curr_n + horiz_n - vert_n

                if self.use_decimal:
                    curr_metric = decimal.Decimal(curr_sum) / decimal.Decimal(curr_n)
                else:
                    curr_metric = curr_sum / curr_n

                left_dist = init_bp_locus - curr_loc
                if curr_metric < min_metric or (
                    curr_metric == min_metric and left_dist < min_distance_right
                ):
                    min_metric = curr_metric
                    min_breakpoint = curr_loc
                    min_metric_details = {"sum": curr_sum, "N_zero": curr_n}

                if curr_loc_ind - 1 >= 0:
                    curr_loc_ind -= 1
                    curr_loc = locus_list[curr_loc_ind]
                else:
                    break
        else:
            log_msg("Warning: no loci to the left of initial breakpoint")

        self.search_complete = True
        log_debug("Search done")
        return min_breakpoint, min_metric_details

    def _search_array(self) -> tuple[int | None, dict | None]:
        log_debug("Starting local search (array)")
        loci = self._array_loci
        sum_vert = self._array_sum_vert
        sum_horiz = self._array_sum_horiz
        if loci is None or sum_vert is None or sum_horiz is None or loci.size == 0:
            log_msg("Array local search has no loci; keeping original")
            return self.breakpoints[self.initial_breakpoint_index], None
        if self.total_n <= 0:
            log_msg("Array local search has no valid denominator; keeping original")
            return self.breakpoints[self.initial_breakpoint_index], None

        try:
            snp_bottom_ind = int(np.searchsorted(loci, self.snp_bottom, side="left"))
            snp_top_ind = int(np.searchsorted(loci, self.snp_top, side="right") - 1)
            if snp_bottom_ind >= loci.size or snp_top_ind < 0:
                raise ValueError("bounds not found")
            bp_ind = int(
                np.searchsorted(
                    loci,
                    self.breakpoints[self.initial_breakpoint_index],
                    side="right",
                )
                - 1
            )
            if bp_ind < 0:
                raise ValueError("breakpoint not found")
        except ValueError as exc:
            log_msg(f"Error finding bounds in array local search: {exc}")
            log_msg("Returning initial breakpoint unchanged")
            return self.breakpoints[self.initial_breakpoint_index], None

        init_bp_locus = int(loci[bp_ind])
        min_metric = self.total_sum / self.total_n
        min_breakpoint: int | None = None
        min_metric_details: dict = {"sum": self.total_sum, "N_zero": self.total_n}
        min_distance_right = 0

        right_stop = int(np.searchsorted(loci, self.snp_last, side="right"))
        if bp_ind + 1 < right_stop:
            right_idx = np.arange(bp_ind + 1, right_stop, dtype=np.int64)
            sum_delta = np.cumsum(-sum_horiz[right_idx] + sum_vert[right_idx])
            n_delta = np.cumsum(
                -(right_idx - snp_bottom_ind - 1) + (snp_top_ind - right_idx)
            )
            sums = self.total_sum + sum_delta
            ns = self.total_n + n_delta
            valid = ns > 0
            if np.any(valid):
                valid_metrics = sums[valid] / ns[valid]
                best_valid = int(np.argmin(valid_metrics))
                valid_idx = np.flatnonzero(valid)
                best = int(valid_idx[best_valid])
            else:
                best = -1
            if best >= 0 and valid_metrics[best_valid] < min_metric:
                min_metric = float(valid_metrics[best_valid])
                min_breakpoint = int(loci[right_idx[best]])
                min_metric_details = {
                    "sum": float(sums[best]),
                    "N_zero": float(ns[best]),
                }
                min_distance_right = min_breakpoint - init_bp_locus
        else:
            log_msg("Warning: no loci to the right of initial breakpoint")

        left_start = int(np.searchsorted(loci, self.snp_first, side="right"))
        if left_start < bp_ind:
            left_idx = np.arange(bp_ind - 1, left_start - 1, -1, dtype=np.int64)
            sum_delta = np.cumsum(sum_horiz[left_idx] - sum_vert[left_idx])
            n_delta = np.cumsum(
                (left_idx - snp_bottom_ind - 1) - (snp_top_ind - left_idx)
            )
            sums = self.total_sum + sum_delta
            ns = self.total_n + n_delta
            valid = ns > 0
            metrics = np.empty_like(sums)
            metrics[valid] = sums[valid] / ns[valid]
            for pos, metric, curr_sum, curr_n, is_valid in zip(
                loci[left_idx],
                metrics,
                sums,
                ns,
                valid,
            ):
                if not is_valid:
                    continue
                left_dist = init_bp_locus - int(pos)
                if metric < min_metric or (
                    metric == min_metric and left_dist < min_distance_right
                ):
                    min_metric = float(metric)
                    min_breakpoint = int(pos)
                    min_metric_details = {
                        "sum": float(curr_sum),
                        "N_zero": float(curr_n),
                    }
        else:
            log_msg("Warning: no loci to the left of initial breakpoint")

        self.search_complete = True
        log_debug("Search done")
        return min_breakpoint, min_metric_details
