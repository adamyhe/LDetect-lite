"""Array-backed covariance-to-vector helpers."""

from __future__ import annotations

import csv
import gzip
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np

from ldetect_lite._util.covariance_array import (
    ChromosomeCovariance,
    CovariancePartition,
)
from ldetect_lite._util.logging import log_debug
from ldetect_lite._util.memory import log_memory_checkpoint
from ldetect_lite.io.covariance_hdf5 import (
    HDF5CovariancePartitionReader,
    open_covariance_reader,
)
from ldetect_lite.io.partitions import CovarianceStore

MATRIX_TO_VECTOR_CHUNK_ROWS = 1_000_000


class _DiagVectorProfile(TypedDict, total=False):
    """Per-partition profiling counters; not every key is set in every path."""

    checkpoint: str
    hdf5_read_seconds: float
    normalize_seconds: float
    center_seconds: float
    open_seconds: float
    loci_seconds: float
    diag_seconds: float
    rows_read: int
    rows_accumulated: int
    chunks: int


@dataclass(frozen=True)
class _DiagVectorPartitionPlan:
    p_index: int
    start: int
    end: int
    next_start: int | None
    center_lower_bound: int
    center_lower_inclusive: bool
    checkpoint: str


@dataclass(frozen=True)
class _DiagVectorPartitionResult:
    loci: np.ndarray
    sum_loci: np.ndarray
    sum_values: np.ndarray
    end_locus: int
    write_cutoff: int
    profile: _DiagVectorProfile


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


def write_diag_vector_array(
    *,
    name: str,
    store: CovarianceStore,
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
    out_path: Path,
    covariance_cache: ChromosomeCovariance | None = None,
    matrix_workers: int = 1,
) -> None:
    """Write the diagonal correlation-sum vector from HDF5 partitions."""
    _log_vector_checkpoint("matrix_to_vector_array_start")
    wall_start = time.perf_counter()
    out_path.unlink(missing_ok=True)
    pending_sums: dict[int, float] = {}
    current_locus = snp_first
    partition_arrays = (
        covariance_cache.partition_arrays if covariance_cache is not None else None
    )
    if (
        covariance_cache is not None
        and tuple(partitions) != covariance_cache.partitions
    ):
        raise ValueError(
            "Chromosome covariance cache partitions do not match matrix-to-vector "
            "partitions"
        )
    plans = _plan_diag_vector_partitions(partitions, snp_first, snp_last)
    parent_profile: dict[str, float | int] = {
        "merge_seconds": 0.0,
        "flush_seconds": 0.0,
        "worker_wait_seconds": 0.0,
        "partitions": 0,
    }
    if (
        matrix_workers > 1
        and covariance_cache is None
        and len(plans) > 1
    ):
        _write_diag_vector_array_hdf5_parallel(
            name=name,
            store=store,
            plans=plans,
            snp_first=snp_first,
            snp_last=snp_last,
            pending_sums=pending_sums,
            out_path=out_path,
            matrix_workers=matrix_workers,
            parent_profile=parent_profile,
        )
    else:
        for plan in plans:
            checkpoint = plan.checkpoint
            _log_vector_checkpoint(f"{checkpoint}_start")
            result = _process_diag_vector_partition(
                name=name,
                store=store,
                partition_arrays=partition_arrays,
                p_index=plan.p_index,
                start=plan.start,
                end=plan.end,
                next_start=plan.next_start,
                snp_first=snp_first,
                snp_last=snp_last,
                current_locus=current_locus,
                center_lower_bound=plan.center_lower_bound,
                center_lower_inclusive=plan.center_lower_inclusive,
                checkpoint=checkpoint,
            )
            current_locus = _merge_diag_vector_partition_result(
                result=result,
                snp_first=snp_first,
                snp_last=snp_last,
                current_locus=current_locus,
                pending_sums=pending_sums,
                out_path=out_path,
                parent_profile=parent_profile,
            )
            _log_vector_checkpoint(f"{checkpoint}_helper_return")
    log_debug(
        "matrix_to_vector_array profile "
        f"partitions={int(parent_profile['partitions'])} "
        f"matrix_workers={matrix_workers} "
        f"wall_seconds={time.perf_counter() - wall_start:.6f} "
        f"merge_seconds={parent_profile['merge_seconds']:.6f} "
        f"flush_seconds={parent_profile['flush_seconds']:.6f} "
        f"worker_wait_seconds={parent_profile['worker_wait_seconds']:.6f}"
    )
    _log_vector_checkpoint("matrix_to_vector_array_end")


def _plan_diag_vector_partitions(
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
) -> list[_DiagVectorPartitionPlan]:
    """Plan matrix-to-vector partitions with independent center-locus bounds."""
    plans: list[_DiagVectorPartitionPlan] = []
    previous_end_locus: int | None = None
    for p_index, (start, end) in enumerate(partitions):
        checkpoint = _partition_checkpoint_label(p_index, start, end)
        if p_index + 1 < len(partitions) and snp_first >= partitions[p_index + 1][0]:
            _log_vector_checkpoint(f"{checkpoint}_skip_before_snp_first")
            continue

        next_start = (
            partitions[p_index + 1][0] if p_index + 1 < len(partitions) else None
        )
        if previous_end_locus is None:
            center_lower_bound = snp_first
            center_lower_inclusive = True
        else:
            center_lower_bound = previous_end_locus
            center_lower_inclusive = False
        if next_start is not None:
            end_locus = int((end + next_start) / 2)
        else:
            end_locus = snp_last
        plans.append(
            _DiagVectorPartitionPlan(
                p_index=p_index,
                start=start,
                end=end,
                next_start=next_start,
                center_lower_bound=center_lower_bound,
                center_lower_inclusive=center_lower_inclusive,
                checkpoint=checkpoint,
            )
        )
        previous_end_locus = end_locus
    return plans


def _write_diag_vector_array_hdf5_parallel(
    *,
    name: str,
    store: CovarianceStore,
    plans: list[_DiagVectorPartitionPlan],
    snp_first: int,
    snp_last: int,
    pending_sums: dict[int, float],
    out_path: Path,
    matrix_workers: int,
    parent_profile: dict[str, float | int],
) -> None:
    """Compute HDF5 matrix-to-vector partitions in workers and merge in order."""
    task_args = [
        (
            name,
            store.root,
            plan.start,
            plan.end,
            plan.next_start,
            snp_first,
            snp_last,
            plan.center_lower_bound,
            plan.center_lower_inclusive,
            plan.checkpoint,
        )
        for plan in plans
    ]
    current_locus = snp_first
    next_submit = 0
    next_emit = 0
    pending: dict[Future[_DiagVectorPartitionResult], int] = {}
    completed: dict[int, _DiagVectorPartitionResult] = {}
    with ProcessPoolExecutor(max_workers=matrix_workers) as pool:
        while next_submit < len(task_args) and len(pending) < matrix_workers:
            pending[
                pool.submit(
                    _compute_diag_vector_partition_hdf5_worker,
                    task_args[next_submit],
                )
            ] = next_submit
            next_submit += 1

        while pending:
            wait_start = time.perf_counter()
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            parent_profile["worker_wait_seconds"] += time.perf_counter() - wait_start
            for future in done:
                result_index = pending.pop(future)
                completed[result_index] = future.result()

            while next_emit in completed:
                current_locus = _merge_diag_vector_partition_result(
                    result=completed.pop(next_emit),
                    snp_first=snp_first,
                    snp_last=snp_last,
                    current_locus=current_locus,
                    pending_sums=pending_sums,
                    out_path=out_path,
                    parent_profile=parent_profile,
                )
                next_emit += 1

            while next_submit < len(task_args) and len(pending) < matrix_workers:
                pending[
                    pool.submit(
                        _compute_diag_vector_partition_hdf5_worker,
                        task_args[next_submit],
                    )
                ] = next_submit
                next_submit += 1


def _compute_diag_vector_partition_hdf5_worker(
    args: tuple[
        str,
        Path,
        int,
        int,
        int | None,
        int,
        int,
        int,
        bool,
        str,
    ],
) -> _DiagVectorPartitionResult:
    (
        name,
        root,
        start,
        end,
        next_start,
        snp_first,
        snp_last,
        center_lower_bound,
        center_lower_inclusive,
        checkpoint,
    ) = args
    return _compute_diag_vector_partition_hdf5(
        name=name,
        store=CovarianceStore(root=root),
        start=start,
        end=end,
        next_start=next_start,
        snp_first=snp_first,
        snp_last=snp_last,
        center_lower_bound=center_lower_bound,
        center_lower_inclusive=center_lower_inclusive,
        checkpoint=checkpoint,
    )


def _merge_diag_vector_partition_result(
    *,
    result: _DiagVectorPartitionResult,
    snp_first: int,
    snp_last: int,
    current_locus: int,
    pending_sums: dict[int, float],
    out_path: Path,
    parent_profile: dict[str, float | int],
) -> int:
    merge_start = time.perf_counter()
    for locus, value in zip(result.sum_loci, result.sum_values):
        pending_sums[int(locus)] = pending_sums.get(int(locus), 0.0) + float(value)
    parent_profile["merge_seconds"] += time.perf_counter() - merge_start
    parent_profile["partitions"] += 1
    flush_before = time.perf_counter()
    next_locus = _finish_diag_vector_partition(
        loci=result.loci,
        end_locus=result.end_locus,
        write_cutoff=result.write_cutoff,
        snp_first=snp_first,
        snp_last=snp_last,
        current_locus=current_locus,
        pending_sums=pending_sums,
        out_path=out_path,
        checkpoint=str(result.profile.get("checkpoint", "matrix_to_vector")),
    )
    parent_profile["flush_seconds"] += time.perf_counter() - flush_before
    return next_locus


def _process_diag_vector_partition(
    *,
    name: str,
    store: CovarianceStore,
    partition_arrays: tuple[CovariancePartition, ...] | None,
    p_index: int,
    start: int,
    end: int,
    next_start: int | None,
    snp_first: int,
    snp_last: int,
    current_locus: int,
    center_lower_bound: int,
    center_lower_inclusive: bool,
    checkpoint: str,
) -> _DiagVectorPartitionResult:
    """Process one matrix-to-vector partition and return the next locus state."""
    if partition_arrays is None:
        return _compute_diag_vector_partition_hdf5(
            name=name,
            store=store,
            start=start,
            end=end,
            next_start=next_start,
            snp_first=snp_first,
            snp_last=snp_last,
            center_lower_bound=center_lower_bound,
            center_lower_inclusive=center_lower_inclusive,
            checkpoint=checkpoint,
        )

    return _compute_diag_vector_partition_array(
        partition_arrays=partition_arrays,
        p_index=p_index,
        start=start,
        end=end,
        next_start=next_start,
        snp_first=snp_first,
        snp_last=snp_last,
        current_locus=current_locus,
        checkpoint=checkpoint,
    )


def _compute_diag_vector_partition_array(
    *,
    partition_arrays: tuple[CovariancePartition, ...],
    p_index: int,
    start: int,
    end: int,
    next_start: int | None,
    snp_first: int,
    snp_last: int,
    current_locus: int,
    checkpoint: str,
) -> _DiagVectorPartitionResult:
    """Process one cached array partition with the materialized reference path."""
    _log_vector_checkpoint(f"{checkpoint}_cache_load_start")
    partition = partition_arrays[p_index]
    i_pos = partition.i_pos
    j_pos = partition.j_pos
    shrink = partition.shrink_ld
    _log_vector_checkpoint(f"{checkpoint}_cache_load_end")
    _log_vector_checkpoint(f"{checkpoint}_filter_start")
    rows_in_partition = (
        (i_pos >= start) & (i_pos <= end) & (j_pos >= start) & (j_pos <= end)
    )
    _log_vector_checkpoint(f"{checkpoint}_filter_mask_end")
    if not np.any(rows_in_partition):
        _log_vector_checkpoint(f"{checkpoint}_no_owned_rows")
        return _empty_diag_vector_partition_result(checkpoint)

    i_pos = i_pos[rows_in_partition]
    j_pos = j_pos[rows_in_partition]
    shrink = shrink[rows_in_partition]
    _log_vector_checkpoint(f"{checkpoint}_filter_apply_end")
    _log_vector_checkpoint(f"{checkpoint}_lo_hi_start")
    if np.all(i_pos <= j_pos):
        lo = i_pos
        hi = j_pos
    else:
        lo = np.minimum(i_pos, j_pos)
        hi = np.maximum(i_pos, j_pos)
    _log_vector_checkpoint(f"{checkpoint}_lo_hi_end")
    _log_vector_checkpoint(f"{checkpoint}_loci_unique_start")
    loci = np.unique(np.concatenate((lo, hi)))
    _log_vector_checkpoint(f"{checkpoint}_loci_unique_end")
    if loci.size == 0:
        _log_vector_checkpoint(f"{checkpoint}_no_loci")
        return _empty_diag_vector_partition_result(checkpoint)

    if next_start is not None:
        end_locus = int((end + next_start) / 2)
        write_cutoff = next_start
    else:
        in_requested_range = loci[loci <= snp_last]
        if in_requested_range.size == 0:
            _log_vector_checkpoint(f"{checkpoint}_no_requested_loci")
            return _empty_diag_vector_partition_result(checkpoint)
        end_locus = int(in_requested_range[-1])
        write_cutoff = end_locus

    _log_vector_checkpoint(f"{checkpoint}_r2_rows_start")
    r2_lo, r2_hi, r2 = _r2_rows(lo, hi, shrink)
    _log_vector_checkpoint(f"{checkpoint}_r2_rows_end")
    partition_sums = np.zeros(loci.size, dtype=np.float64)
    if r2.size:
        _log_vector_checkpoint(f"{checkpoint}_center_search_start")
        lo_idx = np.searchsorted(loci, r2_lo)
        hi_idx = np.searchsorted(loci, r2_hi)
        _log_vector_checkpoint(f"{checkpoint}_center_search_end")
        idx_delta = hi_idx - lo_idx
        even_delta = idx_delta % 2 == 0
        legacy_reachable = even_delta | (lo_idx > 0)
        _log_vector_checkpoint(f"{checkpoint}_legacy_reachable_mask_end")
        if np.any(legacy_reachable):
            lo_idx = lo_idx[legacy_reachable]
            hi_idx = hi_idx[legacy_reachable]
            r2 = r2[legacy_reachable]
            center_idx = (lo_idx + hi_idx) // 2
            center_loci = loci[center_idx]
            _log_vector_checkpoint(f"{checkpoint}_center_filter_start")

            keep_center = (
                (center_loci >= current_locus)
                & (center_loci <= end_locus)
                & (center_loci <= snp_last)
            )
            _log_vector_checkpoint(f"{checkpoint}_center_filter_end")
            if np.any(keep_center):
                _log_vector_checkpoint(f"{checkpoint}_bincount_start")
                partition_sums += np.bincount(
                    center_idx[keep_center],
                    weights=r2[keep_center],
                    minlength=loci.size,
                )
                _log_vector_checkpoint(f"{checkpoint}_bincount_end")
    nonzero = partition_sums > 0.0
    return _DiagVectorPartitionResult(
        loci=loci,
        sum_loci=loci[nonzero],
        sum_values=partition_sums[nonzero],
        end_locus=end_locus,
        write_cutoff=write_cutoff,
        profile={"checkpoint": checkpoint},
    )


def _compute_diag_vector_partition_hdf5(
    *,
    name: str,
    store: CovarianceStore,
    start: int,
    end: int,
    next_start: int | None,
    snp_first: int,
    snp_last: int,
    center_lower_bound: int,
    center_lower_inclusive: bool,
    checkpoint: str,
) -> _DiagVectorPartitionResult:
    """Process one HDF5 partition with chunked normalization and accumulation."""
    path = store.partition_path(name, start, end)
    if not path.exists():
        raise FileNotFoundError(
            f"Covariance partition {path} is missing. The array-backed "
            "matrix-to-vector path requires HDF5 covariance partitions; "
            "regenerate covariance with `ldetect run` or `ldetect calc-covariance`."
        )

    _log_vector_checkpoint(f"{checkpoint}_hdf5_open_start")
    open_start = time.perf_counter()
    with open_covariance_reader(path, start, end) as reader:
        open_seconds = time.perf_counter() - open_start
        _log_vector_checkpoint(f"{checkpoint}_hdf5_open_end")
        _log_vector_checkpoint(f"{checkpoint}_loci_pass_start")
        loci_start = time.perf_counter()
        loci = _chunked_owned_loci(reader, start, end)
        loci_seconds = time.perf_counter() - loci_start
        _log_vector_checkpoint(f"{checkpoint}_loci_pass_end")
        if loci.size == 0:
            _log_vector_checkpoint(f"{checkpoint}_no_loci")
            return _empty_diag_vector_partition_result(checkpoint)

        if next_start is not None:
            end_locus = int((end + next_start) / 2)
            write_cutoff = next_start
        else:
            in_requested_range = loci[loci <= snp_last]
            if in_requested_range.size == 0:
                _log_vector_checkpoint(f"{checkpoint}_no_requested_loci")
                return _empty_diag_vector_partition_result(checkpoint)
            end_locus = int(in_requested_range[-1])
            write_cutoff = end_locus

        _log_vector_checkpoint(f"{checkpoint}_diag_lookup_start")
        diag_start = time.perf_counter()
        diag_pos, diag_val = reader.read_diagonal()
        diag_mask = (diag_pos >= start) & (diag_pos <= end)
        diag_pos = diag_pos[diag_mask]
        diag_val = diag_val[diag_mask]
        diag_seconds = time.perf_counter() - diag_start
        _log_vector_checkpoint(f"{checkpoint}_diag_lookup_end")

        _log_vector_checkpoint(f"{checkpoint}_chunked_r2_accum_start")
        partition_sums = np.zeros(loci.size, dtype=np.float64)
        profile: _DiagVectorProfile = {
            "checkpoint": checkpoint,
            "hdf5_read_seconds": 0.0,
            "normalize_seconds": 0.0,
            "center_seconds": 0.0,
            "open_seconds": open_seconds,
            "loci_seconds": loci_seconds,
            "diag_seconds": diag_seconds,
            "rows_read": 0,
            "rows_accumulated": 0,
            "chunks": 0,
        }
        center_hi = min(end_locus, snp_last)
        center_left = int(
            np.searchsorted(
                loci,
                center_lower_bound,
                side="left" if center_lower_inclusive else "right",
            )
        )
        center_right = int(np.searchsorted(loci, center_hi, side="right"))
        if diag_pos.size and center_left < center_right:
            chunk_iter = reader.iter_owned_rows(
                start,
                end,
                start,
                end,
                MATRIX_TO_VECTOR_CHUNK_ROWS,
            )
            while True:
                read_start = time.perf_counter()
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                profile["hdf5_read_seconds"] += time.perf_counter() - read_start
                profile["rows_read"] += int(chunk.lo.size)
                profile["chunks"] += 1
                _accumulate_vector_chunk(
                    loci=loci,
                    diag_pos=diag_pos,
                    diag_val=diag_val,
                    row_lo=chunk.lo,
                    row_hi=chunk.hi,
                    row_shrink=chunk.shrink_ld,
                    center_left=center_left,
                    center_right=center_right,
                    partition_sums=partition_sums,
                    profile=profile,
                )
        _log_vector_checkpoint(f"{checkpoint}_chunked_r2_accum_end")
        log_debug(
            "matrix_to_vector_hdf5_partition profile "
            f"checkpoint={checkpoint} chunks={int(profile['chunks'])} "
            f"rows_read={int(profile['rows_read'])} "
            f"rows_accumulated={int(profile['rows_accumulated'])} "
            f"open_seconds={profile['open_seconds']:.6f} "
            f"loci_seconds={profile['loci_seconds']:.6f} "
            f"diag_seconds={profile['diag_seconds']:.6f} "
            f"hdf5_read_seconds={profile['hdf5_read_seconds']:.6f} "
            f"normalize_seconds={profile['normalize_seconds']:.6f} "
            f"center_seconds={profile['center_seconds']:.6f}"
        )

    nonzero = partition_sums > 0.0
    return _DiagVectorPartitionResult(
        loci=loci,
        sum_loci=loci[nonzero],
        sum_values=partition_sums[nonzero],
        end_locus=end_locus,
        write_cutoff=write_cutoff,
        profile=profile,
    )


def _empty_diag_vector_partition_result(checkpoint: str) -> _DiagVectorPartitionResult:
    empty_i = np.array([], dtype=np.int64)
    empty_f = np.array([], dtype=np.float64)
    return _DiagVectorPartitionResult(
        loci=empty_i,
        sum_loci=empty_i,
        sum_values=empty_f,
        end_locus=-1,
        write_cutoff=-1,
        profile={"checkpoint": checkpoint},
    )


def _chunked_owned_loci(
    reader: HDF5CovariancePartitionReader, start: int, end: int
) -> np.ndarray:
    """Return sorted unique owned loci from the compact HDF5 index."""
    loci = reader.read_loci()
    mask = (loci >= start) & (loci <= end)
    return _position_array(loci[mask])


def _accumulate_vector_chunk(
    *,
    loci: np.ndarray,
    diag_pos: np.ndarray,
    diag_val: np.ndarray,
    row_lo: np.ndarray,
    row_hi: np.ndarray,
    row_shrink: np.ndarray,
    center_left: int,
    center_right: int,
    partition_sums: np.ndarray,
    profile: _DiagVectorProfile | None = None,
) -> None:
    """Normalize one row chunk and add center-locus sums into ``partition_sums``."""
    if row_lo.size == 0:
        return

    normalize_start = time.perf_counter()
    diag_lo_idx = np.searchsorted(diag_pos, row_lo)
    diag_hi_idx = np.searchsorted(diag_pos, row_hi)
    has_diag = (diag_lo_idx < diag_pos.size) & (diag_hi_idx < diag_pos.size)
    safe_lo_idx = np.minimum(diag_lo_idx, diag_pos.size - 1)
    safe_hi_idx = np.minimum(diag_hi_idx, diag_pos.size - 1)
    has_diag &= (diag_pos[safe_lo_idx] == row_lo) & (diag_pos[safe_hi_idx] == row_hi)
    if not np.any(has_diag):
        if profile is not None:
            profile["normalize_seconds"] += time.perf_counter() - normalize_start
        return

    row_lo = row_lo[has_diag]
    row_hi = row_hi[has_diag]
    row_shrink = row_shrink[has_diag]
    diag_lo = diag_val[diag_lo_idx[has_diag]]
    diag_hi = diag_val[diag_hi_idx[has_diag]]

    positive = (diag_lo > 0.0) & (diag_hi > 0.0)
    if not np.any(positive):
        if profile is not None:
            profile["normalize_seconds"] += time.perf_counter() - normalize_start
        return

    row_lo = row_lo[positive]
    row_hi = row_hi[positive]
    row_shrink = row_shrink[positive]
    diag_lo = diag_lo[positive]
    diag_hi = diag_hi[positive]
    if profile is not None:
        profile["normalize_seconds"] += time.perf_counter() - normalize_start

    center_start = time.perf_counter()
    lo_idx = np.searchsorted(loci, row_lo)
    hi_idx = np.searchsorted(loci, row_hi)
    idx_delta = hi_idx - lo_idx
    legacy_reachable = (idx_delta % 2 == 0) | (lo_idx > 0)
    if not np.any(legacy_reachable):
        if profile is not None:
            profile["center_seconds"] += time.perf_counter() - center_start
        return

    lo_idx = lo_idx[legacy_reachable]
    hi_idx = hi_idx[legacy_reachable]
    row_shrink = row_shrink[legacy_reachable]
    diag_lo = diag_lo[legacy_reachable]
    diag_hi = diag_hi[legacy_reachable]
    center_idx = (lo_idx + hi_idx) // 2
    in_center = (center_idx >= center_left) & (center_idx < center_right)
    if not np.any(in_center):
        if profile is not None:
            profile["center_seconds"] += time.perf_counter() - center_start
        return

    center_idx = center_idx[in_center]
    r2 = row_shrink[in_center] * row_shrink[in_center] / (
        diag_lo[in_center] * diag_hi[in_center]
    )
    _add_grouped_sums(partition_sums, center_idx, r2)
    if profile is not None:
        profile["rows_accumulated"] += int(center_idx.size)
        profile["center_seconds"] += time.perf_counter() - center_start


def _add_grouped_sums(
    partition_sums: np.ndarray,
    center_idx: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Accumulate sparse center-index weights without a dense per-chunk bincount."""
    if center_idx.size == 0:
        return
    order = np.argsort(center_idx, kind="stable")
    sorted_idx = center_idx[order]
    sorted_weights = weights[order]
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_idx[1:] != sorted_idx[:-1]) + 1,
        )
    )
    grouped_idx = sorted_idx[starts]
    grouped_weights = np.add.reduceat(sorted_weights, starts)
    partition_sums[grouped_idx] += grouped_weights


def _finish_diag_vector_partition(
    *,
    loci: np.ndarray,
    end_locus: int,
    write_cutoff: int,
    snp_first: int,
    snp_last: int,
    current_locus: int,
    pending_sums: dict[int, float],
    out_path: Path,
    checkpoint: str,
) -> int:
    """Flush completed vector rows and return the next matrix-to-vector locus."""
    _log_vector_checkpoint(f"{checkpoint}_next_loci_start")
    next_loci = loci[loci > end_locus]
    _log_vector_checkpoint(f"{checkpoint}_next_loci_end")
    if next_loci.size:
        current_locus = int(next_loci[0])

    _log_vector_checkpoint(f"{checkpoint}_writable_loci_start")
    writable_loci = np.array(
        [
            locus
            for locus in sorted(pending_sums)
            if snp_first <= locus < write_cutoff and locus <= snp_last
        ],
        dtype=np.int64,
    )
    _log_vector_checkpoint(f"{checkpoint}_writable_loci_end")
    if writable_loci.size:
        _log_vector_checkpoint(f"{checkpoint}_flush_start")
        _append_vector_rows(
            out_path,
            writable_loci,
            np.array([pending_sums[int(locus)] for locus in writable_loci]),
        )
        for locus in writable_loci:
            pending_sums.pop(int(locus), None)
        _log_vector_checkpoint(f"{checkpoint}_flush_end")
    _log_vector_checkpoint(f"{checkpoint}_end")
    return current_locus


def _partition_checkpoint_label(p_index: int, start: int, end: int) -> str:
    """Return the stable checkpoint prefix for one matrix-to-vector partition."""
    return f"matrix_to_vector_array_partition_{p_index}_start={start}_end={end}"


def _log_vector_checkpoint(label: str) -> None:
    """Log a debug-only matrix-to-vector memory checkpoint."""
    log_memory_checkpoint(label, debug=True)


def _load_hdf5_partition(
    path: Path, start: int, end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Covariance partition {path} is missing. The array-backed "
            "matrix-to-vector path requires HDF5 covariance partitions; "
            "regenerate covariance with `ldetect run` or `ldetect calc-covariance`."
        )

    with open_covariance_reader(path, start, end) as reader:
        rows = reader.read_all()
        return (
            _position_array(rows.lo),
            _position_array(rows.hi),
            np.asarray(rows.shrink_ld, dtype=np.float64),
        )


def _r2_rows(
    lo: np.ndarray, hi: np.ndarray, shrink: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diag_mask = lo == hi
    diag_pos = lo[diag_mask]
    diag_val = shrink[diag_mask]
    if diag_pos.size == 0:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return empty_i, empty_i, empty_f

    order = np.argsort(diag_pos, kind="stable")
    diag_pos = diag_pos[order]
    diag_val = diag_val[order]
    unique_diag_pos, unique_idx = np.unique(diag_pos, return_index=True)
    unique_diag_val = diag_val[unique_idx]

    if _sorted_unique_pairs(lo, hi):
        row_lo = lo
        row_hi = hi
        row_shrink = shrink
    else:
        row_order = np.lexsort((hi, lo))
        row_lo = lo[row_order]
        row_hi = hi[row_order]
        row_shrink = shrink[row_order]
        keep_unique = np.ones(row_lo.size, dtype=bool)
        keep_unique[1:] = (row_lo[1:] != row_lo[:-1]) | (row_hi[1:] != row_hi[:-1])
        row_lo = row_lo[keep_unique]
        row_hi = row_hi[keep_unique]
        row_shrink = row_shrink[keep_unique]

    diag_lo_idx = np.searchsorted(unique_diag_pos, row_lo)
    diag_hi_idx = np.searchsorted(unique_diag_pos, row_hi)
    has_diag = (diag_lo_idx < unique_diag_pos.size) & (
        diag_hi_idx < unique_diag_pos.size
    )
    safe_lo_idx = np.minimum(diag_lo_idx, unique_diag_pos.size - 1)
    safe_hi_idx = np.minimum(diag_hi_idx, unique_diag_pos.size - 1)
    has_diag &= (unique_diag_pos[safe_lo_idx] == row_lo) & (
        unique_diag_pos[safe_hi_idx] == row_hi
    )

    row_lo = row_lo[has_diag]
    row_hi = row_hi[has_diag]
    row_shrink = row_shrink[has_diag]
    diag_lo = unique_diag_val[diag_lo_idx[has_diag]]
    diag_hi = unique_diag_val[diag_hi_idx[has_diag]]

    positive = (diag_lo > 0.0) & (diag_hi > 0.0)
    row_lo = row_lo[positive]
    row_hi = row_hi[positive]
    row_shrink = row_shrink[positive]
    diag_lo = diag_lo[positive]
    diag_hi = diag_hi[positive]

    return row_lo, row_hi, row_shrink * row_shrink / (diag_lo * diag_hi)


def _sorted_unique_pairs(lo: np.ndarray, hi: np.ndarray) -> bool:
    if lo.size <= 1:
        return True
    increasing_lo = lo[1:] > lo[:-1]
    same_lo_increasing_hi = (lo[1:] == lo[:-1]) & (hi[1:] > hi[:-1])
    return bool(np.all(increasing_lo | same_lo_increasing_hi))


def _append_vector_rows(
    out_path: Path, loci: np.ndarray, values: np.ndarray
) -> None:
    with gzip.open(out_path, "at") as f:
        writer = csv.writer(f, delimiter="\t")
        for locus, value in zip(loci, values):
            writer.writerow([int(locus), float(value)])
