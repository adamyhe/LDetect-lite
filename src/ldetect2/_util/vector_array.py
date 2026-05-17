"""Array-backed covariance-to-vector helpers."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

from ldetect2._util.covariance_array import ChromosomeCovariance
from ldetect2.io.partitions import CovarianceStore

_REQUIRED_NPZ_KEYS = frozenset({"i_pos", "j_pos", "shrink_ld"})


def _position_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return arr
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
) -> None:
    """Write the diagonal correlation-sum vector from ``.npz`` partitions."""
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
    for p_index, (start, end) in enumerate(partitions):
        if p_index + 1 < len(partitions) and snp_first >= partitions[p_index + 1][0]:
            continue

        if partition_arrays is None:
            path = store.partition_path(name, start, end)
            i_pos, j_pos, shrink = _load_npz_partition(path)
        else:
            partition = partition_arrays[p_index]
            i_pos = partition.i_pos
            j_pos = partition.j_pos
            shrink = partition.shrink_ld
        rows_in_partition = (
            (i_pos >= start)
            & (i_pos <= end)
            & (j_pos >= start)
            & (j_pos <= end)
        )
        if not np.any(rows_in_partition):
            continue

        i_pos = i_pos[rows_in_partition]
        j_pos = j_pos[rows_in_partition]
        shrink = shrink[rows_in_partition]
        if np.all(i_pos <= j_pos):
            lo = i_pos
            hi = j_pos
        else:
            lo = np.minimum(i_pos, j_pos)
            hi = np.maximum(i_pos, j_pos)
        loci = np.unique(np.concatenate((lo, hi)))
        if loci.size == 0:
            continue

        if p_index + 1 < len(partitions):
            end_locus = int((end + partitions[p_index + 1][0]) / 2)
            write_cutoff = partitions[p_index + 1][0]
        else:
            in_requested_range = loci[loci <= snp_last]
            if in_requested_range.size == 0:
                continue
            end_locus = int(in_requested_range[-1])
            write_cutoff = end_locus

        r2_lo, r2_hi, r2 = _r2_rows(lo, hi, shrink)
        if r2.size:
            lo_idx = np.searchsorted(loci, r2_lo)
            hi_idx = np.searchsorted(loci, r2_hi)
            idx_delta = hi_idx - lo_idx
            even_delta = idx_delta % 2 == 0
            legacy_reachable = even_delta | (lo_idx > 0)
            if np.any(legacy_reachable):
                lo_idx = lo_idx[legacy_reachable]
                hi_idx = hi_idx[legacy_reachable]
                r2 = r2[legacy_reachable]
                center_idx = (lo_idx + hi_idx) // 2
                center_loci = loci[center_idx]

                keep_center = (
                    (center_loci >= current_locus)
                    & (center_loci <= end_locus)
                    & (center_loci <= snp_last)
                )
                if np.any(keep_center):
                    sums = np.bincount(
                        center_idx[keep_center],
                        weights=r2[keep_center],
                        minlength=loci.size,
                    )
                    for locus, value in zip(loci[sums > 0.0], sums[sums > 0.0]):
                        pending_sums[int(locus)] = pending_sums.get(
                            int(locus), 0.0
                        ) + float(value)

        next_loci = loci[loci > end_locus]
        if next_loci.size:
            current_locus = int(next_loci[0])

        writable_loci = np.array(
            [
                locus
                for locus in sorted(pending_sums)
                if snp_first <= locus < write_cutoff and locus <= snp_last
            ],
            dtype=np.int64,
        )
        if writable_loci.size:
            _append_vector_rows(
                out_path,
                writable_loci,
                np.array([pending_sums[int(locus)] for locus in writable_loci]),
            )
            for locus in writable_loci:
                pending_sums.pop(int(locus), None)


def _load_npz_partition(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Covariance partition {path} is missing. The array-backed "
            "matrix-to-vector path requires .npz covariance partitions; "
            "regenerate covariance with `ldetect2 run` or `ldetect2 calc-covariance`."
        )

    with np.load(path) as data:
        missing = sorted(_REQUIRED_NPZ_KEYS - set(data.files))
        if missing:
            raise ValueError(
                f"Covariance partition {path} is missing required field(s): "
                f"{', '.join(missing)}"
            )
        return (
            _position_array(data["i_pos"]),
            _position_array(data["j_pos"]),
            np.asarray(data["shrink_ld"], dtype=np.float64),
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
