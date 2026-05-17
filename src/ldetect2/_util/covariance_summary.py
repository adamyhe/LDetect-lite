"""Covariance partition diagnostics for metric-cache memory planning."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ldetect2.io.partitions import CovarianceStore, first_last, get_final_partitions

MiB = 1024 * 1024


@dataclass(frozen=True)
class CovariancePartitionSummary:
    """Summary statistics for one covariance partition or chromosome total."""

    name: str
    partition_start: int | str
    partition_end: int | str
    rows: int
    diag_rows: int
    offdiag_rows: int
    owned_offdiag_rows: int
    unique_loci: int
    final_int64_mb: float
    final_int32_mb: float
    peak_low_mb: float
    peak_high_mb: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "partition_start": self.partition_start,
            "partition_end": self.partition_end,
            "rows": self.rows,
            "diag_rows": self.diag_rows,
            "offdiag_rows": self.offdiag_rows,
            "owned_offdiag_rows": self.owned_offdiag_rows,
            "unique_loci": self.unique_loci,
            "final_int64_mb": self.final_int64_mb,
            "final_int32_mb": self.final_int32_mb,
            "peak_low_mb": self.peak_low_mb,
            "peak_high_mb": self.peak_high_mb,
        }


SUMMARY_COLUMNS = [
    "name",
    "partition_start",
    "partition_end",
    "rows",
    "diag_rows",
    "offdiag_rows",
    "owned_offdiag_rows",
    "unique_loci",
    "final_int64_mb",
    "final_int32_mb",
    "peak_low_mb",
    "peak_high_mb",
]


def summarize_covariance(
    name: str,
    store: CovarianceStore,
    snp_first: int = -1,
    snp_last: int = -1,
) -> tuple[list[CovariancePartitionSummary], CovariancePartitionSummary]:
    """Summarize covariance partitions and estimate array-cache memory.

    Counts and estimates follow the normal array metric path: rows are first
    restricted to the requested SNP range, then overlapping partitions are
    assigned by lower-endpoint ownership.
    """
    snp_first, snp_last = first_last(name, store, snp_first, snp_last)
    partitions = get_final_partitions(store, name, snp_first, snp_last)

    summaries: list[CovariancePartitionSummary] = []
    total_rows = 0
    total_diag = 0
    total_offdiag = 0
    total_owned_offdiag = 0
    total_loci: set[int] = set()

    for p_index, (start, end) in enumerate(partitions):
        i_pos, j_pos = _read_partition_positions(store.partition_path(name, start, end))
        rows = int(i_pos.size)
        diag_rows = int(np.count_nonzero(i_pos == j_pos))
        offdiag_rows = int(np.count_nonzero(i_pos < j_pos))
        owned = _owned_in_range_mask(
            i_pos,
            j_pos,
            partitions,
            p_index,
            snp_first,
            snp_last,
        )
        owned_offdiag = owned & (i_pos < j_pos)
        loci = np.unique(np.concatenate((i_pos[owned], j_pos[owned])))
        unique_loci = int(loci.size)
        owned_offdiag_rows = int(np.count_nonzero(owned_offdiag))

        total_rows += rows
        total_diag += diag_rows
        total_offdiag += offdiag_rows
        total_owned_offdiag += owned_offdiag_rows
        total_loci.update(int(pos) for pos in loci)

        summaries.append(
            _make_summary(
                name=name,
                partition_start=start,
                partition_end=end,
                rows=rows,
                diag_rows=diag_rows,
                offdiag_rows=offdiag_rows,
                owned_offdiag_rows=owned_offdiag_rows,
                unique_loci=unique_loci,
            )
        )

    total = _make_summary(
        name=name,
        partition_start="TOTAL",
        partition_end="TOTAL",
        rows=total_rows,
        diag_rows=total_diag,
        offdiag_rows=total_offdiag,
        owned_offdiag_rows=total_owned_offdiag,
        unique_loci=len(total_loci),
    )
    return summaries, total


def _read_partition_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.exists():
        data = np.load(path)
        return (
            np.asarray(data["i_pos"], dtype=np.int64),
            np.asarray(data["j_pos"], dtype=np.int64),
        )

    text_path = path.with_suffix(".gz")
    i_pos: list[int] = []
    j_pos: list[int] = []
    with gzip.open(text_path, "rt") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            i_pos.append(int(row[2]))
            j_pos.append(int(row[3]))
    return np.asarray(i_pos, dtype=np.int64), np.asarray(j_pos, dtype=np.int64)


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


def _make_summary(
    *,
    name: str,
    partition_start: int | str,
    partition_end: int | str,
    rows: int,
    diag_rows: int,
    offdiag_rows: int,
    owned_offdiag_rows: int,
    unique_loci: int,
) -> CovariancePartitionSummary:
    final_int64 = _estimate_final_mb(unique_loci, owned_offdiag_rows, "int64")
    final_int32 = _estimate_final_mb(unique_loci, owned_offdiag_rows, "int32")
    return CovariancePartitionSummary(
        name=name,
        partition_start=partition_start,
        partition_end=partition_end,
        rows=rows,
        diag_rows=diag_rows,
        offdiag_rows=offdiag_rows,
        owned_offdiag_rows=owned_offdiag_rows,
        unique_loci=unique_loci,
        final_int64_mb=round(final_int64, 3),
        final_int32_mb=round(final_int32, 3),
        peak_low_mb=round(final_int64 * 3, 3),
        peak_high_mb=round(final_int64 * 6, 3),
    )


def _estimate_final_mb(
    unique_loci: int,
    pair_rows: int,
    position_dtype: Literal["int32", "int64"],
) -> float:
    position_bytes = 4 if position_dtype == "int32" else 8
    pair_bytes = pair_rows * (position_bytes * 2 + 8)
    loci_bytes = unique_loci * position_bytes
    return (pair_bytes + loci_bytes) / MiB
