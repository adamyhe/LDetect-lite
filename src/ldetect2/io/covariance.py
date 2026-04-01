"""Covariance matrix I/O: reading partition files and managing the in-memory matrix."""

from __future__ import annotations

import bisect
import csv
import gzip
from pathlib import Path
from typing import Any

from ldetect2._util.logging import log_msg
from ldetect2.io.partitions import CovarianceStore

# ---------------------------------------------------------------------------
# Column indices for the 8-column covariance partition file format:
#   i_id  j_id  i_pos  j_pos  i_gpos  j_gpos  naive_ld  shrink_ld
# ---------------------------------------------------------------------------
_COL_I_ID = 0
_COL_J_ID = 1
_COL_I_POS = 2
_COL_J_POS = 3
_COL_G_I = 4
_COL_G_J = 5
_COL_NAIVE = 6
_COL_SHRINK = 7

# Type aliases
Matrix = dict[int, Any]
LocusList = list[int]


# ---------------------------------------------------------------------------
# Full matrix (stores all fields per cell — needed for heatmap generation)
# ---------------------------------------------------------------------------

def insert_into_matrix(
    row: list[str],
    matrix: Matrix,
    locus_list: LocusList,
    symmetric: bool = False,
) -> None:
    """Insert one covariance row into the full matrix."""
    loc_i = int(row[_COL_I_POS])
    loc_j = int(row[_COL_J_POS])

    if loc_i <= loc_j:
        lo, hi = loc_i, loc_j
        lo_id, hi_id = row[_COL_I_ID], row[_COL_J_ID]
        lo_g = float(row[_COL_G_I])
        hi_g = float(row[_COL_G_J])
    else:
        lo, hi = loc_j, loc_i
        lo_id, hi_id = row[_COL_J_ID], row[_COL_I_ID]
        lo_g = float(row[_COL_G_J])
        hi_g = float(row[_COL_G_I])

    naive = float(row[_COL_NAIVE])
    shrink = float(row[_COL_SHRINK])

    if lo not in matrix:
        matrix[lo] = {"data": {}, "desc": {"l_id": lo_id, "l_g": lo_g}}
        locus_list.insert(bisect.bisect_left(locus_list, lo), lo)

    if hi not in matrix[lo]["data"]:
        matrix[lo]["data"][hi] = {
            "r_id": hi_id, "r_g": hi_g, "naive": naive, "shrink": shrink
        }

    if symmetric:
        if hi not in matrix:
            matrix[hi] = {"data": {}, "desc": {"l_id": hi_id, "l_g": hi_g}}
            locus_list.insert(bisect.bisect_left(locus_list, hi), hi)
        if lo not in matrix[hi]["data"]:
            matrix[hi]["data"][lo] = {
                "r_id": lo_id, "r_g": lo_g, "naive": naive, "shrink": shrink
            }


def read_partition_into_matrix(
    partitions: list[tuple[int, int]],
    p_index: int,
    matrix: Matrix,
    locus_list: LocusList,
    name: str,
    store: CovarianceStore,
    snp_first: int,
    snp_last: int,
    symmetric: bool = False,
) -> None:
    path = store.partition_path(name, partitions[p_index][0], partitions[p_index][1])
    try:
        with gzip.open(path, "rt") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                insert_into_matrix(row, matrix, locus_list, symmetric)
    except EOFError as exc:
        log_msg(f"EOFError reading {path}: {exc}")


# ---------------------------------------------------------------------------
# Lean matrix (stores only shrink value per cell — primary path)
# ---------------------------------------------------------------------------

def insert_into_matrix_lean(
    row: list[str],
    matrix: Matrix,
    locus_list: LocusList,
) -> None:
    """Insert one covariance row into the lean matrix (shrink values only)."""
    loc_i = int(row[_COL_I_POS])
    loc_j = int(row[_COL_J_POS])

    lo, hi = (loc_i, loc_j) if loc_i <= loc_j else (loc_j, loc_i)
    shrink = float(row[_COL_SHRINK])

    if lo not in matrix:
        matrix[lo] = {}
        locus_list.insert(bisect.bisect_left(locus_list, lo), lo)

    if hi not in matrix[lo]:
        matrix[lo][hi] = shrink


def read_partition_into_matrix_lean(
    partitions: list[tuple[int, int]],
    p_index: int,
    matrix: Matrix,
    locus_list: LocusList,
    name: str,
    store: CovarianceStore,
    snp_first: int,
    snp_last: int,
    symmetric: bool = False,
) -> None:
    path = store.partition_path(name, partitions[p_index][0], partitions[p_index][1])
    try:
        with gzip.open(path, "rt") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                insert_into_matrix_lean(row, matrix, locus_list)
    except EOFError as exc:
        log_msg(f"EOFError reading {path}: {exc}")


# ---------------------------------------------------------------------------
# Matrix deletion helpers
# ---------------------------------------------------------------------------

def _delete_matrix(cutoff: int, locus_list: LocusList, matrix: Matrix) -> int:
    """Delete loci < cutoff from matrix; return count deleted."""
    cnt = 0
    while cnt < len(locus_list) and locus_list[cnt] < cutoff:
        del matrix[locus_list[cnt]]
        cnt += 1
    return cnt


def delete_loci_smaller_than(
    cutoff: int,
    matrix: Matrix,
    locus_list: LocusList,
    locus_list_deleted: LocusList,
) -> None:
    log_msg("Deleting segment of matrix")
    cnt = _delete_matrix(cutoff, locus_list, matrix)
    locus_list_deleted.extend(locus_list[:cnt])
    del locus_list[:cnt]


def delete_loci_smaller_than_leanest(
    cutoff: int,
    matrix: Matrix,
    locus_list: LocusList,
) -> None:
    log_msg("Deleting segment of matrix (leanest)")
    cnt = _delete_matrix(cutoff, locus_list, matrix)
    del locus_list[:cnt]


def delete_loci_smaller_than_lean(
    cutoff: int,
    matrix: Matrix,
    locus_list: LocusList,
    locus_list_deleted: LocusList,
    out_path: Path,
    sum_list: dict[int, float],
) -> None:
    log_msg("Writing segment of matrix to file and deleting")
    cnt = _delete_matrix(cutoff, locus_list, matrix)
    _write_corr_vector_append(out_path, locus_list, 0, cnt, sum_list)
    for i in range(cnt):
        sum_list.pop(locus_list[i], None)
    del locus_list[:cnt]


# ---------------------------------------------------------------------------
# Correlation vector output
# ---------------------------------------------------------------------------

def write_corr_vector(
    out_path: Path,
    locus_list: LocusList,
    locus_list_deleted: LocusList,
    sum_list: dict[int, float],
    sum_list_len: dict[int, int] | None = None,
) -> None:
    """Write the full correlation vector (deleted loci first, then active)."""
    with gzip.open(out_path, "wt") as f:
        writer = csv.writer(f, delimiter="\t")
        n_del = len(locus_list_deleted)
        n_act = len(locus_list)
        log_msg(f"Writing {n_del} deleted + {n_act} active loci")
        for locus_list_part in (locus_list_deleted, locus_list):
            for loc in locus_list_part:
                if loc not in sum_list:
                    log_msg(f"Warning: locus {loc} not in sum_list")
                    continue
                if sum_list_len:
                    val: float = sum_list[loc] / sum_list_len[loc]
                else:
                    val = sum_list[loc]
                writer.writerow([loc, val])
    log_msg("Output done")


def write_corr_vector_slice(
    out_path: Path,
    locus_list: LocusList,
    start: int,
    end: int,
    sum_list: dict[int, float],
    sum_list_len: dict[int, int] | None = None,
) -> None:
    """Append a slice of locus_list to an existing (or new) gzipped vector file."""
    _write_corr_vector_append(out_path, locus_list, start, end, sum_list, sum_list_len)


def _write_corr_vector_append(
    out_path: Path,
    locus_list: LocusList,
    start: int,
    end: int,
    sum_list: dict[int, float],
    sum_list_len: dict[int, int] | None = None,
) -> None:
    with gzip.open(out_path, "at") as f:
        writer = csv.writer(f, delimiter="\t")
        for i in range(start, end):
            loc = locus_list[i]
            if loc not in sum_list:
                log_msg(f"Warning: locus {loc} not in sum_list")
                continue
            if sum_list_len:
                val2: float = sum_list[loc] / sum_list_len[loc]
            else:
                val2 = sum_list[loc]
            writer.writerow([loc, val2])
    log_msg("Output done")
