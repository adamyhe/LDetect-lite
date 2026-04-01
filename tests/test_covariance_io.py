"""Tests for ldetect2.io.covariance."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from ldetect2.io.covariance import (
    delete_loci_smaller_than,
    delete_loci_smaller_than_leanest,
    insert_into_matrix_lean,
    read_partition_into_matrix_lean,
)
from ldetect2.io.partitions import CovarianceStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COV_CONTENT = (
    "snpA snpA 100 100 1.0 1.0 0.5 0.5\n"
    "snpA snpB 100 200 1.0 2.0 0.3 0.3\n"
    "snpB snpB 200 200 2.0 2.0 0.7 0.7\n"
    "snpB snpC 200 300 2.0 3.0 0.1 0.1\n"
    "snpC snpC 300 300 3.0 3.0 0.9 0.9\n"
)


def _make_store(tmp_path: Path) -> tuple[CovarianceStore, list[tuple[int, int]]]:
    """Create a minimal CovarianceStore in *tmp_path* with a 3-SNP fixture."""
    root = tmp_path / "cov"
    (root / "scripts").mkdir(parents=True)
    (root / "testchr").mkdir()
    (root / "scripts" / "testchr_partitions").write_text("100 300\n")
    gz_path = root / "testchr" / "testchr.100.300.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write(_COV_CONTENT)
    store = CovarianceStore(root=root)
    partitions: list[tuple[int, int]] = [(100, 300)]
    return store, partitions


# ---------------------------------------------------------------------------
# insert_into_matrix_lean
# ---------------------------------------------------------------------------

def test_insert_diagonal():
    matrix: dict = {}
    locus_list: list[int] = []
    row = ["snpA", "snpA", "100", "100", "1.0", "1.0", "0.5", "0.5"]
    insert_into_matrix_lean(row, matrix, locus_list)
    assert 100 in matrix
    assert matrix[100][100] == pytest.approx(0.5)
    assert locus_list == [100]


def test_insert_off_diagonal():
    matrix: dict = {}
    locus_list: list[int] = []
    row = ["snpA", "snpB", "100", "200", "1.0", "2.0", "0.3", "0.3"]
    insert_into_matrix_lean(row, matrix, locus_list)
    # lo=100, hi=200 — should create matrix[100][200] = 0.3
    assert matrix[100][200] == pytest.approx(0.3)


def test_insert_locus_list_sorted():
    matrix: dict = {}
    locus_list: list[int] = []
    for row in _COV_CONTENT.strip().split("\n"):
        insert_into_matrix_lean(row.split(), matrix, locus_list)
    assert locus_list == [100, 200, 300]


def test_insert_no_duplicate():
    matrix: dict = {}
    locus_list: list[int] = []
    row = ["snpA", "snpA", "100", "100", "1.0", "1.0", "0.5", "0.5"]
    insert_into_matrix_lean(row, matrix, locus_list)
    insert_into_matrix_lean(row, matrix, locus_list)
    assert locus_list.count(100) == 1


# ---------------------------------------------------------------------------
# read_partition_into_matrix_lean
# ---------------------------------------------------------------------------

def test_read_partition_keys(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    assert set(locus_list) == {100, 200, 300}


def test_read_partition_diagonal_values(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    assert matrix[100][100] == pytest.approx(0.5)
    assert matrix[200][200] == pytest.approx(0.7)
    assert matrix[300][300] == pytest.approx(0.9)


def test_read_partition_off_diagonal_values(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    assert matrix[100][200] == pytest.approx(0.3)
    assert matrix[200][300] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# delete_loci_smaller_than_leanest
# ---------------------------------------------------------------------------

def test_delete_loci_removes_entries(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    delete_loci_smaller_than_leanest(250, matrix, locus_list)
    assert 100 not in matrix
    assert 200 not in matrix
    assert 300 in matrix


def test_delete_loci_updates_locus_list(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    delete_loci_smaller_than_leanest(250, matrix, locus_list)
    assert locus_list == [300]


def test_delete_loci_exact_boundary(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    # cutoff=200 → delete loci < 200 → only locus 100 removed
    delete_loci_smaller_than_leanest(200, matrix, locus_list)
    assert 100 not in matrix
    assert 200 in matrix
    assert locus_list == [200, 300]


# ---------------------------------------------------------------------------
# delete_loci_smaller_than (with deleted list)
# ---------------------------------------------------------------------------

def test_delete_with_deleted_list(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    locus_list_deleted: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    delete_loci_smaller_than(250, matrix, locus_list, locus_list_deleted)
    assert locus_list_deleted == [100, 200]
    assert locus_list == [300]
