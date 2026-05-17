"""Tests for ldetect2.io.partitions."""

from __future__ import annotations

from pathlib import Path

import pytest

from ldetect2.io.partitions import (
    CovarianceStore,
    first_last,
    read_partitions,
    relevant_subpartitions,
)

# ---------------------------------------------------------------------------
# CovarianceStore path construction
# ---------------------------------------------------------------------------


def test_partitions_path():
    store = CovarianceStore(root=Path("/data/cov"))
    assert store.partitions_path("chr2") == Path("/data/cov/chr2_partitions.txt")


def test_partition_path():
    store = CovarianceStore(root=Path("/data/cov"))
    assert store.partition_path("chr2", 100, 200) == Path(
        "/data/cov/chr2/chr2.100.200.npz"
    )


def test_partitions_dir():
    store = CovarianceStore(root=Path("/data/cov"))
    assert store.partitions_dir == Path("/data/cov")


# ---------------------------------------------------------------------------
# read_partitions against the real example fixture
# ---------------------------------------------------------------------------


def test_read_partitions_example(example_store):
    partitions = read_partitions("chr2", example_store)
    assert len(partitions) == 1
    start, end = partitions[0]
    assert start == 39967768
    assert end == 40067768


# ---------------------------------------------------------------------------
# read_partitions with a temporary file
# ---------------------------------------------------------------------------

def test_read_partitions_multi(tmp_path):
    (tmp_path / "testchr_partitions.txt").write_text("100 200\n200 300\n300 400\n")
    store = CovarianceStore(root=tmp_path)
    partitions = read_partitions("testchr", store)
    assert partitions == [(100, 200), (200, 300), (300, 400)]


# ---------------------------------------------------------------------------
# relevant_subpartitions
# ---------------------------------------------------------------------------

PARTS = [(100, 200), (200, 300), (300, 400)]


def test_relevant_subpartitions_single():
    # Entirely within one partition
    result = relevant_subpartitions(PARTS, 150, 180)
    assert result == [(100, 200)]


def test_relevant_subpartitions_two():
    result = relevant_subpartitions(PARTS, 150, 250)
    assert result == [(100, 200), (200, 300)]


def test_relevant_subpartitions_all():
    result = relevant_subpartitions(PARTS, 100, 400)
    assert result == PARTS


def test_relevant_subpartitions_snp_first_on_boundary():
    # snp_first=200 falls in partition 0 (100<=200<=200)
    result = relevant_subpartitions(PARTS, 200, 350)
    assert result[0] == (100, 200)
    assert (300, 400) in result


def test_relevant_subpartitions_snp_last_on_boundary():
    # snp_last=300 falls in both partition 1 and 2 — last match wins (partition 2)
    result = relevant_subpartitions(PARTS, 150, 300)
    assert result[-1] == (300, 400)


def test_relevant_subpartitions_not_found():
    with pytest.raises(ValueError):
        relevant_subpartitions(PARTS, 50, 150)  # snp_first=50 not in any partition

    with pytest.raises(ValueError):
        relevant_subpartitions(PARTS, 150, 500)  # snp_last=500 not in any partition


# ---------------------------------------------------------------------------
# first_last
# ---------------------------------------------------------------------------

def test_first_last_auto(example_store):
    first, last = first_last("chr2", example_store, -1, -1)
    assert first == 39967768
    assert last == 40067768


def test_first_last_explicit(example_store):
    first, last = first_last("chr2", example_store, 39967900, 40000000)
    assert first == 39967900
    assert last == 40000000


def test_first_last_auto_first_only(example_store):
    first, last = first_last("chr2", example_store, -1, 40000000)
    assert first == 39967768
    assert last == 40000000


def test_first_last_auto_last_only(example_store):
    first, last = first_last("chr2", example_store, 39967900, -1)
    assert first == 39967900
    assert last == 40067768
