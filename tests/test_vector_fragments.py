"""Tests for saved matrix-to-vector fragments (_util/vector_fragments.py)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ldetect_lite._util.vector_array import (
    _DiagVectorPartitionPlan,
    _DiagVectorPartitionResult,
)
from ldetect_lite._util.vector_fragments import (
    fragment_path,
    load_vector_fragment,
    remove_vector_fragment,
    save_vector_fragment,
)

PLAN = _DiagVectorPartitionPlan(
    p_index=0,
    start=100,
    end=300,
    next_start=250,
    center_lower_bound=100,
    center_lower_inclusive=True,
    checkpoint="p0",
)
RESULT = _DiagVectorPartitionResult(
    loci=np.array([100, 150, 200, 250, 300], dtype=np.int32),
    sum_loci=np.array([100, 150, 200], dtype=np.int32),
    sum_values=np.array([0.5, 1.25, 2.0]),
    end_locus=275,
    write_cutoff=250,
    profile={"checkpoint": "p0"},
)


def _partition(tmp_path: Path) -> Path:
    path = tmp_path / "1.100.300.h5"
    path.write_bytes(b"partition contents")
    return path


def _load(path: Path, plan=PLAN, snp_first=100, snp_last=450):
    return load_vector_fragment(path, plan=plan, snp_first=snp_first, snp_last=snp_last)


def test_fragment_path_sits_next_to_partition(tmp_path: Path) -> None:
    assert fragment_path(tmp_path / "1.100.300.h5") == tmp_path / "1.100.300.vector.npz"


def test_round_trip_preserves_values_and_dtypes(tmp_path: Path) -> None:
    partition = _partition(tmp_path)
    save_vector_fragment(partition, RESULT, plan=PLAN, snp_first=100, snp_last=450)

    loaded = _load(partition)
    assert loaded is not None
    for field in ("loci", "sum_loci", "sum_values"):
        expected = getattr(RESULT, field)
        actual = getattr(loaded, field)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    assert loaded.end_locus == RESULT.end_locus
    assert loaded.write_cutoff == RESULT.write_cutoff
    assert loaded.profile["checkpoint"] == "p0"
    # No temporary files left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "1.100.300.h5",
        "1.100.300.vector.npz",
    ]


def test_missing_fragment_is_a_miss(tmp_path: Path) -> None:
    assert _load(_partition(tmp_path)) is None


def test_plan_change_is_a_miss(tmp_path: Path) -> None:
    partition = _partition(tmp_path)
    save_vector_fragment(partition, RESULT, plan=PLAN, snp_first=100, snp_last=450)

    other_neighbor = _DiagVectorPartitionPlan(
        p_index=0,
        start=100,
        end=300,
        next_start=260,
        center_lower_bound=100,
        center_lower_inclusive=True,
        checkpoint="p0",
    )
    assert _load(partition, plan=other_neighbor) is None
    assert _load(partition, snp_last=500) is None
    assert _load(partition, snp_first=120) is None
    assert _load(partition) is not None


def test_rewritten_partition_is_a_miss(tmp_path: Path) -> None:
    partition = _partition(tmp_path)
    save_vector_fragment(partition, RESULT, plan=PLAN, snp_first=100, snp_last=450)

    partition.write_bytes(b"different partition contents")
    assert _load(partition) is None


def test_touched_partition_is_a_miss(tmp_path: Path) -> None:
    partition = _partition(tmp_path)
    save_vector_fragment(partition, RESULT, plan=PLAN, snp_first=100, snp_last=450)

    stat = partition.stat()
    os.utime(partition, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert _load(partition) is None


def test_corrupt_fragment_is_a_miss(tmp_path: Path) -> None:
    partition = _partition(tmp_path)
    fragment_path(partition).write_bytes(b"not an npz file")
    assert _load(partition) is None


def test_remove_vector_fragment(tmp_path: Path) -> None:
    partition = _partition(tmp_path)
    save_vector_fragment(partition, RESULT, plan=PLAN, snp_first=100, snp_last=450)
    remove_vector_fragment(partition)
    assert not fragment_path(partition).exists()
    remove_vector_fragment(partition)  # missing file is fine
