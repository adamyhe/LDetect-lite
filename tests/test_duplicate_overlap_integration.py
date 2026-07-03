"""Combined duplicate-VCF-position + cross-partition-overlap integration tests.

These are the one previously-untested combination flagged during the EUR
chr8-12/AFR chr11/chr22 reproduction investigation
(notes/ldetect-original-main-pipeline-audit.md): a duplicate physical VCF
position that also sits inside the overlap zone between two partitions.
Unlike tests/test_covariance_io.py and tests/test_metric.py's
divergent-overlap-pair tests (which use hand-typed HDF5 rows), the fixture
here runs real `calc_covariance()` twice on overlapping VCF slices, so it
exercises the actual duplicate-position dedup and covariance kernel end to
end, not just downstream consumers.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from ldetect2.io.covariance_hdf5 import open_covariance_reader
from ldetect2.local_search import LocalSearch
from ldetect2.matrix_analysis import MatrixAnalysis
from ldetect2.metric import Metric
from tests._partition_fixtures import (
    build_two_overlapping_partitions_with_duplicate_position,
)


def _read_vector(path: Path) -> dict[int, float]:
    data: dict[int, float] = {}
    with gzip.open(path, "rt") as f:
        for raw in f:
            row = raw.strip().split()
            if row:
                data[int(row[0])] = float(row[1])
    return data


def test_duplicate_position_overlap_pair_is_bit_identical_across_partitions(
    tmp_path: Path,
) -> None:
    """Two independent `calc_covariance()` runs on overlapping VCF slices
    that both include the same duplicated physical position must compute
    numerically identical values for any pair present in both files.

    This had only ever been argued from first principles (the shrinkage
    formula for a given pair depends only on genotypes/Ne/genetic distance,
    never on which partition computed it) - never empirically checked."""
    store, partitions, _first, _last = (
        build_two_overlapping_partitions_with_duplicate_position(tmp_path)
    )
    (start_a, end_a), (start_b, end_b) = partitions

    with open_covariance_reader(
        store.partition_path("chr1", start_a, end_a), start_a, end_a
    ) as r:
        rows_a = r.read_all()
    with open_covariance_reader(
        store.partition_path("chr1", start_b, end_b), start_b, end_b
    ) as r:
        rows_b = r.read_all()

    # (300, 300), (300, 400), (400, 400) are computed independently by both
    # partitions (300 and 400 both fall in the overlap zone [250, 400]).
    redundant_pairs = [(300, 300), (300, 400), (400, 400)]
    for lo, hi in redundant_pairs:
        mask_a = (rows_a.lo == lo) & (rows_a.hi == hi)
        mask_b = (rows_b.lo == lo) & (rows_b.hi == hi)
        assert mask_a.sum() == 1, f"expected exactly one ({lo},{hi}) row in partition A"
        assert mask_b.sum() == 1, f"expected exactly one ({lo},{hi}) row in partition B"
        np.testing.assert_array_equal(
            rows_a.shrink_ld[mask_a], rows_b.shrink_ld[mask_b]
        )


def test_matrix_analysis_agrees_across_duplicate_overlap(tmp_path: Path) -> None:
    store, _partitions, _first, _last = (
        build_two_overlapping_partitions_with_duplicate_position(tmp_path)
    )
    array_path = tmp_path / "array.txt.gz"
    legacy_path = tmp_path / "legacy.txt.gz"

    MatrixAnalysis("chr1", store).calc_diag_array(array_path)
    MatrixAnalysis("chr1", store)._calc_diag_lean_legacy(legacy_path)

    array_vector = _read_vector(array_path)
    legacy_vector = _read_vector(legacy_path)
    assert array_vector.keys() == legacy_vector.keys()
    for locus, value in array_vector.items():
        assert value == pytest.approx(legacy_vector[locus])


def test_metric_agrees_across_duplicate_overlap(tmp_path: Path) -> None:
    store, _partitions, first, last = (
        build_two_overlapping_partitions_with_duplicate_position(tmp_path)
    )
    breakpoints = [400]

    legacy = Metric("chr1", store, breakpoints, first, last)._calc_metric_lean()
    fast = Metric("chr1", store, breakpoints, first, last).calc_metric()

    assert fast["sum"] == pytest.approx(legacy["sum"])
    assert fast["N_nonzero"] == legacy["N_nonzero"]
    assert fast["N_zero"] == pytest.approx(legacy["N_zero"])


def test_local_search_agrees_across_duplicate_overlap(tmp_path: Path) -> None:
    store, _partitions, first, last = (
        build_two_overlapping_partitions_with_duplicate_position(tmp_path)
    )
    # Two breakpoints (rather than one) so N_zero (the block-area
    # denominator) is nonzero -- a single breakpoint at the far locus leaves
    # a degenerate one-locus block, unrelated to duplicate/overlap handling.
    breakpoints = [300, 500]
    idx = 0
    start_search, stop_search = first, (breakpoints[0] + breakpoints[1]) // 2
    total = Metric("chr1", store, breakpoints, first, last).calc_metric()

    legacy_search = LocalSearch(
        "chr1",
        start_search,
        stop_search,
        idx,
        breakpoints,
        total["sum"],
        total["N_zero"],
        store,
        use_decimal=True,
    )
    legacy_search.init_search()
    legacy_bp, legacy_metric = legacy_search.search()

    fast_search = LocalSearch(
        "chr1",
        start_search,
        stop_search,
        idx,
        breakpoints,
        total["sum"],
        total["N_zero"],
        store,
        use_decimal=False,
    )
    fast_search.init_search()
    fast_bp, fast_metric = fast_search.search()

    assert fast_bp == legacy_bp
    if fast_metric is not None and legacy_metric is not None:
        assert fast_metric["sum"] == pytest.approx(legacy_metric["sum"])
