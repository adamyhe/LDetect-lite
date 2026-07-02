"""Tests for the r2-zarr benchmark comparison helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_compare_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "r2_zarr_exactness" / "scripts" / "compare_run_modes.py"
    spec = importlib.util.spec_from_file_location("compare_run_modes", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vector_diff_rows_reports_largest_diffs_with_partition_context() -> None:
    compare = _load_compare_module()

    rows = compare.vector_diff_rows(
        comparison="direct_hdf5_vs_matrix_hdf5",
        baseline={100: 1.0, 200: 2.0, 300: 3.0},
        mode={100: 1.5, 200: 2.1, 300: 3.0},
        partitions=[(50, 250), (200, 400)],
        limit=2,
    )

    assert [row["position"] for row in rows] == ["100", "200"]
    assert rows[0]["rank"] == "1"
    assert rows[0]["abs_diff"] == "0.5"
    assert rows[0]["nearest_boundary"] == "50"
    assert rows[0]["distance_to_boundary"] == "50"
    assert rows[1]["rank"] == "2"
    assert rows[1]["nearest_boundary"] == "200"
    assert rows[1]["boundary_role"] == "start"


def test_vector_diff_rows_keeps_missing_positions_at_top() -> None:
    compare = _load_compare_module()

    rows = compare.vector_diff_rows(
        comparison="direct_hdf5_vs_matrix_hdf5",
        baseline={100: 1.0, 200: 2.0},
        mode={100: 1.0, 300: 3.0},
        partitions=[],
        limit=3,
    )

    assert [row["position"] for row in rows[:2]] == ["200", "300"]
    assert rows[0]["baseline_present"] == "True"
    assert rows[0]["mode_present"] == "False"
    assert rows[1]["baseline_present"] == "False"
    assert rows[1]["mode_present"] == "True"
