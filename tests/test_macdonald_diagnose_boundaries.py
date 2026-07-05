"""Tests for the MacDonald2022 boundary diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / "examples"
    / "MacDonald2022"
    / "scripts"
    / "diagnose_boundaries.py"
)
SPEC = importlib.util.spec_from_file_location("diagnose_boundaries", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
diagnose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnose)


def test_nearest_prefers_lower_position_on_tie():
    assert diagnose.nearest(20, [10, 30]) == (10, -10)


def test_classify_reciprocal_shift():
    classification = diagnose.classify_mismatch(
        "ours_to_ref",
        position=100,
        query_boundaries=[0, 100, 200],
        reference_boundaries=[0, 130, 200],
        reference_blocks=[(0, 130), (130, 200)],
    )

    assert classification == "shifted_boundary"


def test_classify_extra_and_missing_splits():
    extra = diagnose.classify_mismatch(
        "ours_to_ref",
        position=100,
        query_boundaries=[0, 100, 200, 300],
        reference_boundaries=[0, 200, 300],
        reference_blocks=[(0, 200), (200, 300)],
    )
    missing = diagnose.classify_mismatch(
        "ref_to_ours",
        position=100,
        query_boundaries=[0, 100, 200, 300],
        reference_boundaries=[0, 200, 300],
        reference_blocks=[(0, 200), (200, 300)],
    )

    assert extra == "extra_split"
    assert missing == "missing_split"


def test_diagnostic_rows_include_optional_context():
    rows = list(
        diagnose.diagnostic_rows(
            ours={"chr1": [(0, 100), (100, 300)]},
            reference={"chr1": [(0, 160), (160, 300)]},
            tolerance=50,
            window=100,
            selected_chroms=set(),
            centromeres={"chr1": (90, 110)},
            genetic_maps={"chr1": [(50, 0.1), (100, 0.2), (150, 0.3)]},
            snp_positions={"chr1": [75, 100, 125]},
        )
    )

    ours_row = next(
        row
        for row in rows
        if row["source"] == "ours_to_ref" and row["position"] == 100
    )
    assert ours_row["classification"] == "shifted_boundary"
    assert ours_row["position_in_centromere"] == "1"
    assert ours_row["map_points_in_window"] == "3"
    assert ours_row["nearest_map_distance_bp"] == "0"
    assert ours_row["snps_in_window"] == "3"
