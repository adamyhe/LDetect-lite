"""Tests for ldetect_lite.interpolate_maps."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from ldetect_lite.interpolate_maps import (
    interpolate,
    interpolate_hapmap,
    interpolate_intervals,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAP_HEADER = "position rate_cM_Mb genetic_position_cM\n"
_MAP_ENTRIES = (
    "1000 1.0 0.1\n"
    "2000 1.0 0.2\n"
    "3000 2.0 0.4\n"
)

# BED columns: chrom start end rs_id
# interpolate() uses col 2 (end) as position and col 3 as rs_id
_BED_CONTENT = (
    "chr1 0 500 rs_before\n"     # pos=500  → before first marker → clamped to 0.0
    "chr1 0 1000 rs_exact1\n"    # pos=1000 → exact match         → 0.1
    "chr1 0 1500 rs_mid1\n"      # pos=1500 → midpoint [1000,2000] → 0.15
    "chr1 0 2500 rs_mid2\n"      # pos=2500 → midpoint [2000,3000] → 0.3  (non-uniform)
    "chr1 0 3500 rs_after\n"     # pos=3500 → after last marker   → 0.4
)

_EXPECTED = {
    "rs_before": 0.0,
    "rs_exact1": 0.1,
    "rs_mid1":   0.15,
    "rs_mid2":   0.3,
    "rs_after":  0.4,
}


def _write_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    snp_file = tmp_path / "snps.bed"
    snp_file.write_text(_BED_CONTENT)

    map_file = tmp_path / "map.gz"
    with gzip.open(map_file, "wt") as f:
        f.write(_MAP_HEADER)
        f.write(_MAP_ENTRIES)

    output = tmp_path / "out.gz"
    return snp_file, map_file, output


def _read_output(output: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with gzip.open(output, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                result[parts[0]] = float(parts[2])
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_before_first_marker(tmp_path):
    snp_file, map_file, output = _write_fixtures(tmp_path)
    interpolate(snp_file, map_file, output)
    result = _read_output(output)
    assert result["rs_before"] == pytest.approx(0.0, abs=1e-10)


def test_exact_match(tmp_path):
    snp_file, map_file, output = _write_fixtures(tmp_path)
    interpolate(snp_file, map_file, output)
    result = _read_output(output)
    assert result["rs_exact1"] == pytest.approx(0.1, abs=1e-10)


def test_interpolation_uniform_interval(tmp_path):
    snp_file, map_file, output = _write_fixtures(tmp_path)
    interpolate(snp_file, map_file, output)
    result = _read_output(output)
    assert result["rs_mid1"] == pytest.approx(0.15, abs=1e-10)


def test_interpolation_nonuniform_interval(tmp_path):
    snp_file, map_file, output = _write_fixtures(tmp_path)
    interpolate(snp_file, map_file, output)
    result = _read_output(output)
    # Interval [2000,3000]: gpos 0.2→0.4, midpoint (2500) → 0.3
    assert result["rs_mid2"] == pytest.approx(0.3, abs=1e-10)


def test_after_last_marker(tmp_path):
    snp_file, map_file, output = _write_fixtures(tmp_path)
    interpolate(snp_file, map_file, output)
    result = _read_output(output)
    assert result["rs_after"] == pytest.approx(0.4, abs=1e-10)


def test_output_row_count(tmp_path):
    snp_file, map_file, output = _write_fixtures(tmp_path)
    interpolate(snp_file, map_file, output)
    result = _read_output(output)
    assert len(result) == 5


def test_output_rs_ids_preserved(tmp_path):
    snp_file, map_file, output = _write_fixtures(tmp_path)
    interpolate(snp_file, map_file, output)
    result = _read_output(output)
    assert set(result.keys()) == set(_EXPECTED.keys())


def test_single_map_entry_all_clamped(tmp_path):
    """When the map has only one entry, all SNPs return that gpos or 0."""
    snp_file = tmp_path / "snps.bed"
    snp_file.write_text(
        "chr1 0 500 rs_before\n"
        "chr1 0 1000 rs_exact\n"
        "chr1 0 2000 rs_after\n"
    )
    map_file = tmp_path / "map.gz"
    with gzip.open(map_file, "wt") as f:
        f.write("pos rate cM\n")
        f.write("1000 1.0 0.5\n")
    output = tmp_path / "out.gz"
    interpolate(snp_file, map_file, output)
    result = _read_output(output)
    assert result["rs_before"] == pytest.approx(0.0, abs=1e-10)
    assert result["rs_exact"] == pytest.approx(0.5, abs=1e-10)
    assert result["rs_after"] == pytest.approx(0.5, abs=1e-10)


# ---------------------------------------------------------------------------
# interpolate_intervals: port of MacDonald et al.'s R interval-rate algorithm
# ---------------------------------------------------------------------------
#
# Map rows are (Begin, rate_cM_Mb, cumulative_cM_at_End), matching
# convert_decode_map.py's output. Three synthetic intervals:
#   I0: Begin=1000, rate=1.0 cM/Mb, End=2000 -> increment=0.001, cumcM=0.001
#   I1: Begin=2000, rate=2.0 cM/Mb, End=3000 -> increment=0.002, cumcM=0.003
#   I2: Begin=3000, rate=0.5 cM/Mb, End=4000 -> increment=0.0005, cumcM=0.0035

_INTERVAL_MAP_HEADER = "position rate_cM_Mb genetic_position_cM\n"
_INTERVAL_MAP_ENTRIES = (
    "1000 1.0 0.001\n"
    "2000 2.0 0.003\n"
    "3000 0.5 0.0035\n"
)

_INTERVAL_BED_CONTENT = (
    "chr1 0 500 rs_before\n"     # before first interval's Begin
    "chr1 0 1000 rs_i0_start\n"  # exact at I0's Begin
    "chr1 0 1500 rs_i0_mid\n"    # mid I0, startcm=0
    "chr1 0 2500 rs_i1_mid\n"    # mid I1, startcm=cumcM[I0]
    "chr1 0 3500 rs_extrap\n"    # past I2's Begin -> extrapolate with I2's rate
)

_INTERVAL_EXPECTED = {
    "rs_before": 0.0,
    "rs_i0_start": 0.0,
    "rs_i0_mid": 0.0005,
    "rs_i1_mid": 0.002,
    "rs_extrap": 0.00325,
}


def _write_interval_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    snp_file = tmp_path / "snps.bed"
    snp_file.write_text(_INTERVAL_BED_CONTENT)

    map_file = tmp_path / "map.gz"
    with gzip.open(map_file, "wt") as f:
        f.write(_INTERVAL_MAP_HEADER)
        f.write(_INTERVAL_MAP_ENTRIES)

    output = tmp_path / "out.gz"
    return snp_file, map_file, output


def test_interval_before_first(tmp_path):
    snp_file, map_file, output = _write_interval_fixtures(tmp_path)
    interpolate_intervals(snp_file, map_file, output)
    result = _read_output(output)
    assert result["rs_before"] == pytest.approx(0.0, abs=1e-10)


def test_interval_exact_at_begin(tmp_path):
    snp_file, map_file, output = _write_interval_fixtures(tmp_path)
    interpolate_intervals(snp_file, map_file, output)
    result = _read_output(output)
    assert result["rs_i0_start"] == pytest.approx(0.0, abs=1e-10)


def test_interval_mid_first_interval(tmp_path):
    snp_file, map_file, output = _write_interval_fixtures(tmp_path)
    interpolate_intervals(snp_file, map_file, output)
    result = _read_output(output)
    assert result["rs_i0_mid"] == pytest.approx(0.0005, abs=1e-10)


def test_interval_mid_later_interval(tmp_path):
    snp_file, map_file, output = _write_interval_fixtures(tmp_path)
    interpolate_intervals(snp_file, map_file, output)
    result = _read_output(output)
    assert result["rs_i1_mid"] == pytest.approx(0.002, abs=1e-10)


def test_interval_extrapolates_past_last(tmp_path):
    """Unlike interpolate(), positions past the last interval keep advancing
    at that interval's rate instead of clamping — matches the R script."""
    snp_file, map_file, output = _write_interval_fixtures(tmp_path)
    interpolate_intervals(snp_file, map_file, output)
    result = _read_output(output)
    assert result["rs_extrap"] == pytest.approx(0.00325, abs=1e-10)
    assert result["rs_extrap"] != pytest.approx(0.0035, abs=1e-10)  # not clamped


def test_interval_output_matches_expected(tmp_path):
    snp_file, map_file, output = _write_interval_fixtures(tmp_path)
    interpolate_intervals(snp_file, map_file, output)
    result = _read_output(output)
    for rs_id, expected in _INTERVAL_EXPECTED.items():
        assert result[rs_id] == pytest.approx(expected, abs=1e-10), rs_id


def test_interval_vs_point_diverge_on_interval_rate_data(tmp_path):
    """Regression guard: feeding interval-rate data (as convert_decode_map.py
    produces) through the point-based interpolate() is the historical bug —
    it uses the *next* interval's rate anchored at the *current* interval's
    endpoint. Confirm the two modes disagree by the predicted amount on the
    same fixture, so neither function can silently change to match the other.
    """
    snp_file, map_file, output_point = _write_interval_fixtures(tmp_path)
    output_interval = tmp_path / "out_interval.gz"

    interpolate(snp_file, map_file, output_point)
    interpolate_intervals(snp_file, map_file, output_interval)

    point_result = _read_output(output_point)
    interval_result = _read_output(output_interval)

    # Mid I0: point-mode brackets (1000, 0.001)-(2000, 0.003) using I1's
    # increment; interval-mode correctly uses I0's own rate.
    assert point_result["rs_i0_mid"] == pytest.approx(0.002, abs=1e-10)
    assert interval_result["rs_i0_mid"] == pytest.approx(0.0005, abs=1e-10)

    # Past the last interval: point-mode clamps, interval-mode extrapolates.
    assert point_result["rs_extrap"] == pytest.approx(0.0035, abs=1e-10)
    assert interval_result["rs_extrap"] == pytest.approx(0.00325, abs=1e-10)


# ---------------------------------------------------------------------------
# interpolate_hapmap: interval-rate maps with cumulative cM at row positions
# ---------------------------------------------------------------------------
#
# HapMap-style rows are (position, rate_cM_Mb, cumulative_cM_at_position).
# The cumulative cM is the anchor for the row's own interval, not the endpoint
# of the previous interval.

_HAPMAP_ENTRIES = (
    "1000 1.0 0.000\n"
    "2000 2.0 0.001\n"
    "3000 0.5 0.003\n"
)

_HAPMAP_SHIFTED_TO_INTERVAL_ENTRIES = (
    "1000 1.0 0.001\n"
    "2000 2.0 0.003\n"
    "3000 0.5 0.003\n"
)

_HAPMAP_EXPECTED = {
    "rs_before": 0.0,
    "rs_i0_start": 0.0,
    "rs_i0_mid": 0.0005,
    "rs_i1_mid": 0.002,
    "rs_extrap": 0.00325,
}


def _write_hapmap_fixtures(
    tmp_path: Path,
    entries: str = _HAPMAP_ENTRIES,
) -> tuple[Path, Path, Path]:
    snp_file = tmp_path / "snps.bed"
    snp_file.write_text(_INTERVAL_BED_CONTENT)

    map_file = tmp_path / "hapmap.gz"
    with gzip.open(map_file, "wt") as f:
        f.write("position rate_cM_Mb map_cM\n")
        f.write(entries)

    output = tmp_path / "out.gz"
    return snp_file, map_file, output


def test_hapmap_output_matches_expected(tmp_path):
    snp_file, map_file, output = _write_hapmap_fixtures(tmp_path)
    interpolate_hapmap(snp_file, map_file, output)
    result = _read_output(output)
    for rs_id, expected in _HAPMAP_EXPECTED.items():
        assert result[rs_id] == pytest.approx(expected, abs=1e-10), rs_id


def test_hapmap_matches_shifted_interval_representation(tmp_path):
    snp_file, hapmap_file, hapmap_output = _write_hapmap_fixtures(tmp_path)
    interval_file = tmp_path / "interval.gz"
    with gzip.open(interval_file, "wt") as f:
        f.write(_INTERVAL_MAP_HEADER)
        f.write(_HAPMAP_SHIFTED_TO_INTERVAL_ENTRIES)
    interval_output = tmp_path / "interval_out.gz"

    interpolate_hapmap(snp_file, hapmap_file, hapmap_output)
    interpolate_intervals(snp_file, interval_file, interval_output)

    assert _read_output(hapmap_output) == pytest.approx(
        _read_output(interval_output),
        abs=1e-10,
    )


def test_interval_mode_on_unshifted_hapmap_reproduces_bad_convention(tmp_path):
    snp_file, map_file, output = _write_hapmap_fixtures(tmp_path)
    interpolate_intervals(snp_file, map_file, output)
    result = _read_output(output)

    # This is the off-by-one cumulative-cM convention behind the nonmonotone
    # MacDonald pyrho maps: interval 1 starts from row 0's cM instead of row 1's.
    assert result["rs_i1_mid"] == pytest.approx(0.001, abs=1e-10)
    assert result["rs_i1_mid"] < result["rs_i0_mid"] + 0.001
