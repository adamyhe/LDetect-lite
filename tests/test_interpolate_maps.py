"""Tests for ldetect2.interpolate_maps."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from ldetect2.interpolate_maps import interpolate

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
