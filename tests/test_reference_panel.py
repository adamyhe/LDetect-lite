"""Tests for reference_panel.py's genetic map reader and its caching."""

from __future__ import annotations

import gzip
from pathlib import Path

from ldetect_lite._util.reference_panel import read_genetic_map


def _write_map(path: Path, rows: list[tuple[int, float]]) -> None:
    with gzip.open(path, "wt") as f:
        for pos, gpos in rows:
            f.write(f"1 {pos} {gpos}\n")


def test_read_genetic_map_parses_position_to_gpos(tmp_path: Path) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path, [(100, 0.0), (200, 0.001), (300, 0.0025)])

    result = read_genetic_map(map_path)

    assert result == {100: 0.0, 200: 0.001, 300: 0.0025}


def test_read_genetic_map_caches_per_path(tmp_path: Path) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path, [(100, 0.0), (200, 0.001)])

    first = read_genetic_map(map_path)
    second = read_genetic_map(map_path)

    assert first is second  # same object: parsed once, not reparsed


def test_read_genetic_map_distinguishes_different_paths(tmp_path: Path) -> None:
    map_a = tmp_path / "a.gz"
    map_b = tmp_path / "b.gz"
    _write_map(map_a, [(100, 0.0)])
    _write_map(map_b, [(200, 0.5)])

    assert read_genetic_map(map_a) == {100: 0.0}
    assert read_genetic_map(map_b) == {200: 0.5}
