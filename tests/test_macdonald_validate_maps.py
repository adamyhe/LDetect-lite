"""Tests for the MacDonald2022 map validator."""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path


def _load_validate_maps_module():
    script = (
        Path(__file__).parents[1]
        / "examples"
        / "MacDonald2022"
        / "scripts"
        / "validate_maps.py"
    )
    spec = importlib.util.spec_from_file_location("validate_maps", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_gz(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as f:
        f.write(text)


def test_summarise_map_reads_ldetect2_interpolated_format(tmp_path: Path) -> None:
    module = _load_validate_maps_module()
    map_path = tmp_path / "chr1.tab.gz"
    _write_gz(
        map_path,
        "rs1 100 0.1\n"
        "rs2 200 0.3\n"
        "rs3 300 0.4\n",
    )

    summary = module.summarise_map(map_path)

    assert summary["n_snps"] == 3
    assert summary["cM_min"] == 0.1
    assert summary["cM_max"] == 0.4
    assert summary["inversions"] == 0


def test_summarise_map_reads_macdonald_chr_position_cm_format(
    tmp_path: Path,
) -> None:
    module = _load_validate_maps_module()
    map_path = tmp_path / "chr1.tab.gz"
    _write_gz(
        map_path,
        "chr1\t100\t0.1\n"
        "chr1\t200\t0.2\n"
        "chr1\t300\t0.15\n",
    )

    summary = module.summarise_map(map_path)

    assert summary["n_snps"] == 3
    assert summary["inversions"] == 1
