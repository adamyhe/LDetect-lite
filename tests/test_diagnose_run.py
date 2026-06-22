"""Tests for ldetect_original diagnostic run summaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_diagnose_run_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "ldetect_original"
        / "scripts"
        / "diagnose_run.py"
    )
    spec = importlib.util.spec_from_file_location("diagnose_run", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarise_breakpoints_accepts_pruned_subsets(tmp_path: Path) -> None:
    diagnose_run = _load_diagnose_run_module()
    path = tmp_path / "breakpoints.json"
    path.write_text(
        json.dumps(
            {
                "n_bpoints": 2,
                "found_width": 33,
                "computed_subsets": ["fourier", "fourier_ls"],
                "skipped_subsets": ["uniform", "uniform_ls"],
                "fourier": {"loci": [100, 200], "metric": {}},
                "fourier_ls": {"loci": [100, 250], "metric": {}},
            }
        )
    )

    row = diagnose_run.summarise_breakpoints(path)

    assert row["n_bpoints"] == "2"
    assert row["found_width"] == "33"
    assert row["fourier_n"] == "2"
    assert row["fourier_ls_n"] == "2"
    assert row["uniform_n"] == ""
    assert row["uniform_ls_n"] == ""
    assert row["fourier_to_fourier_ls_exact"] == "1"
    assert row["uniform_to_uniform_ls_exact"] == ""
