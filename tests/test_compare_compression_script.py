"""Synthetic-fixture tests for the compare_compression.py example script.

Mirrors tests/test_compare_signal_cache_script.py's dynamic-load pattern:
this diagnostic script is meant to run against a real Snakemake-produced
dual-mode `ldetect2 run` (baseline lzf vs zstd), which isn't available in a
fresh checkout, so it is unit-tested here against hand-built fixtures shaped
like real output instead.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ldetect2.io.bed import write_bed

_SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "ldetect_original"
    / "scripts"
)

_BENCHMARK_FIELDS = [
    "s",
    "h:m:s",
    "max_rss",
    "max_vms",
    "max_uss",
    "max_pss",
    "io_in",
    "io_out",
    "mean_load",
    "cpu_time",
]


def _load_compare_compression_module() -> ModuleType:
    """Dynamically load the example diagnostic script (not an importable package)."""
    import importlib.util

    script_path = _SCRIPT_DIR / "compare_compression.py"
    # The script does `from compare_blocks import compare_chrom`, which only
    # resolves when the scripts/ directory is on sys.path (true when run as
    # `python scripts/compare_compression.py`, not automatic for exec_module).
    if str(_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("compare_compression", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_vector(path: Path, rows: list[tuple[int, float]]) -> None:
    with gzip.open(path, "wt") as f:
        writer = csv.writer(f, delimiter="\t")
        for locus, value in rows:
            writer.writerow([locus, value])


def _write_breakpoints(path: Path, subset: str, loci: list[int]) -> None:
    path.write_text(json.dumps({subset: {"loci": loci}}))


def _write_benchmark(path: Path, seconds: float, max_rss: float) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(_BENCHMARK_FIELDS)
        writer.writerow(
            [
                seconds,
                "0:00:00",
                max_rss,
                max_rss * 2,
                max_rss,
                max_rss,
                0,
                0,
                100,
                seconds,
            ]
        )


def _write_covariance_dir(path: Path, file_sizes: list[int]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for i, size in enumerate(file_sizes):
        (path / f"chr.{i}.{i + 1}.h5").write_bytes(b"\x00" * size)


def _run_compare(module: ModuleType, argv: list[str]) -> None:
    old_argv = sys.argv
    sys.argv = ["compare_compression.py", *argv]
    try:
        module.main()
    finally:
        sys.argv = old_argv


def _read_tsv_row(path: Path) -> dict[str, str]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        return next(reader)


def test_compare_compression_reports_exact_match_and_size_reduction(
    tmp_path: Path,
) -> None:
    module = _load_compare_compression_module()
    rows = [(100, 1.0), (200, 2.5), (300, 0.0)]
    loci = [200, 300]

    baseline_vector = tmp_path / "baseline-vector.txt.gz"
    zstd_vector = tmp_path / "zstd-vector.txt.gz"
    _write_vector(baseline_vector, rows)
    _write_vector(zstd_vector, rows)

    baseline_bpoints = tmp_path / "baseline-bpoints.json"
    zstd_bpoints = tmp_path / "zstd-bpoints.json"
    _write_breakpoints(baseline_bpoints, "fourier_ls", loci)
    _write_breakpoints(zstd_bpoints, "fourier_ls", loci)

    baseline_bed = tmp_path / "baseline.bed"
    zstd_bed = tmp_path / "zstd.bed"
    write_bed(name="22", loci=loci, snp_first=100, snp_last=300, output=baseline_bed)
    write_bed(name="22", loci=loci, snp_first=100, snp_last=300, output=zstd_bed)

    baseline_cov_dir = tmp_path / "baseline_cov"
    zstd_cov_dir = tmp_path / "zstd_cov"
    _write_covariance_dir(baseline_cov_dir, [1000, 1000])
    _write_covariance_dir(zstd_cov_dir, [400, 400])

    baseline_bench = tmp_path / "baseline.benchmark.tsv"
    zstd_bench = tmp_path / "zstd.benchmark.tsv"
    _write_benchmark(baseline_bench, seconds=100.0, max_rss=500.0)
    _write_benchmark(zstd_bench, seconds=90.0, max_rss=480.0)

    output = tmp_path / "compare.tsv"
    _run_compare(
        module,
        [
            "--population",
            "EUR",
            "--chromosome",
            "22",
            "--baseline-vector",
            str(baseline_vector),
            "--zstd-vector",
            str(zstd_vector),
            "--baseline-breakpoints",
            str(baseline_bpoints),
            "--zstd-breakpoints",
            str(zstd_bpoints),
            "--baseline-bed",
            str(baseline_bed),
            "--zstd-bed",
            str(zstd_bed),
            "--baseline-covariance-dir",
            str(baseline_cov_dir),
            "--zstd-covariance-dir",
            str(zstd_cov_dir),
            "--baseline-benchmark",
            str(baseline_bench),
            "--zstd-benchmark",
            str(zstd_bench),
            "--tolerance",
            "0",
            "--output",
            str(output),
        ],
    )

    row = _read_tsv_row(output)
    assert row["vector_rows_equal"] == "True"
    assert row["vector_sha256_equal"] == "True"
    assert float(row["vector_max_abs_diff"]) == 0.0
    assert int(row["vector_exact_matches"]) == len(rows)
    assert row["loci_exact_match"] == "True"
    assert float(row["bed_recall"]) == 1.0
    assert float(row["bed_precision"]) == 1.0
    assert int(row["baseline_covariance_bytes"]) == 2000
    assert int(row["zstd_covariance_bytes"]) == 800
    assert float(row["covariance_size_ratio"]) == pytest.approx(0.4)
    assert float(row["covariance_size_reduction_pct"]) == pytest.approx(60.0)
    assert float(row["speedup"]) == pytest.approx(100.0 / 90.0, abs=1e-4)
    assert float(row["max_rss_ratio"]) == pytest.approx(500.0 / 480.0, abs=1e-4)


def test_compare_compression_flags_numeric_divergence_as_a_real_bug(
    tmp_path: Path,
) -> None:
    """Compression is lossless -- unlike the signal-cache diagnostic, any
    numeric divergence here indicates a real bug, not expected floating-point
    noise from a different computation order. The script itself doesn't know
    that distinction; it just reports what it finds."""
    module = _load_compare_compression_module()
    loci = [200]

    baseline_vector = tmp_path / "baseline-vector.txt.gz"
    zstd_vector = tmp_path / "zstd-vector.txt.gz"
    _write_vector(baseline_vector, [(100, 1.0), (200, 2.5)])
    _write_vector(zstd_vector, [(100, 1.0), (200, 2.500001)])

    baseline_bpoints = tmp_path / "baseline-bpoints.json"
    zstd_bpoints = tmp_path / "zstd-bpoints.json"
    _write_breakpoints(baseline_bpoints, "fourier_ls", loci)
    _write_breakpoints(zstd_bpoints, "fourier_ls", loci)

    baseline_bed = tmp_path / "baseline.bed"
    zstd_bed = tmp_path / "zstd.bed"
    write_bed(name="22", loci=loci, snp_first=100, snp_last=300, output=baseline_bed)
    write_bed(name="22", loci=loci, snp_first=100, snp_last=300, output=zstd_bed)

    baseline_cov_dir = tmp_path / "baseline_cov"
    zstd_cov_dir = tmp_path / "zstd_cov"
    _write_covariance_dir(baseline_cov_dir, [1000])
    _write_covariance_dir(zstd_cov_dir, [1000])

    output = tmp_path / "compare.tsv"
    _run_compare(
        module,
        [
            "--population",
            "EUR",
            "--chromosome",
            "22",
            "--baseline-vector",
            str(baseline_vector),
            "--zstd-vector",
            str(zstd_vector),
            "--baseline-breakpoints",
            str(baseline_bpoints),
            "--zstd-breakpoints",
            str(zstd_bpoints),
            "--baseline-bed",
            str(baseline_bed),
            "--zstd-bed",
            str(zstd_bed),
            "--baseline-covariance-dir",
            str(baseline_cov_dir),
            "--zstd-covariance-dir",
            str(zstd_cov_dir),
            "--tolerance",
            "0",
            "--output",
            str(output),
        ],
    )

    row = _read_tsv_row(output)
    assert row["vector_sha256_equal"] == "False"
    assert float(row["vector_max_abs_diff"]) == pytest.approx(0.000001, abs=1e-9)
    assert int(row["vector_exact_matches"]) == 1
    assert row["speedup"] == ""
    assert row["max_rss_ratio"] == ""


def test_compare_compression_handles_empty_covariance_dir(tmp_path: Path) -> None:
    module = _load_compare_compression_module()
    loci = [200]

    baseline_vector = tmp_path / "baseline-vector.txt.gz"
    zstd_vector = tmp_path / "zstd-vector.txt.gz"
    _write_vector(baseline_vector, [(100, 1.0)])
    _write_vector(zstd_vector, [(100, 1.0)])

    baseline_bpoints = tmp_path / "baseline-bpoints.json"
    zstd_bpoints = tmp_path / "zstd-bpoints.json"
    _write_breakpoints(baseline_bpoints, "fourier_ls", loci)
    _write_breakpoints(zstd_bpoints, "fourier_ls", loci)

    baseline_bed = tmp_path / "baseline.bed"
    zstd_bed = tmp_path / "zstd.bed"
    write_bed(name="22", loci=loci, snp_first=100, snp_last=300, output=baseline_bed)
    write_bed(name="22", loci=loci, snp_first=100, snp_last=300, output=zstd_bed)

    baseline_cov_dir = tmp_path / "baseline_cov"
    zstd_cov_dir = tmp_path / "zstd_cov"
    baseline_cov_dir.mkdir()
    zstd_cov_dir.mkdir()

    output = tmp_path / "compare.tsv"
    _run_compare(
        module,
        [
            "--population",
            "EUR",
            "--chromosome",
            "22",
            "--baseline-vector",
            str(baseline_vector),
            "--zstd-vector",
            str(zstd_vector),
            "--baseline-breakpoints",
            str(baseline_bpoints),
            "--zstd-breakpoints",
            str(zstd_bpoints),
            "--baseline-bed",
            str(baseline_bed),
            "--zstd-bed",
            str(zstd_bed),
            "--baseline-covariance-dir",
            str(baseline_cov_dir),
            "--zstd-covariance-dir",
            str(zstd_cov_dir),
            "--tolerance",
            "0",
            "--output",
            str(output),
        ],
    )

    row = _read_tsv_row(output)
    assert int(row["baseline_covariance_bytes"]) == 0
    assert int(row["zstd_covariance_bytes"]) == 0
    assert row["covariance_size_ratio"] == ""
    assert row["covariance_size_reduction_pct"] == ""
