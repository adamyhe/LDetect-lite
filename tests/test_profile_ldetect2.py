"""Tests for ldetect_original profiling log parser."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_profile_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "ldetect_original"
        / "scripts"
        / "profile_ldetect2.py"
    )
    spec = importlib.util.spec_from_file_location("profile_ldetect2", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_gnu_time_log(tmp_path: Path) -> None:
    profile = _load_profile_module()
    path = tmp_path / "timing.log"
    path.write_text(
        "\n".join(
            [
                'Command being timed: "ldetect2 -v debug run ..."',
                "User time (seconds): 12.50",
                "System time (seconds): 1.25",
                "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03",
                "Maximum resident set size (kbytes): 123456",
                "Major (requiring I/O) page faults: 7",
                "Minor (reclaiming a frame) page faults: 89",
                "Swaps: 0",
                "File system inputs: 111",
                "File system outputs: 222",
                "Exit status: 0",
            ]
        )
    )

    row = profile.parse_time_log(path)

    assert row["elapsed_seconds"] == "3723.000000"
    assert row["user_seconds"] == "12.50"
    assert row["system_seconds"] == "1.25"
    assert row["max_rss_kb"] == "123456"
    assert row["major_page_faults"] == "7"
    assert row["minor_page_faults"] == "89"
    assert row["filesystem_inputs"] == "111"
    assert row["filesystem_outputs"] == "222"
    assert row["exit_status"] == "0"


def test_parse_macos_time_log(tmp_path: Path) -> None:
    profile = _load_profile_module()
    path = tmp_path / "timing.log"
    path.write_text(
        "\n".join(
            [
                "12.5 real",
                "10.0 user",
                "2.0 sys",
                "1048576 maximum resident set size",
                "42 page faults",
                "77 page reclaims",
                "0 swaps",
                "5 file system inputs",
                "6 file system outputs",
            ]
        )
    )

    row = profile.parse_time_log(path)

    assert row["elapsed_seconds"] == "12.500000"
    assert row["user_seconds"] == "10.0"
    assert row["system_seconds"] == "2.0"
    assert row["max_rss_kb"] == "1024"
    assert row["major_page_faults"] == "42"
    assert row["minor_page_faults"] == "77"


def test_parse_ldetect2_log_local_search_rows(tmp_path: Path) -> None:
    profile = _load_profile_module()
    path = tmp_path / "ldetect2.log"
    path.write_text(
        "\n".join(
            [
                "[12:00:00] Running local search on Fourier breakpoints",
                "[12:00:01] fourier_ls breakpoint idx=0 start=100 stop=200 "
                "partitions=2 rows=12345 precompute_seconds=1.250 "
                "search_seconds=0.050 total_seconds=1.310 max_rss_mib=512.5",
                "[12:00:02] fourier_ls breakpoint idx=1 start=200 stop=300 "
                "partitions=3 rows=None precompute_seconds=2.500 "
                "search_seconds=0.100 total_seconds=2.650",
                "[12:00:03] Local search fourier_ls done: breakpoints=2 "
                "elapsed_seconds=4.000",
            ]
        )
    )

    sets, breakpoints = profile.parse_ldetect2_log(path)
    by_chrom = profile.aggregate_by_chrom(
        [
            {
                **sets[0],
                "profile_name": "test",
                "population": "EUR",
                "chrom": "chr22",
            }
        ],
        [
            {
                **row,
                "profile_name": "test",
                "population": "EUR",
                "chrom": "chr22",
            }
            for row in breakpoints
        ],
    )

    assert sets == [
        {
            "subset": "fourier_ls",
            "breakpoints": "2",
            "set_elapsed_seconds": "4.000",
        }
    ]
    assert len(breakpoints) == 2
    assert breakpoints[0]["rows"] == "12345"
    assert breakpoints[0]["max_rss_mib"] == "512.5"
    assert breakpoints[1]["rows"] == ""
    assert by_chrom[0]["precompute_seconds"] == "3.750000"
    assert by_chrom[0]["search_seconds"] == "0.150000"
    assert by_chrom[0]["rows"] == "12345"
    assert by_chrom[0]["partitions"] == "5"


def test_missing_debug_lines_produce_empty_local_search_rows(tmp_path: Path) -> None:
    profile = _load_profile_module()
    root = tmp_path / "diagnostics" / "EUR"
    log_dir = root / "22" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "timing.log").write_text("Exit status: 0\n")
    (log_dir / "ldetect2.log").write_text("[12:00:00] Step 1\n")

    run_row, set_rows, breakpoint_rows = profile.profile_chromosome(
        root,
        "EUR",
        "22",
        "test",
    )

    assert run_row["chrom"] == "chr22"
    assert run_row["exit_status"] == "0"
    assert set_rows == []
    assert breakpoint_rows == []
    assert profile.aggregate_by_chrom(set_rows, breakpoint_rows) == []
