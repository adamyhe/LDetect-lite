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
                "[12:00:00] Local search fourier_ls group loaded: "
                "breakpoints=2 partitions=5 rows=22222 load_seconds=0.250 "
                "canonicalize_seconds=0.750",
                "[12:00:01] fourier_ls breakpoint idx=0 start=100 stop=200 "
                "partitions=2 rows=12345 precompute_seconds=1.250 "
                "search_seconds=0.050 total_seconds=1.310 max_rss_mib=512.5 "
                "partition_load_seconds=0.010000 canonicalize_seconds=0.020000 "
                "append_seconds=0.030000 diagonal_seconds=0.040000 "
                "slice_seconds=0.050000 normalize_seconds=0.060000 "
                "vertical_seconds=0.070000 horizontal_seconds=0.080000 "
                "hdf5_read_seconds=0.090000 chunk_filter_seconds=0.100000 "
                "dedup_seconds=0.110000 dedup_merge_seconds=0.015000 "
                "dense_lookup_seconds=0.016000 dense_accumulate_seconds=0.017000 "
                "accumulator_seconds=0.120000 "
                "candidate_rows=1000 eligible_rows=800 normalized_rows=700 "
                "rows_read=1100 rows_after_filter=800 rows_after_dedup=750 "
                "duplicate_rows_skipped=50 chunks=2 segments=1 "
                "active_rows_peak=12345 peak_chunk_rows=600 "
                "hdf5_reader_open_count=3 hdf5_reader_reuse_count=5",
                "[12:00:02] fourier_ls breakpoint idx=1 start=200 stop=300 "
                "partitions=3 rows=None precompute_seconds=2.500 "
                "search_seconds=0.100 total_seconds=2.650",
                "[12:00:03] Local search fourier_ls done: breakpoints=2 "
                "elapsed_seconds=4.000",
            ]
        )
    )

    sets, groups, breakpoints = profile.parse_ldetect2_log(path)
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
            for row in groups
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
    assert groups == [
        {
            "subset": "fourier_ls",
            "group_index": "0",
            "breakpoints": "2",
            "partitions": "5",
            "rows": "22222",
            "load_seconds": "0.250",
            "canonicalize_seconds": "0.750",
            "total_seconds": "1.000000",
        }
    ]
    assert len(breakpoints) == 2
    assert breakpoints[0]["rows"] == "12345"
    assert breakpoints[0]["max_rss_mib"] == "512.5"
    assert breakpoints[0]["slice_seconds"] == "0.050000"
    assert breakpoints[0]["hdf5_read_seconds"] == "0.090000"
    assert breakpoints[0]["dedup_seconds"] == "0.110000"
    assert breakpoints[0]["dedup_merge_seconds"] == "0.015000"
    assert breakpoints[0]["dense_lookup_seconds"] == "0.016000"
    assert breakpoints[0]["dense_accumulate_seconds"] == "0.017000"
    assert breakpoints[0]["candidate_rows"] == "1000"
    assert breakpoints[0]["rows_read"] == "1100"
    assert breakpoints[0]["rows_after_dedup"] == "750"
    assert breakpoints[0]["duplicate_rows_skipped"] == "50"
    assert breakpoints[0]["active_rows_peak"] == "12345"
    assert breakpoints[0]["peak_chunk_rows"] == "600"
    assert breakpoints[0]["hdf5_reader_open_count"] == "3"
    assert breakpoints[0]["hdf5_reader_reuse_count"] == "5"
    assert breakpoints[1]["rows"] == ""
    assert by_chrom[0]["precompute_seconds"] == "3.750000"
    assert by_chrom[0]["search_seconds"] == "0.150000"
    assert by_chrom[0]["rows"] == "12345"
    assert by_chrom[0]["partitions"] == "5"
    assert by_chrom[0]["slice_seconds"] == "0.050000"
    assert by_chrom[0]["hdf5_read_seconds"] == "0.090000"
    assert by_chrom[0]["dedup_seconds"] == "0.110000"
    assert by_chrom[0]["dedup_merge_seconds"] == "0.015000"
    assert by_chrom[0]["dense_lookup_seconds"] == "0.016000"
    assert by_chrom[0]["dense_accumulate_seconds"] == "0.017000"
    assert by_chrom[0]["candidate_rows"] == "1000"
    assert by_chrom[0]["rows_read"] == "1100"
    assert by_chrom[0]["rows_after_dedup"] == "750"
    assert by_chrom[0]["duplicate_rows_skipped"] == "50"
    assert by_chrom[0]["active_rows_peak"] == "12345"
    assert by_chrom[0]["peak_chunk_rows"] == "600"
    assert by_chrom[0]["hdf5_reader_open_count"] == "3"
    assert by_chrom[0]["hdf5_reader_reuse_count"] == "5"
    assert by_chrom[0]["group_count"] == "1"
    assert by_chrom[0]["group_load_seconds"] == "0.250000"
    assert by_chrom[0]["group_canonicalize_seconds"] == "0.750000"
    assert by_chrom[0]["group_total_seconds"] == "1.000000"
    assert by_chrom[0]["local_search_unaccounted_seconds"] == "-0.960000"


def test_missing_debug_lines_produce_empty_local_search_rows(tmp_path: Path) -> None:
    profile = _load_profile_module()
    root = tmp_path / "diagnostics" / "EUR"
    log_dir = root / "22" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "timing.log").write_text("Exit status: 0\n")
    (log_dir / "ldetect2.log").write_text("[12:00:00] Step 1\n")

    run_row, set_rows, group_rows, breakpoint_rows = profile.profile_chromosome(
        root,
        "EUR",
        "22",
        "test",
    )

    assert run_row["chrom"] == "chr22"
    assert run_row["exit_status"] == "0"
    assert set_rows == []
    assert group_rows == []
    assert breakpoint_rows == []
    assert profile.aggregate_by_chrom(set_rows, group_rows, breakpoint_rows) == []


def test_profile_chromosome_prefers_chromosome_prefixed_logs(
    tmp_path: Path,
) -> None:
    profile = _load_profile_module()
    root = tmp_path / "diagnostics" / "EUR"
    log_dir = root / "22" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "timing.log").write_text("Exit status: 1\n")
    (log_dir / "ldetect2.log").write_text("[12:00:00] legacy\n")
    (log_dir / "22.timing.log").write_text("Exit status: 0\n")
    (log_dir / "22.ldetect2.log").write_text("[12:00:00] prefixed\n")

    run_row, set_rows, group_rows, breakpoint_rows = profile.profile_chromosome(
        root,
        "EUR",
        "22",
        "test",
    )

    assert run_row["chrom"] == "chr22"
    assert run_row["exit_status"] == "0"
    assert set_rows == []
    assert group_rows == []
    assert breakpoint_rows == []


def test_missing_optional_phase_fields_stay_blank(tmp_path: Path) -> None:
    profile = _load_profile_module()
    path = tmp_path / "ldetect2.log"
    path.write_text(
        "\n".join(
            [
                "[12:00:01] fourier_ls breakpoint idx=0 start=100 stop=200 "
                "partitions=2 rows=12345 precompute_seconds=1.250 "
                "search_seconds=0.050 total_seconds=1.310 max_rss_mib=512.5",
                "[12:00:02] Local search fourier_ls done: breakpoints=1 "
                "elapsed_seconds=1.400",
            ]
        )
    )

    sets, groups, breakpoints = profile.parse_ldetect2_log(path)
    by_chrom = profile.aggregate_by_chrom(
        [
            {
                **sets[0],
                "profile_name": "test",
                "population": "EUR",
                "chrom": "chr21",
            }
        ],
        groups,
        [
            {
                **breakpoints[0],
                "profile_name": "test",
                "population": "EUR",
                "chrom": "chr21",
            }
        ],
    )

    assert breakpoints[0]["partition_load_seconds"] == ""
    assert by_chrom[0]["partition_load_seconds"] == ""
    assert by_chrom[0]["canonicalize_seconds"] == ""
    assert by_chrom[0]["hdf5_read_seconds"] == ""
    assert by_chrom[0]["candidate_rows"] == ""
    assert by_chrom[0]["rows_read"] == ""
    assert by_chrom[0]["group_total_seconds"] == ""
    assert by_chrom[0]["local_search_unaccounted_seconds"] == "0.090000"
