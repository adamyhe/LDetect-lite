#!/usr/bin/env python3
"""Compare the Step 4 filter-width search phase across `profile_run.py` traces.

Isolates the effect of `--filter-workers` (Numba `prange` threading for the
adaptive exponential/binary-search/minima-extraction convolutions) from
everything else in the pipeline by reading the phase timestamps and CPU
trace that `profile_run.py` + the run's own `Memory checkpoint`/`log_msg`
lines already produce -- no new instrumentation, no rerun of steps 1-3.

Reports, per run: exponential-search, binary-search, and trackback
sub-phase durations, the total filter_width_search duration, and the mean
CPU%/process-count sampled during that window.

Usage:
    uv run python scripts/compare_filter_width_search.py \
        --run baseline results/profiling/EUR-21-serial.log results/profiling/EUR-21-serial.csv \
        --run prange results/profiling/EUR-21-prange.log results/profiling/EUR-21-prange.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

_CHECKPOINT_RE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2})\] Memory checkpoint (\S+):"
)
_MARKER_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\] (.+)$")

_MARKERS = {
    "search_start": "Searching for filter width",
    "exp_end": "Exponential search end:",
    "binary_end": "Binary search found_width:",
    "trackback_end": "Trackback found_width:",
}


def _to_seconds(h: str, m: str, s: str) -> int:
    return int(h) * 3600 + int(m) * 60 + int(s)


def _parse_log(path: Path) -> dict[str, int]:
    timestamps: dict[str, int] = {}
    for line in path.read_text().splitlines():
        checkpoint = _CHECKPOINT_RE.match(line)
        if checkpoint is not None:
            h, m, s, label = checkpoint.groups()
            timestamps.setdefault(label, _to_seconds(h, m, s))
            continue
        marker = _MARKER_RE.match(line)
        if marker is None:
            continue
        h, m, s, rest = marker.groups()
        for key, prefix in _MARKERS.items():
            if key not in timestamps and rest.startswith(prefix):
                timestamps[key] = _to_seconds(h, m, s)
    required = ("run_start", "filter_width_search_start", "filter_width_search_end")
    missing = [key for key in required if key not in timestamps]
    if missing:
        raise ValueError(f"{path}: missing expected log markers {missing}")
    return timestamps


@dataclass
class _CpuWindow:
    mean_cpu_percent: float
    max_cpu_percent: float
    max_n_processes: int


def _cpu_window(csv_path: Path, start_elapsed: float, end_elapsed: float) -> _CpuWindow:
    cpu_samples: list[float] = []
    n_processes: list[int] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            elapsed = float(row["elapsed_s"])
            if start_elapsed <= elapsed <= end_elapsed:
                cpu_samples.append(float(row["cpu_percent_sum"]))
                n_processes.append(int(row["n_processes"]))
    if not cpu_samples:
        return _CpuWindow(0.0, 0.0, 0)
    return _CpuWindow(
        mean_cpu_percent=sum(cpu_samples) / len(cpu_samples),
        max_cpu_percent=max(cpu_samples),
        max_n_processes=max(n_processes),
    )


def _report(label: str, log_path: Path, csv_path: Path) -> None:
    ts = _parse_log(log_path)
    t0 = ts["run_start"]
    search_start = ts.get("search_start", ts["filter_width_search_start"])
    exp_end = ts.get("exp_end")
    binary_end = ts.get("binary_end")
    trackback_end = ts.get("trackback_end", ts["filter_width_search_end"])
    total_start = ts["filter_width_search_start"]
    total_end = ts["filter_width_search_end"]

    window = _cpu_window(csv_path, total_start - t0, total_end - t0)

    print(f"== {label} ({log_path.name}) ==")
    if exp_end is not None:
        print(f"  exponential search: {exp_end - search_start}s")
    if exp_end is not None and binary_end is not None:
        print(f"  binary search:      {binary_end - exp_end}s")
    if binary_end is not None:
        print(f"  trackback:          {trackback_end - binary_end}s")
    print(f"  total filter_width_search: {total_end - total_start}s")
    print(
        f"  CPU during phase: mean={window.mean_cpu_percent:.0f}% "
        f"max={window.max_cpu_percent:.0f}% max_processes={window.max_n_processes}"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("LABEL", "LOG", "CSV"),
        required=True,
        help="One run to include, given as: label log_path csv_path. Repeatable.",
    )
    args = parser.parse_args()

    for label, log_path, csv_path in args.run:
        _report(label, Path(log_path), Path(csv_path))


if __name__ == "__main__":
    main()
