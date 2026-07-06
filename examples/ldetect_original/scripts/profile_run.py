#!/usr/bin/env python3
"""Sample memory + CPU usage across a whole process tree while it runs.

Wraps a single command (typically `ldetect run ...`) in `subprocess.Popen`,
then polls the command's full process tree -- the parent plus every
`ProcessPoolExecutor` worker and `tabix` subprocess it spawns -- at a fixed
interval via `psutil`, summing resident memory and CPU usage across all live
processes. This produces a genuine time-series trace (unlike Snakemake's
`benchmark:` directive, which only records one summary row per job), meant to
be paired with `plot_profile_timeline.py` and the wrapped command's own
`[HH:MM:SS] Memory checkpoint <label>` log lines to show memory/CPU over time
broken down by pipeline step.

Requires the `profiling` extra: `uv sync --extra profiling`.

Usage:
    uv run python scripts/profile_run.py \
        --interval 1.0 \
        --output results/profiling/EUR-chr2.csv \
        --log-output results/profiling/EUR-chr2.log \
        -- uv run ldetect run --genetic-map ... --chromosome 2 ...
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError as e:
    raise SystemExit(
        "psutil is required for profile_run.py. Install it with "
        "`uv sync --extra profiling` (or `pip install ldetect-lite[profiling]`)."
    ) from e


class ProcessTreeSampler:
    """Tracks per-process CPU% state across repeated samples of a process tree."""

    def __init__(self) -> None:
        self._tracked: dict[int, psutil.Process] = {}

    def _live_processes(self, root_pid: int) -> list[psutil.Process]:
        try:
            root = psutil.Process(root_pid)
        except psutil.NoSuchProcess:
            return []
        try:
            all_procs = [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess:
            all_procs = [root]

        live: list[psutil.Process] = []
        for proc in all_procs:
            tracked = self._tracked.get(proc.pid)
            if tracked is None:
                try:
                    proc.cpu_percent(interval=None)  # prime the internal timer
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                tracked = proc
                self._tracked[proc.pid] = tracked
            live.append(tracked)
        return live

    def sample(self, root_pid: int) -> tuple[float, float, int]:
        """Return (rss_total_mib, cpu_percent_sum, n_processes) for the tree."""
        rss_total = 0.0
        cpu_total = 0.0
        n = 0
        for proc in self._live_processes(root_pid):
            try:
                rss_total += proc.memory_info().rss
                cpu_total += proc.cpu_percent(interval=None)
                n += 1
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                continue
        return rss_total / (1024.0 * 1024.0), cpu_total, n


def _split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1 :]


def main() -> None:
    own_argv, command = _split_command(sys.argv[1:])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sample period in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV output path for the sampled time series.",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        required=True,
        help="Where to redirect the wrapped command's stdout/stderr.",
    )
    args = parser.parse_args(own_argv)

    if not command:
        raise SystemExit(
            "No command given; pass it after '--', e.g. "
            "profile_run.py --output out.csv -- ldetect run ..."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_output.parent.mkdir(parents=True, exist_ok=True)

    sampler = ProcessTreeSampler()
    n_samples = 0
    start = time.monotonic()
    with (
        args.log_output.open("w") as log_f,
        args.output.open("w", newline="") as out_f,
    ):
        writer = csv.writer(out_f)
        writer.writerow(
            ["elapsed_s", "rss_total_mib", "cpu_percent_sum", "n_processes"]
        )
        popen = subprocess.Popen(command, stdout=log_f, stderr=subprocess.STDOUT)
        try:
            while popen.poll() is None:
                elapsed = time.monotonic() - start
                rss_mib, cpu_sum, n_processes = sampler.sample(popen.pid)
                writer.writerow(
                    [f"{elapsed:.2f}", f"{rss_mib:.2f}", f"{cpu_sum:.2f}", n_processes]
                )
                out_f.flush()
                n_samples += 1
                time.sleep(args.interval)
        finally:
            returncode = popen.wait()

    total_s = time.monotonic() - start
    print(f"Wrote {args.output} ({n_samples} samples over {total_s:.1f}s)")
    print(f"Wrapped command log: {args.log_output}")
    if returncode != 0:
        raise SystemExit(f"Wrapped command exited {returncode}; see {args.log_output}")


if __name__ == "__main__":
    main()
