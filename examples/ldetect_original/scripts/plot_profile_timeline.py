#!/usr/bin/env python3
"""Plot a memory/CPU time-series trace produced by `profile_run.py`.

Renders a two-panel figure (RSS over time, CPU% over time) from the sampler's
CSV, with vertical shaded spans marking each pipeline step and Step 4
sub-phase, parsed from the wrapped `ldetect run` command's own
`[HH:MM:SS] Memory checkpoint <label>: ...` log lines
(`src/ldetect_lite/_cli/cmd_run.py` and `src/ldetect_lite/pipeline.py` emit these).
Phases whose checkpoints aren't present in the log (e.g. `uniform`/`uniform_ls`
metric and local-search phases when `ldetect run` was invoked with the
default `--subset fourier_ls`, which does not compute the uniform subset) are
skipped rather than erroring.

Usage:
    uv run python scripts/plot_profile_timeline.py \
        --csv results/profiling/EUR-chr2.csv \
        --log results/profiling/EUR-chr2.log \
        --title "EUR chr2 (661 partitions)" \
        --output results/profiling/EUR-chr2-timeline
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CHECKPOINT_RE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2})\] Memory checkpoint (.+?): current_rss_mib="
)

# (start_label, end_label, display_name) -- shaded in this order; phases whose
# labels are absent from the log (e.g. uniform/uniform_ls when not computed)
# are silently skipped.
_PHASES = [
    ("step1_start", "step1_end", "Step 1: partition"),
    ("step2_start", "step2_end", "Step 2: covariance"),
    ("step3_start", "step3_end", "Step 3: matrix→vector"),
    ("vector_read_start", "vector_read_end", "Step 4: read vector"),
    (
        "filter_width_search_start",
        "filter_width_search_end",
        "Step 4: filter width search",
    ),
    ("minima_extraction_start", "minima_extraction_end", "Step 4: minima extraction"),
    ("fourier_metric_start", "fourier_metric_end", "Step 4: fourier metric"),
    ("uniform_metric_start", "uniform_metric_end", "Step 4: uniform metric"),
    (
        "fourier_local_search_start",
        "fourier_local_search_end",
        "Step 4: fourier local search",
    ),
    (
        "uniform_local_search_start",
        "uniform_local_search_end",
        "Step 4: uniform local search",
    ),
    ("fourier_ls_metric_start", "fourier_ls_metric_end", "Step 4: fourier_ls metric"),
    ("uniform_ls_metric_start", "uniform_ls_metric_end", "Step 4: uniform_ls metric"),
    ("step5_start", "run_end", "Step 5: BED extraction"),
]


def parse_checkpoints(log_path: Path) -> dict[str, float]:
    """Return {label: elapsed_seconds_since_first_checkpoint}."""
    checkpoints: dict[str, float] = {}
    first_seconds: int | None = None
    for line in log_path.open():
        m = _CHECKPOINT_RE.match(line)
        if not m:
            continue
        h, mi, s, label = m.groups()
        total_seconds = int(h) * 3600 + int(mi) * 60 + int(s)
        if first_seconds is None:
            first_seconds = total_seconds
        elapsed = total_seconds - first_seconds
        if elapsed < 0:
            elapsed += 86400  # midnight wraparound
        checkpoints.setdefault(label, float(elapsed))
    return checkpoints


def read_samples(csv_path: Path) -> tuple[list[float], list[float], list[float]]:
    elapsed_s: list[float] = []
    rss_mib: list[float] = []
    cpu_pct: list[float] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            elapsed_s.append(float(row["elapsed_s"]))
            rss_mib.append(float(row["rss_total_mib"]))
            cpu_pct.append(float(row["cpu_percent_sum"]))
    return elapsed_s, rss_mib, cpu_pct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", required=True, type=Path, help="profile_run.py output CSV."
    )
    parser.add_argument(
        "--log", required=True, type=Path, help="profile_run.py --log-output file."
    )
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--output", required=True, type=Path, help="Output path stem (no extension)."
    )
    args = parser.parse_args()

    elapsed_s, rss_mib, cpu_pct = read_samples(args.csv)
    checkpoints = parse_checkpoints(args.log)

    spans = [
        (checkpoints[start], checkpoints[end], name)
        for start, end, name in _PHASES
        if start in checkpoints and end in checkpoints
    ]

    fig, (ax_mem, ax_cpu) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, (start, end, name) in enumerate(spans):
        color = cmap(i % 10)
        for ax in (ax_mem, ax_cpu):
            ax.axvspan(start, end, color=color, alpha=0.15)
        ax_mem.text(
            (start + end) / 2,
            1.0,
            name,
            transform=ax_mem.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=7,
            color=color,
        )

    ax_mem.plot(elapsed_s, rss_mib, color="black", linewidth=1.2)
    ax_mem.set_ylabel("RSS across process tree (MiB)")

    ax_cpu.plot(elapsed_s, cpu_pct, color="black", linewidth=1.2)
    ax_cpu.set_ylabel("CPU% summed across process tree")
    ax_cpu.set_xlabel("elapsed time (s)")

    if args.title:
        ax_mem.set_title(args.title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=150)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(
        f"Wrote {args.output.with_suffix('.png')} and "
        f"{args.output.with_suffix('.pdf')}"
    )
    print(
        f"Shaded {len(spans)}/{len(_PHASES)} known phases "
        "(missing ones were not computed in this run)"
    )


if __name__ == "__main__":
    main()
