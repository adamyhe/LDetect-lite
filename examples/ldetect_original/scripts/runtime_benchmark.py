"""Summarize and plot the full chromosome runtime benchmark (default EUR chr21).

This script is intentionally lightweight: the expensive ldetect-lite run is
owned by the main Snakemake workflow, and the legacy runtime can be supplied as
either an elapsed-second value or a `/usr/bin/time -v` log.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

TIMING_FIGSIZE = (3.1, 1.55)
BACKENDS = ("legacy", "lite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", default="EUR")
    parser.add_argument("--chromosome", default="21")
    parser.add_argument(
        "--lite-benchmark",
        type=Path,
        default=Path("results/logs/EUR/21.benchmark.tsv"),
        help="Snakemake benchmark TSV from the ldetect-lite EUR chr21 run.",
    )
    parser.add_argument(
        "--lite-time-log",
        type=Path,
        default=Path("results/logs/EUR/21.timing.log"),
        help="Optional /usr/bin/time -v log from the ldetect-lite EUR chr21 run.",
    )
    parser.add_argument(
        "--legacy-time-log",
        type=Path,
        help="Optional /usr/bin/time -v log from the legacy full EUR chr21 run.",
    )
    parser.add_argument(
        "--legacy-seconds",
        type=float,
        help="Legacy full EUR chr21 elapsed time in seconds.",
    )
    parser.add_argument(
        "--lite-seconds",
        type=float,
        help="Override ldetect-lite elapsed time in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmarks/EUR-chr21"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("plots/timings-full-eur-chr21.svg"),
    )
    parser.add_argument("--plot-format", choices=("svg", "png", "pdf"), default="svg")
    args = parser.parse_args()

    lite_seconds = choose_lite_seconds(args)
    legacy_seconds = choose_legacy_seconds(args)
    rows = [
        timing_row(
            args.population,
            args.chromosome,
            "legacy",
            legacy_seconds,
            legacy_source(args),
        ),
        timing_row(
            args.population, args.chromosome, "lite", lite_seconds, lite_source(args)
        ),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "timings.tsv", rows)
    write_summary(args.output_dir / "summary.tsv", rows)
    write_timing_plot(args.plot, rows)
    print_summary(rows)
    print(f"timings: {args.output_dir / 'timings.tsv'}")
    print(f"summary: {args.output_dir / 'summary.tsv'}")
    print(f"plot: {args.plot}")


def choose_lite_seconds(args: argparse.Namespace) -> float:
    if args.lite_seconds is not None:
        return args.lite_seconds
    if args.lite_time_log.exists():
        return parse_usr_bin_time_elapsed(args.lite_time_log)
    if args.lite_benchmark.exists():
        return parse_snakemake_elapsed(args.lite_benchmark)
    raise SystemExit(
        "Could not determine ldetect-lite runtime. Provide --lite-seconds, "
        "--lite-time-log, or --lite-benchmark after running "
        "`uv run snakemake --cores N --config chromosomes='[21]'` "
        "from examples/ldetect_original."
    )


def choose_legacy_seconds(args: argparse.Namespace) -> float:
    if args.legacy_seconds is not None:
        return args.legacy_seconds
    if args.legacy_time_log is not None:
        return parse_usr_bin_time_elapsed(args.legacy_time_log)
    raise SystemExit(
        "Could not determine legacy LDetect runtime. Provide either "
        "--legacy-seconds or --legacy-time-log."
    )


def parse_usr_bin_time_elapsed(path: Path) -> float:
    text = path.read_text()
    match = re.search(r"Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)", text)
    if not match:
        raise SystemExit(f"Could not parse elapsed wall time from {path}")
    return parse_elapsed_value(match.group(1))


def parse_snakemake_elapsed(path: Path) -> float:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    row = rows[-1]
    for key in ("s", "seconds", "walltime", "runtime"):
        if key in row and row[key] not in ("", None):
            return float(row[key])
    raise SystemExit(f"Could not find elapsed seconds column in {path}")


def parse_elapsed_value(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unrecognized elapsed time: {value}")


def legacy_source(args: argparse.Namespace) -> str:
    if args.legacy_seconds is not None:
        return "manual_seconds"
    return str(args.legacy_time_log)


def lite_source(args: argparse.Namespace) -> str:
    if args.lite_seconds is not None:
        return "manual_seconds"
    if args.lite_time_log.exists():
        return str(args.lite_time_log)
    return str(args.lite_benchmark)


def timing_row(
    population: str,
    chromosome: str,
    backend: str,
    seconds: float,
    source: str,
) -> dict[str, str]:
    return {
        "population": population,
        "chromosome": chromosome,
        "benchmark": "full_chromosome",
        "backend": backend,
        "seconds": f"{seconds:.6f}",
        "minutes": f"{seconds / 60:.6f}",
        "source": source,
    }


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "population",
        "chromosome",
        "benchmark",
        "backend",
        "seconds",
        "minutes",
        "source",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    values = {row["backend"]: float(row["seconds"]) for row in rows}
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "population",
                "chromosome",
                "benchmark",
                "legacy_seconds",
                "lite_seconds",
                "speedup",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "population": rows[0]["population"],
                "chromosome": rows[0]["chromosome"],
                "benchmark": "full_chromosome",
                "legacy_seconds": f"{values['legacy']:.6f}",
                "lite_seconds": f"{values['lite']:.6f}",
                "speedup": f"{values['legacy'] / values['lite']:.6f}",
            }
        )


def write_timing_plot(path: Path, rows: list[dict[str, str]]) -> None:
    plt = setup_matplotlib(path.parent)
    values = {row["backend"]: float(row["minutes"]) for row in rows}
    means = [values["legacy"], values["lite"]]
    labels = ["LDetect", "LDetect-lite"]
    y = [0.0, 0.36]

    fig, ax = plt.subplots(figsize=TIMING_FIGSIZE, constrained_layout=True)
    ax.barh(
        y,
        means,
        height=0.26,
        color=["#0057b8", "#d62728"],
        edgecolor="#222222",
        linewidth=0.6,
    )
    ax.set_yticks(y, labels)
    ax.set_ylim(0.58, -0.24)
    ax.set_xlabel("wall time (minutes)")
    ax.set_title("full EUR chr21")
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def setup_matplotlib(output_dir: Path):
    mpl_config = output_dir / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache = output_dir / ".cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def print_summary(rows: list[dict[str, str]]) -> None:
    values = {row["backend"]: float(row["seconds"]) for row in rows}
    print(
        "full EUR chr21: "
        f"legacy={values['legacy'] / 60:.3f} min, "
        f"ldetect-lite={values['lite'] / 60:.3f} min, "
        f"speedup={values['legacy'] / values['lite']:.3f}x"
    )


if __name__ == "__main__":
    main()
