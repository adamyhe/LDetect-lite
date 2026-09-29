"""Aggregate the LDetect vs ldetect-lite whole-genome runtime benchmark.

Consumes three kinds of measurement and emits one tidy table plus a summary:

- `--lite POP CHROM WORKERS PATH`: a `/usr/bin/time -v` log from `ldetect run`.
- `--legacy-covariance POP CHROM PATH`: the per-partition TSV written by
  `run_legacy_covariance.py`.
- `--legacy-downstream POP CHROM PATH`: a `/usr/bin/time -v` log from
  `run_legacy_ldetect.py` (`P01`/`P02`/`P03`).

CPU-seconds (user + sys) is the primary metric. Legacy covariance is
embarrassingly parallel across partitions and lite parallelizes internally, so
wall clock on its own only reports how many cores each arm was given. Wall
clock is still recorded, tagged with the worker count that produced it.

`max_rss_mib` is a peak for a single process, not a sum across concurrent
workers, so it understates whole-pipeline memory for both parallel arms.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

TIMING_COLUMNS = (
    "population",
    "chromosome",
    "arm",
    "stage",
    "workers",
    "wall_seconds",
    "cpu_seconds",
    "max_rss_mib",
    "source",
)

SUMMARY_COLUMNS = (
    "population",
    "chromosome",
    "legacy_covariance_cpu_seconds",
    "legacy_downstream_cpu_seconds",
    "legacy_total_cpu_seconds",
    "lite_serial_cpu_seconds",
    "lite_parallel_cpu_seconds",
    "lite_parallel_workers",
    "cpu_speedup_vs_lite_serial",
    "cpu_speedup_vs_lite_parallel",
    "legacy_covariance_share",
    "lite_parallel_wall_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lite",
        nargs=4,
        action="append",
        default=[],
        metavar=("POPULATION", "CHROMOSOME", "WORKERS", "PATH"),
    )
    parser.add_argument(
        "--legacy-covariance",
        nargs=3,
        action="append",
        default=[],
        metavar=("POPULATION", "CHROMOSOME", "PATH"),
    )
    parser.add_argument(
        "--legacy-downstream",
        nargs=3,
        action="append",
        default=[],
        metavar=("POPULATION", "CHROMOSOME", "PATH"),
    )
    parser.add_argument("--serial-workers", type=int, default=1)
    parser.add_argument("--timings", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    rows = collect_rows(args)
    if not rows:
        raise SystemExit("No timing inputs supplied")

    write_tsv(args.timings, TIMING_COLUMNS, rows)
    summary = summarize(rows, args.serial_workers)
    write_tsv(args.summary, SUMMARY_COLUMNS, summary)
    if args.plot is not None:
        write_plot(args.plot, summary)
    print_report(summary, args)


def collect_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for population, chromosome, workers, path in args.lite:
        measurement = parse_usr_bin_time(Path(path))
        rows.append(
            timing_row(
                population, chromosome, "lite", "total", int(workers), measurement, path
            )
        )

    for population, chromosome, path in args.legacy_covariance:
        measurement = sum_partition_timings(Path(path))
        rows.append(
            timing_row(
                population,
                chromosome,
                "legacy",
                "covariance",
                measurement.pop("workers"),
                measurement,
                path,
            )
        )

    for population, chromosome, path in args.legacy_downstream:
        measurement = parse_usr_bin_time(Path(path))
        rows.append(
            timing_row(
                population, chromosome, "legacy", "downstream", 1, measurement, path
            )
        )

    rows.sort(key=sort_key)
    return rows


def sort_key(row: dict[str, object]) -> tuple[str, int, str, str, int]:
    return (
        str(row["population"]),
        chromosome_order(str(row["chromosome"])),
        str(row["arm"]),
        str(row["stage"]),
        int(row["workers"]),
    )


def chromosome_order(chromosome: str) -> int:
    return int(chromosome) if chromosome.isdigit() else 99


def timing_row(
    population: str,
    chromosome: str,
    arm: str,
    stage: str,
    workers: int,
    measurement: dict[str, float],
    source: str,
) -> dict[str, object]:
    return {
        "population": population,
        "chromosome": chromosome,
        "arm": arm,
        "stage": stage,
        "workers": workers,
        "wall_seconds": f"{measurement['wall_seconds']:.6f}",
        "cpu_seconds": f"{measurement['cpu_seconds']:.6f}",
        "max_rss_mib": f"{measurement['max_rss_mib']:.3f}",
        "source": source,
    }


def parse_usr_bin_time(path: Path) -> dict[str, float]:
    """Extract wall, CPU and peak RSS from a `/usr/bin/time -v` log.

    GNU time reports RUSAGE_SELF plus RUSAGE_CHILDREN, so the CPU figure covers
    every worker process the timed command spawned and reaped.
    """
    text = path.read_text()
    # The label itself contains colons -- "(h:mm:ss or m:ss)" -- so match
    # lazily up to the last colon that still leaves a time value before EOL.
    wall = search_float(
        text, r"Elapsed \(wall clock\) time[^\n]*?:\s*([0-9:.]+)\s*$", parse_elapsed
    )
    user = search_float(text, r"User time \(seconds\):\s*([0-9.]+)", float)
    system = search_float(text, r"System time \(seconds\):\s*([0-9.]+)", float)
    rss_kb = search_float(
        text, r"Maximum resident set size \(kbytes\):\s*([0-9]+)", float
    )
    return {
        "wall_seconds": wall,
        "cpu_seconds": user + system,
        "max_rss_mib": rss_kb / 1024.0,
    }


def search_float(text: str, pattern: str, convert) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise SystemExit(f"Could not parse {pattern!r} from timing log")
    return convert(match.group(1))


def parse_elapsed(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Unrecognized elapsed time: {value}")


def sum_partition_timings(path: Path) -> dict[str, float]:
    """Total the per-partition rows from `run_legacy_covariance.py`.

    `wall_seconds` is summed rather than taken as an observed elapsed time: it
    is the wall clock this stage would take on a single core, which is the only
    figure comparable across different worker counts.
    """
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise SystemExit(f"No partition timings in {path}")
    skipped = [row for row in rows if row["status"] == "skipped"]
    if skipped:
        raise SystemExit(
            f"{path} has {len(skipped)} skipped partition(s); their runtime was "
            "never measured. Rerun without --skip-existing to time them."
        )
    return {
        "wall_seconds": sum(float(row["wall_seconds"]) for row in rows),
        "cpu_seconds": sum(float(row["cpu_seconds"]) for row in rows),
        "max_rss_mib": max(float(row["max_rss_mib"]) for row in rows),
        "workers": 1,
    }


def summarize(
    rows: list[dict[str, object]], serial_workers: int
) -> list[dict[str, object]]:
    index: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["population"]), str(row["chromosome"]))
        index[key][f"{row['arm']}:{row['stage']}:{row['workers']}"] = row

    summary: list[dict[str, object]] = []
    for (population, chromosome), entries in sorted(
        index.items(), key=lambda item: (item[0][0], chromosome_order(item[0][1]))
    ):
        covariance = entries.get("legacy:covariance:1")
        downstream = entries.get("legacy:downstream:1")
        serial = entries.get(f"lite:total:{serial_workers}")
        parallel = next(
            (
                value
                for key, value in entries.items()
                if key.startswith("lite:total:")
                and int(key.rsplit(":", 1)[1]) != serial_workers
            ),
            None,
        )
        if covariance is None or downstream is None:
            continue  # lite-only chromosome; nothing to compare against

        legacy_cpu = float(covariance["cpu_seconds"]) + float(downstream["cpu_seconds"])
        serial_cpu = float(serial["cpu_seconds"]) if serial else None
        parallel_cpu = float(parallel["cpu_seconds"]) if parallel else None
        summary.append(
            {
                "population": population,
                "chromosome": chromosome,
                "legacy_covariance_cpu_seconds": covariance["cpu_seconds"],
                "legacy_downstream_cpu_seconds": downstream["cpu_seconds"],
                "legacy_total_cpu_seconds": f"{legacy_cpu:.6f}",
                "lite_serial_cpu_seconds": serial["cpu_seconds"] if serial else "",
                "lite_parallel_cpu_seconds": parallel["cpu_seconds"]
                if parallel
                else "",
                "lite_parallel_workers": parallel["workers"] if parallel else "",
                "cpu_speedup_vs_lite_serial": ratio(legacy_cpu, serial_cpu),
                "cpu_speedup_vs_lite_parallel": ratio(legacy_cpu, parallel_cpu),
                "legacy_covariance_share": ratio(
                    float(covariance["cpu_seconds"]), legacy_cpu
                ),
                "lite_parallel_wall_seconds": (
                    parallel["wall_seconds"] if parallel else ""
                ),
            }
        )
    return summary


def ratio(numerator: float, denominator: float | None) -> str:
    if not denominator:
        return ""
    return f"{numerator / denominator:.6f}"


def write_tsv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_plot(path: Path, summary: list[dict[str, object]]) -> None:
    plt = setup_matplotlib(path.parent)

    if not summary:
        # No legacy arm ran, so there is nothing to compare against. Still emit
        # the file: callers (Snakemake) declare it as a required output.
        fig, ax = plt.subplots(figsize=(4.0, 1.4), layout="constrained")
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "no legacy runs to compare\n(ldetect-lite timings only)",
            ha="center",
            va="center",
            fontsize="small",
            color="#666666",
        )
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    labels = [f"chr{row['chromosome']}" for row in summary]
    legacy = [float(row["legacy_total_cpu_seconds"]) / 3600 for row in summary]
    serial = [hours(row["lite_serial_cpu_seconds"]) for row in summary]
    parallel = [hours(row["lite_parallel_cpu_seconds"]) for row in summary]

    positions = range(len(summary))
    width = 0.27
    fig, ax = plt.subplots(
        figsize=(1.6 + 0.9 * len(summary), 2.6), layout="constrained"
    )
    ax.bar(
        [p - width for p in positions], legacy, width, label="LDetect", color="#0057b8"
    )
    ax.bar(
        list(positions), serial, width, label="LDetect-lite (1 worker)", color="#d62728"
    )
    ax.bar(
        [p + width for p in positions],
        parallel,
        width,
        label="LDetect-lite (parallel)",
        color="#f0a202",
    )
    ax.set_xticks(list(positions), labels)
    ax.set_ylabel("CPU hours")
    ax.set_yscale("log")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, fontsize="small")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def hours(value: object) -> float:
    return float(value) / 3600 if value not in ("", None) else 0.0


def setup_matplotlib(output_dir: Path):
    for variable, subdir in (
        ("MPLCONFIGDIR", ".mplconfig"),
        ("XDG_CACHE_HOME", ".cache"),
    ):
        target = output_dir / subdir
        target.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(target))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def print_report(summary: list[dict[str, object]], args: argparse.Namespace) -> None:
    for row in summary:
        print(
            f"{row['population']} chr{row['chromosome']}: "
            f"legacy={float(row['legacy_total_cpu_seconds']) / 3600:.3f} CPU-h "
            f"(covariance {float(row['legacy_covariance_share']) * 100:.1f}%), "
            f"lite_serial={hours(row['lite_serial_cpu_seconds']):.3f} CPU-h, "
            f"speedup={row['cpu_speedup_vs_lite_serial'] or 'n/a'}x"
        )
    print(f"timings: {args.timings}")
    print(f"summary: {args.summary}")


if __name__ == "__main__":
    main()
