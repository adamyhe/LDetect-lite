#!/usr/bin/env python3
"""Aggregate per-chromosome resource usage across the full replication.

Reads the Snakemake `benchmark:` TSVs that `run_ldetect` already produces
for every chromosome x population job
(`results/logs/{population}/{chrom}.benchmark.tsv`, columns: s, h:m:s,
max_rss, max_vms, max_uss, max_pss, io_in, io_out, mean_load, cpu_time),
joins each row with the chromosome's partition count
(`results/{population}/{chrom}/{chrom}_partitions.txt` line count) as a
dataset-size proxy, and plots peak memory, wall-clock runtime, and core
utilization against dataset size, one series per population. No new run is
required -- this only aggregates data a completed replication already left
on disk.

Usage:
    uv run python scripts/plot_resource_scaling.py \
        --results-root results \
        --output-dir results/profiling \
        --workers-cap 4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TSV_FIELDS = [
    "population",
    "chrom",
    "n_partitions",
    "wall_s",
    "max_rss_mb",
    "mean_load_pct",
    "cores_used",
]


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    try:
        return (0, f"{int(chrom):03d}")
    except ValueError:
        return (1, chrom)


def collect_rows(results_root: Path, populations: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for population in populations:
        logs_dir = results_root / "logs" / population
        if not logs_dir.is_dir():
            print(f"Skipping {population}: no {logs_dir}")
            continue
        for benchmark_path in sorted(logs_dir.glob("*.benchmark.tsv")):
            chrom = benchmark_path.name.removesuffix(".benchmark.tsv")
            partitions_path = (
                results_root / population / chrom / f"{chrom}_partitions.txt"
            )
            if not partitions_path.exists():
                print(f"Skipping {population}/{chrom}: no partitions file")
                continue
            n_partitions = sum(1 for _ in partitions_path.open())

            with benchmark_path.open() as f:
                reader = csv.DictReader(f, delimiter="\t")
                record = next(reader, None)
            if record is None:
                print(f"Skipping {population}/{chrom}: empty benchmark TSV")
                continue

            mean_load_pct = float(record["mean_load"])
            rows.append(
                {
                    "population": population,
                    "chrom": chrom,
                    "n_partitions": n_partitions,
                    "wall_s": float(record["s"]),
                    "max_rss_mb": float(record["max_rss"]),
                    "mean_load_pct": mean_load_pct,
                    "cores_used": mean_load_pct / 100.0,
                }
            )
    rows.sort(key=lambda r: (r["population"], _chrom_sort_key(str(r["chrom"]))))
    return rows


def write_tsv(rows: list[dict[str, object]], output_tsv: Path) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _scatter_by_population(
    ax: plt.Axes, rows: list[dict[str, object]], y_field: str
) -> None:
    populations = sorted({str(r["population"]) for r in rows})
    for population in populations:
        pop_rows = [r for r in rows if r["population"] == population]
        x = [r["n_partitions"] for r in pop_rows]
        y = [r[y_field] for r in pop_rows]
        ax.scatter(x, y, label=population, alpha=0.8)


def plot_metric(
    rows: list[dict[str, object]],
    y_field: str,
    ylabel: str,
    title: str,
    output_stem: Path,
    reference_line: float | None = None,
    reference_label: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _scatter_by_population(ax, rows, y_field)
    if reference_line is not None:
        ax.axhline(
            reference_line,
            color="0.4",
            linestyle="--",
            linewidth=1.2,
            label=reference_label,
        )
    ax.set_xlabel("covariance partitions per chromosome (dataset-size proxy)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=150)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
    print(
        f"Wrote {output_stem.with_suffix('.png')} and "
        f"{output_stem.with_suffix('.pdf')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--populations", nargs="+", default=["EUR", "AFR", "ASN"], metavar="POP"
    )
    parser.add_argument(
        "--workers-cap",
        type=float,
        default=4.0,
        help="Configured --workers value, drawn as a reference line on the "
        "core-utilization plot (default: 4.0, matching config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/profiling"),
        help="Directory for the aggregated TSV and plots (default: results/profiling).",
    )
    args = parser.parse_args()

    rows = collect_rows(args.results_root, args.populations)
    if not rows:
        raise SystemExit("No benchmark rows found -- check --results-root")

    write_tsv(rows, args.output_dir / "resource_scaling.tsv")
    print(f"Aggregated {len(rows)} chromosome x population rows")

    plot_metric(
        rows,
        "max_rss_mb",
        "peak RSS (MB)",
        "Peak memory vs. dataset size",
        args.output_dir / "resource_scaling_memory",
    )
    plot_metric(
        rows,
        "wall_s",
        "wall-clock time (s)",
        "Runtime vs. dataset size",
        args.output_dir / "resource_scaling_runtime",
    )
    plot_metric(
        rows,
        "cores_used",
        "mean cores used (mean_load / 100)",
        "Core utilization vs. dataset size",
        args.output_dir / "resource_scaling_cores",
        reference_line=args.workers_cap,
        reference_label=f"--workers {args.workers_cap:g} cap",
    )


if __name__ == "__main__":
    main()
