#!/usr/bin/env python
"""Plot genome-wide LD-neighborhood summaries from per-chromosome TSVs."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from pathlib import Path

CATEGORY_ORDER = ("left", "across", "right")
COLORS = {
    "left": "#4c78a8",
    "across": "#e45756",
    "right": "#54a24b",
}


def chrom_sort_key(chrom: str) -> tuple[int, int | str]:
    value = chrom.removeprefix("chr")
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def read_chrom_summary(path: Path) -> dict[str, object]:
    medians: dict[str, list[float]] = {category: [] for category in CATEGORY_ORDER}
    chrom = ""
    window_bp = ""
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            chrom = chrom or row["chrom"]
            window_bp = window_bp or row["window_bp"]
            if int(row["n"]) <= 0 or not row["median_r2"]:
                continue
            medians[row["category"]].append(float(row["median_r2"]))

    summary: dict[str, object] = {
        "chrom": chrom or path.stem,
        "window_bp": window_bp,
    }
    for category, values in medians.items():
        summary[f"{category}_median_r2"] = (
            statistics.median(values) if values else float("nan")
        )
        summary[f"{category}_n_boundaries"] = len(values)
    left = float(summary["left_median_r2"])
    right = float(summary["right_median_r2"])
    across = float(summary["across_median_r2"])
    within = (left + right) / 2.0
    summary["within_median_avg_r2"] = within
    summary["across_over_within"] = across / within if within > 0 else float("nan")
    return summary


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "chrom",
        "window_bp",
        "left_median_r2",
        "across_median_r2",
        "right_median_r2",
        "within_median_avg_r2",
        "across_over_within",
        "left_n_boundaries",
        "across_n_boundaries",
        "right_n_boundaries",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib_cache(path: Path) -> None:
    cache_root = path.parent / ".matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


def write_plot(path: Path, rows: list[dict[str, object]], title: str) -> None:
    configure_matplotlib_cache(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: chrom_sort_key(str(row["chrom"])))
    chroms = [str(row["chrom"]).removeprefix("chr") for row in rows]
    x = list(range(len(rows)))
    width = 0.22
    offsets = {"left": -width, "across": 0.0, "right": width}

    fig_width = max(5.0, 0.27 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, 2.25))
    for category in CATEGORY_ORDER:
        values = [float(row[f"{category}_median_r2"]) for row in rows]
        ax.bar(
            [idx + offsets[category] for idx in x],
            values,
            width=width,
            color=COLORS[category],
            alpha=0.82,
            linewidth=0,
            label=category,
        )

    ax.set_title(title, fontsize=8, pad=2)
    ax.set_ylabel("median $r^2$", labelpad=1)
    ax.set_xticks(x)
    ax.set_xticklabels(chroms, fontsize=6)
    ax.tick_params(axis="y", labelsize=7, pad=1)
    ax.tick_params(axis="x", pad=0)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=6,
        handlelength=1.0,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.16, top=0.82)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--plot", required=True, type=Path)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    rows = [read_chrom_summary(path) for path in args.inputs]
    write_summary(args.output_tsv, rows)
    write_plot(args.plot, rows, args.title or "LD neighborhood separation")
    print(f"Wrote {args.output_tsv} and {args.plot}")


if __name__ == "__main__":
    main()
