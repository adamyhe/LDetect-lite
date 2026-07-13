"""Generate schematic diagrams for the LDetect/LDetect-lite pipeline.

The figures are conceptual, publication-friendly SVG/PDF schematics of the
five pipeline stages. They are intentionally data-free: the goal is to show
what each command transforms, not to reproduce a particular chromosome run.

Usage:
    uv run --extra heatmap python schematics/plot_pipeline_schematics.py
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable


BLUE = "#0057b8"
RED = "#d62728"
GREEN = "#009e73"
ORANGE = "#e69f00"
GRAY = "#6b6b6b"
LIGHT_BLUE = "#dce9f8"
LIGHT_RED = "#f8dedb"
LIGHT_GREEN = "#ddf1e9"
LIGHT_ORANGE = "#f7ecd3"
LIGHT_GRAY = "#ededed"
DARK = "#222222"
GRID = "#d0d0d0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("schematics/plots"),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["svg", "pdf"],
        choices=["svg", "pdf", "png"],
        help="One or more output formats to write.",
    )
    args = parser.parse_args()

    configure_matplotlib_cache(args.output_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "step1-partition-chromosome": plot_step1_partition(plt),
        "step2-calc-covariance": plot_step2_covariance(plt),
        "step3-matrix-to-vector": plot_step3_vector(plt),
        "step4-find-breakpoints": plot_step4_minima(plt),
        "step5-extract-bed": plot_step5_bed(plt),
    }
    for stem, fig in figures.items():
        save_figure(fig, args.output_dir / stem, args.formats)
        plt.close(fig)
    print(f"Wrote {len(figures)} schematic figure(s) to {args.output_dir.resolve()}")


def configure_matplotlib_cache(output_dir: Path) -> None:
    root = output_dir.parent.parent if output_dir.parent.name == "plots" else output_dir
    mpl_config = root / ".cache" / "matplotlib"
    xdg_cache = root / ".cache"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache.resolve()))


def save_figure(fig, stem: Path, formats: Iterable[str]) -> None:
    for fmt in formats:
        fig.savefig(
            stem.with_suffix(f".{fmt}"),
            dpi=180,
            bbox_inches="tight",
            pad_inches=0.03,
        )


def setup_axis(ax, xlim: tuple[float, float] = (0, 1), ylim: tuple[float, float] = (0, 1)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    *,
    facecolor: str = LIGHT_GRAY,
    edgecolor: str = DARK,
    fontsize: int = 10,
):
    import matplotlib.patches as patches

    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1 if subtitle else 0
    title_y = 0.68 if subtitle and title_lines > 1 else 0.60 if subtitle else 0.5
    subtitle_y = 0.20 if title_lines > 1 or subtitle_lines > 1 else 0.28
    ax.text(
        x + width / 2,
        y + height * title_y,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=DARK,
        linespacing=1.08,
    )
    if subtitle:
        ax.text(
            x + width / 2,
            y + height * subtitle_y,
            subtitle,
            ha="center",
            va="center",
            fontsize=fontsize - 1,
            color=DARK,
            linespacing=1.08,
        )
    return box


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = DARK,
    lw: float = 1.1,
    rad: float = 0.0,
):
    import matplotlib.patches as patches

    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arrow)
    return arrow


def add_chromosome(ax, y: float, x0: float = 0.08, x1: float = 0.92, *, color: str = DARK):
    ax.plot([x0, x1], [y, y], color=color, linewidth=2.0, solid_capstyle="round")
    snp_fractions = [0.05, 0.14, 0.27, 0.40, 0.52, 0.64, 0.74, 0.86, 0.95]
    snps = [x0 + (x1 - x0) * fraction for fraction in snp_fractions]
    for x in snps:
        ax.plot([x, x], [y - 0.035, y + 0.035], color=color, linewidth=1.0)
    return snps


def plot_overview(plt):
    fig, ax = plt.subplots(figsize=(9.8, 2.1))
    setup_axis(ax)
    steps = [
        ("1", "partition-chromosome", "map -> windows", LIGHT_BLUE),
        ("2", "calc-covariance", "VCF -> LD cache", LIGHT_GREEN),
        ("3", "matrix-to-vector", "cache -> vector", LIGHT_ORANGE),
        ("4", "find-minima", "smooth + refine", LIGHT_RED),
        ("5", "extract-bpoints", "JSON -> BED", LIGHT_GRAY),
    ]
    box_w = 0.15
    gap = 0.04
    x = 0.04
    y = 0.32
    for i, (num, title, subtitle, fill) in enumerate(steps):
        add_box(ax, (x, y), box_w, 0.34, f"Step {num}", title, facecolor=fill, fontsize=9)
        ax.text(
            x + box_w / 2,
            y - 0.06,
            subtitle,
            ha="center",
            va="top",
            fontsize=8,
            color=DARK,
        )
        if i < len(steps) - 1:
            add_arrow(ax, (x + box_w + 0.008, y + 0.17), (x + box_w + gap - 0.008, y + 0.17))
        x += box_w + gap
    ax.text(
        0.5,
        0.82,
        "LDetect/LDetect-lite pipeline: reference haplotypes to LD blocks",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=DARK,
    )
    return fig


def plot_step1_partition(plt):
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    setup_axis(ax)
    ax.text(0.04, 0.84, "Step 1 - partition chromosome", fontsize=11, fontweight="bold", color=DARK)
    add_box(ax, (0.04, 0.57), 0.19, 0.20, "genetic map", "position + cM", facecolor=LIGHT_BLUE, fontsize=9)
    add_box(ax, (0.04, 0.27), 0.19, 0.20, "sample size", "sets threshold", facecolor=LIGHT_GRAY, fontsize=9)
    add_arrow(ax, (0.25, 0.67), (0.33, 0.62))
    add_arrow(ax, (0.25, 0.37), (0.33, 0.50))
    add_chromosome(ax, 0.52, x0=0.35, x1=0.94)
    windows = [(0.36, 0.52), (0.50, 0.66), (0.64, 0.80), (0.78, 0.93)]
    colors = [LIGHT_GREEN, LIGHT_ORANGE, LIGHT_GREEN, LIGHT_ORANGE]
    for i, ((start, end), fill) in enumerate(zip(windows, colors)):
        ax.add_patch(
            plt.Rectangle(
                (start, 0.38),
                end - start,
                0.09,
                facecolor=fill,
                edgecolor=DARK,
                linewidth=0.8,
            )
        )
        ax.text((start + end) / 2, 0.30, f"window {i + 1}", ha="center", fontsize=8, color=DARK)
    for x in [0.50, 0.64, 0.78]:
        ax.plot([x, x], [0.38, 0.68], color=RED, linewidth=1.0, linestyle="--")
    ax.text(0.64, 0.19, "overlapping windows", ha="center", fontsize=9, color=DARK)
    return fig


def plot_step2_covariance(plt):
    fig, ax = plt.subplots(figsize=(7.2, 2.1))
    setup_axis(ax)
    ax.text(0.04, 0.84, "Step 2 - calculate covariance per partition", fontsize=11, fontweight="bold", color=DARK)
    add_box(ax, (0.04, 0.58), 0.18, 0.18, "phased VCF", "indexed region", facecolor=LIGHT_BLUE, fontsize=9)
    add_box(ax, (0.04, 0.30), 0.18, 0.18, "samples", "individual list", facecolor=LIGHT_GRAY, fontsize=9)
    add_box(ax, (0.29, 0.44), 0.16, 0.18, "haplotypes", "SNP x hap", facecolor=LIGHT_GREEN, fontsize=9)
    draw_small_matrix(ax, 0.53, 0.33, 0.16, 0.36, filled=True)
    ax.text(0.61, 0.73, "shrinkage LD", ha="center", fontsize=9, color=DARK)
    add_box(ax, (0.75, 0.44), 0.18, 0.18, "HDF5 cache", "shrink LD", facecolor=LIGHT_ORANGE, fontsize=9)
    add_arrow(ax, (0.22, 0.67), (0.29, 0.56))
    add_arrow(ax, (0.22, 0.39), (0.29, 0.50))
    add_arrow(ax, (0.45, 0.53), (0.53, 0.53))
    add_arrow(ax, (0.69, 0.53), (0.75, 0.53))
    ax.text(0.84, 0.28, "per partition", ha="center", fontsize=9, color=DARK)
    return fig


def draw_small_matrix(ax, x: float, y: float, w: float, h: float, *, filled: bool) -> None:
    cells = 6
    for i in range(cells):
        for j in range(cells):
            alpha = 0.18 + 0.12 * ((i + j) % 3)
            if abs(i - j) <= 1:
                alpha = 0.85
            color = BLUE if filled else LIGHT_GRAY
            ax.add_patch(
                rectangle(
                    x + w * j / cells,
                    y + h * (cells - 1 - i) / cells,
                    w / cells,
                    h / cells,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.4,
                    alpha=alpha,
                )
            )
    ax.add_patch(rectangle(x, y, w, h, facecolor="none", edgecolor=DARK, linewidth=0.9))


def rectangle(x: float, y: float, width: float, height: float, **kwargs):
    import matplotlib.patches as patches

    return patches.Rectangle((x, y), width, height, **kwargs)


def plot_step3_vector(plt):
    fig, ax = plt.subplots(figsize=(7.2, 2.1))
    setup_axis(ax)
    ax.text(0.04, 0.84, "Step 3 - matrix to vector", fontsize=11, fontweight="bold", color=DARK)
    for offset in [0.00, 0.032, 0.064]:
        draw_small_matrix(ax, 0.05 + offset, 0.40 + offset, 0.15, 0.30, filled=True)
    add_box(ax, (0.38, 0.46), 0.17, 0.18, "sum r2", "per SNP", facecolor=LIGHT_ORANGE, fontsize=9)
    add_arrow(ax, (0.28, 0.55), (0.365, 0.55))
    add_arrow(ax, (0.565, 0.55), (0.64, 0.55))
    xs = [0.64 + i * 0.027 for i in range(10)]
    values = [0.50, 0.64, 0.70, 0.43, 0.58, 0.72, 0.66, 0.40, 0.54, 0.62]
    ax.plot(xs, values, color=BLUE, linewidth=1.8)
    ax.scatter(xs, values, s=18, color=BLUE, zorder=3)
    ax.plot([0.63, 0.91], [0.36, 0.36], color=DARK, linewidth=0.8)
    ax.plot([0.63, 0.63], [0.36, 0.76], color=DARK, linewidth=0.8)
    ax.text(0.78, 0.25, "diagonal-sum vector", ha="center", fontsize=9, color=DARK)
    for idx in [3, 7]:
        ax.plot([xs[idx], xs[idx]], [0.36, values[idx]], color=RED, linewidth=0.9, linestyle="--")
    return fig


def plot_step4_minima(plt):
    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    setup_axis(ax)
    ax.text(0.04, 0.86, "Step 4 - find and refine breakpoints", fontsize=11, fontweight="bold", color=DARK)
    x = [0.08 + i * 0.025 for i in range(25)]
    raw = [0.48 + 0.12 * math.sin(i * 0.8) + 0.04 * math.sin(i * 2.2) for i in range(25)]
    smooth = [0.49 + 0.10 * math.sin(i * 0.8) for i in range(25)]
    ax.plot(x, raw, color=GRAY, linewidth=1.0, alpha=0.75, label="raw vector")
    ax.plot(x, smooth, color=BLUE, linewidth=1.8, label="Hann-smoothed")
    minima_idx = [5, 13, 21]
    ax.scatter([x[i] for i in minima_idx], [smooth[i] for i in minima_idx], s=36, color=RED, zorder=4)
    for i in minima_idx:
        ax.plot([x[i], x[i]], [0.30, smooth[i]], color=RED, linewidth=0.9, linestyle="--")
    ax.plot([0.07, 0.70], [0.30, 0.30], color=DARK, linewidth=0.8)
    ax.plot([0.07, 0.07], [0.30, 0.68], color=DARK, linewidth=0.8)
    ax.text(0.38, 0.21, "smoothed minima", ha="center", fontsize=9, color=DARK)
    add_box(ax, (0.77, 0.57), 0.19, 0.15, "candidates", "fourier / uniform", facecolor=LIGHT_RED, fontsize=9)
    add_box(ax, (0.77, 0.35), 0.19, 0.15, "local search", "score nearby SNPs", facecolor=LIGHT_ORANGE, fontsize=9)
    add_box(ax, (0.77, 0.13), 0.19, 0.15, "breakpoints", "fourier_ls JSON", facecolor=LIGHT_GREEN, fontsize=9)
    add_arrow(ax, (0.865, 0.57), (0.865, 0.50))
    add_arrow(ax, (0.865, 0.35), (0.865, 0.28))
    return fig


def plot_step5_bed(plt):
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    setup_axis(ax)
    ax.text(0.04, 0.84, "Step 5 - extract LD blocks to BED", fontsize=11, fontweight="bold", color=DARK)
    add_box(ax, (0.04, 0.50), 0.18, 0.18, "breakpoint\nJSON", facecolor=LIGHT_GREEN, fontsize=8)
    add_arrow(ax, (0.23, 0.59), (0.35, 0.59))
    add_chromosome(ax, 0.58, x0=0.37, x1=0.94)
    breakpoints = [0.49, 0.64, 0.80]
    block_edges = [0.37, *breakpoints, 0.94]
    fills = [LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GREEN, LIGHT_RED]
    for i, (start, end) in enumerate(zip(block_edges[:-1], block_edges[1:])):
        ax.add_patch(
            plt.Rectangle(
                (start, 0.39),
                end - start,
                0.11,
                facecolor=fills[i],
                edgecolor=DARK,
                linewidth=0.8,
            )
        )
        ax.text((start + end) / 2, 0.32, f"block {i + 1}", ha="center", fontsize=8, color=DARK)
    for x in breakpoints:
        ax.plot([x, x], [0.36, 0.71], color=RED, linewidth=1.1, linestyle="--")
    add_box(ax, (0.44, 0.10), 0.38, 0.13, "BED output", facecolor=LIGHT_GRAY, fontsize=8)
    return fig


if __name__ == "__main__":
    main()
