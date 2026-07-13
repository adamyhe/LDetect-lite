"""Compare our BED output against the ldetect reference BED.

Usage:
    uv run python scripts/compare_bed.py \
        --ours   work/chr2-ld-blocks.bed \
        --ref    ref/bed/EUR-chr2-50-39967768-40067768.bed \
        --output results/compare_bed.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ldetect_lite.io.bed import read_single_chrom_bed

GENOMIC_FIG_WIDTH = 7.2
GENOMIC_LEFT = 0.12
GENOMIC_RIGHT = 0.98


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours",   required=True, type=Path)
    parser.add_argument("--ref",    required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    _, ours = read_single_chrom_bed(args.ours)
    _, ref = read_single_chrom_bed(args.ref)

    n_ours = len(ours)
    n_ref  = len(ref)

    exact_blocks = sum(1 for o, r in zip(ours, ref) if o == r)
    start_match  = sum(1 for o, r in zip(ours, ref) if o[0] == r[0])
    end_match    = sum(1 for o, r in zip(ours, ref) if o[1] == r[1])

    all_our_bounds = sorted({b for blk in ours for b in blk})
    all_ref_bounds = sorted({b for blk in ref  for b in blk})
    exact_bounds   = len(set(all_our_bounds) & set(all_ref_bounds))

    rows = [
        ("metric", "value"),
        ("n_ours_blocks", n_ours),
        ("n_ref_blocks",  n_ref),
        ("exact_block_matches", exact_blocks),
        ("start_position_matches", start_match),
        ("end_position_matches", end_match),
        ("n_our_boundaries", len(all_our_bounds)),
        ("n_ref_boundaries", len(all_ref_bounds)),
        ("exact_boundary_matches", exact_bounds),
        ("boundary_recall",    round(exact_bounds / len(all_ref_bounds), 4)
                                if all_ref_bounds else "nan"),
        ("boundary_precision", round(exact_bounds / len(all_our_bounds), 4)
                                if all_our_bounds else "nan"),
    ]

    print(f"\nBED comparison ({args.ours.name} vs {args.ref.name})")
    for k, v in rows[1:]:
        print(f"  {k}: {v}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
    if args.plot is not None:
        write_plot(ours, ref, args.ref, args.plot)
    if args.require_exact and ours != ref:
        raise SystemExit("BED output is not exact")
    print(f"\nWritten to {args.output}")


def write_plot(
    ours: list[tuple[int, int]],
    ref: list[tuple[int, int]],
    ref_path: Path,
    path: Path,
) -> None:
    configure_matplotlib_cache(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(GENOMIC_FIG_WIDTH, 1.7))
    for idx, blocks in enumerate((ref, ours)):
        y = 1 - idx
        for block_idx, (start, end) in enumerate(blocks):
            face = "black" if block_idx % 2 == 0 else "white"
            ax.add_patch(
                Rectangle(
                    (start, y - 0.14),
                    end - start,
                    0.28,
                    facecolor=face,
                    edgecolor="black",
                    linewidth=0.8,
                )
            )
    ax.set_yticks([0, 1], ["ours", "ref"])
    ax.set_xlabel("chr2 (hg19)")
    ax.set_title("BED block intervals")
    ax.autoscale_view()
    xlim = fixture_xlim_from_path(ref_path) or interval_xlim(ref)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.5, 1.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.subplots_adjust(left=GENOMIC_LEFT, right=GENOMIC_RIGHT, bottom=0.28, top=0.78)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fixture_xlim_from_path(path: Path) -> tuple[int, int] | None:
    stem = path.name.removesuffix(".bed")
    parts = stem.split("-")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return int(parts[-2]), int(parts[-1])
    return None


def interval_xlim(intervals: list[tuple[int, int]]) -> tuple[int, int]:
    return min(item[0] for item in intervals), max(item[1] for item in intervals)


def configure_matplotlib_cache(path: Path) -> None:
    import os

    root = path.parent.parent if path.parent.name == "plots" else path.parent
    mpl_config = root / ".mplconfig"
    xdg_cache = root / ".cache"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache.resolve()))


if __name__ == "__main__":
    main()
