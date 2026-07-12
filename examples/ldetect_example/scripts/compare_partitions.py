"""Compare our generated partition file against the ldetect reference.

Usage:
    uv run python scripts/compare_partitions.py \
        --ours   work/generated_partitions/chr2_partitions \
        --ref    ref/cov_matrix/scripts/chr2_partitions \
        --output results/compare_partitions.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

GENOMIC_FIG_WIDTH = 7.2
GENOMIC_LEFT = 0.12
GENOMIC_RIGHT = 0.98


def read_partitions(path: Path) -> list[tuple[int, int]]:
    partitions = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                partitions.append((int(parts[0]), int(parts[1])))
    return partitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours",   required=True, type=Path)
    parser.add_argument("--ref",    required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    ours = read_partitions(args.ours)
    ref  = read_partitions(args.ref)

    n_ours = len(ours)
    n_ref  = len(ref)
    exact = sum(1 for o, r in zip(ours, ref) if o == r)

    rows = [
        ("metric", "value"),
        ("n_ours",   n_ours),
        ("n_ref",    n_ref),
        ("n_exact_match", exact),
        ("match", "yes" if ours == ref else "no"),
    ]

    # Per-partition detail
    detail_cols = ["idx", "our_start", "our_end", "ref_start", "ref_end", "match"]
    detail_rows = []
    for i, (o, r) in enumerate(zip(ours, ref)):
        detail_rows.append({
            "idx": i, "our_start": o[0], "our_end": o[1],
            "ref_start": r[0], "ref_end": r[1],
            "match": "yes" if o == r else "no",
        })

    print(f"\nPartition comparison ({args.ours.name} vs {args.ref.name})")
    for k, v in rows[1:]:
        print(f"  {k}: {v}")
    if ours != ref:
        print("\n  Per-partition detail:")
        print("  " + "\t".join(detail_cols))
        for row in detail_rows:
            print("  " + "\t".join(str(row[c]) for c in detail_cols))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
        if detail_rows:
            writer.writerow([])
            writer.writerow(detail_cols)
            for row in detail_rows:
                writer.writerow([str(row[c]) for c in detail_cols])
    if args.plot is not None:
        write_plot(ours, ref, args.plot)
    if args.require_exact and ours != ref:
        raise SystemExit("partition output is not exact")
    print(f"\nWritten to {args.output}")


def write_plot(
    ours: list[tuple[int, int]],
    ref: list[tuple[int, int]],
    path: Path,
) -> None:
    configure_matplotlib_cache(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    path.parent.mkdir(parents=True, exist_ok=True)
    view_start, view_end = interval_view(ref)
    fig, ax = plt.subplots(figsize=(GENOMIC_FIG_WIDTH, 1.7))
    for idx, partitions in enumerate((ref, ours)):
        y = 1 - idx
        for part_idx, (start, end) in enumerate(partitions):
            if end < view_start or start > view_end:
                continue
            clipped_start = max(start, view_start)
            clipped_end = min(end, view_end)
            face = "black" if part_idx % 2 == 0 else "white"
            ax.add_patch(
                Rectangle(
                    (clipped_start, y - 0.14),
                    clipped_end - clipped_start,
                    0.28,
                    facecolor=face,
                    edgecolor="black",
                    linewidth=0.8,
                )
            )
    ax.set_yticks([0, 1], ["ours", "ref"])
    ax.set_xlabel("chr2 (hg19)")
    ax.set_title("Partition intervals near reference fixture")
    ax.set_xlim(view_start, view_end)
    ax.set_ylim(-0.5, 1.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.subplots_adjust(left=GENOMIC_LEFT, right=GENOMIC_RIGHT, bottom=0.28, top=0.78)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def interval_view(ref: list[tuple[int, int]]) -> tuple[int, int]:
    start = min(item[0] for item in ref)
    end = max(item[1] for item in ref)
    return start, end


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
