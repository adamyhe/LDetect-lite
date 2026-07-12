"""Compare our correlation-sum vector against the ldetect reference vector.

Both files are gzipped TSV with columns: position  value

Usage:
    uv run python scripts/compare_vector.py \
        --ours   work/vector-chr2.txt.gz \
        --ref    ref/vector/vector-EUR-chr2-39967768-40067768.txt.gz \
        --output results/compare_vector.tsv
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

GENOMIC_FIG_WIDTH = 7.2
GENOMIC_LEFT = 0.12
GENOMIC_RIGHT = 0.98
REFERENCE_COLOR = "#0057b8"
LDETECT_LITE_COLOR = "#d62728"


def read_vector(path: Path) -> dict[int, float]:
    opener = gzip.open(path, "rt") if path.suffix in (".gz", ".gzip") else open(path)
    out: dict[int, float] = {}
    with opener as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            out[int(row[0])] = float(row[1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours",   required=True, type=Path)
    parser.add_argument("--ref",    required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--require-equivalent", action="store_true")
    args = parser.parse_args()

    ours = read_vector(args.ours)
    ref  = read_vector(args.ref)

    all_pos = sorted(set(ours) | set(ref))
    only_ours = sum(1 for p in all_pos if p not in ref)
    only_ref  = sum(1 for p in all_pos if p not in ours)
    shared    = [p for p in all_pos if p in ours and p in ref]

    abs_diffs = [abs(ours[p] - ref[p]) for p in shared]
    rel_diffs = [
        abs(ours[p] - ref[p]) / max(abs(ref[p]), 1e-30)
        for p in shared
    ]

    max_abs = max(abs_diffs) if abs_diffs else float("nan")
    mean_abs = sum(abs_diffs) / len(abs_diffs) if abs_diffs else float("nan")
    max_rel = max(rel_diffs) if rel_diffs else float("nan")
    exact_match = sum(1 for d in abs_diffs if d == 0.0)
    equivalent_match = sum(1 for d in abs_diffs if d <= args.atol)

    rows = [
        ("metric", "value"),
        ("n_ours", len(ours)),
        ("n_ref", len(ref)),
        ("n_shared", len(shared)),
        ("only_in_ours", only_ours),
        ("only_in_ref", only_ref),
        ("exact_matches", exact_match),
        ("equivalent_matches", equivalent_match),
        ("max_abs_diff", f"{max_abs:.6e}"),
        ("mean_abs_diff", f"{mean_abs:.6e}"),
        ("max_rel_diff", f"{max_rel:.6e}"),
        ("all_equivalent", "yes" if equivalent_match == len(shared) else "no"),
    ]

    print(f"\nVector comparison ({args.ours.name} vs {args.ref.name})")
    for k, v in rows[1:]:
        print(f"  {k}: {v}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
    if args.plot is not None:
        write_plot(ours, ref, shared, args.ref, args.plot)
    if args.require_exact and (
        only_ours != 0 or only_ref != 0 or exact_match != len(shared)
    ):
        raise SystemExit("vector output is not exact")
    if args.require_equivalent and (
        only_ours != 0 or only_ref != 0 or equivalent_match != len(shared)
    ):
        raise SystemExit("vector output is not numerically equivalent")
    print(f"\nWritten to {args.output}")


def write_plot(
    ours: dict[int, float],
    ref: dict[int, float],
    shared: list[int],
    ref_path: Path,
    path: Path,
) -> None:
    configure_matplotlib_cache(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    path.parent.mkdir(parents=True, exist_ok=True)
    x = shared
    ref_values = [ref[p] for p in shared]
    ours_values = [ours[p] for p in shared]
    diffs = [ours[p] - ref[p] for p in shared]

    fig, axes = plt.subplots(2, 1, figsize=(GENOMIC_FIG_WIDTH, 3.2), sharex=True)
    axes[0].plot(x, ref_values, label="reference", linewidth=1.5, color=REFERENCE_COLOR)
    axes[0].plot(
        x,
        ours_values,
        label="ldetect-lite",
        linewidth=1,
        linestyle="--",
        color=LDETECT_LITE_COLOR,
    )
    axes[0].set_ylabel("diagonal sum")
    axes[0].set_title("Matrix-to-vector output")
    axes[0].legend()

    axes[1].plot(x, diffs, linewidth=1, color=LDETECT_LITE_COLOR)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("chr2 (hg19)")
    axes[1].set_ylabel("difference")
    xlim = fixture_xlim_from_path(ref_path) or (min(shared), max(shared))
    axes[1].set_xlim(*xlim)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    fig.subplots_adjust(left=GENOMIC_LEFT, right=GENOMIC_RIGHT, hspace=0.38)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fixture_xlim_from_path(path: Path) -> tuple[int, int] | None:
    name = path.name
    for suffix in (".txt.gz", ".txt", ".gz"):
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
            break
    parts = name.split("-")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return int(parts[-2]), int(parts[-1])
    return None


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
