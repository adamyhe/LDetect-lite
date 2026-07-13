"""Compare our breakpoints (JSON) against the ldetect reference (pickle).

Compares all four subsets: fourier, fourier_ls, uniform, uniform_ls.

Usage:
    uv run python scripts/compare_bpoints.py \
        --ours   work/breakpoints-chr2.json \
        --ref    ref/minima/minima-EUR-chr2-50-39967768-40067768.pickle \
        --output results/compare_bpoints.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

SUBSETS = ("fourier", "fourier_ls", "uniform", "uniform_ls")
PLOT_SUBSETS = ("uniform", "uniform_ls", "fourier", "fourier_ls")
GENOMIC_FIG_WIDTH = 7.2
GENOMIC_LEFT = 0.12
GENOMIC_RIGHT = 0.98


def compare_loci(ours: list[int], ref: list[int]) -> dict:
    n_ours = len(ours)
    n_ref  = len(ref)
    ours_set = set(ours)
    ref_set  = set(ref)
    exact    = len(ours_set & ref_set)
    return {
        "n_ours":   n_ours,
        "n_ref":    n_ref,
        "n_exact":  exact,
        "recall":   round(exact / n_ref,  4) if n_ref  else "nan",
        "precision": round(exact / n_ours, 4) if n_ours else "nan",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours",   required=True, type=Path)
    parser.add_argument("--ref",    required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    ours_data = json.loads(args.ours.read_text())
    with open(args.ref, "rb") as f:
        ref_data = pickle.load(f)

    cols = ["subset", "n_ours", "n_ref", "n_exact", "recall", "precision"]
    rows: list[dict] = []

    for subset in SUBSETS:
        our_loci = ours_data.get(subset, {}).get("loci", [])
        ref_loci = ref_data.get(subset, {}).get("loci", [])
        row = {"subset": subset, **compare_loci(our_loci, ref_loci)}
        rows.append(row)
    all_exact = all(
        row["n_ours"] == row["n_ref"] == row["n_exact"] for row in rows
    ) and ours_data.get("n_bpoints") == ref_data.get("n_bpoints") and (
        ours_data.get("found_width") == ref_data.get("found_width")
    )

    print(f"\nBreakpoint comparison ({args.ours.name} vs {args.ref.name})")
    print("  " + "\t".join(cols))
    for row in rows:
        print("  " + "\t".join(str(row[c]) for c in cols))

    # Also compare n_bpoints and found_width
    print(
        f"\n  n_bpoints : ours={ours_data.get('n_bpoints')}  "
        f"ref={ref_data.get('n_bpoints')}"
    )
    print(
        f"  found_width: ours={ours_data.get('found_width')}  "
        f"ref={ref_data.get('found_width')}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

        # Append scalar metadata as extra rows
        writer.writerow(
            {
                "subset": "n_bpoints",
                "n_ours": ours_data.get("n_bpoints"),
                "n_ref": ref_data.get("n_bpoints"),
            }
        )
        writer.writerow(
            {
                "subset": "found_width",
                "n_ours": ours_data.get("found_width"),
                "n_ref": ref_data.get("found_width"),
            }
        )

    if args.plot is not None:
        write_plot(ours_data, ref_data, args.ref, args.plot)
    if args.require_exact and not all_exact:
        raise SystemExit("breakpoint output is not exact")
    print(f"\nWritten to {args.output}")


def write_plot(ours_data: dict, ref_data: dict, ref_path: Path, path: Path) -> None:
    configure_matplotlib_cache(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(PLOT_SUBSETS),
        1,
        figsize=(GENOMIC_FIG_WIDTH, 1.0 * len(PLOT_SUBSETS)),
        sharex=True,
    )
    if len(PLOT_SUBSETS) == 1:
        axes = [axes]
    for ax, subset in zip(axes, PLOT_SUBSETS, strict=True):
        our_loci = ours_data.get(subset, {}).get("loci", [])
        ref_loci = ref_data.get(subset, {}).get("loci", [])
        ax.eventplot(
            [ref_loci, our_loci],
            lineoffsets=[1, 0],
            colors=["#0057b8", "#d62728"],
            linewidths=1.9,
        )
        ax.set_yticks([0, 1], ["ours", "ref"])
        ax.set_title(subset)
    xlim = fixture_xlim_from_path(ref_path)
    if xlim is not None:
        axes[-1].set_xlim(*xlim)
    axes[-1].set_xlabel("chr2 (hg19)")
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.subplots_adjust(left=GENOMIC_LEFT, right=GENOMIC_RIGHT, hspace=0.62)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fixture_xlim_from_path(path: Path) -> tuple[int, int] | None:
    stem = path.name.removesuffix(".pickle")
    parts = stem.split("-")
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
