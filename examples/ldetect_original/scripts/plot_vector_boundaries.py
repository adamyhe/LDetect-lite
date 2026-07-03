#!/usr/bin/env python3
"""Plot the diagonal-sum vector with our and reference LD-block boundaries.

Visualizes the same signal the algorithm actually sees (Step 3's raw
[position, diagonal_sum] vector, plus the Hann-filtered/smoothed version used
to find minima in Step 4), overlaid with both our own final breakpoints and
the published reference breakpoints, for a chosen genomic window. Useful for
visually inspecting *why* a boundary ended up where it did -- e.g. whether a
divergent boundary corresponds to a real, comparably-deep trough at a
different position, or something less clear-cut. Similar in spirit to the
vector/minima figures in the LDetect paper (Berisa & Pickrell 2016, Fig. 1).

Usage:
    uv run python scripts/plot_vector_boundaries.py \
        --vector results/EUR/10/vector-10.txt.gz \
        --ours-bed results/EUR/10/10-ld-blocks.bed \
        --ref-bed resources/ldetect_ref/EUR_fourier_ls-all.bed \
        --chrom 10 \
        --region-start 53000000 --region-end 57500000 \
        --found-width 4305 \
        --title "EUR chr10: divergent boundary between two concordant ones" \
        --output /tmp/eur_chr10_mixed.png
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_vector(path: Path) -> tuple[np.ndarray, np.ndarray]:
    positions: list[int] = []
    values: list[float] = []
    opener = gzip.open if path.suffix in (".gz", ".gzip") else open
    with opener(path, "rt") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            positions.append(int(parts[0]))
            values.append(float(parts[1]))
    return np.array(positions, dtype=np.int64), np.array(values, dtype=np.float64)


def read_bed_boundaries(path: Path, chrom: str) -> list[int]:
    """Return sorted unique boundary positions (block starts+ends) for *chrom*.

    Accepts either bare ("10") or "chr"-prefixed ("chr10") chromosome labels
    in the file, matched against the bare *chrom* value.
    """
    want = {chrom, f"chr{chrom}"}
    positions: set[int] = set()
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3 or parts[0] not in want:
                continue
            try:
                positions.add(int(parts[1]))
                positions.add(int(parts[2]))
            except ValueError:
                continue
    return sorted(positions)


def hann_smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Same Hann-window convolution as ldetect2.filters.apply_filter."""
    from scipy import ndimage

    window = np.hanning(2 * width + 1)
    return ndimage.convolve1d(values, window / window.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector", required=True, type=Path)
    parser.add_argument("--ours-bed", required=True, type=Path)
    parser.add_argument("--ref-bed", required=True, type=Path)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--region-start", required=True, type=int)
    parser.add_argument("--region-end", required=True, type=int)
    parser.add_argument(
        "--found-width",
        type=int,
        default=None,
        help="Hann filter half-width from breakpoints-<chrom>.json's "
        "found_width; if given, overlays the smoothed vector actually used "
        "for minima detection.",
    )
    parser.add_argument("--tolerance", type=int, default=25_000)
    parser.add_argument("--title", default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    positions, values = read_vector(args.vector)
    ours = read_bed_boundaries(args.ours_bed, args.chrom)
    ref = read_bed_boundaries(args.ref_bed, args.chrom)

    smoothed = hann_smooth(values, args.found_width) if args.found_width else None

    mask = (positions >= args.region_start) & (positions <= args.region_end)
    x = positions[mask] / 1e6
    y_raw = values[mask]
    y_smooth = smoothed[mask] if smoothed is not None else None

    ours_in_window = [p for p in ours if args.region_start <= p <= args.region_end]
    ref_in_window = [p for p in ref if args.region_start <= p <= args.region_end]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y_raw, color="0.75", linewidth=0.8, label="raw vector (diagonal sum)")
    if y_smooth is not None:
        ax.plot(
            x, y_smooth, color="black", linewidth=1.6,
            label=f"Hann-smoothed (width={args.found_width})",
        )

    for i, p in enumerate(ref_in_window):
        ax.axvline(
            p / 1e6, color="tab:red", linestyle="--", linewidth=1.4, alpha=0.85,
            label="reference boundary" if i == 0 else None,
        )

    seen_labels: set[str] = set()
    for p in ours_in_window:
        matched = any(abs(p - r) <= args.tolerance for r in ref)
        label = "our boundary (matched)" if matched else "our boundary (divergent)"
        ax.axvline(
            p / 1e6,
            color="tab:blue" if matched else "tab:orange",
            linestyle="-",
            linewidth=1.2,
            alpha=0.85,
            label=label if label not in seen_labels else None,
        )
        seen_labels.add(label)

    ax.set_xlabel(f"chr{args.chrom} position (Mb)")
    ax.set_ylabel("diagonal sum of r²")
    if args.title:
        ax.set_title(args.title)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=9)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")
    print(f"  our boundaries in window: {ours_in_window}")
    print(f"  ref boundaries in window: {ref_in_window}")


if __name__ == "__main__":
    main()
