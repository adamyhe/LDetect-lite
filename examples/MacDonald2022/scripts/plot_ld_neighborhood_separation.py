#!/usr/bin/env python
"""Plot LD separation around called block boundaries.

For each internal LD-block boundary, this diagnostic samples normalized
pairwise LD (r^2) from three local neighborhoods:

  left    pairs wholly in [boundary - window, boundary)
  across  pairs crossing the boundary, one SNP on each side
  right   pairs wholly in [boundary, boundary + window]

The resulting box/whisker plot is an orthogonal benchmark to exact boundary
matching: useful boundaries should tend to make the across-boundary LD
distribution lower than the within-neighborhood distributions.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ldetect_lite.io.bed import Block, read_genome_bed
from ldetect_lite.io.covariance_hdf5 import open_covariance_reader
from ldetect_lite.io.partitions import CovarianceStore, read_partitions

CATEGORY_ORDER = ("left", "across", "right")
COLORS = {
    "left": "#4c78a8",
    "across": "#e45756",
    "right": "#54a24b",
}


def internal_boundaries(blocks: list[Block]) -> list[int]:
    return [end for _start, end in blocks[:-1]]


def read_diagonal_index(
    name: str,
    store: CovarianceStore,
    partitions: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    pos_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []
    for start, end in partitions:
        path = store.partition_path(name, start, end)
        with open_covariance_reader(path, start, end) as reader:
            pos, val = reader.read_diagonal()
        if pos.size:
            pos_chunks.append(pos.astype(np.int64, copy=False))
            val_chunks.append(val.astype(np.float64, copy=False))

    if not pos_chunks:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    pos = np.concatenate(pos_chunks)
    val = np.concatenate(val_chunks)
    order = np.argsort(pos, kind="stable")
    pos = pos[order]
    val = val[order]
    unique_pos, unique_idx = np.unique(pos, return_index=True)
    return unique_pos, val[unique_idx]


def r2_for_pairs(
    lo: np.ndarray,
    hi: np.ndarray,
    shrink_ld: np.ndarray,
    diag_pos: np.ndarray,
    diag_val: np.ndarray,
) -> np.ndarray:
    if lo.size == 0 or diag_pos.size == 0:
        return np.array([], dtype=np.float64)

    lo_idx = np.searchsorted(diag_pos, lo)
    hi_idx = np.searchsorted(diag_pos, hi)
    has_diag = (lo_idx < diag_pos.size) & (hi_idx < diag_pos.size)
    safe_lo_idx = np.minimum(lo_idx, diag_pos.size - 1)
    safe_hi_idx = np.minimum(hi_idx, diag_pos.size - 1)
    has_diag &= (diag_pos[safe_lo_idx] == lo) & (diag_pos[safe_hi_idx] == hi)
    if not np.any(has_diag):
        return np.array([], dtype=np.float64)

    lo_idx = lo_idx[has_diag]
    hi_idx = hi_idx[has_diag]
    shrink = shrink_ld[has_diag]
    denom = diag_val[lo_idx] * diag_val[hi_idx]
    positive = denom > 0.0
    if not np.any(positive):
        return np.array([], dtype=np.float64)

    values = shrink[positive] * shrink[positive] / denom[positive]
    return values[np.isfinite(values)]


def owned_bounds(
    partitions: list[tuple[int, int]],
    p_index: int,
    snp_first: int,
    snp_last: int,
) -> tuple[int, int, bool]:
    start = partitions[p_index][0]
    lower_min = snp_first if p_index == 0 else start
    lower_max = (
        partitions[p_index + 1][0] if p_index + 1 < len(partitions) else snp_last
    )
    return lower_min, lower_max, p_index == 0


def category_masks(
    lo: np.ndarray,
    hi: np.ndarray,
    boundary: int,
    left: int,
    right: int,
) -> dict[str, np.ndarray]:
    in_window = (lo >= left) & (hi <= right) & (lo < hi)
    return {
        "left": in_window & (hi < boundary),
        "across": in_window & (lo < boundary) & (hi >= boundary),
        "right": in_window & (lo >= boundary),
    }


def summarize(values: list[float]) -> dict[str, object]:
    if not values:
        return {
            "n": 0,
            "mean_r2": "",
            "median_r2": "",
            "q1_r2": "",
            "q3_r2": "",
            "p90_r2": "",
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_r2": f"{float(np.mean(arr)):.8g}",
        "median_r2": f"{float(np.median(arr)):.8g}",
        "q1_r2": f"{float(np.quantile(arr, 0.25)):.8g}",
        "q3_r2": f"{float(np.quantile(arr, 0.75)):.8g}",
        "p90_r2": f"{float(np.quantile(arr, 0.90)):.8g}",
    }


def reservoir_extend(
    sample: list[float],
    values: Iterable[float],
    *,
    seen: int,
    limit: int,
    rng: random.Random,
) -> int:
    for value in values:
        seen += 1
        if len(sample) < limit:
            sample.append(float(value))
            continue
        replacement = rng.randrange(seen)
        if replacement < limit:
            sample[replacement] = float(value)
    return seen


def boundary_rows_and_samples(
    *,
    chrom: str,
    boundaries: list[int],
    store: CovarianceStore,
    name: str,
    window_bp: int,
    sample_limit: int,
    seed: int,
    chunk_rows: int,
) -> tuple[list[dict[str, object]], dict[str, list[float]]]:
    partitions = read_partitions(name, store)
    diag_pos, diag_val = read_diagonal_index(name, store, partitions)
    if diag_pos.size == 0:
        raise RuntimeError(f"No diagonal rows found for {name} in {store.root}")

    rows: list[dict[str, object]] = []
    samples: dict[str, list[float]] = {category: [] for category in CATEGORY_ORDER}
    seen = {category: 0 for category in CATEGORY_ORDER}
    rng = random.Random(seed)

    for boundary in boundaries:
        left = boundary - window_bp
        right = boundary + window_bp
        per_boundary: dict[str, list[float]] = {
            category: [] for category in CATEGORY_ORDER
        }

        for p_index, (start, end) in enumerate(partitions):
            if end < left or start > right:
                continue
            lower_min, lower_max, include_lower_min = owned_bounds(
                partitions,
                p_index,
                left,
                right,
            )
            if lower_min > right or lower_max < left:
                continue
            path = store.partition_path(name, start, end)
            with open_covariance_reader(path, start, end) as reader:
                for chunk in reader.iter_owned_rows(
                    lower_min,
                    lower_max,
                    left,
                    right,
                    chunk_rows,
                    include_lower_min=include_lower_min,
                ):
                    masks = category_masks(chunk.lo, chunk.hi, boundary, left, right)
                    for category, mask in masks.items():
                        if not np.any(mask):
                            continue
                        values = r2_for_pairs(
                            chunk.lo[mask],
                            chunk.hi[mask],
                            chunk.shrink_ld[mask],
                            diag_pos,
                            diag_val,
                        )
                        if values.size == 0:
                            continue
                        per_boundary[category].extend(values.tolist())
                        seen[category] = reservoir_extend(
                            samples[category],
                            values,
                            seen=seen[category],
                            limit=sample_limit,
                            rng=rng,
                        )

        for category in CATEGORY_ORDER:
            row = {
                "chrom": chrom,
                "boundary": boundary,
                "category": category,
                "window_bp": window_bp,
            }
            row.update(summarize(per_boundary[category]))
            rows.append(row)

    return rows, samples


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "chrom",
        "boundary",
        "category",
        "window_bp",
        "n",
        "mean_r2",
        "median_r2",
        "q1_r2",
        "q3_r2",
        "p90_r2",
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


def write_boxplot(
    path: Path,
    samples: dict[str, list[float]],
    *,
    title: str,
    window_bp: int,
) -> None:
    configure_matplotlib_cache(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["left", "across", "right"]
    data = [samples[category] for category in CATEGORY_ORDER]
    fig, ax = plt.subplots(figsize=(3.4, 2.35), constrained_layout=True)
    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.34,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "0.35", "linewidth": 0.8},
        capprops={"color": "0.35", "linewidth": 0.8},
    )
    for patch, category in zip(box["boxes"], CATEGORY_ORDER, strict=True):
        patch.set_facecolor(COLORS[category])
        patch.set_alpha(0.65)
        patch.set_edgecolor("0.25")

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_ylabel("$r^2$", labelpad=2)
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8, pad=1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bed", required=True, type=Path)
    parser.add_argument("--covariance-root", required=True, type=Path)
    parser.add_argument("--chrom", required=True)
    parser.add_argument(
        "--covariance-name",
        default="",
        help=(
            "Covariance partition basename. Defaults to --chrom. Use this when "
            "the BED chromosome name and covariance cache name differ."
        ),
    )
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--plot", required=True, type=Path)
    parser.add_argument("--window-bp", type=int, default=500_000)
    parser.add_argument("--sample-limit", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    chrom = args.chrom if args.chrom.startswith("chr") else f"chr{args.chrom}"
    blocks = read_genome_bed(args.bed).get(chrom, [])
    if len(blocks) < 2:
        raise RuntimeError(f"{args.bed} has fewer than two blocks for {chrom}")

    rows, samples = boundary_rows_and_samples(
        chrom=chrom,
        boundaries=internal_boundaries(blocks),
        store=CovarianceStore(root=args.covariance_root),
        name=args.covariance_name or chrom,
        window_bp=args.window_bp,
        sample_limit=args.sample_limit,
        seed=args.seed,
        chunk_rows=args.chunk_rows,
    )
    write_summary(args.output_tsv, rows)

    title = args.title
    if not title:
        block_set = args.covariance_root.parent.name
        title = f"{block_set} {chrom}: LD neighborhood separation"
    write_boxplot(args.plot, samples, title=title, window_bp=args.window_bp)

    aggregate = {
        category: statistics.median(values) if values else float("nan")
        for category, values in samples.items()
    }
    print(
        "median r2: "
        + ", ".join(
            f"{category}={aggregate[category]:.4g}" for category in CATEGORY_ORDER
        )
    )
    print(f"Wrote {args.output_tsv} and {args.plot}")


if __name__ == "__main__":
    main()
