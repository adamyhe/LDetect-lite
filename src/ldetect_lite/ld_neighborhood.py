"""LD-neighborhood separation diagnostic for computed breakpoints.

For each computed breakpoint, samples pairwise LD (r^2) from the covariance
cache into three local neighborhoods relative to that breakpoint:

    left    pairs wholly in [boundary - window, boundary)
    across  pairs crossing the boundary, one SNP on each side
    right   pairs wholly in [boundary, boundary + window]

The resulting box/whisker plot (aggregated across every breakpoint in the
chromosome) is an orthogonal sanity check to exact boundary matching against
a reference: useful boundaries should tend to make the across-boundary LD
distribution lower than the within-neighborhood (left/right) distributions.
Needs no reference -- only the covariance cache and breakpoints from a single
run.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ldetect_lite.io.covariance_hdf5 import open_covariance_reader
from ldetect_lite.io.partitions import CovarianceStore, read_partitions

CATEGORY_ORDER = ("left", "across", "right")
CATEGORY_COLORS = {
    "left": "#4c78a8",
    "across": "#e45756",
    "right": "#54a24b",
}

# Reservoir-sampled r^2 values keyed by category ("left"/"across"/"right").
SeparationSamples = dict[str, list[float]]


def category_masks(
    lo: np.ndarray, hi: np.ndarray, boundary: int, left: int, right: int
) -> dict[str, np.ndarray]:
    """Classify pairs ``(lo, hi)`` within ``[left, right]`` around *boundary*."""
    in_window = (lo >= left) & (hi <= right) & (lo < hi)
    return {
        "left": in_window & (hi < boundary),
        "across": in_window & (lo < boundary) & (hi >= boundary),
        "right": in_window & (lo >= boundary),
    }


def _read_diagonal_index(
    name: str, store: CovarianceStore, partitions: list[tuple[int, int]]
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


def _r2_for_pairs(
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
    return np.asarray(values[np.isfinite(values)], dtype=np.float64)


def _owned_bounds(
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


def _reservoir_extend(
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


def chromosome_separation_samples(
    *,
    boundaries: list[int],
    store: CovarianceStore,
    name: str,
    window_bp: int = 500_000,
    sample_limit: int = 200_000,
    seed: int = 1,
    chunk_rows: int = 1_000_000,
) -> SeparationSamples:
    """Reservoir-sample r^2 values per category, aggregated across every
    boundary in *boundaries* (a chromosome's own computed breakpoints).

    Reservoir sampling bounds memory for chromosomes with many breakpoints
    and/or dense covariance data; each category's sample is an unbiased
    random subset (capped at *sample_limit*) of every matching pair seen.
    """
    samples: SeparationSamples = {"left": [], "across": [], "right": []}
    if not boundaries:
        return samples

    partitions = read_partitions(name, store)
    diag_pos, diag_val = _read_diagonal_index(name, store, partitions)
    if diag_pos.size == 0:
        raise RuntimeError(f"No diagonal rows found for {name} in {store.root}")

    seen = {category: 0 for category in CATEGORY_ORDER}
    rng = random.Random(seed)

    for boundary in boundaries:
        left = boundary - window_bp
        right = boundary + window_bp

        for p_index, (start, end) in enumerate(partitions):
            if end < left or start > right:
                continue
            lower_min, lower_max, include_lower_min = _owned_bounds(
                partitions, p_index, left, right
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
                        values = _r2_for_pairs(
                            chunk.lo[mask],
                            chunk.hi[mask],
                            chunk.shrink_ld[mask],
                            diag_pos,
                            diag_val,
                        )
                        if values.size:
                            seen[category] = _reservoir_extend(
                                samples[category],
                                values.tolist(),
                                seen=seen[category],
                                limit=sample_limit,
                                rng=rng,
                            )
    return samples


def write_separation_boxplot(
    path: Path,
    samples: SeparationSamples,
    *,
    title: str = "LD neighborhood separation",
) -> None:
    """Write a left/across/right box-and-whisker plot of sampled r^2 values.

    A sanity check for computed breakpoints -- a useful set of breakpoints
    should show lower "across" LD than "left"/"right" neighborhood LD. Lazily
    imports matplotlib so invocations that don't request this plot stay fast.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    if not any(samples[category] for category in CATEGORY_ORDER):
        fig, ax = plt.subplots(figsize=(2.25, 1.9))
        ax.text(
            0.5,
            0.52,
            "no internal\nboundaries",
            ha="center",
            va="center",
            fontsize=8,
            color="0.35",
            transform=ax.transAxes,
        )
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.85")
        fig.subplots_adjust(left=0.04, right=0.995, bottom=0.04, top=0.88)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.005)
        plt.close(fig)
        return

    labels = list(CATEGORY_ORDER)
    data = [samples[category] for category in CATEGORY_ORDER]
    positions = [1.0, 1.26, 1.52]
    fig, ax = plt.subplots(figsize=(2.25, 1.9))
    box = ax.boxplot(
        data,
        positions=positions,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.15,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "0.35", "linewidth": 0.8},
        capprops={"color": "0.35", "linewidth": 0.8},
    )
    for patch, category in zip(box["boxes"], CATEGORY_ORDER, strict=True):
        patch.set_facecolor(CATEGORY_COLORS[category])
        patch.set_alpha(0.65)
        patch.set_edgecolor("0.25")

    ax.set_title(title, fontsize=8, pad=2)
    ax.set_ylabel("$r^2$", labelpad=0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(0.89, 1.63)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=7, pad=0)
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.16, top=0.88)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.005)
    plt.close(fig)
