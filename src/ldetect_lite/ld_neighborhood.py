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

import os
from collections.abc import Iterator
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


def read_diagonal_index(
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


def r2_for_pairs(
    lo: np.ndarray,
    hi: np.ndarray,
    shrink_ld: np.ndarray,
    diag_pos: np.ndarray,
    diag_val: np.ndarray,
) -> np.ndarray:
    values, valid = r2_with_mask(lo, hi, shrink_ld, diag_pos, diag_val)
    return np.asarray(values[valid], dtype=np.float64)


def r2_with_mask(
    lo: np.ndarray,
    hi: np.ndarray,
    shrink_ld: np.ndarray,
    diag_pos: np.ndarray,
    diag_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """r^2 for each pair, aligned with the input, plus a validity mask.

    A pair is valid when both positions have a diagonal entry, the diagonal
    product is positive and the result is finite; invalid entries of the
    returned values are 0.0.
    """
    n = lo.size
    values = np.zeros(n, dtype=np.float64)
    if n == 0 or diag_pos.size == 0:
        return values, np.zeros(n, dtype=bool)

    lo_idx = np.minimum(np.searchsorted(diag_pos, lo), diag_pos.size - 1)
    hi_idx = np.minimum(np.searchsorted(diag_pos, hi), diag_pos.size - 1)
    valid = (diag_pos[lo_idx] == lo) & (diag_pos[hi_idx] == hi)
    denom = diag_val[lo_idx] * diag_val[hi_idx]
    valid &= denom > 0.0
    shrink = np.asarray(shrink_ld, dtype=np.float64)
    np.divide(shrink * shrink, denom, out=values, where=valid)
    valid &= np.isfinite(values)
    values[~valid] = 0.0
    return values, valid


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


class BottomKSampler:
    """Uniform random sample, without replacement, of at most *limit* values.

    Bottom-k sampling: every value seen gets an independent uniform key and
    the *limit* values with the smallest keys are kept. That is the same
    distribution as classic reservoir sampling, but each batch is handled in
    NumPy instead of a per-value Python loop. Once the sample is full, a value
    whose key is not below the current *limit*-th smallest key can never be
    kept, so it is dropped before buffering.
    """

    def __init__(self, limit: int, rng: np.random.Generator) -> None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        self.limit = limit
        self.seen = 0
        self._rng = rng
        self._keys: list[np.ndarray] = []
        self._values: list[np.ndarray] = []
        self._buffered = 0
        # Keys >= this can never make the final sample (inf until full).
        self._threshold = np.inf

    def extend(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        n = int(values.size)
        if n == 0:
            return
        self.seen += n
        if self.limit == 0:
            return
        keys = self._rng.random(n)
        if self._threshold < np.inf:
            keep = keys < self._threshold
            keys = keys[keep]
            values = values[keep]
            if keys.size == 0:
                return
        self._keys.append(keys)
        self._values.append(values)
        self._buffered += int(keys.size)
        if self._buffered > 2 * self.limit:
            self._compact()

    def _compact(self) -> None:
        if len(self._keys) > 1:
            keys = np.concatenate(self._keys)
            values = np.concatenate(self._values)
        elif self._keys:
            keys, values = self._keys[0], self._values[0]
        else:
            return
        if keys.size > self.limit:
            kept = np.argpartition(keys, self.limit - 1)[: self.limit]
            keys = keys[kept]
            values = values[kept]
        if keys.size >= self.limit:
            self._threshold = float(keys.max())
        self._keys = [keys]
        self._values = [values]
        self._buffered = int(keys.size)

    def sample(self) -> list[float]:
        """The kept values, in key order (a deterministic random order)."""
        self._compact()
        if not self._keys:
            return []
        order = np.argsort(self._keys[0], kind="stable")
        values: list[float] = self._values[0][order].tolist()
        return values


def window_row_bounds(
    lower_min: int,
    lower_max: int,
    include_lower_min: bool,
    left: int,
    right: int,
) -> tuple[int, int, bool]:
    """Clip a partition's owned ``lo`` range to the window ``[left, right]``.

    The window mask drops every row whose ``lo`` is outside ``[left, right]``,
    so reading only the clipped ``lo`` range yields the same rows while
    skipping the rest of the partition.
    """
    if left > lower_min:
        lower_min, include_lower_min = left, True
    return lower_min, min(lower_max, right), include_lower_min


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
    """Randomly sample r^2 values per category, aggregated across every
    boundary in *boundaries* (a chromosome's own computed breakpoints).

    Sampling bounds memory for chromosomes with many breakpoints and/or dense
    covariance data; each category's sample is a uniform random subset
    (capped at *sample_limit*) of every matching pair seen, deterministic for
    a given *seed*.
    """
    samples: SeparationSamples = {"left": [], "across": [], "right": []}
    if not boundaries:
        return samples

    partitions = read_partitions(name, store)
    diag_pos, diag_val = read_diagonal_index(name, store, partitions)
    if diag_pos.size == 0:
        raise RuntimeError(f"No diagonal rows found for {name} in {store.root}")

    rng = np.random.default_rng(seed)
    samplers = {
        category: BottomKSampler(sample_limit, rng) for category in CATEGORY_ORDER
    }

    for boundary in boundaries:
        for category, values in iter_boundary_r2(
            boundary=boundary,
            partitions=partitions,
            store=store,
            name=name,
            window_bp=window_bp,
            diag_pos=diag_pos,
            diag_val=diag_val,
            chunk_rows=chunk_rows,
        ):
            samplers[category].extend(values)
    return {category: samplers[category].sample() for category in CATEGORY_ORDER}


def iter_boundary_r2(
    *,
    boundary: int,
    partitions: list[tuple[int, int]],
    store: CovarianceStore,
    name: str,
    window_bp: int,
    diag_pos: np.ndarray,
    diag_val: np.ndarray,
    chunk_rows: int = 1_000_000,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(category, r2_values)`` batches for one boundary's window.

    Each pair is read from the one partition that owns it, and only rows
    whose ``lo`` falls inside ``[boundary - window_bp, boundary + window_bp]``
    are read from disk.
    """
    left = boundary - window_bp
    right = boundary + window_bp
    # Every position looked up below lies in [left, right]; searching only
    # that slice of the chromosome-wide diagonal keeps lookups cache-resident.
    d0 = int(np.searchsorted(diag_pos, left, side="left"))
    d1 = int(np.searchsorted(diag_pos, right, side="right"))
    win_pos = diag_pos[d0:d1]
    win_val = diag_val[d0:d1]

    for p_index, (start, end) in enumerate(partitions):
        if end < left or start > right:
            continue
        lower_min, lower_max, include_lower_min = owned_bounds(
            partitions, p_index, left, right
        )
        if lower_min > right or lower_max < left:
            continue
        lower_min, lower_max, include_lower_min = window_row_bounds(
            lower_min, lower_max, include_lower_min, left, right
        )
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
                # r^2 once per chunk, then split by category (the same
                # classification as category_masks).
                lo, hi = chunk.lo, chunk.hi
                pairs = (lo >= left) & (hi <= right) & (lo < hi)
                if not np.any(pairs):
                    continue
                lo, hi = lo[pairs], hi[pairs]
                values, valid = r2_with_mask(
                    lo, hi, chunk.shrink_ld[pairs], win_pos, win_val
                )
                hi_left = hi < boundary
                lo_right = lo >= boundary
                for category, mask in (
                    ("left", hi_left),
                    ("across", ~hi_left & ~lo_right),
                    ("right", lo_right),
                ):
                    selected = values[mask & valid]
                    if selected.size:
                        yield category, selected


def _configure_matplotlib_cache(path: Path) -> None:
    """Point matplotlib's font/config cache at a directory next to *path*.

    Avoids matplotlib trying to write its cache to a shared or read-only
    home directory, which can fail or warn noisily on cluster filesystems
    (e.g. under Slurm-managed shared-node scheduling).
    """
    cache_root = path.parent / ".matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


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
    _configure_matplotlib_cache(path)
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
