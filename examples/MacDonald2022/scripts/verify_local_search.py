"""Replay ldetect-lite's LocalSearch for one breakpoint and inspect the full metric
curve across its search window, not just the reported optimum.

Motivation: for several "Category B" MacDonald2022 pyrho boundary mismatches
(see notes/logs/macdonald2022-pyrho-handoff.md), the raw Hanning-filter stage finds
a candidate very close to the published reference boundary, but local search
then refines it away to a worse position. This script reproduces the exact
LocalSearch computation (using the real total_sum/total_n stored in the
pipeline's breakpoints-*.json) and prints the metric at every candidate locus
in the search window, so we can see whether the reported optimum is genuinely
the best point by the algorithm's own sum(r^2)/N_zero criterion, or whether a
point closer to the reference boundary would have scored better (which would
indicate a bug rather than a genuine data difference from legacy).

Usage:
    uv run python scripts/verify_local_search.py \
        --results-dir results/pyrho_EAS/chr4 \
        --chrom chr4 \
        --breakpoint-index 39 \
        --highlight 84513834 88318340

--breakpoint-index is the index into breakpoints-<chrom>.json's "fourier"
loci list (the *raw*, pre-local-search candidate whose refinement you want to
inspect). --highlight marks extra positions of interest (e.g. the published
reference boundaries) in the printed curve, even if they aren't the reported
optimum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ldetect_lite.io.partitions import CovarianceStore
from ldetect_lite.local_search import LocalSearch


def _midpoint(a: int, b: int) -> int:
    lo, hi = (a, b) if a <= b else (b, a)
    return lo + (hi - lo) // 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--breakpoint-index", required=True, type=int)
    parser.add_argument(
        "--highlight",
        nargs="*",
        type=int,
        default=[],
        help="Extra genomic positions to report the metric at (e.g. reference "
        "boundaries).",
    )
    args = parser.parse_args()

    chrom = args.chrom
    idx = args.breakpoint_index
    bp_path = args.results_dir / f"breakpoints-{chrom}.json"
    data = json.loads(bp_path.read_text())

    fourier_loci = data["fourier"]["loci"]
    fourier_ls_loci = data["fourier_ls"]["loci"]
    total_sum = float(data["fourier"]["metric"]["sum"])
    total_n = float(data["fourier"]["metric"]["N_zero"])

    if not (0 <= idx < len(fourier_loci)):
        raise SystemExit(
            f"--breakpoint-index {idx} out of range [0, {len(fourier_loci)})"
        )

    start_search = (
        fourier_loci[0]
        if idx == 0
        else _midpoint(fourier_loci[idx - 1], fourier_loci[idx])
    )
    stop_search = (
        fourier_loci[-1]
        if idx == len(fourier_loci) - 1
        else _midpoint(fourier_loci[idx], fourier_loci[idx + 1])
    )

    print(f"raw fourier candidate:      {fourier_loci[idx]}")
    print(f"reported fourier_ls result: {fourier_ls_loci[idx]}")
    print(f"search window:              [{start_search}, {stop_search}]")
    print(f"total_sum={total_sum}  total_n={total_n}")
    print()

    store = CovarianceStore(root=args.results_dir)
    ls = LocalSearch(
        chrom,
        start_search,
        stop_search,
        idx,
        fourier_loci,
        total_sum,
        total_n,
        store,
        use_decimal=False,
    )
    ls.init_search()
    best_locus, best_metric = ls.search()
    print(f"LocalSearch.search() result: {best_locus}")
    matches = best_locus == fourier_ls_loci[idx]
    print(f"  matches JSON's fourier_ls loci[{idx}]? {matches}")
    if best_metric is not None:
        print(f"  metric at result: {best_metric['sum'] / best_metric['N_zero']:.10g}")
    print()

    loci = ls._array_loci
    sum_vert = ls._array_sum_vert
    sum_horiz = ls._array_sum_horiz
    if loci is None or sum_vert is None or sum_horiz is None:
        raise SystemExit("LocalSearch did not populate array-search precomputed data")

    snp_bottom_ind = int(np.searchsorted(loci, ls.snp_bottom, side="left"))
    snp_top_ind = int(np.searchsorted(loci, ls.snp_top, side="right") - 1)
    bp_ind = int(np.searchsorted(loci, fourier_loci[idx], side="right") - 1)

    # Full metric curve across the window, both directions from bp_ind,
    # mirroring _search_array's cumulative-sum construction exactly. Also
    # track N_zero (= n_left * n_right for a single breakpoint) at every
    # point, to check whether divergent candidates sit in a numerically
    # sparse (small-denominator) pocket relative to the rest of the window.
    curve_loci = []
    curve_metric = []
    curve_n = []

    right_stop = int(np.searchsorted(loci, ls.snp_last, side="right"))
    if bp_ind + 1 < right_stop:
        right_idx = np.arange(bp_ind + 1, right_stop, dtype=np.int64)
        sums = total_sum + np.cumsum(-sum_horiz[right_idx] + sum_vert[right_idx])
        ns = total_n + np.cumsum(
            -(right_idx - snp_bottom_ind - 1) + (snp_top_ind - right_idx)
        )
        valid = ns > 0
        curve_loci.extend(loci[right_idx[valid]].tolist())
        curve_metric.extend((sums[valid] / ns[valid]).tolist())
        curve_n.extend(ns[valid].tolist())

    left_start = int(np.searchsorted(loci, ls.snp_first, side="right"))
    if left_start < bp_ind:
        left_idx = np.arange(bp_ind - 1, left_start - 1, -1, dtype=np.int64)
        sums = total_sum + np.cumsum(sum_horiz[left_idx] - sum_vert[left_idx])
        ns = total_n + np.cumsum(
            (left_idx - snp_bottom_ind - 1) - (snp_top_ind - left_idx)
        )
        valid = ns > 0
        curve_loci.extend(loci[left_idx[valid]].tolist())
        curve_metric.extend((sums[valid] / ns[valid]).tolist())
        curve_n.extend(ns[valid].tolist())

    order = np.argsort(curve_loci)
    curve_loci = np.array(curve_loci)[order]
    curve_metric = np.array(curve_metric)[order]
    curve_n = np.array(curve_n)[order]

    best_curve_i = int(np.argmin(curve_metric))
    median_n = float(np.median(curve_n))
    print(
        f"Global minimum over full search window: locus={curve_loci[best_curve_i]} "
        f"metric={curve_metric[best_curve_i]:.10g} N_zero={curve_n[best_curve_i]:.6g}"
    )
    print(
        f"N_zero across window: min={curve_n.min():.6g} median={median_n:.6g} "
        f"max={curve_n.max():.6g}"
    )

    for h in args.highlight + [fourier_ls_loci[idx]]:
        i = int(np.searchsorted(curve_loci, h))
        i = max(0, min(i, len(curve_loci) - 1))
        nearest_locus = curve_loci[i]
        dist = abs(nearest_locus - h)
        metric = curve_metric[i]
        n_zero = curve_n[i]
        n_ratio = n_zero / median_n if median_n else float("nan")
        print(
            f"  near highlighted position {h}: nearest evaluated locus="
            f"{nearest_locus} (dist={dist}) metric={metric:.10g} "
            f"N_zero={n_zero:.6g} (N_zero/median={n_ratio:.4g})"
        )


if __name__ == "__main__":
    main()
