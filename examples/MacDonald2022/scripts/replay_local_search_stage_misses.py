#!/usr/bin/env python
"""Replay local search for stage-diagnostic boundaries moved away by refinement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ldetect_lite.io.partitions import CovarianceStore
from ldetect_lite.local_search import LocalSearch


def midpoint(a: int, b: int) -> int:
    lo, hi = (a, b) if a <= b else (b, a)
    return lo + (hi - lo) // 2


def nearest_index(values: np.ndarray, target: int) -> int:
    if values.size == 0:
        raise ValueError("cannot find nearest index in an empty array")
    index = int(np.searchsorted(values, target))
    candidates = []
    if index < values.size:
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    return min(candidates, key=lambda idx: (abs(int(values[idx]) - target), idx))


def metric_ratio_pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return (numerator / denominator - 1.0) * 100.0


def local_search_curve(
    ls: LocalSearch,
    fourier_loci: list[int],
    breakpoint_index: int,
    total_sum: float,
    total_n: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loci = ls._array_loci
    sum_vert = ls._array_sum_vert
    sum_horiz = ls._array_sum_horiz
    if loci is None or sum_vert is None or sum_horiz is None:
        raise RuntimeError("LocalSearch did not populate array-search data")

    snp_bottom_ind = int(np.searchsorted(loci, ls.snp_bottom, side="left"))
    snp_top_ind = int(np.searchsorted(loci, ls.snp_top, side="right") - 1)
    bp_ind = int(
        np.searchsorted(loci, fourier_loci[breakpoint_index], side="right") - 1
    )

    curve_loci: list[int] = []
    curve_metric: list[float] = []
    curve_n: list[float] = []

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
    return (
        np.asarray(curve_loci, dtype=np.int64)[order],
        np.asarray(curve_metric, dtype=np.float64)[order],
        np.asarray(curve_n, dtype=np.float64)[order],
    )


def replay_row(
    row: dict[str, str],
    results_dir: Path,
    chrom: str,
    breakpoints: dict,
) -> dict[str, object]:
    reference_boundary = int(row["reference_boundary"])
    nearest_fourier = int(row["nearest_fourier"])
    fourier_loci = [int(x) for x in breakpoints["fourier"]["loci"]]
    fourier_ls_loci = [int(x) for x in breakpoints["fourier_ls"]["loci"]]
    total_sum = float(breakpoints["fourier"]["metric"]["sum"])
    total_n = float(breakpoints["fourier"]["metric"]["N_zero"])

    try:
        breakpoint_index = fourier_loci.index(nearest_fourier)
    except ValueError as exc:
        raise RuntimeError(
            f"nearest Fourier locus {nearest_fourier} is not in {results_dir}"
        ) from exc

    start_search = (
        fourier_loci[0]
        if breakpoint_index == 0
        else midpoint(
            fourier_loci[breakpoint_index - 1],
            fourier_loci[breakpoint_index],
        )
    )
    stop_search = (
        fourier_loci[-1]
        if breakpoint_index == len(fourier_loci) - 1
        else midpoint(
            fourier_loci[breakpoint_index],
            fourier_loci[breakpoint_index + 1],
        )
    )

    store = CovarianceStore(root=results_dir)
    ls = LocalSearch(
        chrom,
        start_search,
        stop_search,
        breakpoint_index,
        fourier_loci,
        total_sum,
        total_n,
        store,
        use_decimal=False,
    )
    ls.init_search()
    search_locus, search_metric = ls.search()
    if search_locus is None or search_metric is None:
        raise RuntimeError(f"LocalSearch returned no locus for {results_dir}")

    curve_loci, curve_metric, curve_n = local_search_curve(
        ls,
        fourier_loci,
        breakpoint_index,
        total_sum,
        total_n,
    )
    if curve_loci.size == 0:
        raise RuntimeError(f"empty metric curve for {results_dir}")

    best_index = int(np.argmin(curve_metric))
    ref_index = nearest_index(curve_loci, reference_boundary)
    reported_ls_index = nearest_index(curve_loci, fourier_ls_loci[breakpoint_index])
    search_index = nearest_index(curve_loci, int(search_locus))

    best_metric = float(curve_metric[best_index])
    ref_metric = float(curve_metric[ref_index])
    reported_ls_metric = float(curve_metric[reported_ls_index])
    search_metric_value = float(curve_metric[search_index])

    return {
        **row,
        "breakpoint_index": breakpoint_index,
        "raw_fourier_locus": fourier_loci[breakpoint_index],
        "reported_fourier_ls_locus": fourier_ls_loci[breakpoint_index],
        "search_result_locus": int(search_locus),
        "search_matches_reported": int(
            search_locus == fourier_ls_loci[breakpoint_index]
        ),
        "search_start": start_search,
        "search_stop": stop_search,
        "reference_in_search_window": int(
            start_search <= reference_boundary <= stop_search
        ),
        "curve_best_locus": int(curve_loci[best_index]),
        "curve_best_metric": repr(best_metric),
        "curve_best_n_zero": repr(float(curve_n[best_index])),
        "curve_best_matches_reported": int(
            int(curve_loci[best_index]) == fourier_ls_loci[breakpoint_index]
        ),
        "reference_nearest_evaluated_locus": int(curve_loci[ref_index]),
        "reference_nearest_evaluated_distance_bp": abs(
            int(curve_loci[ref_index]) - reference_boundary
        ),
        "reference_metric": repr(ref_metric),
        "reference_n_zero": repr(float(curve_n[ref_index])),
        "reference_metric_vs_best_pct": repr(metric_ratio_pct(ref_metric, best_metric)),
        "reference_metric_vs_reported_ls_pct": repr(
            metric_ratio_pct(ref_metric, reported_ls_metric)
        ),
        "reported_ls_nearest_evaluated_locus": int(curve_loci[reported_ls_index]),
        "reported_ls_nearest_evaluated_distance_bp": abs(
            int(curve_loci[reported_ls_index]) - fourier_ls_loci[breakpoint_index]
        ),
        "reported_ls_metric": repr(reported_ls_metric),
        "reported_ls_n_zero": repr(float(curve_n[reported_ls_index])),
        "search_result_metric": repr(search_metric_value),
    }


FIELDNAMES = [
    "chrom",
    "reference_boundary",
    "nearest_fourier",
    "fourier_signed_offset_bp",
    "fourier_abs_offset_bp",
    "fourier_within_tolerance",
    "nearest_fourier_ls",
    "fourier_ls_signed_offset_bp",
    "fourier_ls_abs_offset_bp",
    "fourier_ls_within_tolerance",
    "local_search_delta_abs_bp",
    "stage_class",
    "breakpoint_index",
    "raw_fourier_locus",
    "reported_fourier_ls_locus",
    "search_result_locus",
    "search_matches_reported",
    "search_start",
    "search_stop",
    "reference_in_search_window",
    "curve_best_locus",
    "curve_best_metric",
    "curve_best_n_zero",
    "curve_best_matches_reported",
    "reference_nearest_evaluated_locus",
    "reference_nearest_evaluated_distance_bp",
    "reference_metric",
    "reference_n_zero",
    "reference_metric_vs_best_pct",
    "reference_metric_vs_reported_ls_pct",
    "reported_ls_nearest_evaluated_locus",
    "reported_ls_nearest_evaluated_distance_bp",
    "reported_ls_metric",
    "reported_ls_n_zero",
    "search_result_metric",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-diagnostics", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--stage-class",
        default="local_search_moved_away",
        help="Only replay rows with this stage_class.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Replay the largest N selected rows by fourier_ls_abs_offset_bp; "
        "0 means all selected rows.",
    )
    args = parser.parse_args()

    with args.stage_diagnostics.open() as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["stage_class"] == args.stage_class
        ]
    rows.sort(key=lambda row: int(row["fourier_ls_abs_offset_bp"]), reverse=True)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    bp_path = args.results_dir / f"breakpoints-{args.chrom}.json"
    breakpoints = json.loads(bp_path.read_text())
    out_rows = [
        replay_row(row, args.results_dir, args.chrom, breakpoints) for row in rows
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
