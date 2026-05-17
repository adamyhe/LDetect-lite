"""find_minima pipeline: binary search → filter → local search → JSON output."""

from __future__ import annotations

import gzip
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ldetect2._util.binary_search import find_ge_ind, find_le_ind
from ldetect2._util.covariance_array import (
    ChromosomeCovariance,
    load_covariance_arrays,  # noqa: F401 - kept for monkeypatch compatibility
    load_metric_covariance,
    metric_from_arrays,
)
from ldetect2._util.logging import log_msg
from ldetect2.filters import apply_filter, apply_filter_get_minima, get_minima_loc
from ldetect2.find_minima import custom_binary_search_with_trackback
from ldetect2.io.partitions import CovarianceStore, first_last, get_final_partitions
from ldetect2.local_search import LocalSearch
from ldetect2.metric import Metric

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def find_breakpoints(
    input_path: Path,
    chr_name: str,
    store: CovarianceStore,
    n_snps_bw_bpoints: int,
    output_path: Path,
    snp_first: int = -1,
    snp_last: int = -1,
    trackback_delta: int = 200,
    trackback_step: int = 20,
    init_search_location: int = 1000,
    workers: int = 1,
    use_decimal: bool = False,
    n_bpoints: int | None = None,
    covariance_cache: ChromosomeCovariance | None = None,
) -> None:
    """Run the full minima-detection pipeline and write breakpoints to a JSON file.

    Pipeline stages:
    1. Read the correlation-sum vector.
    2. Binary-search for the Hanning filter width that yields the target number
       of breakpoints.
    3. Apply the filter and extract minima positions.
    4. Compute the global LD metric for Fourier and uniform breakpoints.
    5. Run local search on both sets.
    6. Write results to *output_path* as JSON.

    Args:
        input_path: Gzipped vector file (position \\t corr_sum).
        chr_name: Chromosome name (e.g. ``"chr2"``).
        store: :class:`~ldetect2.io.partitions.CovarianceStore` pointing at the
            covariance matrix directory.
        n_snps_bw_bpoints: Target mean number of SNPs between breakpoints.
            Ignored when *n_bpoints* is provided.
        output_path: JSON output path.
        snp_first: Start position; auto-detected from partitions if ``-1``.
        snp_last: End position; auto-detected from partitions if ``-1``.
        trackback_delta: Coarse trackback search range.
        trackback_step: Coarse trackback step size.
        init_search_location: Starting width for exponential search.
        workers: Number of parallel workers for local search (default: 1).
        use_decimal: Use 50-digit Decimal arithmetic instead of float
            (default: False).
        n_bpoints: Direct target breakpoint count.  When provided,
            *n_snps_bw_bpoints* is ignored.
        covariance_cache: Optional in-memory chromosome covariance cache reused
            for normal float metrics.
    """
    snp_first, snp_last = first_last(chr_name, store, snp_first, snp_last)

    # 1. Read vector
    log_msg("Reading vector data")
    raw_vals, raw_x = _read_vector(input_path)

    begin_ind = find_ge_ind(raw_x, snp_first)
    end_ind = find_le_ind(raw_x, snp_last)

    np_array = np.array(raw_vals[begin_ind : end_ind + 1])
    np_array_x = np.array(raw_x[begin_ind : end_ind + 1])

    # 2. Target breakpoint count
    if n_bpoints is None:
        n_bpoints = int(math.ceil(len(np_array_x) / n_snps_bw_bpoints - 1))
    log_msg(f"Target breakpoints: {n_bpoints}")

    # 3. Binary search for filter width
    log_msg("Searching for filter width...")
    found_width = custom_binary_search_with_trackback(
        np_array,
        apply_filter_get_minima,
        n_bpoints,
        trackback_delta=trackback_delta,
        trackback_step=trackback_step,
        init_search_location=init_search_location,
    )
    log_msg(f"Found width: {found_width}")

    # 4. Extract minima positions
    log_msg("Applying filter and extracting minima")
    g = apply_filter(np_array, found_width)
    fourier_loci = get_minima_loc(g, np_array_x)

    metric_cov = None if use_decimal else covariance_cache
    if not use_decimal:
        if metric_cov is None:
            log_msg("Loading metric covariance arrays")
            metric_cov = load_metric_covariance(
                chr_name,
                store,
                get_final_partitions(store, chr_name, snp_first, snp_last),
                snp_first,
                snp_last,
            )
        else:
            log_msg("Using cached covariance arrays for metrics")

    # 5a. Metric for Fourier breakpoints
    log_msg("Computing Fourier metric")
    fourier_metric = _apply_metric(
        chr_name, snp_first, snp_last, store, fourier_loci, use_decimal, metric_cov
    )
    _log_metric(fourier_metric)

    # 5b. Uniform breakpoints + metric
    log_msg("Computing uniform breakpoints")
    step = int(len(raw_x) / (len(fourier_loci) + 1))
    uniform_loci = [raw_x[i] for i in range(step, len(raw_x) - step + 1, step)]
    uniform_metric = _apply_metric(
        chr_name, snp_first, snp_last, store, uniform_loci, use_decimal, metric_cov
    )
    _log_metric(uniform_metric)
    if covariance_cache is None:
        metric_cov = None

    # 6. Local search on Fourier
    log_msg("Running local search on Fourier breakpoints")
    fourier_ls = _run_local_search(
        chr_name,
        fourier_loci,
        snp_first,
        snp_last,
        store,
        fourier_metric,
        workers=workers,
        use_decimal=use_decimal,
        covariance_cache=covariance_cache,
    )
    # 7. Local search on uniform
    log_msg("Running local search on uniform breakpoints")
    uniform_ls = _run_local_search(
        chr_name,
        uniform_loci,
        snp_first,
        snp_last,
        store,
        uniform_metric,
        workers=workers,
        use_decimal=use_decimal,
        covariance_cache=covariance_cache,
    )
    if not use_decimal and metric_cov is None:
        log_msg("Reloading metric covariance arrays for final metric reuse")
        metric_cov = load_metric_covariance(
            chr_name,
            store,
            get_final_partitions(store, chr_name, snp_first, snp_last),
            snp_first,
            snp_last,
        )
    fourier_ls_metric = _apply_metric(
        chr_name,
        snp_first,
        snp_last,
        store,
        fourier_ls["loci"],
        use_decimal,
        metric_cov,
    )
    uniform_ls_metric = _apply_metric(
        chr_name,
        snp_first,
        snp_last,
        store,
        uniform_ls["loci"],
        use_decimal,
        metric_cov,
    )

    # 8. Serialise to JSON
    result = {
        "n_bpoints": n_bpoints,
        "found_width": found_width,
        "fourier": {
            "loci": fourier_loci,
            "metric": _metric_to_json(fourier_metric),
        },
        "fourier_ls": {
            "loci": fourier_ls["loci"],
            "metric": _metric_to_json(fourier_ls_metric),
        },
        "uniform": {
            "loci": uniform_loci,
            "metric": _metric_to_json(uniform_metric),
        },
        "uniform_ls": {
            "loci": uniform_ls["loci"],
            "metric": _metric_to_json(uniform_ls_metric),
        },
    }

    output_path.write_text(json.dumps(result, indent=2))
    log_msg(f"Breakpoints written to {output_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_vector(path: Path) -> tuple[list[float], list[int]]:
    """Read a gzipped or plain-text (position, value) TSV vector file."""
    import csv

    vals: list[float] = []
    positions: list[int] = []
    opener = (
        gzip.open(path, "rt") if path.suffix.lower() in (".gz", ".gzip") else open(path)
    )
    with opener as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            positions.append(int(row[0]))
            vals.append(float(row[1]))
    return vals, positions


def _apply_metric(
    chr_name: str,
    snp_first: int,
    snp_last: int,
    store: CovarianceStore,
    loci: list[int],
    use_decimal: bool = False,
    covariance_arrays=None,
) -> dict:
    if covariance_arrays is not None and not use_decimal:
        return metric_from_arrays(covariance_arrays, loci)
    m = Metric(chr_name, store, loci, snp_first, snp_last, use_decimal=use_decimal)
    return m.calc_metric()


def _log_metric(metric_out: dict) -> None:
    n_zero = metric_out["N_zero"]
    if n_zero > 0:
        log_msg(
            f"  sum={metric_out['sum']:.6f}  "
            f"N_zero={n_zero}  "
            f"metric={metric_out['sum'] / n_zero:.6e}"
        )


def _midpoint(a: int, b: int) -> int:
    lo, hi = (a, b) if a <= b else (b, a)
    return lo + (hi - lo) // 2


def _local_search_worker(
    chr_name: str,
    start: int,
    stop: int,
    idx: int,
    breakpoint_loci: list[int],
    total_sum,
    total_n,
    store: CovarianceStore,
    use_decimal: bool,
    covariance_cache: ChromosomeCovariance | None = None,
) -> tuple[int, dict | None]:
    """Run a single LocalSearch in a worker process (module-level for pickling)."""
    from ldetect2._util.logging import log_msg

    try:
        ls = LocalSearch(
            chr_name,
            start,
            stop,
            idx,
            breakpoint_loci,
            total_sum,
            total_n,
            store,
            use_decimal=use_decimal,
            covariance_cache=covariance_cache,
        )
        bp, m = ls.search()
        return (bp if bp is not None else breakpoint_loci[idx]), m
    except Exception as exc:
        log_msg(f"LocalSearch error at index {idx}: {exc}; keeping original")
        return breakpoint_loci[idx], None


def _run_local_search(
    chr_name: str,
    breakpoint_loci: list[int],
    snp_first: int,
    snp_last: int,
    store: CovarianceStore,
    metric_out: dict,
    workers: int = 1,
    use_decimal: bool = False,
    covariance_cache: ChromosomeCovariance | None = None,
) -> dict:
    total_sum = metric_out["sum"]
    total_n = metric_out["N_zero"]

    # Build (idx, start, stop) triples
    if len(breakpoint_loci) == 1:
        tasks = [(0, snp_first, snp_last)]
    else:
        tasks = []
        tasks.append((0, snp_first, _midpoint(breakpoint_loci[0], breakpoint_loci[1])))
        for idx in range(1, len(breakpoint_loci) - 1):
            b_start = _midpoint(breakpoint_loci[idx - 1], breakpoint_loci[idx])
            b_stop = _midpoint(breakpoint_loci[idx], breakpoint_loci[idx + 1])
            tasks.append((idx, b_start, b_stop))
        tasks.append(
            (
                len(breakpoint_loci) - 1,
                _midpoint(breakpoint_loci[-2], breakpoint_loci[-1]),
                snp_last,
            )
        )

    results: dict[int, tuple[int, dict | None]] = {}

    if covariance_cache is not None and not use_decimal:
        if workers > 1:
            log_msg(
                "Using cached in-memory array local search in a single process; "
                "ignoring local-search worker parallelism"
            )
        for idx, start, stop in tasks:
            results[idx] = _local_search_worker(
                chr_name,
                start,
                stop,
                idx,
                breakpoint_loci,
                total_sum,
                total_n,
                store,
                use_decimal,
                covariance_cache,
            )
    elif workers == 1:
        for idx, start, stop in tasks:
            results[idx] = _local_search_worker(
                chr_name,
                start,
                stop,
                idx,
                breakpoint_loci,
                total_sum,
                total_n,
                store,
                use_decimal,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _local_search_worker,
                    chr_name,
                    start,
                    stop,
                    idx,
                    breakpoint_loci,
                    total_sum,
                    total_n,
                    store,
                    use_decimal,
                ): idx
                for idx, start, stop in tasks
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()

    new_loci = [results[i][0] for i in range(len(breakpoint_loci))]
    new_metrics = [results[i][1] for i in range(len(breakpoint_loci))]

    return {"loci": new_loci, "metrics": new_metrics}


def _metric_to_json(metric_out: dict) -> dict:
    """Serialise a metric dict; Decimal values become strings to preserve precision."""
    return {
        "sum": str(metric_out["sum"]),
        "N_nonzero": int(metric_out["N_nonzero"]),
        "N_zero": str(metric_out["N_zero"]),
    }
