"""find_minima pipeline: binary search → filter → local search → JSON output."""

from __future__ import annotations

import gzip
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ldetect2._util.binary_search import find_ge_ind, find_le_ind
from ldetect2._util.covariance_array import (
    ChromosomeCovariance,
    load_covariance_arrays,  # noqa: F401 - kept for monkeypatch compatibility
    metric_from_arrays,
)
from ldetect2._util.logging import log_msg
from ldetect2._util.memory import log_memory_checkpoint, max_rss_mib
from ldetect2.filters import apply_filter, apply_filter_get_minima, get_minima_loc
from ldetect2.find_minima import custom_binary_search_with_trackback
from ldetect2.io.partitions import CovarianceStore, first_last, get_final_partitions
from ldetect2.io.r2_nocache import R2NoCacheConfig
from ldetect2.local_search import (
    LocalSearch,
    local_search_hdf5_partition,
    local_search_r2_nocache_partition,
    local_search_r2_zarr_partition,
)
from ldetect2.metric import Metric

_VALID_SUBSETS = frozenset({"fourier", "fourier_ls", "uniform", "uniform_ls"})

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
    metric_workers: int = 1,
    use_decimal: bool = False,
    n_bpoints: int | None = None,
    covariance_cache: ChromosomeCovariance | None = None,
    subsets: set[str] | None = None,
    pair_cache: str = "hdf5",
    r2_nocache_config: R2NoCacheConfig | None = None,
) -> None:
    """Run minima detection and write selected breakpoint subsets to JSON.

    Pipeline stages:
    1. Read the correlation-sum vector.
    2. Binary-search for the Hanning filter width that yields the target number
       of breakpoints.
    3. Apply the filter and extract minima positions.
    4. Compute the requested raw Fourier/uniform metrics.
    5. Run requested local-search refinements.
    6. Write computed subsets and explicit skip metadata to *output_path*.

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
        metric_workers: Number of parallel workers for streaming metric row
            passes when not using Decimal arithmetic or an in-memory covariance
            cache (default: 1).
        use_decimal: Use 50-digit Decimal arithmetic instead of float
            (default: False).
        n_bpoints: Direct target breakpoint count.  When provided,
            *n_snps_bw_bpoints* is ignored.
        covariance_cache: Optional in-memory chromosome covariance cache reused
            for normal float metrics.
        subsets: Optional breakpoint subsets to compute and write.  ``None``
            preserves the historical behavior and writes all four subsets.
    """
    if pair_cache != "hdf5" and use_decimal:
        raise ValueError(
            "Experimental r2 pair caches do not support Decimal mode"
        )
    if pair_cache == "r2-nocache" and r2_nocache_config is None:
        raise ValueError("r2-nocache requires an R2NoCacheConfig")

    requested_subsets, explicit_subsets = _normalise_subsets(subsets)
    needs_fourier_metric = bool(requested_subsets & {"fourier", "fourier_ls"})
    needs_uniform = bool(requested_subsets & {"uniform", "uniform_ls"})
    needs_fourier_ls = "fourier_ls" in requested_subsets
    needs_uniform_ls = "uniform_ls" in requested_subsets

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
    if metric_cov is not None:
        log_msg("Using cached covariance arrays for metrics")

    fourier_metric = None
    if needs_fourier_metric:
        log_msg("Computing Fourier metric")
        log_memory_checkpoint("fourier_metric_start")
        fourier_metric = _apply_metric(
            chr_name,
            snp_first,
            snp_last,
            store,
            fourier_loci,
            use_decimal,
            metric_cov,
            metric_workers,
            pair_cache,
            r2_nocache_config,
        )
        _log_metric(fourier_metric)
        log_memory_checkpoint("fourier_metric_end")

    uniform_loci = None
    uniform_metric = None
    if needs_uniform:
        log_msg("Computing uniform breakpoints")
        step = int(len(raw_x) / (len(fourier_loci) + 1))
        uniform_loci = [raw_x[i] for i in range(step, len(raw_x) - step + 1, step)]
        log_msg("Computing uniform metric")
        log_memory_checkpoint("uniform_metric_start")
        uniform_metric = _apply_metric(
            chr_name,
            snp_first,
            snp_last,
            store,
            uniform_loci,
            use_decimal,
            metric_cov,
            metric_workers,
            pair_cache,
            r2_nocache_config,
        )
        _log_metric(uniform_metric)
        log_memory_checkpoint("uniform_metric_end")
    if covariance_cache is None:
        metric_cov = None

    # 6. Local search on Fourier
    fourier_ls = None
    if needs_fourier_ls:
        if fourier_metric is None:
            raise RuntimeError("Fourier local search requires the Fourier metric")
        log_msg("Running local search on Fourier breakpoints")
        log_memory_checkpoint("fourier_local_search_start")
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
            subset_name="fourier_ls",
            pair_cache=pair_cache,
            r2_nocache_config=r2_nocache_config,
        )
        log_memory_checkpoint("fourier_local_search_end")
    # 7. Local search on uniform
    uniform_ls = None
    if needs_uniform_ls:
        if uniform_loci is None or uniform_metric is None:
            raise RuntimeError("Uniform local search requires uniform breakpoints")
        log_msg("Running local search on uniform breakpoints")
        log_memory_checkpoint("uniform_local_search_start")
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
            subset_name="uniform_ls",
            pair_cache=pair_cache,
            r2_nocache_config=r2_nocache_config,
        )
        log_memory_checkpoint("uniform_local_search_end")
    fourier_ls_metric = None
    if fourier_ls is not None:
        log_memory_checkpoint("fourier_ls_metric_start")
        fourier_ls_metric = _apply_metric(
            chr_name,
            snp_first,
            snp_last,
            store,
            fourier_ls["loci"],
            use_decimal,
            metric_cov,
            metric_workers,
            pair_cache,
            r2_nocache_config,
        )
        log_memory_checkpoint("fourier_ls_metric_end")
    uniform_ls_metric = None
    if uniform_ls is not None:
        log_memory_checkpoint("uniform_ls_metric_start")
        uniform_ls_metric = _apply_metric(
            chr_name,
            snp_first,
            snp_last,
            store,
            uniform_ls["loci"],
            use_decimal,
            metric_cov,
            metric_workers,
            pair_cache,
            r2_nocache_config,
        )
        log_memory_checkpoint("uniform_ls_metric_end")

    # 8. Serialise to JSON
    result = {
        "n_bpoints": n_bpoints,
        "found_width": found_width,
        "computed_subsets": sorted(requested_subsets),
        "skipped_subsets": sorted(_VALID_SUBSETS - requested_subsets)
        if explicit_subsets
        else [],
    }
    if "fourier" in requested_subsets:
        if fourier_metric is None:
            raise RuntimeError("Fourier output requires the Fourier metric")
        result["fourier"] = {
            "loci": fourier_loci,
            "metric": _metric_to_json(fourier_metric),
        }
    if "fourier_ls" in requested_subsets:
        if fourier_ls is None or fourier_ls_metric is None:
            raise RuntimeError("Fourier local-search output was not computed")
        result["fourier_ls"] = {
            "loci": fourier_ls["loci"],
            "metric": _metric_to_json(fourier_ls_metric),
        }
    if "uniform" in requested_subsets:
        if uniform_loci is None or uniform_metric is None:
            raise RuntimeError("Uniform output requires the uniform metric")
        result["uniform"] = {
            "loci": uniform_loci,
            "metric": _metric_to_json(uniform_metric),
        }
    if "uniform_ls" in requested_subsets:
        if uniform_ls is None or uniform_ls_metric is None:
            raise RuntimeError("Uniform local-search output was not computed")
        result["uniform_ls"] = {
            "loci": uniform_ls["loci"],
            "metric": _metric_to_json(uniform_ls_metric),
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


def _normalise_subsets(subsets: set[str] | None) -> tuple[set[str], bool]:
    """Validate requested subsets and add dependencies for local-search outputs.

    Returns the expanded subset set plus a flag indicating whether the caller
    explicitly requested a subset selection.  ``None`` means historical full
    output and is not treated as an explicit selection.
    """
    if subsets is None:
        return set(_VALID_SUBSETS), False
    invalid = set(subsets) - _VALID_SUBSETS
    if invalid:
        raise ValueError(f"Invalid breakpoint subset(s): {', '.join(sorted(invalid))}")
    if not subsets:
        raise ValueError("At least one breakpoint subset must be requested")
    requested = set(subsets)
    if "fourier_ls" in requested:
        requested.add("fourier")
    if "uniform_ls" in requested:
        requested.add("uniform")
    return requested, True


def _apply_metric(
    chr_name: str,
    snp_first: int,
    snp_last: int,
    store: CovarianceStore,
    loci: list[int],
    use_decimal: bool = False,
    covariance_arrays=None,
    metric_workers: int = 1,
    pair_cache: str = "hdf5",
    r2_nocache_config: R2NoCacheConfig | None = None,
) -> dict:
    if covariance_arrays is not None and not use_decimal and pair_cache == "hdf5":
        return metric_from_arrays(covariance_arrays, loci)
    m = Metric(
        chr_name,
        store,
        loci,
        snp_first,
        snp_last,
        use_decimal=use_decimal,
        workers=metric_workers,
        pair_cache=pair_cache,
        r2_nocache_config=r2_nocache_config,
    )
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
    local_search_partitions=None,
    local_search_hdf5_partitions=None,
    local_search_r2_zarr_partitions=None,
    local_search_r2_nocache_partitions=None,
    subset_name: str = "local_search",
) -> tuple[int, dict | None]:
    """Run one breakpoint refinement and emit debug timing/memory diagnostics.

    This function is module-level so it can be submitted to
    :class:`ProcessPoolExecutor`.  It keeps the previous fail-soft behavior:
    errors are logged and the original breakpoint is returned unchanged.
    """
    from ldetect2._util.logging import log_debug, log_msg

    init_seconds = 0.0
    search_seconds = 0.0
    try:
        start_time = time.perf_counter()
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
            local_search_partitions=local_search_partitions,
            local_search_hdf5_partitions=local_search_hdf5_partitions,
            local_search_r2_zarr_partitions=local_search_r2_zarr_partitions,
            local_search_r2_nocache_partitions=local_search_r2_nocache_partitions,
        )
        init_start = time.perf_counter()
        ls.init_search()
        init_seconds = time.perf_counter() - init_start
        search_start = time.perf_counter()
        bp, m = ls.search()
        search_seconds = time.perf_counter() - search_start
        total_seconds = time.perf_counter() - start_time
        row_count = getattr(ls, "loaded_row_count", None)
        partition_count = getattr(ls, "loaded_partition_count", len(ls.partitions))
        precompute_stats = getattr(ls, "precompute_stats", None)
        precompute_stats_text = (
            f" {precompute_stats.log_fields()}" if precompute_stats is not None else ""
        )
        rss = max_rss_mib()
        rss_text = f" max_rss_mib={rss:.1f}" if rss is not None else ""
        log_debug(
            f"{subset_name} breakpoint idx={idx} start={start} stop={stop} "
            f"partitions={partition_count} rows={row_count} "
            f"precompute_seconds={init_seconds:.3f} "
            f"search_seconds={search_seconds:.3f} "
            f"total_seconds={total_seconds:.3f}{rss_text}{precompute_stats_text}"
        )
        return (bp if bp is not None else breakpoint_loci[idx]), m
    except Exception as exc:
        log_msg(
            f"LocalSearch error at index {idx} after "
            f"precompute_seconds={init_seconds:.3f} "
            f"search_seconds={search_seconds:.3f}: {exc}; keeping original"
        )
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
    subset_name: str = "local_search",
    pair_cache: str = "hdf5",
    r2_nocache_config: R2NoCacheConfig | None = None,
) -> dict:
    """Refine all breakpoints for one subset and log aggregate elapsed time.

    Each breakpoint search is independent.  The function runs them sequentially
    or through a process pool depending on *workers*, except that an in-memory
    covariance cache is intentionally kept single-process to avoid copying a
    large cache into worker processes.
    """
    run_start = time.perf_counter()
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

    if covariance_cache is not None and not use_decimal and pair_cache == "hdf5":
        if workers > 1:
            log_msg(
                "Using cached in-memory array local search in a single process; "
                "ignoring local-search worker parallelism"
            )
        for idx, start, stop in tasks:
            if idx == 0:
                log_memory_checkpoint(f"{subset_name}_breakpoint0_start")
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
                covariance_cache=covariance_cache,
                subset_name=subset_name,
            )
            if idx == 0:
                log_memory_checkpoint(f"{subset_name}_breakpoint0_end")
    elif workers == 1:
        if use_decimal:
            if pair_cache != "hdf5":
                raise ValueError("Decimal local search requires the HDF5 pair cache")
            for idx, start, stop in tasks:
                if idx == 0:
                    log_memory_checkpoint(f"{subset_name}_breakpoint0_start")
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
                    subset_name=subset_name,
                )
                if idx == 0:
                    log_memory_checkpoint(f"{subset_name}_breakpoint0_end")
        else:
            grouped_tasks = _group_local_search_tasks(
                chr_name, tasks, breakpoint_loci, store
            )
            for partition_bounds, group in grouped_tasks:
                group_load_start = time.perf_counter()
                if pair_cache == "r2-zarr":
                    group_hdf5_partitions = None
                    group_r2_nocache_partitions = None
                    group_r2_zarr_partitions = tuple(
                        local_search_r2_zarr_partition(chr_name, store, start, end)
                        for start, end in partition_bounds
                    )
                    group_row_count = sum(
                        partition.source_row_count
                        for partition in group_r2_zarr_partitions
                    )
                elif pair_cache == "r2-nocache":
                    if r2_nocache_config is None:
                        raise ValueError("r2-nocache local search requires config")
                    group_hdf5_partitions = None
                    group_r2_zarr_partitions = None
                    group_r2_nocache_partitions = tuple(
                        local_search_r2_nocache_partition(
                            r2_nocache_config, start, end
                        )
                        for start, end in partition_bounds
                    )
                    group_row_count = sum(
                        partition.source_row_count
                        for partition in group_r2_nocache_partitions
                    )
                elif pair_cache == "hdf5":
                    group_r2_zarr_partitions = None
                    group_r2_nocache_partitions = None
                    group_hdf5_partitions = tuple(
                        local_search_hdf5_partition(chr_name, store, start, end)
                        for start, end in partition_bounds
                    )
                    group_row_count = sum(
                        partition.source_row_count
                        for partition in group_hdf5_partitions
                    )
                else:
                    raise ValueError(f"Unsupported pair cache backend: {pair_cache}")
                group_load_seconds = time.perf_counter() - group_load_start
                log_msg(
                    f"Local search {subset_name} group loaded: "
                    f"breakpoints={len(group)} partitions={len(partition_bounds)} "
                    f"rows={group_row_count} "
                    f"load_seconds={group_load_seconds:.3f} "
                    "canonicalize_seconds=0.000"
                )
                group_cache = ChromosomeCovariance(
                    loci=np.array([], dtype=np.int64),
                    i_pos=np.array([], dtype=np.int64),
                    j_pos=np.array([], dtype=np.int64),
                    r2=np.array([], dtype=np.float64),
                    partitions=partition_bounds,
                    partition_arrays=(),
                )
                for idx, start, stop in group:
                    if idx == 0:
                        log_memory_checkpoint(f"{subset_name}_breakpoint0_start")
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
                        covariance_cache=group_cache,
                        local_search_hdf5_partitions=group_hdf5_partitions,
                        local_search_r2_zarr_partitions=group_r2_zarr_partitions,
                        local_search_r2_nocache_partitions=(
                            group_r2_nocache_partitions
                        ),
                        subset_name=subset_name,
                    )
                    if idx == 0:
                        log_memory_checkpoint(f"{subset_name}_breakpoint0_end")
    else:
        if pair_cache != "hdf5":
            log_msg(
                "Using experimental r2-zarr local search in a single process; "
                "ignoring local-search worker parallelism"
            )
            return _run_local_search(
                chr_name,
                breakpoint_loci,
                snp_first,
                snp_last,
                store,
                metric_out,
                workers=1,
                use_decimal=use_decimal,
                covariance_cache=covariance_cache,
                subset_name=subset_name,
                pair_cache=pair_cache,
                r2_nocache_config=r2_nocache_config,
            )
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
                    subset_name=subset_name,
                ): idx
                for idx, start, stop in tasks
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()

    new_loci = [results[i][0] for i in range(len(breakpoint_loci))]
    new_metrics = [results[i][1] for i in range(len(breakpoint_loci))]

    elapsed = time.perf_counter() - run_start
    log_msg(
        f"Local search {subset_name} done: breakpoints={len(breakpoint_loci)} "
        f"elapsed_seconds={elapsed:.3f}"
    )

    return {"loci": new_loci, "metrics": new_metrics}


def _group_local_search_tasks(
    chr_name: str,
    tasks: list[tuple[int, int, int]],
    breakpoint_loci: list[int],
    store: CovarianceStore,
) -> list[tuple[tuple[tuple[int, int], ...], list[tuple[int, int, int]]]]:
    """Group local-search tasks that require the same covariance partitions."""
    grouped: dict[tuple[tuple[int, int], ...], list[tuple[int, int, int]]] = {}
    order: list[tuple[tuple[int, int], ...]] = []
    for idx, start, stop in tasks:
        tmp_partitions = get_final_partitions(store, chr_name, start, stop)
        snp_top = (
            breakpoint_loci[idx + 1]
            if idx + 1 < len(breakpoint_loci)
            else tmp_partitions[-1][1]
        )
        snp_bottom = breakpoint_loci[idx - 1] if idx - 1 >= 0 else tmp_partitions[0][0]
        partition_bounds = tuple(
            get_final_partitions(store, chr_name, snp_bottom, snp_top)
        )
        if partition_bounds not in grouped:
            grouped[partition_bounds] = []
            order.append(partition_bounds)
        grouped[partition_bounds].append((idx, start, stop))
    return [(partition_bounds, grouped[partition_bounds]) for partition_bounds in order]


def _metric_to_json(metric_out: dict) -> dict:
    """Serialise a metric dict; Decimal values become strings to preserve precision."""
    return {
        "sum": str(metric_out["sum"]),
        "N_nonzero": int(metric_out["N_nonzero"]),
        "N_zero": str(metric_out["N_zero"]),
    }
