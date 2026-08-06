"""CLI: run subcommand — chains all five pipeline steps end-to-end."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from ldetect_lite.io.covariance_hdf5 import validate_covariance_hdf5

if TYPE_CHECKING:
    from ldetect_lite._util.vector_array import (
        _DiagVectorPartitionPlan,
        _DiagVectorPartitionResult,
    )

_VALID_SUBSETS = ("fourier", "fourier_ls", "uniform", "uniform_ls")

# Numpy/BLAS/numba read these once at library-init time to size their own
# internal thread pools. Left unset, they default to the *whole machine's*
# core count, not --workers -- harmless when a job has the machine to itself,
# but oversubscribes real CPUs when many jobs run concurrently on a shared
# node (e.g. several Slurm array tasks on one allocation).
_THREAD_CAP_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "run",
        help="Run the full LD block detection pipeline end-to-end.",
    )
    p.add_argument(
        "--genetic-map",
        required=True,
        type=Path,
        metavar="PATH",
        help="Gzipped genetic map (chr, position, cM).",
    )
    p.add_argument(
        "--reference-panel",
        required=True,
        metavar="PATH",
        help=(
            "VCF/BCF reference panel path (accessed via cyvcf2; must be "
            "indexed with tabix/bcftools index)."
        ),
    )
    p.add_argument(
        "--individuals",
        required=True,
        type=Path,
        metavar="PATH",
        help="Plain-text file; one individual ID per line.",
    )
    p.add_argument(
        "--chromosome",
        required=True,
        metavar="TEXT",
        help="Chromosome name as in the VCF (e.g. chr2 or 2).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="PATH",
        help="Directory where all outputs are written.",
    )
    p.add_argument(
        "--ne",
        type=float,
        default=11418.0,
        metavar="FLOAT",
        help="Effective population size (default: 11418.0).",
    )
    p.add_argument(
        "--cov-cutoff",
        type=float,
        default=1e-7,
        metavar="FLOAT",
        help="LD cutoff for covariance calculation (default: 1e-7).",
    )
    p.add_argument(
        "--covariance-cache",
        choices=("compact", "full"),
        default="compact",
        help=(
            "Covariance partition cache schema for this run. 'compact' writes "
            "only i_pos, j_pos, and shrink_ld; 'full' writes the archival "
            "debug schema (default: compact)."
        ),
    )
    p.add_argument(
        "--covariance-compression",
        choices=("lzf", "zstd"),
        default="zstd",
        help=(
            "HDF5 compression codec for covariance partitions. 'zstd' is "
            "smaller and faster to read/write than 'lzf' at equal precision "
            "(default: zstd)."
        ),
    )
    p.add_argument(
        "--ld-kernel",
        choices=("bitpacked", "uint8"),
        default="bitpacked",
        help=(
            "Pair-count backend for compact covariance output. 'bitpacked' "
            "uses packed haplotypes and popcounts; 'uint8' keeps the older "
            "array-sum backend available for reference and diagnostics "
            "(default: bitpacked). Use --ld-kernel uint8 with "
            "--covariance-cache full."
        ),
    )
    p.add_argument(
        "--n-snps-bw-bpoints",
        type=int,
        default=10_000,
        metavar="N",
        help="Target mean SNPs between breakpoints (default: 10000).",
    )
    p.add_argument(
        "--n-bpoints",
        type=int,
        default=None,
        metavar="N",
        help="Direct target breakpoint count (overrides --n-snps-bw-bpoints).",
    )
    p.add_argument(
        "--subset",
        choices=_VALID_SUBSETS,
        default="fourier_ls",
        metavar="SUBSET",
        help="Breakpoint set for final BED output (default: fourier_ls).",
    )
    p.add_argument(
        "--all-breakpoint-subsets",
        action="store_true",
        help=(
            "Compute all four breakpoint subsets in the JSON output. By default, "
            "only the subset requested by --subset and its dependencies are "
            "computed to avoid unused local-search work."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Parallel workers for the pipeline; used directly for covariance "
            "calculation (Step 2) and as the default for --matrix-workers, "
            "--local-search-workers, and --metric-workers when those are not "
            "set explicitly (default: 1)."
        ),
    )
    p.add_argument(
        "--local-search-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel workers for local search. Higher values may multiply "
            "memory use because each worker loads its own covariance window "
            "(default: inherit --workers)."
        ),
    )
    p.add_argument(
        "--matrix-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel workers for Step 3 matrix-to-vector partition computation "
            "(default: inherit --workers)."
        ),
    )
    p.add_argument(
        "--matrix-backend",
        choices=("array", "legacy"),
        default="array",
        help=(
            "Backend for Step 3 matrix-to-vector conversion. 'array' is faster; "
            "'legacy' uses the dictionary-backed ldetect-compatible summation "
            "path for replication diagnostics (default: array)."
        ),
    )
    p.add_argument(
        "--fused-vector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build the correlation-sum vector during covariance generation "
            "(Step 2) by tee-ing each partition's row stream, instead of "
            "re-reading persisted partitions afterward (Step 3). Bit-exact "
            "with the post-hoc read; skips Step 3's partition re-read "
            "entirely when every partition is freshly computed this run. "
            "Falls back to the normal Step 3 read otherwise (e.g. a resumed "
            "run with skipped/cached partitions). Use --no-fused-vector to "
            "always use the post-hoc Step 3 read, e.g. for diagnostics "
            "(default: enabled)."
        ),
    )
    p.add_argument(
        "--metric-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel workers for Step 4 streaming metric row passes "
            "(default: inherit --workers)."
        ),
    )
    p.add_argument(
        "--high-precision",
        action="store_true",
        help="Use 50-digit Decimal arithmetic for local search (slower).",
    )
    p.add_argument(
        "--filter-window",
        choices=("symmetric", "scipy-periodic"),
        default="symmetric",
        help=(
            "Hanning window mode for breakpoint filtering. 'symmetric' uses "
            "np.hanning and is the default: it matches the window Berisa & "
            "Pickrell (2016) specify in their supplement, is the "
            "conventional choice for a windowed-FIR/convolution filter "
            "kernel, and reproduces published reference blocks generated "
            "under scipy <1.1. 'scipy-periodic' matches modern "
            "scipy.signal.get_window(..., fftbins=True) behavior; use it to "
            "reproduce modern-scipy-era published output (e.g. MacDonald et "
            "al. 2022) (default: symmetric)."
        ),
    )
    p.add_argument(
        "--filter-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Minimum threads for Step 4 adaptive single-filter "
            "convolutions. Trackback keeps candidate-width threading and "
            "forces each candidate filter to one thread to avoid nested "
            "parallelism (default: 1)."
        ),
    )
    p.add_argument(
        "--delete-covariance-cache",
        action="store_true",
        help=(
            "Delete this chromosome's covariance partition cache after the run "
            "completes successfully, to reclaim disk space. Trades away Step "
            "2's skip-already-computed-partitions restart/resume speedup for "
            "this chromosome and output directory (default: keep the cache)."
        ),
    )
    p.add_argument(
        "--force-covariance",
        action="store_true",
        help=(
            "Recompute this chromosome's covariance partitions even if cached "
            "partition files already exist. Useful when changing diagnostic "
            "covariance backends or cache parameters."
        ),
    )
    p.add_argument(
        "--generate-ld-neighborhood-plot",
        action="store_true",
        help=(
            "Write a left/across/right LD-separation box plot alongside the "
            "BED output: r^2 distributions sampled around every computed "
            "breakpoint, comparing LD within each neighboring window against "
            "pairs that cross it. A useful set of breakpoints should show "
            "lower 'across' LD than 'left'/'right' neighborhood LD. "
            "Self-contained sanity check -- needs no reference BED."
        ),
    )
    p.add_argument(
        "--ld-neighborhood-window-bp",
        type=int,
        default=500_000,
        metavar="BP",
        help=(
            "Window size (bp) on each side of a breakpoint sampled for "
            "--generate-ld-neighborhood-plot (default: 500000)."
        ),
    )
    p.set_defaults(func=_run)


def _calc_partition(
    start: int,
    end: int,
    chrom: str,
    reference_panel: str,
    genetic_map_path: Path,
    individuals_path: Path,
    output_path: Path,
    ne: float,
    cutoff: float,
    compact_output: bool,
    compression: str,
    ld_kernel: str,
    vector_plan: _DiagVectorPartitionPlan | None = None,
    snp_last: int | None = None,
) -> _DiagVectorPartitionResult | None:
    """
    Wraps an indexed region fetch > calc_covariance so we can run as a
    worker process.

    When *vector_plan* is given (the --fused-vector fast path), also tees
    this partition's row stream through a CovarianceSidecarAccumulator and
    returns the resulting vector fragment -- see
    `_util/covariance_sidecars.py`. *snp_last* is the chromosome-wide value,
    required by `finalize_vector` alongside the plan's own fields. Returns
    ``None`` if no fragment was requested, or if the single-pass write path
    fell back partway through (the sidecar's buffered rows would be an
    incomplete view of the partition in that case).
    """
    from ldetect_lite._util.covariance_sidecars import CovarianceSidecarAccumulator
    from ldetect_lite._util.memory import log_memory_checkpoint
    from ldetect_lite.shrinkage import calc_covariance

    region = f"{chrom}:{start}-{end}"
    sidecar = CovarianceSidecarAccumulator() if vector_plan is not None else None
    calc_covariance(
        vcf_path=Path(reference_panel),
        region=region,
        genetic_map_path=genetic_map_path,
        individuals_path=individuals_path,
        output_path=output_path,
        ne=ne,
        cutoff=cutoff,
        compact_output=compact_output,
        compression=compression,
        ld_kernel=ld_kernel,
        sidecar=sidecar,
    )
    log_memory_checkpoint(f"covariance_partition_end start={start} end={end}")

    if sidecar is None or vector_plan is None or snp_last is None:
        return None
    if sidecar.invalid:
        return None
    return sidecar.finalize_vector(
        end=vector_plan.end,
        next_start=vector_plan.next_start,
        snp_last=snp_last,
        center_lower_bound=vector_plan.center_lower_bound,
        center_lower_inclusive=vector_plan.center_lower_inclusive,
        checkpoint=vector_plan.checkpoint,
    )


def _resolve_workers(explicit: int | None, default: int) -> int:
    """Resolve a per-stage worker override, falling back to --workers."""
    return default if explicit is None else explicit


def _missing_thread_cap_env_vars() -> list[str]:
    """Return native thread-pool caps that are absent from the environment."""
    return [name for name in _THREAD_CAP_ENV_VARS if not os.environ.get(name)]


def _fused_vector_ready(
    fused_vector_flag: bool,
    pending: list[tuple[int, int]],
    partitions: list[tuple[int, int]],
    vector_fragments: Mapping[tuple[int, int], object],
) -> bool:
    """Whether Step 2's fused-vector fragments can replace Step 3 entirely.

    Only true when every partition was freshly (re)computed this run --
    otherwise some partitions have no fragment (skipped as already-valid
    from a prior run) and Step 3's post-hoc read is the only way to get a
    complete vector. Also false for any partition whose sidecar was marked
    invalid (single-pass write fell back partway through). Sidesteps "some
    partitions have fragments, some don't" entirely rather than half-solving
    it.
    """
    return (
        fused_vector_flag
        and pending == partitions
        and len(vector_fragments) == len(partitions)
    )


def _run(args: argparse.Namespace) -> int:
    import json

    import ldetect_lite
    from ldetect_lite._util.logging import log_msg
    from ldetect_lite._util.memory import log_memory_checkpoint
    from ldetect_lite.io.bed import write_bed
    from ldetect_lite.io.partitions import CovarianceStore, read_partitions
    from ldetect_lite.matrix_analysis import MatrixAnalysis
    from ldetect_lite.pipeline import find_breakpoints
    from ldetect_lite.shrinkage import partition_chromosome

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    chrom = args.chromosome
    cov_dir = output_dir / chrom
    cov_dir.mkdir(exist_ok=True)

    store = CovarianceStore(root=output_dir)
    log_msg(
        "ldetect-lite runtime: "
        f"version={getattr(ldetect_lite, '__version__', 'unknown')} "
        f"source={Path(ldetect_lite.__file__).resolve()}"
    )
    log_memory_checkpoint("run_start")

    missing_thread_caps = _missing_thread_cap_env_vars()
    if args.workers > 1 and missing_thread_caps:
        log_msg(
            f"Warning: --workers {args.workers} is set but these native "
            "thread-pool caps are missing from the environment: "
            f"{', '.join(missing_thread_caps)}. Numpy/BLAS/numba may size "
            "uncapped pools to the whole machine, causing nested CPU "
            "oversubscription. For the default process-parallel pipeline, "
            "export the BLAS/OpenMP/NumExpr/Numba thread caps to 1 unless "
            "you are deliberately using an intra-operation threading option "
            "such as --filter-workers."
        )

    # ------------------------------------------------------------------ #
    # Step 1: Partition chromosome                                         #
    # ------------------------------------------------------------------ #
    partitions_path = output_dir / f"{chrom}_partitions.txt"
    log_msg("Step 1: Partitioning chromosome")
    log_memory_checkpoint("step1_start")
    partition_chromosome(
        genetic_map_path=args.genetic_map,
        n_individuals=_count_individuals(args.individuals),
        output_path=partitions_path,
        ne=args.ne,
    )
    log_memory_checkpoint("step1_end")

    # ------------------------------------------------------------------ #
    # Step 2: Calculate covariance for each partition                     #
    # ------------------------------------------------------------------ #
    compact_output = args.covariance_cache == "compact"
    if args.ld_kernel == "bitpacked" and not compact_output:
        print(
            "Error: --ld-kernel bitpacked requires --covariance-cache compact",
            file=sys.stderr,
        )
        return 1
    log_msg(
        "Step 2: Calculating covariance matrices "
        f"(workers={args.workers}, cache={args.covariance_cache}, "
        f"compression={args.covariance_compression}, "
        f"ld_kernel={args.ld_kernel})"
    )
    log_memory_checkpoint("step2_start")
    partitions = read_partitions(chrom, store)

    pending = []
    invalid = 0
    forced = 0
    for start, end in partitions:
        partition_path = store.partition_path(chrom, start, end)
        if args.force_covariance and partition_path.exists():
            partition_path.unlink()
            forced += 1
            pending.append((start, end))
            continue
        if not partition_path.exists():
            pending.append((start, end))
            continue
        if not _is_valid_covariance_partition(
            partition_path, require_full=not compact_output
        ):
            invalid += 1
            partition_path.unlink()
            pending.append((start, end))
    skipped = len(partitions) - len(pending)
    if skipped:
        log_msg(f"  Skipping {skipped} already-completed partition(s)")
    if forced:
        log_msg(f"  Forcing recomputation of {forced} cached partition(s)")
    if invalid:
        log_msg(f"  Regenerating {invalid} invalid cached partition(s)")

    snp_first = partitions[0][0]
    snp_last = partitions[-1][1]

    matrix_workers = _resolve_workers(args.matrix_workers, args.workers)
    local_search_workers = _resolve_workers(args.local_search_workers, args.workers)
    metric_workers = _resolve_workers(args.metric_workers, args.workers)

    # Fused-vector plans, keyed by partition bounds so Step 2's futures can
    # look theirs up below. Requires the compact cache (the sidecar only
    # tees the compact single-pass write path); silently unavailable
    # otherwise rather than erroring, since --fused-vector defaults to on.
    vector_plans_by_bounds: dict[tuple[int, int], _DiagVectorPartitionPlan] = {}
    if args.fused_vector and compact_output:
        from ldetect_lite._util.vector_array import _plan_diag_vector_partitions

        plans = _plan_diag_vector_partitions(partitions, snp_first, snp_last)
        if len(plans) == len(partitions):
            vector_plans_by_bounds = {(p.start, p.end): p for p in plans}
        else:
            log_msg(
                "  --fused-vector: partition plan count mismatch "
                f"({len(plans)} != {len(partitions)}); falling back to Step 3"
            )

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _calc_partition,
                start,
                end,
                chrom,
                args.reference_panel,
                args.genetic_map,
                args.individuals,
                store.partition_path(chrom, start, end),
                args.ne,
                args.cov_cutoff,
                compact_output,
                args.covariance_compression,
                args.ld_kernel,
                vector_plans_by_bounds.get((start, end)),
                snp_last if vector_plans_by_bounds else None,
            ): (start, end)
            for start, end in pending
        }
        vector_fragments: dict[tuple[int, int], _DiagVectorPartitionResult] = {}
        for fut in as_completed(futures):
            start, end = futures[fut]
            try:
                result = fut.result()
            except (RuntimeError, ValueError) as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
            if result is not None:
                vector_fragments[(start, end)] = result
            log_msg(f"  Partition {start}-{end} done")
    log_memory_checkpoint("step2_end")

    # ------------------------------------------------------------------ #
    # Step 3: Matrix → vector                                             #
    # ------------------------------------------------------------------ #
    vector_path = output_dir / f"vector-{chrom}.txt.gz"
    if _fused_vector_ready(
        bool(vector_plans_by_bounds), pending, partitions, vector_fragments
    ):
        log_msg("Step 3: Writing vector from fused direct-vector sidecar fragments")
        log_memory_checkpoint("step3_start")
        from ldetect_lite._util.vector_array import _merge_diag_vector_partition_result

        vector_path.unlink(missing_ok=True)
        pending_sums: dict[int, float] = {}
        parent_profile: dict[str, float | int] = {
            "merge_seconds": 0.0,
            "flush_seconds": 0.0,
            "worker_wait_seconds": 0.0,
            "partitions": 0,
        }
        current_locus = snp_first
        for start, end in partitions:
            current_locus = _merge_diag_vector_partition_result(
                result=vector_fragments[(start, end)],
                snp_first=snp_first,
                snp_last=snp_last,
                current_locus=current_locus,
                pending_sums=pending_sums,
                out_path=vector_path,
                parent_profile=parent_profile,
            )
        if args.matrix_workers is not None:
            log_msg(
                "  --matrix-workers ignored (fused-vector path skips Step 3's "
                "worker pool)"
            )
        log_memory_checkpoint("step3_end")
    else:
        if args.fused_vector and compact_output:
            log_msg(
                "  --fused-vector requested but not all partitions were freshly "
                "computed this run; falling back to Step 3"
            )
        log_msg(
            "Step 3: Converting matrix to vector "
            f"(backend={args.matrix_backend}, workers={matrix_workers})"
        )
        log_memory_checkpoint("step3_start")
        analysis = MatrixAnalysis(name=chrom, store=store)
        analysis.calc_diag_lean(
            vector_path,
            matrix_workers=matrix_workers,
            backend=args.matrix_backend,
        )
        log_memory_checkpoint("step3_end")

    # ------------------------------------------------------------------ #
    # Step 4: Find minima                                                 #
    # ------------------------------------------------------------------ #
    breakpoints_path = output_dir / f"breakpoints-{chrom}.json"
    log_msg(
        "Step 4: Finding breakpoints "
        f"(local_search_workers={local_search_workers}, "
        f"metric_workers={metric_workers})"
    )
    log_memory_checkpoint("step4_start")
    find_breakpoints(
        input_path=vector_path,
        chr_name=chrom,
        store=store,
        n_snps_bw_bpoints=args.n_snps_bw_bpoints,
        output_path=breakpoints_path,
        workers=local_search_workers,
        metric_workers=metric_workers,
        use_decimal=args.high_precision,
        n_bpoints=args.n_bpoints,
        subsets=_breakpoint_subsets_for_run(args.subset, args.all_breakpoint_subsets),
        filter_window=args.filter_window,
        filter_workers=args.filter_workers,
    )
    log_memory_checkpoint("step4_end")

    # ------------------------------------------------------------------ #
    # Step 5: Extract breakpoints to BED                                  #
    # ------------------------------------------------------------------ #
    bed_path = output_dir / f"{chrom}-ld-blocks.bed"
    log_msg(f"Step 5: Extracting {args.subset} breakpoints to {bed_path}")
    log_memory_checkpoint("step5_start")
    data = json.loads(breakpoints_path.read_text())
    if args.subset not in data:
        computed = ", ".join(data.get("computed_subsets", [])) or "(none)"
        print(
            f"Error: requested subset {args.subset!r} was not computed. "
            f"Computed subset(s): {computed}",
            file=sys.stderr,
        )
        return 1
    loci: list[int] = data[args.subset]["loci"]

    write_bed(
        name=chrom,
        loci=loci,
        snp_first=snp_first,
        snp_last=snp_last,
        output=bed_path,
    )

    if args.generate_ld_neighborhood_plot:
        from ldetect_lite.ld_neighborhood import (
            chromosome_separation_samples,
            write_separation_boxplot,
        )

        log_msg("Generating LD-neighborhood separation plot")
        neighborhood_path = output_dir / f"{chrom}-ld-neighborhood.svg"
        samples = chromosome_separation_samples(
            boundaries=loci,
            store=store,
            name=chrom,
            window_bp=args.ld_neighborhood_window_bp,
        )
        write_separation_boxplot(
            neighborhood_path,
            samples,
            title=f"{chrom}: LD neighborhood separation",
        )
        log_msg(f"LD-neighborhood plot: {neighborhood_path}")

    if args.delete_covariance_cache:
        log_msg(f"Deleting covariance cache: {cov_dir}")
        _delete_covariance_cache(cov_dir)

    log_msg(f"Done. BED file: {bed_path}")
    log_memory_checkpoint("run_end")
    return 0


def _delete_covariance_cache(cov_dir: Path) -> None:
    """Remove a chromosome's covariance partition directory, if present."""
    if cov_dir.exists():
        shutil.rmtree(cov_dir)


def _count_individuals(path: Path) -> int:
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _breakpoint_subsets_for_run(
    subset: str, all_breakpoint_subsets: bool
) -> set[str] | None:
    """Return the breakpoint subset request passed from ``run`` to the pipeline.

    ``None`` intentionally preserves the full historical JSON output; otherwise
    ``run`` asks the pipeline to compute only the final BED subset and its
    dependencies.
    """
    return None if all_breakpoint_subsets else {subset}


def _is_valid_covariance_partition(
    path: Path, require_full: bool = True
) -> bool:
    return validate_covariance_hdf5(path, require_full=require_full)
