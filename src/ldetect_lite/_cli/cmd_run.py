"""CLI: run subcommand — chains all five pipeline steps end-to-end."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ldetect_lite.io.covariance_hdf5 import validate_covariance_hdf5

if TYPE_CHECKING:
    from ldetect_lite.io.partitions import CovarianceStore

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
        "--covariance-mode",
        choices=("partition", "chromosome"),
        default="partition",
        help=(
            "Covariance calculation strategy. 'partition' preserves the "
            "historical one-region-per-partition path and supports "
            "partition-level workers. 'chromosome' loads this chromosome's "
            "genotypes once and slices partitions from the prepared arrays; "
            "it currently requires --covariance-cache compact (default: "
            "partition)."
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
        choices=("uint8", "bitpacked"),
        default="uint8",
        help=(
            "Pair-count backend for compact covariance output. 'uint8' is the "
            "established backend; 'bitpacked' uses packed haplotypes and "
            "popcounts (default: uint8)."
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
        "--profile-covariance",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write covariance timing diagnostics as TSV. In chromosome mode "
            "this includes one chromosome-load row plus per-partition writer "
            "rows."
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
) -> None:
    """
    Wraps an indexed region fetch > calc_covariance so we can run as a
    worker process.
    """
    from ldetect_lite._util.memory import log_memory_checkpoint
    from ldetect_lite.shrinkage import calc_covariance

    region = f"{chrom}:{start}-{end}"
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
    )
    log_memory_checkpoint(f"covariance_partition_end start={start} end={end}")


def _resolve_workers(explicit: int | None, default: int) -> int:
    """Resolve a per-stage worker override, falling back to --workers."""
    return default if explicit is None else explicit


def _calc_chromosome_partitions(
    *,
    pending: list[tuple[int, int]],
    chrom: str,
    reference_panel: str,
    genetic_map_path: Path,
    individuals_path: Path,
    store: CovarianceStore,
    ne: float,
    cutoff: float,
    compression: str,
    ld_kernel: str,
) -> list[dict[str, str]]:
    from ldetect_lite._util.memory import log_memory_checkpoint
    from ldetect_lite.shrinkage import (
        calc_covariance_from_genotypes,
        load_chromosome_genotypes,
    )

    profile_rows: list[dict[str, str]] = []
    load_profile: dict[str, float] = {}
    storage: Literal["uint8", "packed"] = (
        "packed" if ld_kernel == "bitpacked" else "uint8"
    )
    kernel: Literal["uint8", "bitpacked"] = (
        "bitpacked" if ld_kernel == "bitpacked" else "uint8"
    )
    genotypes = load_chromosome_genotypes(
        vcf_path=Path(reference_panel),
        chrom=chrom,
        genetic_map_path=genetic_map_path,
        individuals_path=individuals_path,
        storage=storage,
        profile=load_profile,
    )
    profile_rows.append(
        _profile_row(
            row_type="chromosome",
            chrom=chrom,
            start="",
            end="",
            ld_kernel=ld_kernel,
            profile=load_profile,
        )
    )
    log_memory_checkpoint(f"covariance_chromosome_loaded chrom={chrom}")

    for start, end in pending:
        profile: dict[str, float] = {}
        calc_covariance_from_genotypes(
            genotypes,
            start,
            end,
            store.partition_path(chrom, start, end),
            ne=ne,
            cutoff=cutoff,
            compression=compression,
            ld_kernel=kernel,
            profile=profile,
        )
        profile_rows.append(
            _profile_row(
                row_type="partition",
                chrom=chrom,
                start=str(start),
                end=str(end),
                ld_kernel=ld_kernel,
                profile=profile,
            )
        )
        log_memory_checkpoint(f"covariance_partition_end start={start} end={end}")

    return profile_rows


def _profile_row(
    *,
    row_type: str,
    chrom: str,
    start: str,
    end: str,
    ld_kernel: str,
    profile: dict[str, float],
) -> dict[str, str]:
    row = {
        "row_type": row_type,
        "chrom": chrom,
        "start": start,
        "end": end,
        "ld_kernel": ld_kernel,
    }
    for key, value in sorted(profile.items()):
        row[key] = f"{value:.9g}"
    return row


def _write_profile_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = ["row_type", "chrom", "start", "end", "ld_kernel"]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = preferred + extras
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


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

    if args.workers > 1 and not any(os.environ.get(v) for v in _THREAD_CAP_ENV_VARS):
        log_msg(
            f"Warning: --workers {args.workers} is set but none of "
            f"{', '.join(_THREAD_CAP_ENV_VARS)} are set in the environment. "
            "Numpy/BLAS/numba may each size their own thread pools to the "
            "whole machine instead of --workers, oversubscribing CPUs if "
            "other jobs are running concurrently on the same node (e.g. "
            "under Slurm). Consider exporting these to match --workers."
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
    if args.covariance_mode == "chromosome" and not compact_output:
        print(
            "Error: --covariance-mode chromosome requires "
            "--covariance-cache compact",
            file=sys.stderr,
        )
        return 1
    log_msg(
        "Step 2: Calculating covariance matrices "
        f"(workers={args.workers}, cache={args.covariance_cache}, "
        f"compression={args.covariance_compression}, "
        f"ld_kernel={args.ld_kernel}, mode={args.covariance_mode})"
    )
    log_memory_checkpoint("step2_start")
    partitions = read_partitions(chrom, store)

    pending = []
    invalid = 0
    for start, end in partitions:
        partition_path = store.partition_path(chrom, start, end)
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
    if invalid:
        log_msg(f"  Regenerating {invalid} invalid cached partition(s)")

    profile_rows: list[dict[str, str]] = []
    if args.covariance_mode == "chromosome":
        if pending:
            if args.workers != 1:
                log_msg(
                    "  Note: --covariance-mode chromosome processes this "
                    "single chromosome serially; --workers still applies to "
                    "later pipeline stages unless overridden."
                )
            try:
                profile_rows = _calc_chromosome_partitions(
                    pending=pending,
                    chrom=chrom,
                    reference_panel=args.reference_panel,
                    genetic_map_path=args.genetic_map,
                    individuals_path=args.individuals,
                    store=store,
                    ne=args.ne,
                    cutoff=args.cov_cutoff,
                    compression=args.covariance_compression,
                    ld_kernel=args.ld_kernel,
                )
            except (RuntimeError, ValueError) as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
            for start, end in pending:
                log_msg(f"  Partition {start}-{end} done")
    else:
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
                ): (start, end)
                for start, end in pending
            }
            for fut in as_completed(futures):
                start, end = futures[fut]
                try:
                    fut.result()
                except (RuntimeError, ValueError) as e:
                    print(f"Error: {e}", file=sys.stderr)
                    return 1
                log_msg(f"  Partition {start}-{end} done")
    if args.profile_covariance is not None and profile_rows:
        _write_profile_tsv(args.profile_covariance, profile_rows)
    log_memory_checkpoint("step2_end")

    snp_first = partitions[0][0]
    snp_last = partitions[-1][1]

    matrix_workers = _resolve_workers(args.matrix_workers, args.workers)
    local_search_workers = _resolve_workers(args.local_search_workers, args.workers)
    metric_workers = _resolve_workers(args.metric_workers, args.workers)

    # ------------------------------------------------------------------ #
    # Step 3: Matrix → vector                                             #
    # ------------------------------------------------------------------ #
    vector_path = output_dir / f"vector-{chrom}.txt.gz"
    log_msg(f"Step 3: Converting matrix to vector (workers={matrix_workers})")
    log_memory_checkpoint("step3_start")
    analysis = MatrixAnalysis(name=chrom, store=store)
    analysis.calc_diag_lean(vector_path, matrix_workers=matrix_workers)
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
        name=chrom, loci=loci, snp_first=snp_first, snp_last=snp_last, output=bed_path
    )

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
