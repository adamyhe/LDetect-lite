"""CLI: run subcommand — chains all five pipeline steps end-to-end."""

from __future__ import annotations

import argparse
import gzip
import io
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ldetect2.io.covariance_hdf5 import validate_covariance_hdf5
from ldetect2.io.r2_zarr import validate_r2_zarr_partition

_VALID_SUBSETS = ("fourier", "fourier_ls", "uniform", "uniform_ls")


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
        help="VCF reference panel path (accessed via tabix).",
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
        "--pair-cache",
        choices=("hdf5", "r2-zarr"),
        default="hdf5",
        help=(
            "Pair cache used by metric/local-search. 'hdf5' preserves the "
            "current covariance cache behavior; 'r2-zarr' is experimental and "
            "writes normalized float64 r2 rows plus direct vector fragments "
            "(default: hdf5)."
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
        help="Parallel workers for covariance calculation (default: 1).",
    )
    p.add_argument(
        "--local-search-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Parallel workers for local search. Higher values may multiply "
            "memory use because each worker loads its own covariance window "
            "(default: 1)."
        ),
    )
    p.add_argument(
        "--matrix-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Parallel workers for Step 3 matrix-to-vector partition computation "
            "(default: 1)."
        ),
    )
    p.add_argument(
        "--vector-mode",
        choices=("matrix", "direct"),
        default="matrix",
        help=(
            "How to produce the Step 3 correlation-sum vector. 'matrix' reads "
            "covariance partitions with matrix-to-vector; 'direct' writes "
            "ownership-bounded vector fragments during covariance calculation "
            "and concatenates them (default: matrix)."
        ),
    )
    p.add_argument(
        "--metric-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Parallel workers for Step 4 streaming metric row passes "
            "(default: 1)."
        ),
    )
    p.add_argument(
        "--high-precision",
        action="store_true",
        help="Use 50-digit Decimal arithmetic for local search (slower).",
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
    pair_cache: str = "hdf5",
    vector_output_path: Path | None = None,
    center_lower_bound: int | None = None,
    center_lower_inclusive: bool = True,
    center_upper_bound: int | None = None,
    center_upper_inclusive: bool = True,
) -> None:
    """
    Wraps tabix > calc_covariance so we can run as a worker process.
    """
    from ldetect2._util.memory import log_memory_checkpoint
    from ldetect2.shrinkage import (
        calc_covariance,
        calc_covariance_vector,
        calc_r2_zarr_partition,
    )

    region = f"{chrom}:{start}-{end}"
    try:
        tabix_proc = subprocess.Popen(
            ["tabix", "-h", reference_panel, region],
            stdout=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "tabix not found. Install htslib and ensure tabix is on PATH."
        )

    if pair_cache == "r2-zarr":
        if vector_output_path is None:
            raise RuntimeError("r2-zarr pair cache requires a direct vector fragment")
        with tabix_proc.stdout:  # type: ignore[union-attr]
            calc_r2_zarr_partition(
                vcf_stream=tabix_proc.stdout,
                genetic_map_path=genetic_map_path,
                individuals_path=individuals_path,
                output_root=output_path,
                name=chrom,
                start=start,
                end=end,
                ne=ne,
                cutoff=cutoff,
                vector_output_path=vector_output_path,
                center_lower_bound=center_lower_bound,
                center_lower_inclusive=center_lower_inclusive,
                center_upper_bound=center_upper_bound,
                center_upper_inclusive=center_upper_inclusive,
            )
    elif vector_output_path is not None:
        with tabix_proc.stdout:  # type: ignore[union-attr]
            vcf_text = tabix_proc.stdout.read()
        calc_covariance(
            vcf_stream=io.StringIO(vcf_text),
            genetic_map_path=genetic_map_path,
            individuals_path=individuals_path,
            output_path=output_path,
            ne=ne,
            cutoff=cutoff,
            compact_output=compact_output,
        )
        calc_covariance_vector(
            vcf_stream=io.StringIO(vcf_text),
            genetic_map_path=genetic_map_path,
            individuals_path=individuals_path,
            output_path=vector_output_path,
            ne=ne,
            cutoff=cutoff,
            center_lower_bound=center_lower_bound,
            center_lower_inclusive=center_lower_inclusive,
            center_upper_bound=center_upper_bound,
            center_upper_inclusive=center_upper_inclusive,
        )
    else:
        with tabix_proc.stdout:  # type: ignore[union-attr]
            calc_covariance(
                vcf_stream=tabix_proc.stdout,
                genetic_map_path=genetic_map_path,
                individuals_path=individuals_path,
                output_path=output_path,
                ne=ne,
                cutoff=cutoff,
                compact_output=compact_output,
            )
    tabix_proc.wait()
    log_memory_checkpoint(f"covariance_partition_end start={start} end={end}")


def _direct_vector_plan(
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
) -> dict[tuple[int, int], tuple[int, bool, int, bool]]:
    """Return ownership bounds for direct vector fragments."""
    bounds: dict[tuple[int, int], tuple[int, bool, int, bool]] = {}
    previous_end_locus: int | None = None
    for p_index, (start, end) in enumerate(partitions):
        if previous_end_locus is None:
            lower_bound = snp_first
            lower_inclusive = True
        else:
            lower_bound = previous_end_locus
            lower_inclusive = False

        if p_index + 1 < len(partitions):
            upper_bound = int((end + partitions[p_index + 1][0]) / 2)
            upper_inclusive = True
        else:
            upper_bound = snp_last
            upper_inclusive = False
        bounds[(start, end)] = (
            lower_bound,
            lower_inclusive,
            upper_bound,
            upper_inclusive,
        )
        previous_end_locus = (
            int((end + partitions[p_index + 1][0]) / 2)
            if p_index + 1 < len(partitions)
            else snp_last
        )
    return bounds


def _concatenate_direct_vector_fragments(
    fragments: list[Path],
    output_path: Path,
) -> None:
    output_path.unlink(missing_ok=True)
    with gzip.open(output_path, "wt") as out:
        for fragment in fragments:
            if not fragment.exists():
                continue
            with gzip.open(fragment, "rt") as inp:
                for line in inp:
                    out.write(line)


def _run(args: argparse.Namespace) -> int:
    import json

    import ldetect2
    from ldetect2._util.logging import log_msg
    from ldetect2._util.memory import log_memory_checkpoint
    from ldetect2.io.bed import write_bed
    from ldetect2.io.partitions import CovarianceStore, read_partitions
    from ldetect2.matrix_analysis import MatrixAnalysis
    from ldetect2.pipeline import find_breakpoints
    from ldetect2.shrinkage import partition_chromosome

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.pair_cache == "r2-zarr" and args.high_precision:
        print(
            "Error: --pair-cache r2-zarr does not support --high-precision.",
            file=sys.stderr,
        )
        return 1
    vector_mode = "direct" if args.pair_cache == "r2-zarr" else args.vector_mode

    chrom = args.chromosome
    cov_dir = output_dir / chrom
    cov_dir.mkdir(exist_ok=True)

    store = CovarianceStore(root=output_dir)
    log_msg(
        "ldetect2 runtime: "
        f"version={getattr(ldetect2, '__version__', 'unknown')} "
        f"source={Path(ldetect2.__file__).resolve()}"
    )
    log_memory_checkpoint("run_start")

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
    log_msg(
        "Step 2: Calculating covariance matrices "
        f"(workers={args.workers}, cache={args.covariance_cache}, "
        f"pair_cache={args.pair_cache})"
    )
    log_memory_checkpoint("step2_start")
    partitions = read_partitions(chrom, store)
    snp_first = partitions[0][0]
    snp_last = partitions[-1][1]
    direct_vector_bounds = (
        _direct_vector_plan(partitions, snp_first, snp_last)
        if vector_mode == "direct"
        else {}
    )
    direct_vector_dir = output_dir / "direct_vector_fragments" / chrom
    if vector_mode == "direct":
        direct_vector_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    invalid = 0
    for start, end in partitions:
        partition_path = store.partition_path(chrom, start, end)
        vector_fragment_path = direct_vector_dir / f"{chrom}.{start}.{end}.txt.gz"
        if args.pair_cache == "r2-zarr":
            if not validate_r2_zarr_partition(output_dir, chrom, start, end):
                pending.append((start, end))
                continue
            if not vector_fragment_path.exists():
                pending.append((start, end))
                continue
            continue

        if not partition_path.exists():
            pending.append((start, end))
            continue
        if vector_mode == "direct" and not vector_fragment_path.exists():
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
                (
                    output_dir
                    if args.pair_cache == "r2-zarr"
                    else store.partition_path(chrom, start, end)
                ),
                args.ne,
                args.cov_cutoff,
                compact_output,
                args.pair_cache,
                (
                    direct_vector_dir / f"{chrom}.{start}.{end}.txt.gz"
                    if vector_mode == "direct"
                    else None
                ),
                *direct_vector_bounds.get((start, end), (None, True, None, True)),
            ): (start, end)
            for start, end in pending
        }
        for fut in as_completed(futures):
            start, end = futures[fut]
            try:
                fut.result()
            except RuntimeError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
            log_msg(f"  Partition {start}-{end} done")
    log_memory_checkpoint("step2_end")

    # ------------------------------------------------------------------ #
    # Step 3: Matrix/direct → vector                                      #
    # ------------------------------------------------------------------ #
    vector_path = output_dir / f"vector-{chrom}.txt.gz"
    log_memory_checkpoint("step3_start")
    if vector_mode == "direct":
        log_msg("Step 3: Concatenating direct vector fragments")
        _concatenate_direct_vector_fragments(
            [
                direct_vector_dir / f"{chrom}.{start}.{end}.txt.gz"
                for start, end in partitions
            ],
            vector_path,
        )
    else:
        log_msg("Step 3: Converting matrix to vector")
        analysis = MatrixAnalysis(name=chrom, store=store)
        analysis.calc_diag_lean(vector_path, matrix_workers=args.matrix_workers)
    log_memory_checkpoint("step3_end")

    # ------------------------------------------------------------------ #
    # Step 4: Find minima                                                 #
    # ------------------------------------------------------------------ #
    breakpoints_path = output_dir / f"breakpoints-{chrom}.json"
    log_msg("Step 4: Finding breakpoints")
    log_memory_checkpoint("step4_start")
    find_breakpoints(
        input_path=vector_path,
        chr_name=chrom,
        store=store,
        n_snps_bw_bpoints=args.n_snps_bw_bpoints,
        output_path=breakpoints_path,
        workers=args.local_search_workers,
        metric_workers=args.metric_workers,
        use_decimal=args.high_precision,
        n_bpoints=args.n_bpoints,
        subsets=_breakpoint_subsets_for_run(args.subset, args.all_breakpoint_subsets),
        pair_cache=args.pair_cache,
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

    log_msg(f"Done. BED file: {bed_path}")
    log_memory_checkpoint("run_end")
    return 0


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
