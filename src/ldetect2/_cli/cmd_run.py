"""CLI: run subcommand — chains all five pipeline steps end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_VALID_SUBSETS = ("fourier", "fourier_ls", "uniform", "uniform_ls")


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "run",
        help="Run the full LD block detection pipeline end-to-end.",
    )
    p.add_argument("--genetic-map", required=True, type=Path, metavar="PATH",
                   help="Gzipped genetic map (chr, position, cM).")
    p.add_argument("--reference-panel", required=True, metavar="PATH",
                   help="VCF reference panel path (accessed via tabix).")
    p.add_argument("--individuals", required=True, type=Path, metavar="PATH",
                   help="Plain-text file; one individual ID per line.")
    p.add_argument("--chromosome", required=True, metavar="TEXT",
                   help="Chromosome name as in the VCF (e.g. chr2 or 2).")
    p.add_argument("--output-dir", required=True, type=Path, metavar="PATH",
                   help="Directory where all outputs are written.")
    p.add_argument("--ne", type=float, default=11418.0, metavar="FLOAT",
                   help="Effective population size (default: 11418.0).")
    p.add_argument("--cov-cutoff", type=float, default=1e-7, metavar="FLOAT",
                   help="LD cutoff for covariance calculation (default: 1e-7).")
    p.add_argument("--n-snps-bw-bpoints", type=int, default=50, metavar="N",
                   help="Target mean SNPs between breakpoints (default: 50).")
    p.add_argument("--subset", choices=_VALID_SUBSETS, default="fourier_ls",
                   metavar="SUBSET",
                   help=f"Breakpoint set for final BED output "
                        f"(default: fourier_ls).")
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from ldetect2._util.logging import log_msg
    from ldetect2.io.partitions import CovarianceStore
    from ldetect2.io.bed import write_bed
    from ldetect2.pipeline import find_breakpoints
    from ldetect2.shrinkage import calc_covariance, partition_chromosome
    from ldetect2.matrix_analysis import MatrixAnalysis
    from ldetect2.io.partitions import read_partitions
    import json

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    chrom = args.chromosome
    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    cov_dir = output_dir / chrom
    cov_dir.mkdir(exist_ok=True)

    store = CovarianceStore(root=output_dir)

    # ------------------------------------------------------------------ #
    # Step 1: Partition chromosome                                         #
    # ------------------------------------------------------------------ #
    partitions_path = scripts_dir / f"{chrom}_partitions"
    log_msg("Step 1: Partitioning chromosome")
    partition_chromosome(
        genetic_map_path=args.genetic_map,
        n_individuals=_count_individuals(args.individuals),
        output_path=partitions_path,
    )

    # ------------------------------------------------------------------ #
    # Step 2: Calculate covariance for each partition                     #
    # ------------------------------------------------------------------ #
    log_msg("Step 2: Calculating covariance matrices")
    partitions = read_partitions(chrom, store)

    for start, end in partitions:
        cov_file = store.partition_path(chrom, start, end)
        if cov_file.exists():
            log_msg(f"  Partition {start}-{end} already exists, skipping")
            continue
        log_msg(f"  Partition {start}-{end}")
        region = f"{chrom}:{start}-{end}"
        tabix_cmd = ["tabix", "-h", args.reference_panel, region]
        try:
            tabix_proc = subprocess.Popen(
                tabix_cmd,
                stdout=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            print(
                "Error: tabix not found. Install htslib and ensure tabix is on PATH.",
                file=sys.stderr,
            )
            return 1

        with tabix_proc.stdout:  # type: ignore[union-attr]
            calc_covariance(
                vcf_stream=tabix_proc.stdout,
                genetic_map_path=args.genetic_map,
                individuals_path=args.individuals,
                output_path=cov_file,
                ne=args.ne,
                cutoff=args.cov_cutoff,
            )
        tabix_proc.wait()

    # ------------------------------------------------------------------ #
    # Step 3: Matrix → vector                                             #
    # ------------------------------------------------------------------ #
    vector_path = output_dir / f"vector-{chrom}.txt.gz"
    log_msg("Step 3: Converting matrix to vector")
    analysis = MatrixAnalysis(name=chrom, store=store)
    analysis.calc_diag_lean(vector_path)

    # ------------------------------------------------------------------ #
    # Step 4: Find minima                                                 #
    # ------------------------------------------------------------------ #
    breakpoints_path = output_dir / f"breakpoints-{chrom}.json"
    log_msg("Step 4: Finding breakpoints")
    find_breakpoints(
        input_path=vector_path,
        chr_name=chrom,
        store=store,
        n_snps_bw_bpoints=args.n_snps_bw_bpoints,
        output_path=breakpoints_path,
    )

    # ------------------------------------------------------------------ #
    # Step 5: Extract breakpoints to BED                                  #
    # ------------------------------------------------------------------ #
    bed_path = output_dir / f"{chrom}-ld-blocks.bed"
    log_msg(f"Step 5: Extracting {args.subset} breakpoints to {bed_path}")
    data = json.loads(breakpoints_path.read_text())
    loci: list[int] = data[args.subset]["loci"]

    snp_first = partitions[0][0]
    snp_last = partitions[-1][1]
    write_bed(name=chrom, loci=loci, snp_first=snp_first, snp_last=snp_last,
              output=bed_path)

    log_msg(f"Done. BED file: {bed_path}")
    return 0


def _count_individuals(path: Path) -> int:
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count
