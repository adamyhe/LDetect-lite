"""CLI: calc-covariance subcommand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "calc-covariance",
        help=(
            "Compute Wen/Stephens shrinkage LD from a VCF/BCF file "
            "(or stdin if --reference-panel is omitted)."
        ),
    )
    p.add_argument(
        "--reference-panel",
        default=None,
        metavar="PATH",
        help=(
            "VCF/BCF reference panel path (indexed with tabix/bcftools index; "
            "accessed via cyvcf2). If omitted, reads from stdin instead."
        ),
    )
    p.add_argument(
        "--region",
        default=None,
        metavar="CHROM:START-END",
        help=(
            "Restrict to this region via an indexed fetch. Requires "
            "--reference-panel; omit to read the whole file/stream."
        ),
    )
    p.add_argument(
        "--genetic-map",
        required=True,
        type=Path,
        metavar="PATH",
        help="Gzipped genetic map (chr, position, cM).",
    )
    p.add_argument(
        "--individuals",
        required=True,
        type=Path,
        metavar="PATH",
        help="Plain-text file; one individual ID per line.",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Gzipped 8-column covariance output file.",
    )
    p.add_argument(
        "--ne",
        type=float,
        default=11418.0,
        metavar="FLOAT",
        help="Effective population size (default: 11418.0).",
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=1e-7,
        metavar="FLOAT",
        help="LD cutoff; pairs below this are not written (default: 1e-7).",
    )
    p.add_argument(
        "--covariance-compression",
        choices=("lzf", "zstd"),
        default="zstd",
        help=(
            "HDF5 compression codec for the covariance partition. 'zstd' is "
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
            "(default: bitpacked)."
        ),
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from ldetect_lite.shrinkage import calc_covariance

    if args.region is not None and args.reference_panel is None:
        print("Error: --region requires --reference-panel", file=sys.stderr)
        return 1

    vcf_path = Path(args.reference_panel) if args.reference_panel else Path("-")
    try:
        calc_covariance(
            vcf_path=vcf_path,
            region=args.region,
            genetic_map_path=args.genetic_map,
            individuals_path=args.individuals,
            output_path=args.output,
            ne=args.ne,
            cutoff=args.cutoff,
            compression=args.covariance_compression,
            compact_output=args.ld_kernel == "bitpacked",
            ld_kernel=args.ld_kernel,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0
