"""CLI: calc-covariance subcommand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "calc-covariance",
        help="Compute Wen/Stephens shrinkage LD from a VCF stream (reads stdin).",
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
        "--shrink-ld-precision",
        choices=("float64", "float32"),
        default="float64",
        help=(
            "Precision for shrink_ld values. 'float32' rounds values before "
            "writing (still stored as float64, no schema change) to improve "
            "compressibility; not yet validated against real breakpoints, "
            "so it defaults off (default: float64)."
        ),
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from ldetect2.shrinkage import calc_covariance

    calc_covariance(
        vcf_stream=sys.stdin,
        genetic_map_path=args.genetic_map,
        individuals_path=args.individuals,
        output_path=args.output,
        ne=args.ne,
        cutoff=args.cutoff,
        compression=args.covariance_compression,
        shrink_ld_precision=args.shrink_ld_precision,
    )
    return 0
