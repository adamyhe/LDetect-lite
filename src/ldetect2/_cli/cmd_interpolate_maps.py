"""CLI: interpolate-maps subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "interpolate-maps",
        help="Interpolate genetic map positions onto SNP physical positions.",
    )
    p.add_argument(
        "--snp-file",
        required=True,
        type=Path,
        metavar="PATH",
        help="BED file of SNP positions (columns: chrom start end rs_id).",
    )
    p.add_argument(
        "--genetic-map",
        required=True,
        type=Path,
        metavar="PATH",
        help="Gzipped recombination map (position, rate, cM).",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Gzipped output file (rs_id, position, genetic_position).",
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from ldetect2.interpolate_maps import interpolate

    interpolate(
        snp_file=args.snp_file,
        genetic_map=args.genetic_map,
        output=args.output,
    )
    return 0
