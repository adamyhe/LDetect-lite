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
        "--signal-output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Optional path to also write a per-partition signal sidecar, "
            "letting `matrix-to-vector --prefer-signal-cache` skip rereading "
            "and renormalizing this partition's covariance rows."
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
        signal_output_path=args.signal_output,
    )
    return 0
