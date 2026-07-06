"""CLI: partition-chromosome subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "partition-chromosome",
        help="Split a chromosome into overlapping windows using a genetic map.",
    )
    p.add_argument(
        "--genetic-map",
        required=True,
        type=Path,
        metavar="PATH",
        help="Gzipped genetic map (chr, position, cM).",
    )
    p.add_argument(
        "--n-individuals",
        required=True,
        type=int,
        metavar="N",
        help="Number of individuals in the reference panel.",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Output partition file (space-delimited start/end pairs).",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=5000,
        metavar="N",
        help="Target SNPs per partition window (default: 5000).",
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
        default=1.5e-8,
        metavar="FLOAT",
        help="Recombination fraction threshold for window extension (default: 1.5e-8).",
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from ldetect_lite.shrinkage import partition_chromosome

    partition_chromosome(
        genetic_map_path=args.genetic_map,
        n_individuals=args.n_individuals,
        output_path=args.output,
        window_size=args.window_size,
        ne=args.ne,
        cutoff=args.cutoff,
    )
    return 0
