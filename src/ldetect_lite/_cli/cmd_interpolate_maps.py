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
    p.add_argument(
        "--mode",
        choices=("point", "interval", "hapmap"),
        default="point",
        help=(
            "Interpolation algorithm. 'point' (default) treats the map as "
            "discrete points and interpolates between the two bracketing "
            "points. 'interval' treats each map row as the start of a "
            "genomic interval with its own recombination rate and expects "
            "the cM column to be the cumulative position at the interval "
            "end; use this for interval-rate maps such as the deCODE map "
            "converted by convert_decode_map.py. 'hapmap' treats each map "
            "row as the start of an interval whose cM column is the "
            "cumulative position at the row itself; use this for pyrho "
            "HapMap-format maps."
        ),
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from ldetect_lite.interpolate_maps import (
        interpolate,
        interpolate_hapmap,
        interpolate_intervals,
    )

    fns = {
        "point": interpolate,
        "interval": interpolate_intervals,
        "hapmap": interpolate_hapmap,
    }
    fn = fns[args.mode]
    fn(
        snp_file=args.snp_file,
        genetic_map=args.genetic_map,
        output=args.output,
    )
    return 0
