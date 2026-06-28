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
        required=False,
        type=Path,
        metavar="PATH",
        help="Gzipped 8-column covariance output file.",
    )
    p.add_argument(
        "--vector-output",
        type=Path,
        metavar="PATH",
        help=(
            "Optional gzipped direct correlation-sum vector output "
            "(position, corr_sum), computed without materializing covariance rows."
        ),
    )
    p.add_argument(
        "--center-lower-bound",
        type=int,
        metavar="INT",
        help="Only write direct-vector center loci at or above this position.",
    )
    p.add_argument(
        "--center-lower-exclusive",
        action="store_true",
        help="Treat --center-lower-bound as exclusive for direct-vector output.",
    )
    p.add_argument(
        "--center-upper-bound",
        type=int,
        metavar="INT",
        help="Only write direct-vector center loci at or below this position.",
    )
    p.add_argument(
        "--center-upper-exclusive",
        action="store_true",
        help="Treat --center-upper-bound as exclusive for direct-vector output.",
    )
    p.add_argument(
        "--append-vector-output",
        action="store_true",
        help="Append direct-vector rows instead of truncating --vector-output.",
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
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from ldetect2.shrinkage import calc_covariance, calc_covariance_vector

    if args.output is None and args.vector_output is None:
        raise SystemExit("calc-covariance requires --output or --vector-output")

    if args.output is not None and args.vector_output is not None:
        import io

        vcf_text = sys.stdin.read()
        calc_covariance(
            vcf_stream=io.StringIO(vcf_text),
            genetic_map_path=args.genetic_map,
            individuals_path=args.individuals,
            output_path=args.output,
            ne=args.ne,
            cutoff=args.cutoff,
        )
        calc_covariance_vector(
            vcf_stream=io.StringIO(vcf_text),
            genetic_map_path=args.genetic_map,
            individuals_path=args.individuals,
            output_path=args.vector_output,
            ne=args.ne,
            cutoff=args.cutoff,
            center_lower_bound=args.center_lower_bound,
            center_lower_inclusive=not args.center_lower_exclusive,
            center_upper_bound=args.center_upper_bound,
            center_upper_inclusive=not args.center_upper_exclusive,
            append_output=args.append_vector_output,
        )
        return 0

    if args.output is not None:
        calc_covariance(
            vcf_stream=sys.stdin,
            genetic_map_path=args.genetic_map,
            individuals_path=args.individuals,
            output_path=args.output,
            ne=args.ne,
            cutoff=args.cutoff,
        )
    else:
        calc_covariance_vector(
            vcf_stream=sys.stdin,
            genetic_map_path=args.genetic_map,
            individuals_path=args.individuals,
            output_path=args.vector_output,
            ne=args.ne,
            cutoff=args.cutoff,
            center_lower_bound=args.center_lower_bound,
            center_lower_inclusive=not args.center_lower_exclusive,
            center_upper_bound=args.center_upper_bound,
            center_upper_inclusive=not args.center_upper_exclusive,
            append_output=args.append_vector_output,
        )
    return 0
