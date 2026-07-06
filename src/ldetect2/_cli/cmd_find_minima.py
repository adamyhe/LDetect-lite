"""CLI: find-minima subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path

_VALID_SUBSETS = ("fourier", "fourier_ls", "uniform", "uniform_ls")


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "find-minima",
        help="Detect LD block breakpoints via Hanning filter + local search.",
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help="Gzipped vector file from matrix-to-vector.",
    )
    p.add_argument(
        "--chr-name", required=True, metavar="TEXT", help="Chromosome name (e.g. chr2)."
    )
    p.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        metavar="PATH",
        help="Root directory of the covariance matrix dataset.",
    )
    p.add_argument(
        "--n-snps-bw-bpoints",
        required=True,
        type=int,
        metavar="N",
        help="Target mean SNPs between breakpoints (default: 10000).",
    )
    p.add_argument(
        "--output", required=True, type=Path, metavar="PATH", help="JSON output file."
    )
    p.add_argument(
        "--snp-first",
        type=int,
        default=-1,
        metavar="INT",
        help="First SNP position (auto-detected if omitted).",
    )
    p.add_argument(
        "--snp-last",
        type=int,
        default=-1,
        metavar="INT",
        help="Last SNP position (auto-detected if omitted).",
    )
    p.add_argument(
        "--trackback-delta",
        type=int,
        default=200,
        metavar="INT",
        help="Coarse trackback search range (default: 200).",
    )
    p.add_argument(
        "--trackback-step",
        type=int,
        default=20,
        metavar="INT",
        help="Coarse trackback step size (default: 20).",
    )
    p.add_argument(
        "--init-search-loc",
        type=int,
        default=1000,
        metavar="INT",
        help="Starting width for exponential search (default: 1000).",
    )
    p.add_argument(
        "--workers",
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
        "--metric-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel workers for streaming metric row passes "
            "(default: inherit --workers)."
        ),
    )
    p.add_argument(
        "--high-precision",
        action="store_true",
        help="Use 50-digit Decimal arithmetic for local search (slower).",
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
        action="append",
        default=None,
        metavar="SUBSET",
        help=(
            "Breakpoint subset to compute. Repeat to compute multiple subsets. "
            "By default, all subsets are computed for backward compatibility."
        ),
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from ldetect2._cli.cmd_run import _resolve_workers
    from ldetect2.io.partitions import CovarianceStore
    from ldetect2.pipeline import find_breakpoints

    metric_workers = _resolve_workers(args.metric_workers, args.workers)

    store = CovarianceStore(root=args.dataset_path)
    find_breakpoints(
        input_path=args.input,
        chr_name=args.chr_name,
        store=store,
        n_snps_bw_bpoints=args.n_snps_bw_bpoints,
        output_path=args.output,
        snp_first=args.snp_first,
        snp_last=args.snp_last,
        trackback_delta=args.trackback_delta,
        trackback_step=args.trackback_step,
        init_search_location=args.init_search_loc,
        workers=args.workers,
        metric_workers=metric_workers,
        use_decimal=args.high_precision,
        n_bpoints=args.n_bpoints,
        subsets=set(args.subset) if args.subset else None,
    )
    return 0
