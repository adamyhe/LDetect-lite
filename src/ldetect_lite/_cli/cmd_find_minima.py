"""CLI: find-minima subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path

_VALID_SUBSETS = ("fourier", "fourier_ls", "uniform", "uniform_ls", "dp")


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
        "--filter-window",
        choices=("symmetric", "scipy-periodic"),
        default="symmetric",
        help=(
            "Hanning window mode for breakpoint filtering. 'symmetric' uses "
            "np.hanning and is the default: it matches the window Berisa & "
            "Pickrell (2016) specify in their supplement, is the "
            "conventional choice for a windowed-FIR/convolution filter "
            "kernel, and reproduces published reference blocks generated "
            "under scipy <1.1. 'scipy-periodic' matches modern "
            "scipy.signal.get_window(..., fftbins=True) behavior; use it to "
            "reproduce modern-scipy-era published output (e.g. MacDonald et "
            "al. 2022) (default: symmetric)."
        ),
    )
    p.add_argument(
        "--filter-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Minimum Numba threads for adaptive single-filter convolutions. "
            "Trackback keeps candidate-width threading and forces each "
            "candidate filter to one Numba thread to avoid nested parallelism "
            "(default: 1)."
        ),
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
            "By default, the four LDetect subsets are computed for backward "
            "compatibility ('dp' must be requested explicitly)."
        ),
    )
    p.add_argument(
        "--dp-min-size",
        type=int,
        default=1,
        metavar="N",
        help="Minimum SNPs per block for the 'dp' subset (default: 1).",
    )
    p.add_argument(
        "--dp-max-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum SNPs per block for the 'dp' subset "
            "(default: the full requested region, i.e. no limit)."
        ),
    )
    p.add_argument(
        "--dp-max-k",
        type=int,
        default=500,
        metavar="N",
        help=(
            "Solve the 'dp' subset for every block count from 1 up to this "
            "many in one pass (default: 500)."
        ),
    )
    p.add_argument(
        "--dp-candidate-mode",
        choices=("filter", "all"),
        default="filter",
        help=(
            "'filter' (default) restricts 'dp' candidate breakpoints to the "
            "local minima of a small, fixed-width Hann filter "
            "(--dp-candidate-width); 'all' allows a cut at every SNP with "
            "covariance data, which is exact over every position but more "
            "expensive."
        ),
    )
    p.add_argument(
        "--dp-candidate-width",
        type=int,
        default=25,
        metavar="N",
        help=(
            "Half-width of the dense candidate-generating Hann filter used "
            "when --dp-candidate-mode=filter (default: 25)."
        ),
    )
    p.add_argument(
        "--dp-thr-r2",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help=(
            "Ignore pairs with r^2 below this threshold in the 'dp' "
            "objective (default: 0.0, i.e. no filtering)."
        ),
    )
    p.add_argument(
        "--dp-max-r2",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help=(
            "Forbid any 'dp' breakpoint that would separate a pair whose "
            "r^2 exceeds this (default: 1.0, i.e. no constraint)."
        ),
    )
    p.add_argument(
        "--dp-min-size-bp",
        type=int,
        default=None,
        metavar="BP",
        help=(
            "Minimum physical (bp) width per block for the 'dp' subset "
            "(default: no minimum). A block must satisfy this and "
            "--dp-min-size."
        ),
    )
    p.add_argument(
        "--dp-max-size-bp",
        type=int,
        default=None,
        metavar="BP",
        help=(
            "Maximum physical (bp) width per block for the 'dp' subset "
            "(default: no maximum). A block must satisfy this and "
            "--dp-max-size."
        ),
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from ldetect_lite._cli.cmd_run import _resolve_workers
    from ldetect_lite.io.partitions import CovarianceStore
    from ldetect_lite.pipeline import find_breakpoints

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
        filter_window=args.filter_window,
        filter_workers=args.filter_workers,
        dp_min_size=args.dp_min_size,
        dp_max_size=args.dp_max_size,
        dp_max_k=args.dp_max_k,
        dp_candidate_mode=args.dp_candidate_mode,
        dp_candidate_width=args.dp_candidate_width,
        dp_thr_r2=args.dp_thr_r2,
        dp_max_r2=args.dp_max_r2,
        dp_min_size_bp=args.dp_min_size_bp,
        dp_max_size_bp=args.dp_max_size_bp,
    )
    return 0
