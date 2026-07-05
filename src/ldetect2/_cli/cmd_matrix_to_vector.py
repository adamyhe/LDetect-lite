"""CLI: matrix-to-vector subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "matrix-to-vector",
        help="Convert covariance matrix partitions to a correlation-sum vector.",
    )
    p.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        metavar="PATH",
        help="Root directory of the covariance matrix dataset.",
    )
    p.add_argument(
        "--name", required=True, metavar="TEXT", help="Chromosome name (e.g. chr2)."
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Gzipped output vector file (position, corr_sum).",
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
        "--mode",
        choices=["diag", "vert"],
        default="diag",
        help="Calculation mode (default: diag).",
    )
    p.add_argument(
        "--generate-heatmap",
        action="store_true",
        help="Write a PNG heatmap alongside the output file.",
    )
    p.add_argument(
        "--matrix-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Parallel workers for HDF5 matrix-to-vector partition computation "
            "(default: 1)."
        ),
    )
    p.add_argument(
        "--prefer-signal-cache",
        action="store_true",
        help=(
            "Assemble the vector purely from precomputed signal sidecars "
            "(written by `calc-covariance --signal-output`), skipping the "
            "pair-level covariance read-and-renormalize pass. Every "
            "partition must have a sidecar; incompatible with "
            "--generate-heatmap."
        ),
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from ldetect2.io.partitions import CovarianceStore
    from ldetect2.matrix_analysis import MatrixAnalysis

    store = CovarianceStore(root=args.dataset_path)
    analysis = MatrixAnalysis(
        name=args.name,
        store=store,
        snp_first=args.snp_first,
        snp_last=args.snp_last,
    )

    if args.mode == "diag":
        if args.generate_heatmap:
            if args.prefer_signal_cache:
                raise ValueError(
                    "--prefer-signal-cache cannot be combined with "
                    "--generate-heatmap, which requires the full matrix"
                )
            # Full matrix needed for heatmap — use non-lean path
            analysis.calc_diag()
            analysis.write_output_to_file(args.output)
        elif args.prefer_signal_cache:
            analysis.calc_diag_signal(args.output)
        else:
            analysis.calc_diag_lean(args.output, matrix_workers=args.matrix_workers)
    else:
        raise NotImplementedError("vert mode is deprecated; use diag")

    if args.generate_heatmap:
        img_path = args.output.with_suffix(".png")
        analysis.generate_img(img_path)

    return 0
