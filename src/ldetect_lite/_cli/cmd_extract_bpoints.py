"""CLI: extract-bpoints subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_VALID_SUBSETS = ("fourier", "fourier_ls", "uniform", "uniform_ls", "dp")


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "extract-bpoints",
        help="Extract breakpoints from a find-minima JSON file and write a BED file.",
    )
    p.add_argument(
        "--name", required=True, metavar="TEXT", help="Chromosome name (e.g. chr2)."
    )
    p.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        metavar="PATH",
        help="Root directory of the covariance matrix dataset.",
    )
    p.add_argument(
        "--breakpoints",
        required=True,
        type=Path,
        metavar="PATH",
        help="JSON file from find-minima.",
    )
    p.add_argument(
        "--subset",
        required=True,
        choices=_VALID_SUBSETS,
        metavar="SUBSET",
        help=f"Breakpoint set to extract: {{{', '.join(_VALID_SUBSETS)}}}.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output BED file (default: stdout).",
    )
    p.add_argument(
        "--dp-n-block",
        type=int,
        default=None,
        metavar="K",
        help="Required when --subset=dp: which block count's solution to extract.",
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    import sys

    from ldetect_lite.io.bed import write_bed
    from ldetect_lite.io.partitions import CovarianceStore, read_partitions

    store = CovarianceStore(root=args.dataset_path)
    partitions = read_partitions(args.name, store)
    snp_first = partitions[0][0]
    snp_last = partitions[-1][1]

    data = json.loads(args.breakpoints.read_text())
    if args.subset == "dp":
        if args.dp_n_block is None:
            print("Error: --dp-n-block is required when --subset=dp.", file=sys.stderr)
            return 1
        candidates = data["dp"]["candidates"]
        matching = [c for c in candidates if c["n_block"] == args.dp_n_block]
        if not matching:
            available = ", ".join(str(c["n_block"]) for c in candidates) or "(none)"
            print(
                f"Error: no dp solution for --dp-n-block={args.dp_n_block}. "
                f"Available block counts: {available}",
                file=sys.stderr,
            )
            return 1
        loci: list[int] = matching[0]["loci"]
    else:
        loci = data[args.subset]["loci"]

    write_bed(
        name=args.name,
        loci=loci,
        snp_first=snp_first,
        snp_last=snp_last,
        output=args.output,
    )
    return 0
