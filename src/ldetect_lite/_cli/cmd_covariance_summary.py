"""CLI: covariance-summary subcommand."""

from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "covariance-summary",
        help="Estimate covariance array memory from partition row counts.",
    )
    p.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        metavar="PATH",
        help="Root directory of the covariance matrix dataset.",
    )
    p.add_argument(
        "--name",
        required=True,
        metavar="TEXT",
        help="Chromosome name (e.g. chr2).",
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
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write summary to a file instead of stdout.",
    )
    p.add_argument(
        "--format",
        choices=["tsv", "json"],
        default="tsv",
        help="Output format (default: tsv).",
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from ldetect_lite._util.covariance_summary import (
        SUMMARY_COLUMNS,
        summarize_covariance,
    )
    from ldetect_lite.io.partitions import CovarianceStore

    store = CovarianceStore(root=args.dataset_path)
    partitions, total = summarize_covariance(
        args.name,
        store,
        snp_first=args.snp_first,
        snp_last=args.snp_last,
    )
    rows = [summary.as_dict() for summary in partitions]
    rows.append(total.as_dict())

    if args.format == "json":
        text = json.dumps({"partitions": rows[:-1], "total": rows[-1]}, indent=2)
    else:
        text = _format_tsv(rows, SUMMARY_COLUMNS)

    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


def _format_tsv(rows: list[dict[str, str | int | float]], columns: list[str]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().rstrip("\n")
