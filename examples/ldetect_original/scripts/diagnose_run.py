"""Summarise ldetect2 diagnostic runs and combine per-chromosome summaries."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

SUMMARY_COLS = [
    "population",
    "chrom",
    "vector_rows",
    "vector_first_locus",
    "vector_last_locus",
    "vector_sha256",
    "vector_min",
    "vector_max",
    "vector_mean",
    "vector_sd",
    "n_bpoints",
    "found_width",
    "fourier_n",
    "fourier_ls_n",
    "uniform_n",
    "uniform_ls_n",
    "fourier_to_fourier_ls_exact",
    "uniform_to_uniform_ls_exact",
    "cov_partitions",
    "cov_rows",
    "cov_schema",
    "cov_bytes",
]

COMPARISON_COLS = [
    "our_n",
    "ref_n",
    "recall",
    "precision",
    "jaccard",
    "our_median_offset_kb",
    "our_p90_offset_kb",
    "bp_jaccard",
]


def _fmt_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.12g}"


def summarise_vector(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    values: list[float] = []
    first_locus = ""
    last_locus = ""
    with gzip.open(path, "rb") as raw:
        for line in raw:
            digest.update(line)
            parts = line.decode().strip().split()
            if not parts:
                continue
            locus = parts[0]
            if first_locus == "":
                first_locus = locus
            last_locus = locus
            values.append(float(parts[1]))

    if not values:
        return {
            "vector_rows": "0",
            "vector_first_locus": "",
            "vector_last_locus": "",
            "vector_sha256": digest.hexdigest(),
            "vector_min": "nan",
            "vector_max": "nan",
            "vector_mean": "nan",
            "vector_sd": "nan",
        }

    return {
        "vector_rows": str(len(values)),
        "vector_first_locus": first_locus,
        "vector_last_locus": last_locus,
        "vector_sha256": digest.hexdigest(),
        "vector_min": _fmt_float(min(values)),
        "vector_max": _fmt_float(max(values)),
        "vector_mean": _fmt_float(statistics.mean(values)),
        "vector_sd": _fmt_float(statistics.pstdev(values)),
    }


def summarise_breakpoints(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    fourier = _subset_loci(data, "fourier")
    fourier_ls = _subset_loci(data, "fourier_ls")
    uniform = _subset_loci(data, "uniform")
    uniform_ls = _subset_loci(data, "uniform_ls")
    return {
        "n_bpoints": str(data["n_bpoints"]),
        "found_width": str(data["found_width"]),
        "fourier_n": _subset_count(fourier),
        "fourier_ls_n": _subset_count(fourier_ls),
        "uniform_n": _subset_count(uniform),
        "uniform_ls_n": _subset_count(uniform_ls),
        "fourier_to_fourier_ls_exact": _pairwise_exact_count(fourier, fourier_ls),
        "uniform_to_uniform_ls_exact": _pairwise_exact_count(uniform, uniform_ls),
    }


def _subset_loci(data: dict, subset: str) -> list[int] | None:
    if subset not in data:
        return None
    return data[subset]["loci"]


def _subset_count(loci: list[int] | None) -> str:
    return "" if loci is None else str(len(loci))


def _pairwise_exact_count(left: list[int] | None, right: list[int] | None) -> str:
    if left is None or right is None:
        return ""
    return str(sum(a == b for a, b in zip(left, right)))


def summarise_covariance(run_dir: Path, chrom: str) -> dict[str, str]:
    cov_dir = run_dir / chrom
    if not cov_dir.exists():
        cov_dir = run_dir / "cov_matrix" / chrom
    partitions = sorted(cov_dir.glob(f"{chrom}.*.*.npz"))
    rows = 0
    schemas: set[str] = set()
    total_bytes = 0
    for path in partitions:
        total_bytes += path.stat().st_size
        with np.load(path) as data:
            rows += len(data["i_pos"])
            keys = set(data.files)
            if {"i_pos", "j_pos", "shrink_ld"} == keys:
                schemas.add("compact")
            elif {"i_pos", "j_pos", "shrink_ld"}.issubset(keys):
                schemas.add("full")
            else:
                schemas.add("unknown")
    return {
        "cov_partitions": str(len(partitions)),
        "cov_rows": str(rows),
        "cov_schema": ",".join(sorted(schemas)),
        "cov_bytes": str(total_bytes),
    }


def write_run_summary(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    chrom = args.chromosome
    row = {"population": args.population, "chrom": f"chr{chrom}"}
    row.update(summarise_vector(run_dir / f"vector-{chrom}.txt.gz"))
    row.update(summarise_breakpoints(run_dir / f"breakpoints-{chrom}.json"))
    row.update(summarise_covariance(run_dir, chrom))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLS, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


def _read_single_summary(path: Path) -> dict[str, str]:
    with path.open() as f:
        return next(csv.DictReader(f, delimiter="\t"))


def _read_matching_comparison(path: Path, chrom: str) -> dict[str, str]:
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["chrom"] == chrom:
                return row
    return {"chrom": chrom}


def write_combined(args: argparse.Namespace) -> None:
    rows: list[dict[str, str]] = []
    summary_by_chrom: dict[str, dict[str, str]] = {}
    for summary_path, comparison_path in zip(
        args.combine_summaries, args.combine_comparisons
    ):
        summary = _read_single_summary(summary_path)
        chrom = summary["chrom"]
        comparison = _read_matching_comparison(comparison_path, chrom)
        row = dict(summary)
        for col in COMPARISON_COLS:
            row[f"compare_{col}"] = comparison.get(col, "")
        rows.append(row)
        summary_by_chrom[chrom.removeprefix("chr")] = row

    cols = SUMMARY_COLS + [f"compare_{col}" for col in COMPARISON_COLS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    case = summary_by_chrom.get(str(args.case_chromosome), {})
    control = summary_by_chrom.get(str(args.control_chromosome), {})
    cc_cols = ["field", "case", "control"]
    compare_fields = [
        "vector_rows",
        "vector_sha256",
        "found_width",
        "fourier_n",
        "fourier_ls_n",
        "cov_partitions",
        "cov_rows",
        "compare_recall",
        "compare_precision",
        "compare_jaccard",
        "compare_our_median_offset_kb",
        "compare_our_p90_offset_kb",
    ]
    with args.case_vs_control.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cc_cols, delimiter="\t")
        writer.writeheader()
        for field in compare_fields:
            writer.writerow(
                {
                    "field": field,
                    "case": case.get(field, ""),
                    "control": control.get(field, ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--chromosome")
    parser.add_argument("--population")
    parser.add_argument("--combine-summaries", nargs="+", type=Path)
    parser.add_argument("--combine-comparisons", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-vs-control", type=Path)
    parser.add_argument("--case-chromosome")
    parser.add_argument("--control-chromosome")
    args = parser.parse_args()

    if args.combine_summaries:
        write_combined(args)
    else:
        write_run_summary(args)


if __name__ == "__main__":
    main()
