"""Compare filtered VCF position sets for provenance diagnostics."""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import Counter
from pathlib import Path

FIELDNAMES = [
    "population",
    "chrom",
    "baseline_label",
    "candidate_label",
    "baseline_records",
    "candidate_records",
    "baseline_unique_positions",
    "candidate_unique_positions",
    "shared_positions",
    "baseline_only_positions",
    "candidate_only_positions",
    "position_jaccard",
    "baseline_duplicate_positions",
    "candidate_duplicate_positions",
]


def read_positions(path: Path) -> Counter[int]:
    result = subprocess.run(
        ["bcftools", "query", "-f", "%POS\n", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    counts: Counter[int] = Counter()
    for line in result.stdout.splitlines():
        if line:
            counts[int(line)] += 1
    return counts


def fmt_float(value: float) -> str:
    return f"{value:.12g}"


def compare(args: argparse.Namespace) -> dict[str, str]:
    baseline_counts = read_positions(args.baseline_vcf)
    candidate_counts = read_positions(args.candidate_vcf)
    baseline = set(baseline_counts)
    candidate = set(candidate_counts)
    union = baseline | candidate
    shared = baseline & candidate

    return {
        "population": args.population,
        "chrom": f"chr{args.chromosome.removeprefix('chr')}",
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "baseline_records": str(sum(baseline_counts.values())),
        "candidate_records": str(sum(candidate_counts.values())),
        "baseline_unique_positions": str(len(baseline)),
        "candidate_unique_positions": str(len(candidate)),
        "shared_positions": str(len(shared)),
        "baseline_only_positions": str(len(baseline - candidate)),
        "candidate_only_positions": str(len(candidate - baseline)),
        "position_jaccard": fmt_float(len(shared) / len(union)) if union else "nan",
        "baseline_duplicate_positions": str(
            sum(count - 1 for count in baseline_counts.values() if count > 1)
        ),
        "candidate_duplicate_positions": str(
            sum(count - 1 for count in candidate_counts.values() if count > 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", required=True)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-vcf", required=True, type=Path)
    parser.add_argument("--candidate-vcf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    row = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
