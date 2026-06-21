"""Summarise ldetect_original diagnostic inputs for one chromosome."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import statistics
import subprocess
from pathlib import Path

FIELDNAMES = [
    "population",
    "chrom",
    "raw_vcf_records",
    "raw_vcf_samples",
    "filtered_vcf_records",
    "filtered_vcf_samples",
    "map_rows",
    "map_first_pos",
    "map_last_pos",
    "map_cm_min",
    "map_cm_max",
    "map_cm_range",
    "map_inversions",
    "map_gap_median_cm",
    "map_gap_max_cm",
    "partitions",
    "partition_first_start",
    "partition_last_end",
    "partition_mean_bp",
    "partition_median_bp",
    "partition_min_bp",
    "partition_max_bp",
]


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _run_text(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def count_vcf_records(path: Path) -> str:
    return _run_text(["bcftools", "index", "-n", str(path)])


def count_vcf_samples(path: Path) -> str:
    samples = _run_text(["bcftools", "query", "-l", str(path)])
    if not samples:
        return "0"
    return str(len(samples.splitlines()))


def summarise_map(path: Path) -> dict[str, str]:
    positions: list[int] = []
    cms: list[float] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                pos = int(parts[1])
                cm = float(parts[2])
            except ValueError:
                continue
            positions.append(pos)
            cms.append(cm)

    if not positions:
        return {
            "map_rows": "0",
            "map_first_pos": "",
            "map_last_pos": "",
            "map_cm_min": "nan",
            "map_cm_max": "nan",
            "map_cm_range": "nan",
            "map_inversions": "0",
            "map_gap_median_cm": "nan",
            "map_gap_max_cm": "nan",
        }

    gaps = [b - a for a, b in zip(cms, cms[1:])]
    inversions = sum(gap < 0 for gap in gaps)
    return {
        "map_rows": str(len(positions)),
        "map_first_pos": str(positions[0]),
        "map_last_pos": str(positions[-1]),
        "map_cm_min": _fmt(min(cms)),
        "map_cm_max": _fmt(max(cms)),
        "map_cm_range": _fmt(max(cms) - min(cms)),
        "map_inversions": str(inversions),
        "map_gap_median_cm": _fmt(statistics.median(gaps) if gaps else math.nan),
        "map_gap_max_cm": _fmt(max(gaps) if gaps else math.nan),
    }


def summarise_partitions(path: Path) -> dict[str, str]:
    intervals: list[tuple[int, int]] = []
    with path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            start_s, end_s, *_ = line.split()
            intervals.append((int(start_s), int(end_s)))

    if not intervals:
        return {
            "partitions": "0",
            "partition_first_start": "",
            "partition_last_end": "",
            "partition_mean_bp": "nan",
            "partition_median_bp": "nan",
            "partition_min_bp": "nan",
            "partition_max_bp": "nan",
        }

    sizes = [end - start for start, end in intervals]
    return {
        "partitions": str(len(intervals)),
        "partition_first_start": str(intervals[0][0]),
        "partition_last_end": str(intervals[-1][1]),
        "partition_mean_bp": _fmt(statistics.mean(sizes)),
        "partition_median_bp": _fmt(statistics.median(sizes)),
        "partition_min_bp": str(min(sizes)),
        "partition_max_bp": str(max(sizes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", required=True)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--raw-vcf", required=True, type=Path)
    parser.add_argument("--filtered-vcf", required=True, type=Path)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--partitions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    row = {
        "population": args.population,
        "chrom": f"chr{args.chromosome.removeprefix('chr')}",
        "raw_vcf_records": count_vcf_records(args.raw_vcf),
        "raw_vcf_samples": count_vcf_samples(args.raw_vcf),
        "filtered_vcf_records": count_vcf_records(args.filtered_vcf),
        "filtered_vcf_samples": count_vcf_samples(args.filtered_vcf),
    }
    row.update(summarise_map(args.genetic_map))
    row.update(summarise_partitions(args.partitions))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
