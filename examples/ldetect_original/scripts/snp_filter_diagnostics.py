#!/usr/bin/env python3
"""Diagnose population-specific SNP filtering against an ldetect VCF stream.

Example:
    tabix -h data/raw/ALL.chr2...genotypes.vcf.gz 2:39967768-40067768 |
    uv run python scripts/snp_filter_diagnostics.py \
        --vcf - \
        --individuals resources/EUR_inds.txt \
        --genetic-map data/maps/chr2.interpolated_genetic_map.gz \
        --reference-cov tests/data/cov_matrix/chr2/chr2.39967768.40067768.gz
"""

from __future__ import annotations

import argparse
import gzip
import sys
from collections import Counter
from pathlib import Path
from typing import TextIO


def read_individuals(path: Path) -> list[str]:
    with open(path) as f:
        return [line.split()[0] for line in f if line.strip()]


def read_map_positions(path: Path) -> set[int]:
    positions: set[int] = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                positions.add(int(parts[1]))
    return positions


def read_reference_cov_positions(path: Path) -> set[int]:
    positions: set[int] = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                positions.add(int(parts[2]))
                positions.add(int(parts[3]))
    return positions


def open_vcf(path: str) -> TextIO:
    if path == "-":
        return sys.stdin
    if path.endswith((".gz", ".bgz")):
        return gzip.open(path, "rt")
    return open(path)


def summarize_vcf(
    vcf: TextIO,
    individuals: list[str],
    map_positions: set[int] | None,
) -> tuple[Counter[str], Counter[int], set[int], set[int]]:
    counts: Counter[str] = Counter()
    mac_hist: Counter[int] = Counter()
    polymorphic_positions: set[int] = set()
    all_positions: set[int] = set()
    ind2col: dict[str, int] = {}

    for raw in vcf:
        raw = raw.rstrip("\n")
        if raw.startswith("##"):
            continue

        parts = raw.split("\t")
        if raw.startswith("#CHROM"):
            for col_idx, sample in enumerate(parts[9:], start=9):
                if sample in individuals:
                    ind2col[sample] = col_idx
            missing = sorted(set(individuals) - set(ind2col))
            if missing:
                raise SystemExit(f"Missing {len(missing)} selected samples from VCF")
            continue

        counts["vcf_records"] += 1
        pos = int(parts[1])
        all_positions.add(pos)

        if map_positions is not None and pos not in map_positions:
            counts["not_in_map"] += 1
            continue
        counts["in_map"] += 1

        alleles: list[int] = []
        for ind in individuals:
            gt = parts[ind2col[ind]].split(":", 1)[0]
            if "|" not in gt or "." in gt:
                counts["unphased_or_missing"] += 1
                alleles = []
                break
            alleles.extend(int(a) for a in gt.split("|"))
        if not alleles:
            continue

        alt_count = sum(alleles)
        mac = min(alt_count, len(alleles) - alt_count)
        mac_hist[mac] += 1
        if mac == 0:
            counts["population_monomorphic"] += 1
        else:
            counts["population_polymorphic"] += 1
            polymorphic_positions.add(pos)

    return counts, mac_hist, all_positions, polymorphic_positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", required=True, help="VCF path, or '-' for stdin")
    parser.add_argument("--individuals", required=True, type=Path)
    parser.add_argument("--genetic-map", type=Path)
    parser.add_argument("--reference-cov", type=Path)
    parser.add_argument("--mac-bins", type=int, default=10)
    args = parser.parse_args()

    individuals = read_individuals(args.individuals)
    map_positions = read_map_positions(args.genetic_map) if args.genetic_map else None

    with open_vcf(args.vcf) as vcf:
        counts, mac_hist, all_positions, poly_positions = summarize_vcf(
            vcf, individuals, map_positions
        )

    print(f"selected_individuals\t{len(individuals)}")
    for key in (
        "vcf_records",
        "in_map",
        "not_in_map",
        "unphased_or_missing",
        "population_monomorphic",
        "population_polymorphic",
    ):
        print(f"{key}\t{counts[key]}")

    if args.reference_cov:
        ref_positions = read_reference_cov_positions(args.reference_cov)
        print(f"reference_cov_unique_positions\t{len(ref_positions)}")
        print(f"reference_overlap_all_vcf\t{len(ref_positions & all_positions)}")
        print(f"reference_overlap_polymorphic\t{len(ref_positions & poly_positions)}")
        print(f"polymorphic_not_in_reference\t{len(poly_positions - ref_positions)}")

    print("mac_histogram")
    for mac in range(args.mac_bins + 1):
        print(f"{mac}\t{mac_hist[mac]}")
    higher = sum(n for mac, n in mac_hist.items() if mac > args.mac_bins)
    print(f">{args.mac_bins}\t{higher}")


if __name__ == "__main__":
    main()
