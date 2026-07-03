#!/usr/bin/env python3
"""Check whether redundant cross-partition covariance pairs actually match.

Partitions deliberately overlap (see ``ldetect2.shrinkage.partition_chromosome``),
so the same canonical ``(lo, hi)`` SNP-position pair is independently,
redundantly computed by two adjacent partitions' ``calc_covariance`` runs.
Every downstream consumer (matrix-to-vector, metric, local search) resolves
this by picking one partition's value rather than averaging or re-deriving
it, on the assumption that both partitions computed the *same* number in the
first place (the shrinkage formula for a given pair depends only on
genotypes/Ne/genetic distance, never on which partition computed it). This
script checks that assumption directly against real, materialized covariance
partitions, and separately reports how many duplicate physical VCF positions
(two VCF records sharing the same POS) fall inside an overlap zone, since
that is the combination most likely to expose an edge case.

NOTE: this cannot be run end-to-end in a fresh checkout — it requires real,
already-materialized HDF5 covariance partitions for the target chromosome
(e.g. from a completed `ldetect2 run` or the main `examples/ldetect_original`
Snakefile). It is unit-tested against a synthetic fixture with a deliberately
mismatched redundant pair; see
tests/test_covariance_io.py::test_compare_partition_overlap_duplicates_flags_mismatch.

Usage:
    uv run python scripts/compare_partition_overlap_duplicates.py \
        --population EUR \
        --chromosome 10 \
        --store-root results/EUR/10 \
        --name 10 \
        --vcf-path resources/provenance/v3/filtered_vcf/v3/all/EUR/chr10.vcf.gz \
        --output results/provenance_diagnostics/EUR/chr10/duplicate_overlap_report.tsv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
from pathlib import Path

from ldetect2.io.covariance_hdf5 import open_covariance_reader
from ldetect2.io.partitions import CovarianceStore, read_partitions

FIELDNAMES = [
    "population",
    "chrom",
    "n_partitions",
    "n_overlapping_partition_pairs",
    "n_duplicate_vcf_positions",
    "n_duplicate_positions_in_overlap_zone",
    "n_redundant_pairs_checked",
    "n_redundant_pairs_bit_identical",
    "n_redundant_pairs_mismatched",
    "max_abs_shrink_diff",
    "mean_abs_shrink_diff",
    "first_mismatch_lo",
    "first_mismatch_hi",
    "first_mismatch_value_a",
    "first_mismatch_value_b",
]

_CHUNK_ROWS = 1_000_000


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def vcf_positions(path: Path) -> list[int]:
    out = _run(["bcftools", "query", "-f", "%POS\n", str(path)])
    return [int(line) for line in out.splitlines() if line]


def duplicate_positions(positions: list[int]) -> set[int]:
    seen: set[int] = set()
    dupes: set[int] = set()
    for pos in positions:
        if pos in seen:
            dupes.add(pos)
        seen.add(pos)
    return dupes


def overlapping_partition_pairs(
    partitions: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs = []
    for (start_a, end_a), (start_b, end_b) in zip(partitions, partitions[1:]):
        if start_b <= end_a:
            pairs.append(((start_a, end_a), (start_b, end_b)))
    return pairs


def _read_overlap_rows(
    store: CovarianceStore,
    name: str,
    partition: tuple[int, int],
    overlap_lo: int,
    overlap_hi: int,
) -> dict[tuple[int, int], float]:
    """Read every row of *partition* whose ``lo`` falls in the overlap zone."""
    start, end = partition
    rows: dict[tuple[int, int], float] = {}
    path = store.partition_path(name, start, end)
    with open_covariance_reader(path, start, end) as reader:
        for chunk in reader.iter_rows(overlap_lo, overlap_hi, _CHUNK_ROWS):
            for lo, hi, shrink in zip(
                chunk.lo.tolist(), chunk.hi.tolist(), chunk.shrink_ld.tolist()
            ):
                rows[(lo, hi)] = shrink
    return rows


def compare(args: argparse.Namespace) -> dict[str, str]:
    store = CovarianceStore(root=args.store_root)
    partitions = read_partitions(args.name, store)
    overlaps = overlapping_partition_pairs(partitions)

    n_checked = 0
    n_identical = 0
    abs_diffs: list[float] = []
    first_mismatch: tuple[int, int, float, float] | None = None
    duplicate_positions_in_overlap = 0

    all_positions = vcf_positions(args.vcf_path) if args.vcf_path else []
    dupes = duplicate_positions(all_positions)

    for partition_a, partition_b in overlaps:
        overlap_lo, overlap_hi = partition_b[0], partition_a[1]
        rows_a = _read_overlap_rows(
            store, args.name, partition_a, overlap_lo, overlap_hi
        )
        rows_b = _read_overlap_rows(
            store, args.name, partition_b, overlap_lo, overlap_hi
        )

        for key in rows_a.keys() & rows_b.keys():
            n_checked += 1
            value_a, value_b = rows_a[key], rows_b[key]
            if value_a == value_b:
                n_identical += 1
            else:
                diff = abs(value_a - value_b)
                abs_diffs.append(diff)
                if first_mismatch is None:
                    first_mismatch = (key[0], key[1], value_a, value_b)

        duplicate_positions_in_overlap += sum(
            1 for pos in dupes if overlap_lo <= pos <= overlap_hi
        )

    return {
        "population": args.population,
        "chrom": f"chr{args.chromosome.removeprefix('chr')}",
        "n_partitions": str(len(partitions)),
        "n_overlapping_partition_pairs": str(len(overlaps)),
        "n_duplicate_vcf_positions": str(len(dupes)) if args.vcf_path else "nan",
        "n_duplicate_positions_in_overlap_zone": (
            str(duplicate_positions_in_overlap) if args.vcf_path else "nan"
        ),
        "n_redundant_pairs_checked": str(n_checked),
        "n_redundant_pairs_bit_identical": str(n_identical),
        "n_redundant_pairs_mismatched": str(n_checked - n_identical),
        "max_abs_shrink_diff": f"{max(abs_diffs):.12g}" if abs_diffs else "0",
        "mean_abs_shrink_diff": (
            f"{statistics.mean(abs_diffs):.12g}" if abs_diffs else "0"
        ),
        "first_mismatch_lo": str(first_mismatch[0]) if first_mismatch else "",
        "first_mismatch_hi": str(first_mismatch[1]) if first_mismatch else "",
        "first_mismatch_value_a": f"{first_mismatch[2]:.12g}" if first_mismatch else "",
        "first_mismatch_value_b": f"{first_mismatch[3]:.12g}" if first_mismatch else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", required=True)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument(
        "--name", required=True, help="Chromosome label used by CovarianceStore"
    )
    parser.add_argument(
        "--vcf-path",
        type=Path,
        default=None,
        help="Filtered VCF to count duplicate physical positions from (optional)",
    )
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
