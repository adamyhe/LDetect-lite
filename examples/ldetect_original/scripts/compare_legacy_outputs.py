"""Compare ldetect2 and vendored legacy ldetect diagnostic outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from compare_blocks import compare_chrom

from ldetect2.io.bed import read_genome_bed


def _vector_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    with gzip.open(path, "rb") as f:
        for raw in f:
            digest.update(raw)
            if raw.strip():
                rows += 1
    return rows, digest.hexdigest()


def _read_loci(path: Path, subset: str) -> list[int]:
    data = json.loads(path.read_text())
    return [int(x) for x in data[subset]["loci"]]


def _compare_blocks_for_chrom(
    ours: Path, ref: Path, chrom: str, tolerance: int
) -> dict:
    ours_by_chrom = read_genome_bed(ours)
    ref_by_chrom = read_genome_bed(ref)
    chrom_key = f"chr{chrom}"
    return compare_chrom(
        chrom_key,
        ours_by_chrom.get(chrom_key, ours_by_chrom.get(chrom, [])),
        ref_by_chrom.get(chrom_key, ref_by_chrom.get(chrom, [])),
        tolerance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--ldetect2-vector", required=True, type=Path)
    parser.add_argument("--legacy-vector", required=True, type=Path)
    parser.add_argument("--ldetect2-breakpoints", required=True, type=Path)
    parser.add_argument("--legacy-breakpoints", required=True, type=Path)
    parser.add_argument("--ldetect2-bed", required=True, type=Path)
    parser.add_argument("--legacy-bed", required=True, type=Path)
    parser.add_argument("--reference-bed", required=True, type=Path)
    parser.add_argument("--subset", default="fourier_ls")
    parser.add_argument("--tolerance", type=int, default=100_000)
    parser.add_argument("--ldetect2-vs-legacy", required=True, type=Path)
    parser.add_argument("--legacy-vs-reference", required=True, type=Path)
    args = parser.parse_args()

    l2_rows, l2_hash = _vector_digest(args.ldetect2_vector)
    legacy_rows, legacy_hash = _vector_digest(args.legacy_vector)
    l2_loci = _read_loci(args.ldetect2_breakpoints, args.subset)
    legacy_loci = _read_loci(args.legacy_breakpoints, args.subset)
    paired = list(zip(l2_loci, legacy_loci))
    identical_loci = sum(a == b for a, b in paired)
    bed_row = _compare_blocks_for_chrom(
        args.ldetect2_bed, args.legacy_bed, args.chromosome, args.tolerance
    )

    args.ldetect2_vs_legacy.parent.mkdir(parents=True, exist_ok=True)
    with args.ldetect2_vs_legacy.open("w", newline="") as f:
        cols = [
            "chrom",
            "vector_rows_equal",
            "vector_sha256_equal",
            "ldetect2_vector_rows",
            "legacy_vector_rows",
            "ldetect2_vector_sha256",
            "legacy_vector_sha256",
            "ldetect2_n_loci",
            "legacy_n_loci",
            "identical_loci",
            "all_loci_equal",
            "bed_recall",
            "bed_precision",
            "bed_jaccard",
            "bed_bp_jaccard",
            "bed_our_median_offset_kb",
            "bed_our_p90_offset_kb",
        ]
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "chrom": f"chr{args.chromosome}",
                "vector_rows_equal": l2_rows == legacy_rows,
                "vector_sha256_equal": l2_hash == legacy_hash,
                "ldetect2_vector_rows": l2_rows,
                "legacy_vector_rows": legacy_rows,
                "ldetect2_vector_sha256": l2_hash,
                "legacy_vector_sha256": legacy_hash,
                "ldetect2_n_loci": len(l2_loci),
                "legacy_n_loci": len(legacy_loci),
                "identical_loci": identical_loci,
                "all_loci_equal": l2_loci == legacy_loci,
                "bed_recall": bed_row["recall"],
                "bed_precision": bed_row["precision"],
                "bed_jaccard": bed_row["jaccard"],
                "bed_bp_jaccard": bed_row["bp_jaccard"],
                "bed_our_median_offset_kb": bed_row["our_median_offset_kb"],
                "bed_our_p90_offset_kb": bed_row["our_p90_offset_kb"],
            }
        )

    row = _compare_blocks_for_chrom(
        args.legacy_bed, args.reference_bed, args.chromosome, args.tolerance
    )
    cols = [
        "chrom",
        "our_n",
        "ref_n",
        "our_mean_kb",
        "ref_mean_kb",
        "our_median_kb",
        "ref_median_kb",
        "recall",
        "precision",
        "jaccard",
        "our_median_offset_kb",
        "our_p90_offset_kb",
        "ref_median_offset_kb",
        "ref_p90_offset_kb",
        "bp_jaccard",
    ]
    args.legacy_vs_reference.parent.mkdir(parents=True, exist_ok=True)
    with args.legacy_vs_reference.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in cols})


if __name__ == "__main__":
    main()
