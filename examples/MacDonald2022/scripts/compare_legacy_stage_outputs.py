"""Compare ldetect-lite and legacy-from-VCF diagnostic outputs for one chrom."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from compare_blocks import compare_chrom

from ldetect_lite.io.bed import read_genome_bed


def _vector_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    with gzip.open(path, "rb") as f:
        for raw in f:
            digest.update(raw)
            if raw.strip():
                rows += 1
    return rows, digest.hexdigest()


def _compare_vectors_numeric(
    left: Path,
    right: Path,
    tolerance: float,
) -> dict[str, int | float | str | bool]:
    left_rows = 0
    right_rows = 0
    shared_rows = 0
    position_mismatches = 0
    value_mismatches = 0
    max_abs_diff = 0.0
    first_diff_position = ""
    first_left_value = ""
    first_right_value = ""

    with gzip.open(left, "rt") as left_f, gzip.open(right, "rt") as right_f:
        for left_raw, right_raw in zip(left_f, right_f, strict=False):
            left_parts = left_raw.split()
            right_parts = right_raw.split()
            if left_parts:
                left_rows += 1
            if right_parts:
                right_rows += 1
            if not left_parts or not right_parts:
                continue
            shared_rows += 1
            if left_parts[0] != right_parts[0]:
                position_mismatches += 1
                if not first_diff_position:
                    first_diff_position = f"{left_parts[0]}!={right_parts[0]}"
                continue
            left_val = float(left_parts[1])
            right_val = float(right_parts[1])
            diff = abs(left_val - right_val)
            max_abs_diff = max(max_abs_diff, diff)
            if diff > tolerance:
                value_mismatches += 1
                if not first_diff_position:
                    first_diff_position = left_parts[0]
                    first_left_value = repr(left_val)
                    first_right_value = repr(right_val)
    return {
        "vector_numeric_equal": (
            left_rows == right_rows
            and position_mismatches == 0
            and value_mismatches == 0
        ),
        "vector_shared_rows": shared_rows,
        "vector_position_mismatches": position_mismatches,
        "vector_value_mismatches": value_mismatches,
        "vector_max_abs_diff": max_abs_diff,
        "vector_first_diff_position": first_diff_position,
        "vector_first_lite_value": first_left_value,
        "vector_first_legacy_value": first_right_value,
    }


def _read_loci(path: Path, subset: str) -> list[int]:
    data = json.loads(path.read_text())
    return [int(x) for x in data[subset]["loci"]]


def _breakpoint_row(
    lite_path: Path,
    legacy_path: Path,
    subset: str,
) -> dict[str, int | bool]:
    lite_loci = _read_loci(lite_path, subset)
    legacy_loci = _read_loci(legacy_path, subset)
    paired = list(zip(lite_loci, legacy_loci))
    return {
        f"{subset}_ldetect_lite_n_loci": len(lite_loci),
        f"{subset}_legacy_n_loci": len(legacy_loci),
        f"{subset}_identical_loci": sum(a == b for a, b in paired),
        f"{subset}_all_loci_equal": lite_loci == legacy_loci,
    }


def _compare_blocks_for_chrom(
    ours: Path,
    ref: Path,
    chrom: str,
    tolerance: int,
) -> dict:
    ours_by_chrom = read_genome_bed(ours)
    ref_by_chrom = read_genome_bed(ref)
    chrom_key = chrom if chrom.startswith("chr") else f"chr{chrom}"
    chrom_plain = chrom_key.removeprefix("chr")
    return compare_chrom(
        chrom_key,
        ours_by_chrom.get(chrom_key, ours_by_chrom.get(chrom_plain, [])),
        ref_by_chrom.get(chrom_key, ref_by_chrom.get(chrom_plain, [])),
        tolerance,
    )


def _write_single_row(path: Path, cols: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in cols})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--ldetect-lite-vector", required=True, type=Path)
    parser.add_argument("--legacy-vector", required=True, type=Path)
    parser.add_argument("--ldetect-lite-breakpoints", required=True, type=Path)
    parser.add_argument("--legacy-breakpoints", required=True, type=Path)
    parser.add_argument("--ldetect-lite-raw-bed", required=True, type=Path)
    parser.add_argument("--legacy-raw-bed", required=True, type=Path)
    parser.add_argument("--legacy-final-bed", required=True, type=Path)
    parser.add_argument("--reference-bed", required=True, type=Path)
    parser.add_argument("--subset", default="fourier_ls")
    parser.add_argument("--tolerance", type=int, default=50_000)
    parser.add_argument("--vector-tolerance", type=float, default=1e-9)
    parser.add_argument("--raw-ldetect-lite-vs-legacy", required=True, type=Path)
    parser.add_argument("--legacy-final-vs-reference", required=True, type=Path)
    args = parser.parse_args()

    lite_rows, lite_hash = _vector_digest(args.ldetect_lite_vector)
    legacy_rows, legacy_hash = _vector_digest(args.legacy_vector)
    vector_numeric = _compare_vectors_numeric(
        args.ldetect_lite_vector,
        args.legacy_vector,
        args.vector_tolerance,
    )
    subset_compare = _breakpoint_row(
        args.ldetect_lite_breakpoints,
        args.legacy_breakpoints,
        args.subset,
    )
    all_subset_compare = {}
    for subset_name in ("fourier", "fourier_ls", "uniform", "uniform_ls"):
        all_subset_compare.update(
            _breakpoint_row(
                args.ldetect_lite_breakpoints,
                args.legacy_breakpoints,
                subset_name,
            )
        )
    raw_bed_row = _compare_blocks_for_chrom(
        args.ldetect_lite_raw_bed,
        args.legacy_raw_bed,
        args.chromosome,
        args.tolerance,
    )

    raw_cols = [
        "chrom",
        "vector_rows_equal",
        "vector_sha256_equal",
        "vector_numeric_equal",
        "ldetect_lite_vector_rows",
        "legacy_vector_rows",
        "vector_shared_rows",
        "vector_position_mismatches",
        "vector_value_mismatches",
        "vector_max_abs_diff",
        "vector_first_diff_position",
        "vector_first_lite_value",
        "vector_first_legacy_value",
        "ldetect_lite_vector_sha256",
        "legacy_vector_sha256",
        "ldetect_lite_n_loci",
        "legacy_n_loci",
        "identical_loci",
        "all_loci_equal",
        "fourier_ldetect_lite_n_loci",
        "fourier_legacy_n_loci",
        "fourier_identical_loci",
        "fourier_all_loci_equal",
        "fourier_ls_ldetect_lite_n_loci",
        "fourier_ls_legacy_n_loci",
        "fourier_ls_identical_loci",
        "fourier_ls_all_loci_equal",
        "uniform_ldetect_lite_n_loci",
        "uniform_legacy_n_loci",
        "uniform_identical_loci",
        "uniform_all_loci_equal",
        "uniform_ls_ldetect_lite_n_loci",
        "uniform_ls_legacy_n_loci",
        "uniform_ls_identical_loci",
        "uniform_ls_all_loci_equal",
        "raw_bed_recall",
        "raw_bed_precision",
        "raw_bed_jaccard",
        "raw_bed_bp_jaccard",
        "raw_bed_our_median_offset_kb",
        "raw_bed_our_p90_offset_kb",
    ]
    _write_single_row(
        args.raw_ldetect_lite_vs_legacy,
        raw_cols,
        {
            "chrom": args.chromosome,
            "vector_rows_equal": lite_rows == legacy_rows,
            "vector_sha256_equal": lite_hash == legacy_hash,
            **vector_numeric,
            "ldetect_lite_vector_rows": lite_rows,
            "legacy_vector_rows": legacy_rows,
            "ldetect_lite_vector_sha256": lite_hash,
            "legacy_vector_sha256": legacy_hash,
            "ldetect_lite_n_loci": subset_compare[
                f"{args.subset}_ldetect_lite_n_loci"
            ],
            "legacy_n_loci": subset_compare[f"{args.subset}_legacy_n_loci"],
            "identical_loci": subset_compare[f"{args.subset}_identical_loci"],
            "all_loci_equal": subset_compare[f"{args.subset}_all_loci_equal"],
            **all_subset_compare,
            "raw_bed_recall": raw_bed_row["recall"],
            "raw_bed_precision": raw_bed_row["precision"],
            "raw_bed_jaccard": raw_bed_row["jaccard"],
            "raw_bed_bp_jaccard": raw_bed_row["bp_jaccard"],
            "raw_bed_our_median_offset_kb": raw_bed_row["our_median_offset_kb"],
            "raw_bed_our_p90_offset_kb": raw_bed_row["our_p90_offset_kb"],
        },
    )

    final_row = _compare_blocks_for_chrom(
        args.legacy_final_bed,
        args.reference_bed,
        args.chromosome,
        args.tolerance,
    )
    final_cols = [
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
    _write_single_row(args.legacy_final_vs_reference, final_cols, final_row)


if __name__ == "__main__":
    main()
