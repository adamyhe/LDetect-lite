#!/usr/bin/env python
"""Compare published LD-block boundaries with raw and local-search breakpoints."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

from ldetect_lite.io.bed import Block, read_genome_bed


def normalize_chrom(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def internal_boundaries(blocks: list[Block]) -> list[int]:
    positions: set[int] = set()
    starts = {start for start, _end in blocks}
    ends = {end for _start, end in blocks}
    for position in starts & ends:
        positions.add(position)
    return sorted(positions)


def nearest(position: int, candidates: list[int]) -> tuple[int | str, int | str]:
    if not candidates:
        return "", ""
    index = bisect.bisect_left(candidates, position)
    nearby = candidates[max(0, index - 1) : index + 1]
    nearest_position = min(nearby, key=lambda value: (abs(value - position), value))
    return nearest_position, nearest_position - position


def read_stage_loci(path: Path, subset: str) -> list[int]:
    data = json.loads(path.read_text())
    return [int(position) for position in data[subset]["loci"]]


def classify_stage(
    fourier_abs_offset: int | str,
    fourier_ls_abs_offset: int | str,
    tolerance: int,
) -> str:
    if not isinstance(fourier_abs_offset, int) or not isinstance(
        fourier_ls_abs_offset, int
    ):
        return "missing_stage"

    fourier_close = fourier_abs_offset <= tolerance
    fourier_ls_close = fourier_ls_abs_offset <= tolerance
    if fourier_close and fourier_ls_close:
        return "both_close"
    if fourier_close and not fourier_ls_close:
        return "local_search_moved_away"
    if not fourier_close and fourier_ls_close:
        return "local_search_improved"
    if fourier_ls_abs_offset < fourier_abs_offset:
        return "local_search_improved_but_still_far"
    if fourier_ls_abs_offset > fourier_abs_offset:
        return "local_search_worsened_already_shifted"
    return "fourier_already_shifted"


def row_for_boundary(
    chrom: str,
    ref_boundary: int,
    fourier_loci: list[int],
    fourier_ls_loci: list[int],
    tolerance: int,
) -> dict[str, int | str]:
    nearest_fourier, fourier_signed = nearest(ref_boundary, fourier_loci)
    nearest_fourier_ls, fourier_ls_signed = nearest(ref_boundary, fourier_ls_loci)
    fourier_abs = abs(fourier_signed) if isinstance(fourier_signed, int) else ""
    fourier_ls_abs = (
        abs(fourier_ls_signed) if isinstance(fourier_ls_signed, int) else ""
    )
    return {
        "chrom": chrom,
        "reference_boundary": ref_boundary,
        "nearest_fourier": nearest_fourier,
        "fourier_signed_offset_bp": fourier_signed,
        "fourier_abs_offset_bp": fourier_abs,
        "fourier_within_tolerance": (
            int(fourier_abs <= tolerance) if isinstance(fourier_abs, int) else 0
        ),
        "nearest_fourier_ls": nearest_fourier_ls,
        "fourier_ls_signed_offset_bp": fourier_ls_signed,
        "fourier_ls_abs_offset_bp": fourier_ls_abs,
        "fourier_ls_within_tolerance": (
            int(fourier_ls_abs <= tolerance) if isinstance(fourier_ls_abs, int) else 0
        ),
        "local_search_delta_abs_bp": (
            fourier_ls_abs - fourier_abs
            if isinstance(fourier_abs, int) and isinstance(fourier_ls_abs, int)
            else ""
        ),
        "stage_class": classify_stage(fourier_abs, fourier_ls_abs, tolerance),
    }


FIELDNAMES = [
    "chrom",
    "reference_boundary",
    "nearest_fourier",
    "fourier_signed_offset_bp",
    "fourier_abs_offset_bp",
    "fourier_within_tolerance",
    "nearest_fourier_ls",
    "fourier_ls_signed_offset_bp",
    "fourier_ls_abs_offset_bp",
    "fourier_ls_within_tolerance",
    "local_search_delta_abs_bp",
    "stage_class",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bed", required=True, type=Path)
    parser.add_argument("--breakpoints", required=True, type=Path)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=int, default=50_000)
    args = parser.parse_args()

    chrom = normalize_chrom(args.chromosome)
    reference_by_chrom = read_genome_bed(args.reference_bed)
    chrom_plain = chrom.removeprefix("chr")
    reference_blocks = reference_by_chrom.get(
        chrom,
        reference_by_chrom.get(chrom_plain, []),
    )
    if not reference_blocks:
        raise SystemExit(f"No reference blocks for {chrom} in {args.reference_bed}")

    fourier_loci = read_stage_loci(args.breakpoints, "fourier")
    fourier_ls_loci = read_stage_loci(args.breakpoints, "fourier_ls")
    rows = [
        row_for_boundary(
            chrom,
            boundary,
            fourier_loci,
            fourier_ls_loci,
            args.tolerance,
        )
        for boundary in internal_boundaries(reference_blocks)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
