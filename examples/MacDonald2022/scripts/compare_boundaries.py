#!/usr/bin/env python
"""Write nearest-boundary offsets for two genome-wide BED files."""

from __future__ import annotations

import argparse
import bisect
import csv
from pathlib import Path

from ldetect2._util.intervals import boundaries
from ldetect2.io.bed import read_genome_bed


def nearest_boundary(
    position: int,
    reference: list[int],
) -> tuple[int | str, int | str]:
    if not reference:
        return "", ""

    index = bisect.bisect_left(reference, position)
    candidates: list[int] = []
    if index < len(reference):
        candidates.append(reference[index])
    if index > 0:
        candidates.append(reference[index - 1])
    nearest = min(candidates, key=lambda value: (abs(value - position), value))
    return nearest, nearest - position


def iter_offset_rows(
    chrom: str,
    source: str,
    query: list[int],
    reference: list[int],
    tolerance: int,
):
    for position in query:
        nearest, signed_offset = nearest_boundary(position, reference)
        abs_offset = abs(signed_offset) if isinstance(signed_offset, int) else ""
        within_tolerance = (
            abs_offset <= tolerance if isinstance(abs_offset, int) else False
        )
        yield {
            "chrom": chrom,
            "source": source,
            "position": position,
            "nearest_position": nearest,
            "signed_offset_bp": signed_offset,
            "abs_offset_bp": abs_offset,
            "within_tolerance": int(within_tolerance),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=int, default=50_000)
    args = parser.parse_args()

    ours = read_genome_bed(args.ours)
    ref = read_genome_bed(args.ref)
    chroms = sorted(set(ours) | set(ref), key=lambda value: (len(value), value))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chrom",
        "source",
        "position",
        "nearest_position",
        "signed_offset_bp",
        "abs_offset_bp",
        "within_tolerance",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for chrom in chroms:
            our_bounds = boundaries(ours.get(chrom, []))
            ref_bounds = boundaries(ref.get(chrom, []))
            writer.writerows(
                iter_offset_rows(
                    chrom,
                    "ours_to_ref",
                    our_bounds,
                    ref_bounds,
                    args.tolerance,
                )
            )
            writer.writerows(
                iter_offset_rows(
                    chrom,
                    "ref_to_ours",
                    ref_bounds,
                    our_bounds,
                    args.tolerance,
                )
            )


if __name__ == "__main__":
    main()
