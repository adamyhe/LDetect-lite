#!/usr/bin/env python
"""Diagnose LD-block boundary mismatches against a reference BED.

The output contains one row for each boundary whose nearest counterpart is
farther away than ``--tolerance``. Reciprocal nearest boundaries are classified
as shifted boundaries. Non-reciprocal internal boundaries are classified as
extra or missing splits, depending on the comparison direction.

Optional genetic-map and SNP-position inputs add local map and marker-density
context without changing the mismatch classification.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from ldetect2._util.intervals import boundaries
from ldetect2.io.bed import Block, read_genome_bed


def open_text(path: Path) -> TextIO:
    if path.suffix.lower() in {".gz", ".bgz", ".gzip"}:
        return gzip.open(path, "rt")
    return path.open()


def chrom_key(chrom: str) -> tuple[int, int | str]:
    value = chrom.removeprefix("chr")
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def nearest(position: int, positions: list[int]) -> tuple[int | None, int | None]:
    if not positions:
        return None, None

    index = bisect.bisect_left(positions, position)
    candidates = positions[max(0, index - 1) : index + 1]
    nearest_position = min(
        candidates,
        key=lambda candidate: (abs(candidate - position), candidate),
    )
    return nearest_position, nearest_position - position


def adjacent_blocks(
    position: int,
    blocks: list[Block],
) -> tuple[Block | None, Block | None]:
    left = None
    right = None
    for block in blocks:
        if block[1] == position:
            left = block
        if block[0] == position:
            right = block
    return left, right


def containing_block(position: int, blocks: list[Block]) -> Block | None:
    for block in blocks:
        if block[0] < position < block[1]:
            return block
    return None


def format_block(block: Block | None) -> str:
    if block is None:
        return ""
    return f"{block[0]}-{block[1]}"


def format_blocks(blocks: Iterable[Block]) -> str:
    return ";".join(format_block(block) for block in blocks)


def blocks_in_window(
    position: int,
    blocks: list[Block],
    window: int,
) -> list[Block]:
    start = position - window
    end = position + window
    return [block for block in blocks if block[1] >= start and block[0] <= end]


def classify_mismatch(
    source: str,
    position: int,
    query_boundaries: list[int],
    reference_boundaries: list[int],
    reference_blocks: list[Block],
) -> str:
    nearest_position, _ = nearest(position, reference_boundaries)
    if nearest_position is None:
        return "unmatched_boundary"

    reciprocal_position, _ = nearest(nearest_position, query_boundaries)
    if reciprocal_position == position:
        return "shifted_boundary"

    query_is_edge = position in {query_boundaries[0], query_boundaries[-1]}
    if query_is_edge:
        return "chromosome_edge_mismatch"

    if containing_block(position, reference_blocks) is not None:
        if source == "ours_to_ref":
            return "extra_split"
        return "missing_split"

    return "nonreciprocal_boundary"


def read_centromeres(path: Path | None) -> dict[str, tuple[int, int]]:
    if path is None:
        return {}

    intervals: dict[str, tuple[int, int]] = {}
    with open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            if parts[0].startswith("chr"):
                chrom_index = 0
            elif len(parts) >= 4 and parts[1].startswith("chr"):
                chrom_index = 1
            else:
                continue
            try:
                chrom = parts[chrom_index]
                start = int(parts[chrom_index + 1])
                end = int(parts[chrom_index + 2])
            except ValueError:
                continue
            if chrom in intervals:
                old_start, old_end = intervals[chrom]
                intervals[chrom] = min(old_start, start), max(old_end, end)
            else:
                intervals[chrom] = start, end
    return intervals


def centromere_context(
    chrom: str,
    position: int,
    centromeres: dict[str, tuple[int, int]],
) -> tuple[str, str, str, str]:
    interval = centromeres.get(chrom)
    if interval is None:
        return "", "", "", ""
    start, end = interval
    overlaps = start <= position <= end
    distance = 0 if overlaps else min(abs(position - start), abs(position - end))
    return str(start), str(end), str(int(overlaps)), str(distance)


def read_genetic_maps(
    paths: list[Path],
) -> dict[str, list[tuple[int, float]]]:
    maps: dict[str, list[tuple[int, float]]] = {}
    for path in paths:
        with open_text(path) as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    position = int(parts[1])
                    cm = float(parts[2])
                except ValueError:
                    continue
                maps.setdefault(parts[0], []).append((position, cm))
    for values in maps.values():
        values.sort()
    return maps


def read_snp_positions(paths: list[Path]) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for path in paths:
        with open_text(path) as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    position = int(parts[1])
                except ValueError:
                    continue
                positions.setdefault(parts[0], []).append(position)
    for values in positions.values():
        values.sort()
    return positions


def local_position_context(
    position: int,
    positions: list[int],
    window: int,
) -> tuple[str, str, str]:
    if not positions:
        return "", "", ""
    left = bisect.bisect_left(positions, position - window)
    right = bisect.bisect_right(positions, position + window)
    nearest_position, offset = nearest(position, positions)
    return (
        str(right - left),
        "" if nearest_position is None else str(nearest_position),
        "" if offset is None else str(abs(offset)),
    )


def local_map_context(
    position: int,
    genetic_map: list[tuple[int, float]],
    window: int,
) -> tuple[str, str, str, str, str]:
    if not genetic_map:
        return "", "", "", "", ""

    map_positions = [item[0] for item in genetic_map]
    left = bisect.bisect_left(map_positions, position - window)
    right = bisect.bisect_right(map_positions, position + window)
    local = genetic_map[left:right]
    nearest_position, offset = nearest(position, map_positions)
    nearest_cm = ""
    if nearest_position is not None:
        nearest_index = bisect.bisect_left(map_positions, nearest_position)
        nearest_cm = str(genetic_map[nearest_index][1])
    cm_span = ""
    if len(local) >= 2:
        cm_span = str(local[-1][1] - local[0][1])
    return (
        str(len(local)),
        "" if nearest_position is None else str(nearest_position),
        "" if offset is None else str(abs(offset)),
        nearest_cm,
        cm_span,
    )


def diagnostic_rows(
    ours: dict[str, list[Block]],
    reference: dict[str, list[Block]],
    tolerance: int,
    window: int,
    selected_chroms: set[str],
    centromeres: dict[str, tuple[int, int]],
    genetic_maps: dict[str, list[tuple[int, float]]],
    snp_positions: dict[str, list[int]],
) -> Iterable[dict[str, object]]:
    chroms = sorted(set(ours) | set(reference), key=chrom_key)
    for chrom in chroms:
        if selected_chroms and chrom not in selected_chroms:
            continue

        our_blocks = ours.get(chrom, [])
        ref_blocks = reference.get(chrom, [])
        our_boundaries = boundaries(our_blocks)
        ref_boundaries = boundaries(ref_blocks)
        directions = [
            (
                "ours_to_ref",
                our_boundaries,
                ref_boundaries,
                our_blocks,
                ref_blocks,
            ),
            (
                "ref_to_ours",
                ref_boundaries,
                our_boundaries,
                ref_blocks,
                our_blocks,
            ),
        ]
        for source, query_bounds, ref_bounds, query_blocks, target_blocks in directions:
            for position in query_bounds:
                nearest_position, signed_offset = nearest(position, ref_bounds)
                if signed_offset is not None and abs(signed_offset) <= tolerance:
                    continue

                query_left, query_right = adjacent_blocks(position, query_blocks)
                ref_left, ref_right = adjacent_blocks(
                    nearest_position if nearest_position is not None else position,
                    target_blocks,
                )
                cent_start, cent_end, in_centromere, cent_distance = (
                    centromere_context(chrom, position, centromeres)
                )
                map_count, map_nearest, map_distance, map_cm, map_cm_span = (
                    local_map_context(
                        position,
                        genetic_maps.get(chrom, []),
                        window,
                    )
                )
                snp_count, snp_nearest, snp_distance = local_position_context(
                    position,
                    snp_positions.get(chrom, []),
                    window,
                )
                yield {
                    "chrom": chrom,
                    "source": source,
                    "position": position,
                    "nearest_position": (
                        nearest_position if nearest_position is not None else ""
                    ),
                    "signed_offset_bp": (
                        signed_offset if signed_offset is not None else ""
                    ),
                    "abs_offset_bp": (
                        abs(signed_offset) if signed_offset is not None else ""
                    ),
                    "classification": classify_mismatch(
                        source,
                        position,
                        query_bounds,
                        ref_bounds,
                        target_blocks,
                    ),
                    "query_left_block": format_block(query_left),
                    "query_right_block": format_block(query_right),
                    "nearest_ref_left_block": format_block(ref_left),
                    "nearest_ref_right_block": format_block(ref_right),
                    "query_blocks_in_window": format_blocks(
                        blocks_in_window(position, query_blocks, window)
                    ),
                    "ref_blocks_in_window": format_blocks(
                        blocks_in_window(position, target_blocks, window)
                    ),
                    "centromere_start": cent_start,
                    "centromere_end": cent_end,
                    "position_in_centromere": in_centromere,
                    "distance_to_centromere_bp": cent_distance,
                    "map_points_in_window": map_count,
                    "nearest_map_position": map_nearest,
                    "nearest_map_distance_bp": map_distance,
                    "nearest_map_cm": map_cm,
                    "map_cm_span_in_window": map_cm_span,
                    "snps_in_window": snp_count,
                    "nearest_snp_position": snp_nearest,
                    "nearest_snp_distance_bp": snp_distance,
                }


FIELDNAMES = [
    "chrom",
    "source",
    "position",
    "nearest_position",
    "signed_offset_bp",
    "abs_offset_bp",
    "classification",
    "query_left_block",
    "query_right_block",
    "nearest_ref_left_block",
    "nearest_ref_right_block",
    "query_blocks_in_window",
    "ref_blocks_in_window",
    "centromere_start",
    "centromere_end",
    "position_in_centromere",
    "distance_to_centromere_bp",
    "map_points_in_window",
    "nearest_map_position",
    "nearest_map_distance_bp",
    "nearest_map_cm",
    "map_cm_span_in_window",
    "snps_in_window",
    "nearest_snp_position",
    "nearest_snp_distance_bp",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=int, default=50_000)
    parser.add_argument("--window", type=int, default=2_000_000)
    parser.add_argument(
        "--chrom",
        action="append",
        default=[],
        help="Restrict diagnostics to a chromosome; may be repeated.",
    )
    parser.add_argument("--centromeres", type=Path)
    parser.add_argument(
        "--genetic-map",
        action="append",
        default=[],
        type=Path,
        help="Three-column chrom/position/cM map; may be repeated.",
    )
    parser.add_argument(
        "--snp-positions",
        action="append",
        default=[],
        type=Path,
        help="Two-column chrom/position file; may be repeated.",
    )
    args = parser.parse_args()

    rows = diagnostic_rows(
        ours=read_genome_bed(args.ours),
        reference=read_genome_bed(args.ref),
        tolerance=args.tolerance,
        window=args.window,
        selected_chroms=set(args.chrom),
        centromeres=read_centromeres(args.centromeres),
        genetic_maps=read_genetic_maps(args.genetic_map),
        snp_positions=read_snp_positions(args.snp_positions),
    )

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
