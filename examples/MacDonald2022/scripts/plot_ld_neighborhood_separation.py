#!/usr/bin/env python
"""Plot LD separation around called block boundaries.

For each internal LD-block boundary, this diagnostic samples normalized
pairwise LD (r^2) from three local neighborhoods:

  left    pairs wholly in [boundary - window, boundary)
  across  pairs crossing the boundary, one SNP on each side
  right   pairs wholly in [boundary, boundary + window]

The resulting box/whisker plot is an orthogonal benchmark to exact boundary
matching: useful boundaries should tend to make the across-boundary LD
distribution lower than the within-neighborhood distributions.

Unlike `ldetect run --generate-ld-neighborhood-plot` (which only ever
samples around a run's own just-computed breakpoints), this script also
supports sampling around an arbitrary reference BED's boundaries -- used by
this example's "published" comparison rules to check LD separation quality
at the published ldetect blocks, not just our own. It also writes a
per-boundary summary TSV, consumed by `plot_ld_neighborhood_genomewide.py`
to build a cross-chromosome comparison.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

import numpy as np

from ldetect_lite.io.bed import Block, read_genome_bed
from ldetect_lite.io.covariance_hdf5 import open_covariance_reader
from ldetect_lite.io.partitions import CovarianceStore, read_partitions
from ldetect_lite.ld_neighborhood import (
    CATEGORY_ORDER,
    SeparationSamples,
    category_masks,
    owned_bounds,
    r2_for_pairs,
    read_diagonal_index,
    reservoir_extend,
    write_separation_boxplot,
)


def internal_boundaries(blocks: list[Block]) -> list[int]:
    return [end for _start, end in blocks[:-1]]


def chrom_aliases(chrom: str) -> list[str]:
    if chrom.startswith("chr"):
        bare = chrom.removeprefix("chr")
        return [chrom, bare]
    return [chrom, f"chr{chrom}"]


def resolve_chrom_blocks(
    genome_bed: dict[str, list[Block]],
    requested_chrom: str,
) -> tuple[str, list[Block]]:
    for chrom in chrom_aliases(requested_chrom):
        if chrom in genome_bed:
            return chrom, genome_bed[chrom]
    available = ", ".join(sorted(genome_bed)) or "none"
    raise RuntimeError(
        f"No blocks for {requested_chrom}; available BED chromosomes: {available}"
    )


def summarize(values: list[float]) -> dict[str, object]:
    if not values:
        return {
            "n": 0,
            "mean_r2": "",
            "median_r2": "",
            "q1_r2": "",
            "q3_r2": "",
            "p90_r2": "",
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_r2": f"{float(np.mean(arr)):.8g}",
        "median_r2": f"{float(np.median(arr)):.8g}",
        "q1_r2": f"{float(np.quantile(arr, 0.25)):.8g}",
        "q3_r2": f"{float(np.quantile(arr, 0.75)):.8g}",
        "p90_r2": f"{float(np.quantile(arr, 0.90)):.8g}",
    }


def boundary_rows_and_samples(
    *,
    chrom: str,
    boundaries: list[int],
    store: CovarianceStore,
    name: str,
    window_bp: int,
    sample_limit: int,
    seed: int,
    chunk_rows: int,
) -> tuple[list[dict[str, object]], SeparationSamples]:
    partitions = read_partitions(name, store)
    diag_pos, diag_val = read_diagonal_index(name, store, partitions)
    if diag_pos.size == 0:
        raise RuntimeError(f"No diagonal rows found for {name} in {store.root}")

    rows: list[dict[str, object]] = []
    samples: SeparationSamples = {category: [] for category in CATEGORY_ORDER}
    seen = {category: 0 for category in CATEGORY_ORDER}
    rng = random.Random(seed)

    for boundary in boundaries:
        left = boundary - window_bp
        right = boundary + window_bp
        per_boundary: dict[str, list[float]] = {
            category: [] for category in CATEGORY_ORDER
        }

        for p_index, (start, end) in enumerate(partitions):
            if end < left or start > right:
                continue
            lower_min, lower_max, include_lower_min = owned_bounds(
                partitions,
                p_index,
                left,
                right,
            )
            if lower_min > right or lower_max < left:
                continue
            path = store.partition_path(name, start, end)
            with open_covariance_reader(path, start, end) as reader:
                for chunk in reader.iter_owned_rows(
                    lower_min,
                    lower_max,
                    left,
                    right,
                    chunk_rows,
                    include_lower_min=include_lower_min,
                ):
                    masks = category_masks(chunk.lo, chunk.hi, boundary, left, right)
                    for category, mask in masks.items():
                        if not np.any(mask):
                            continue
                        values = r2_for_pairs(
                            chunk.lo[mask],
                            chunk.hi[mask],
                            chunk.shrink_ld[mask],
                            diag_pos,
                            diag_val,
                        )
                        if values.size == 0:
                            continue
                        per_boundary[category].extend(values.tolist())
                        seen[category] = reservoir_extend(
                            samples[category],
                            values,
                            seen=seen[category],
                            limit=sample_limit,
                            rng=rng,
                        )

        for category in CATEGORY_ORDER:
            row = {
                "chrom": chrom,
                "boundary": boundary,
                "category": category,
                "window_bp": window_bp,
            }
            row.update(summarize(per_boundary[category]))
            rows.append(row)

    return rows, samples


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "chrom",
        "boundary",
        "category",
        "window_bp",
        "n",
        "mean_r2",
        "median_r2",
        "q1_r2",
        "q3_r2",
        "p90_r2",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bed", required=True, type=Path)
    parser.add_argument("--covariance-root", required=True, type=Path)
    parser.add_argument("--chrom", required=True)
    parser.add_argument(
        "--covariance-name",
        default="",
        help=(
            "Covariance partition basename. Defaults to --chrom. Use this when "
            "the BED chromosome name and covariance cache name differ."
        ),
    )
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--plot", required=True, type=Path)
    parser.add_argument("--window-bp", type=int, default=500_000)
    parser.add_argument("--sample-limit", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    chrom, blocks = resolve_chrom_blocks(read_genome_bed(args.bed), args.chrom)
    title = args.title
    if not title:
        block_set = args.covariance_root.parent.name
        title = f"{block_set} {chrom}: LD neighborhood separation"

    if len(blocks) < 2:
        write_summary(args.output_tsv, [])
        write_separation_boxplot(
            args.plot,
            {category: [] for category in CATEGORY_ORDER},
            title=title,
        )
        print(f"{args.bed} has fewer than two blocks for {chrom}; wrote empty plot")
        print(f"Wrote {args.output_tsv} and {args.plot}")
        return

    rows, samples = boundary_rows_and_samples(
        chrom=chrom,
        boundaries=internal_boundaries(blocks),
        store=CovarianceStore(root=args.covariance_root),
        name=args.covariance_name or chrom,
        window_bp=args.window_bp,
        sample_limit=args.sample_limit,
        seed=args.seed,
        chunk_rows=args.chunk_rows,
    )
    write_summary(args.output_tsv, rows)
    write_separation_boxplot(args.plot, samples, title=title)

    aggregate = {
        category: statistics.median(values) if values else float("nan")
        for category, values in samples.items()
    }
    print(
        "median r2: "
        + ", ".join(
            f"{category}={aggregate[category]:.4g}" for category in CATEGORY_ORDER
        )
    )
    print(f"Wrote {args.output_tsv} and {args.plot}")


if __name__ == "__main__":
    main()
