"""Utility helpers for the end-to-end run pipeline."""

from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

from ldetect2.io.covariance_hdf5 import validate_covariance_hdf5


def calc_partition(
    start: int,
    end: int,
    chrom: str,
    reference_panel: str,
    genetic_map_path: Path,
    individuals_path: Path,
    output_path: Path,
    ne: float,
    cutoff: float,
    compact_output: bool,
    pair_cache: str = "hdf5",
    vector_output_path: Path | None = None,
    center_lower_bound: int | None = None,
    center_lower_inclusive: bool = True,
    center_upper_bound: int | None = None,
    center_upper_inclusive: bool = True,
) -> None:
    """Run one partition's tabix stream through the selected cache writer."""
    from ldetect2._util.memory import log_memory_checkpoint
    from ldetect2.shrinkage import (
        calc_covariance,
        calc_covariance_vector,
        calc_r2_zarr_partition,
    )

    region = f"{chrom}:{start}-{end}"
    tabix_proc = open_tabix_process(reference_panel, region)

    if pair_cache == "r2-zarr":
        if vector_output_path is None:
            raise RuntimeError("r2-zarr pair cache requires a direct vector fragment")
        with tabix_proc.stdout:  # type: ignore[union-attr]
            calc_r2_zarr_partition(
                vcf_stream=tabix_proc.stdout,
                genetic_map_path=genetic_map_path,
                individuals_path=individuals_path,
                output_root=output_path,
                name=chrom,
                start=start,
                end=end,
                ne=ne,
                cutoff=cutoff,
                vector_output_path=vector_output_path,
                center_lower_bound=center_lower_bound,
                center_lower_inclusive=center_lower_inclusive,
                center_upper_bound=center_upper_bound,
                center_upper_inclusive=center_upper_inclusive,
            )
    elif vector_output_path is not None:
        with tabix_proc.stdout:  # type: ignore[union-attr]
            calc_covariance(
                vcf_stream=tabix_proc.stdout,
                genetic_map_path=genetic_map_path,
                individuals_path=individuals_path,
                output_path=output_path,
                ne=ne,
                cutoff=cutoff,
                compact_output=compact_output,
            )
        tabix_proc.wait()
        tabix_proc = open_tabix_process(reference_panel, region)
        with tabix_proc.stdout:  # type: ignore[union-attr]
            calc_covariance_vector(
                vcf_stream=tabix_proc.stdout,
                genetic_map_path=genetic_map_path,
                individuals_path=individuals_path,
                output_path=vector_output_path,
                ne=ne,
                cutoff=cutoff,
                center_lower_bound=center_lower_bound,
                center_lower_inclusive=center_lower_inclusive,
                center_upper_bound=center_upper_bound,
                center_upper_inclusive=center_upper_inclusive,
            )
    else:
        with tabix_proc.stdout:  # type: ignore[union-attr]
            calc_covariance(
                vcf_stream=tabix_proc.stdout,
                genetic_map_path=genetic_map_path,
                individuals_path=individuals_path,
                output_path=output_path,
                ne=ne,
                cutoff=cutoff,
                compact_output=compact_output,
            )
    tabix_proc.wait()
    log_memory_checkpoint(f"covariance_partition_end start={start} end={end}")


def open_tabix_process(reference_panel: str, region: str) -> subprocess.Popen:
    """Open a tabix process streaming *region* with header lines."""
    try:
        return subprocess.Popen(
            ["tabix", "-h", reference_panel, region],
            stdout=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "tabix not found. Install htslib and ensure tabix is on PATH."
        )


def direct_vector_plan(
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
) -> dict[tuple[int, int], tuple[int, bool, int | None, bool]]:
    """Return ownership bounds for direct vector fragments."""
    bounds: dict[tuple[int, int], tuple[int, bool, int | None, bool]] = {}
    previous_end_locus: int | None = None
    for p_index, (start, end) in enumerate(partitions):
        if previous_end_locus is None:
            lower_bound = snp_first
            lower_inclusive = True
        else:
            lower_bound = previous_end_locus
            lower_inclusive = False

        if p_index + 1 < len(partitions):
            upper_bound = int((end + partitions[p_index + 1][0]) / 2)
            upper_inclusive = True
        else:
            upper_bound = None
            upper_inclusive = False
        bounds[(start, end)] = (
            lower_bound,
            lower_inclusive,
            upper_bound,
            upper_inclusive,
        )
        previous_end_locus = (
            int((end + partitions[p_index + 1][0]) / 2)
            if p_index + 1 < len(partitions)
            else snp_last
        )
    return bounds


def concatenate_direct_vector_fragments(
    fragments: list[Path],
    output_path: Path,
) -> None:
    """Concatenate per-partition direct vector fragments into one gzip TSV."""
    output_path.unlink(missing_ok=True)
    with gzip.open(output_path, "wt") as out:
        for fragment in fragments:
            if not fragment.exists():
                continue
            with gzip.open(fragment, "rt") as inp:
                for line in inp:
                    out.write(line)


def count_individuals(path: Path) -> int:
    """Count non-empty individual rows."""
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def breakpoint_subsets_for_run(
    subset: str, all_breakpoint_subsets: bool
) -> set[str] | None:
    """Return the breakpoint subset request passed from ``run`` to the pipeline.

    ``None`` intentionally preserves the full historical JSON output; otherwise
    ``run`` asks the pipeline to compute only the final BED subset and its
    dependencies.
    """
    return None if all_breakpoint_subsets else {subset}


def is_valid_covariance_partition(path: Path, require_full: bool = True) -> bool:
    """Return whether a cached covariance partition satisfies the needed schema."""
    return validate_covariance_hdf5(path, require_full=require_full)
