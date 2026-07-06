"""Prototype: on-demand covariance recompute from source VCF (priority 5).

Local search currently reads *persisted* HDF5 covariance partitions for a
breakpoint's window (`get_final_partitions` / `local_search_hdf5_partition`
in `local_search.py`), inheriting whatever precision the bulk cache stores.
That's the last consumer standing between the bulk cache and being safe to
compress aggressively (see `notes/logs/covariance-cache-redesign-plan.md`).

The fix doesn't need new numerical code: `calc_covariance` is already
exactly the function `cmd_run.py`'s `_calc_partition` uses to generate one
partition on demand (tabix-slice the region, pipe into `calc_covariance`).
"On-demand recompute" is calling that same function again, later, for the
same `(start, end)` bounds a local-search call already needs via
`get_final_partitions` -- not a new kernel, and not a re-derivation of the
ASN22-relevant multi-partition boundary semantics (those live entirely in
`local_search.py`'s existing, already-fixed downstream reading code, which
this module never touches -- it only changes where the partition HDF5 file
comes from).

This is a prototype: not wired into `pipeline.py`/the CLI. See
`tests/test_local_search_vcf_recompute.py` for the two things that actually
need proving -- that recompute reproduces bit-identical rows to whatever was
persisted at generation time, and that `LocalSearch` itself makes identical
decisions when pointed at a recomputed partition instead of a cached one.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from ldetect_lite._util.logging import log_debug
from ldetect_lite.shrinkage import calc_covariance


def recompute_partition_to_hdf5(
    *,
    vcf_path: Path,
    genetic_map_path: Path,
    individuals_path: Path,
    chrom: str,
    start: int,
    end: int,
    output_path: Path,
    ne: float = 11418.0,
    cutoff: float = 1e-7,
    compression: str | None = "zstd",
) -> None:
    """Recompute one covariance partition from source VCF, on demand.

    Mirrors `cmd_run.py`'s `_calc_partition` tabix-to-`calc_covariance`
    invocation exactly, so the result is the same deterministic function of
    `(vcf slice, map, individuals, ne, cutoff)` a pre-generated partition
    already is -- just invoked at query time instead of ahead of time.
    """
    region = f"{chrom}:{start}-{end}"
    tabix_start = time.perf_counter()
    try:
        tabix_proc = subprocess.Popen(
            ["tabix", "-h", str(vcf_path), region],
            stdout=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "tabix not found. Install htslib and ensure tabix is on PATH."
        ) from exc

    stdout = tabix_proc.stdout
    if stdout is None:
        raise RuntimeError("tabix subprocess produced no stdout stream")

    with stdout:
        calc_covariance(
            vcf_stream=stdout,
            genetic_map_path=genetic_map_path,
            individuals_path=individuals_path,
            output_path=output_path,
            ne=ne,
            cutoff=cutoff,
            compact_output=True,
            compression=compression,
        )
    tabix_proc.wait()
    log_debug(
        "recompute_partition_to_hdf5 "
        f"chrom={chrom} start={start} end={end} "
        f"seconds={time.perf_counter() - tabix_start:.3f}"
    )
