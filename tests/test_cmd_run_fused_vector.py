"""Integration test for --fused-vector's Step 2/3 wiring in cmd_run.py.

Exercises the actual new glue (`_calc_partition`'s `vector_plan`/`snp_last`
params, `_fused_vector_ready`, and the `_merge_diag_vector_partition_result`
loop `_run()`'s Step 3 uses when the fused path is eligible) rather than
just the already-validated underlying sidecar math
(`tests/test_covariance_sidecars.py`). `_calc_partition` shells out to a real
`tabix` subprocess, so this builds a small real bgzip+tabix-indexed
synthetic VCF rather than an in-memory stream.

Reuses the overlap dataset from `test_covariance_sidecars.py` (two
overlapping partitions, 10 individuals, 16 SNPs) so the fixture data is
already characterized rather than inventing a fourth copy of similar
genotype-generation logic.
"""

from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

from ldetect_lite._cli.cmd_run import _calc_partition, _fused_vector_ready
from ldetect_lite._util.vector_array import (
    _merge_diag_vector_partition_result,
    _plan_diag_vector_partitions,
)
from ldetect_lite.io.partitions import CovarianceStore
from ldetect_lite.matrix_analysis import MatrixAnalysis
from tests.test_covariance_sidecars import (
    _OVERLAP_PARTITIONS,
    _overlap_genotype_rows,
    _overlap_individuals,
    _overlap_positions,
)


def _write_indexed_vcf(path: Path, individuals: list[str]) -> None:
    genotype_rows = _overlap_genotype_rows()
    positions = _overlap_positions()
    header = (
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(individuals)
        + "\n"
    )
    body = "\n".join(
        f"1\t{pos}\trs{pos}\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(genotype_rows[pos])
        for pos in positions
    )
    raw_path = path.with_suffix("")
    raw_path.write_text(f"{header}{body}\n")
    subprocess.run(["bgzip", "-f", str(raw_path)], check=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", str(path)], check=True)


def test_fused_vector_matches_posthoc_step3_read(tmp_path: Path) -> None:
    individuals = _overlap_individuals()
    vcf_path = tmp_path / "panel.vcf.gz"
    _write_indexed_vcf(vcf_path, individuals)

    map_path = tmp_path / "map.gz"
    with gzip.open(map_path, "wt") as f:
        for i, pos in enumerate(_overlap_positions()):
            f.write(f"1 {pos} {i * 0.001}\n")
    individuals_path = tmp_path / "inds.txt"
    individuals_path.write_text("\n".join(individuals) + "\n")

    partitions = _OVERLAP_PARTITIONS
    snp_first, snp_last = partitions[0][0], partitions[-1][1]
    plans = _plan_diag_vector_partitions(partitions, snp_first, snp_last)
    assert len(plans) == len(partitions)
    plans_by_bounds = {(p.start, p.end): p for p in plans}

    # Reference: persist partitions plainly, then read them back post-hoc,
    # exactly like _run()'s non-fused Step 3 path.
    cache_store = CovarianceStore(root=tmp_path / "cache")
    (cache_store.root / "1").mkdir(parents=True)
    with (cache_store.root / "1_partitions.txt").open("w") as f:
        for start, end in partitions:
            f.write(f"{start} {end}\n")
    for start, end in partitions:
        _calc_partition(
            start,
            end,
            "1",
            str(vcf_path),
            map_path,
            individuals_path,
            cache_store.partition_path("1", start, end),
            11418.0,
            1e-7,
            True,
            "zstd",
        )
    reference_vector_path = tmp_path / "reference_vector.txt.gz"
    MatrixAnalysis(name="1", store=cache_store).calc_diag_lean(reference_vector_path)

    # Fused: same _calc_partition call, but with a vector_plan -- mirrors
    # _run()'s Step 2 loop when --fused-vector is set.
    fused_store = CovarianceStore(root=tmp_path / "fused")
    (fused_store.root / "1").mkdir(parents=True)
    vector_fragments = {}
    for start, end in partitions:
        fragment = _calc_partition(
            start,
            end,
            "1",
            str(vcf_path),
            map_path,
            individuals_path,
            fused_store.partition_path("1", start, end),
            11418.0,
            1e-7,
            True,
            "zstd",
            plans_by_bounds[(start, end)],
            snp_last,
        )
        assert fragment is not None
        vector_fragments[(start, end)] = fragment

    assert _fused_vector_ready(True, partitions, partitions, vector_fragments)

    # Mirrors _run()'s Step 3 fused branch exactly.
    fused_vector_path = tmp_path / "fused_vector.txt.gz"
    pending_sums: dict[int, float] = {}
    parent_profile: dict[str, float | int] = {
        "merge_seconds": 0.0,
        "flush_seconds": 0.0,
        "worker_wait_seconds": 0.0,
        "partitions": 0,
    }
    current_locus = snp_first
    for start, end in partitions:
        current_locus = _merge_diag_vector_partition_result(
            result=vector_fragments[(start, end)],
            snp_first=snp_first,
            snp_last=snp_last,
            current_locus=current_locus,
            pending_sums=pending_sums,
            out_path=fused_vector_path,
            parent_profile=parent_profile,
        )

    def _read_vector_gz(path: Path) -> dict[int, float]:
        data: dict[int, float] = {}
        with gzip.open(path, "rt") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    data[int(parts[0])] = float(parts[1])
        return data

    reference = _read_vector_gz(reference_vector_path)
    fused = _read_vector_gz(fused_vector_path)
    assert reference, "reference vector build produced no rows; fixture too sparse"
    assert fused.keys() == reference.keys()
    for locus in reference:
        assert fused[locus] == reference[locus], f"locus {locus} mismatch"
