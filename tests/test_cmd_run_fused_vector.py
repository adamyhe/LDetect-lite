"""Tests for the --fused-vector Step 2/3 glue in cmd_run.py.

Validates that CovarianceSidecarAccumulator-derived vector fragments, merged
via `_merge_diag_vector_partition_result`, produce a bit-exact match against
the existing post-hoc `MatrixAnalysis.calc_diag_lean` read of the same
persisted partitions -- and that teeing the row stream through the sidecar
does not change what gets persisted.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
from pathlib import Path

import pytest

from ldetect_lite._cli.cmd_run import (
    _calc_partition,
    _collect_vector_fragments,
    register,
)
from ldetect_lite._util.vector_array import _plan_diag_vector_partitions
from ldetect_lite.io.partitions import CovarianceStore, read_partitions
from ldetect_lite.matrix_analysis import MatrixAnalysis

_TABIX_TOOLS_AVAILABLE = all(
    shutil.which(tool) is not None for tool in ("bgzip", "tabix")
)
requires_htslib_tools = pytest.mark.skipif(
    not _TABIX_TOOLS_AVAILABLE, reason="bgzip/tabix not found on PATH"
)

_SAMPLES = [f"s{i}" for i in range(6)]
_POSITIONS = [100, 150, 200, 250, 300, 350, 400, 450]
# Deterministic, not-all-identical phased genotypes per sample per SNP.
_GENOTYPES = [
    ["0|1", "1|0", "0|0", "1|1", "0|1", "1|0"],
    ["1|1", "0|0", "0|1", "1|0", "0|0", "1|1"],
    ["0|0", "1|1", "1|0", "0|1", "1|1", "0|0"],
    ["1|0", "0|1", "1|1", "0|0", "0|1", "1|0"],
    ["0|1", "0|1", "0|0", "1|0", "1|1", "0|1"],
    ["1|1", "1|0", "0|1", "0|1", "0|0", "1|0"],
    ["0|0", "0|0", "1|1", "1|1", "0|1", "0|1"],
    ["1|0", "1|1", "0|0", "0|1", "1|0", "1|1"],
]
_PARTITIONS = [(100, 300), (250, 450)]


def _write_indexed_vcf(tmp_path: Path) -> Path:
    header = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "##contig=<ID=1>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(_SAMPLES),
    ]
    lines = list(header)
    for pos, row in zip(_POSITIONS, _GENOTYPES):
        lines.append(
            f"1\t{pos}\trs{pos}\tA\tG\t.\t.\t.\tGT\t" + "\t".join(row)
        )
    raw_path = tmp_path / "panel.vcf"
    raw_path.write_text("\n".join(lines) + "\n")
    subprocess.run(["bgzip", "-f", str(raw_path)], check=True)
    gz_path = tmp_path / "panel.vcf.gz"
    subprocess.run(["tabix", "-f", "-p", "vcf", str(gz_path)], check=True)
    return gz_path


def _write_map(path: Path) -> None:
    with gzip.open(path, "wt") as f:
        for i, pos in enumerate(_POSITIONS):
            f.write(f"1 {pos} {i * 0.001:.3f}\n")


def _write_individuals(path: Path) -> None:
    path.write_text("\n".join(_SAMPLES) + "\n")


def _write_partitions_file(store: CovarianceStore, name: str) -> None:
    lines = "\n".join(f"{start} {end}" for start, end in _PARTITIONS)
    store.partitions_path(name).write_text(lines + "\n")


@requires_htslib_tools
def test_fused_vector_fragments_match_post_hoc_read(tmp_path: Path) -> None:
    vcf_path = _write_indexed_vcf(tmp_path)
    genetic_map_path = tmp_path / "map.txt.gz"
    _write_map(genetic_map_path)
    individuals_path = tmp_path / "individuals.txt"
    _write_individuals(individuals_path)

    store_root = tmp_path / "store"
    store_root.mkdir()
    store = CovarianceStore(root=store_root)
    name = "1"
    (store_root / name).mkdir()
    _write_partitions_file(store, name)

    partitions = read_partitions(name, store)
    snp_first, snp_last = partitions[0][0], partitions[-1][1]
    plans = _plan_diag_vector_partitions(partitions, snp_first, snp_last)
    assert len(plans) == len(partitions)
    plans_by_bounds = {(p.start, p.end): p for p in plans}

    fragments = {}
    for start, end in partitions:
        result = _calc_partition(
            start,
            end,
            name,
            str(vcf_path),
            genetic_map_path,
            individuals_path,
            store.partition_path(name, start, end),
            11418.0,
            1e-7,
            True,
            "zstd",
            "bitpacked",
            plans_by_bounds[(start, end)],
            snp_last,
        )
        assert result is not None
        fragments[(start, end)] = result

    from ldetect_lite._util.vector_array import _merge_diag_vector_partition_result

    fused_vector_path = tmp_path / "fused-vector.txt.gz"
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
            result=fragments[(start, end)],
            snp_first=snp_first,
            snp_last=snp_last,
            current_locus=current_locus,
            pending_sums=pending_sums,
            out_path=fused_vector_path,
            parent_profile=parent_profile,
        )

    post_hoc_vector_path = tmp_path / "post-hoc-vector.txt.gz"
    analysis = MatrixAnalysis(name=name, store=store)
    analysis.calc_diag_lean(post_hoc_vector_path, backend="array")

    with gzip.open(fused_vector_path, "rt") as f:
        fused_lines = f.read()
    with gzip.open(post_hoc_vector_path, "rt") as f:
        post_hoc_lines = f.read()
    assert fused_lines == post_hoc_lines
    assert fused_lines  # sanity: the fixture actually produced output


@requires_htslib_tools
def test_fused_vector_does_not_change_persisted_partition(tmp_path: Path) -> None:
    vcf_path = _write_indexed_vcf(tmp_path)
    genetic_map_path = tmp_path / "map.txt.gz"
    _write_map(genetic_map_path)
    individuals_path = tmp_path / "individuals.txt"
    _write_individuals(individuals_path)

    start, end = _PARTITIONS[0]

    def _run_partition(store_root: Path, vector_plan) -> Path:
        store_root.mkdir()
        store = CovarianceStore(root=store_root)
        name = "1"
        (store_root / name).mkdir()
        partition_path = store.partition_path(name, start, end)
        _calc_partition(
            start,
            end,
            name,
            str(vcf_path),
            genetic_map_path,
            individuals_path,
            partition_path,
            11418.0,
            1e-7,
            True,
            "zstd",
            "bitpacked",
            vector_plan,
            end if vector_plan is not None else None,
        )
        return partition_path

    from ldetect_lite._util.vector_array import _DiagVectorPartitionPlan

    plan = _DiagVectorPartitionPlan(
        p_index=0,
        start=start,
        end=end,
        next_start=None,
        center_lower_bound=start,
        center_lower_inclusive=True,
        checkpoint="test",
    )

    without_sidecar = _run_partition(tmp_path / "no_sidecar", None)
    with_sidecar = _run_partition(tmp_path / "with_sidecar", plan)

    assert without_sidecar.read_bytes() == with_sidecar.read_bytes()


def _parse_run_args(extra: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    register(subparsers)
    return parser.parse_args(
        [
            "run",
            "--genetic-map",
            "map.gz",
            "--reference-panel",
            "panel.vcf.gz",
            "--individuals",
            "inds.txt",
            "--chromosome",
            "chr1",
            "--output-dir",
            "out",
            *extra,
        ]
    )


def test_fused_vector_defaults_to_enabled() -> None:
    assert _parse_run_args([]).fused_vector is True


def test_fused_vector_can_be_disabled() -> None:
    assert _parse_run_args(["--no-fused-vector"]).fused_vector is False


def _merge_vector(fragments, partitions, snp_first, snp_last, out_path: Path) -> str:
    from ldetect_lite._util.vector_array import _merge_diag_vector_partition_result

    pending_sums: dict[int, float] = {}
    parent_profile: dict[str, float | int] = {
        "merge_seconds": 0.0,
        "flush_seconds": 0.0,
        "worker_wait_seconds": 0.0,
        "partitions": 0,
    }
    out_path.unlink(missing_ok=True)
    current_locus = snp_first
    for bounds in partitions:
        current_locus = _merge_diag_vector_partition_result(
            result=fragments[bounds],
            snp_first=snp_first,
            snp_last=snp_last,
            current_locus=current_locus,
            pending_sums=pending_sums,
            out_path=out_path,
            parent_profile=parent_profile,
        )
    with gzip.open(out_path, "rt") as f:
        return f.read()


@requires_htslib_tools
def test_saved_fragments_reproduce_post_hoc_vector_on_reruns(tmp_path: Path) -> None:
    """Fresh, saved and recomputed fragments all give the Step 3 vector."""
    import os

    from ldetect_lite._util.vector_fragments import fragment_path

    vcf_path = _write_indexed_vcf(tmp_path)
    genetic_map_path = tmp_path / "map.txt.gz"
    _write_map(genetic_map_path)
    individuals_path = tmp_path / "individuals.txt"
    _write_individuals(individuals_path)

    store_root = tmp_path / "store"
    store_root.mkdir()
    store = CovarianceStore(root=store_root)
    name = "1"
    (store_root / name).mkdir()
    _write_partitions_file(store, name)
    partitions = read_partitions(name, store)
    snp_first, snp_last = partitions[0][0], partitions[-1][1]
    plans_by_bounds = {
        (p.start, p.end): p
        for p in _plan_diag_vector_partitions(partitions, snp_first, snp_last)
    }

    fresh = {}
    for start, end in partitions:
        result = _calc_partition(
            start,
            end,
            name,
            str(vcf_path),
            genetic_map_path,
            individuals_path,
            store.partition_path(name, start, end),
            11418.0,
            1e-7,
            True,
            "zstd",
            "bitpacked",
            plans_by_bounds[(start, end)],
            snp_last,
        )
        assert result is not None
        fresh[(start, end)] = result

    reference_path = tmp_path / "post-hoc-vector.txt.gz"
    MatrixAnalysis(name=name, store=store).calc_diag_lean(
        reference_path, backend="array"
    )
    with gzip.open(reference_path, "rt") as f:
        reference = f.read()
    assert reference

    def collect(fresh_subset, workers=1):
        fragments, counts = _collect_vector_fragments(
            chrom=name,
            store=store,
            partitions=partitions,
            plans_by_bounds=plans_by_bounds,
            fresh=fresh_subset,
            snp_first=snp_first,
            snp_last=snp_last,
            workers=workers,
        )
        vector = _merge_vector(
            fragments, partitions, snp_first, snp_last, tmp_path / "vector.txt.gz"
        )
        return vector, counts

    # First run: every partition fresh; fragments get saved.
    vector, counts = collect(fresh)
    assert vector == reference
    assert counts == {"fresh": 2, "saved": 0, "recomputed": 0}
    for bounds in partitions:
        assert fragment_path(store.partition_path(name, *bounds)).exists()

    # Full cached rerun: everything comes from saved fragments.
    vector, counts = collect({})
    assert vector == reference
    assert counts == {"fresh": 0, "saved": 2, "recomputed": 0}

    # Partial rerun: one partition recomputed in Step 2, one cached.
    vector, counts = collect({partitions[1]: fresh[partitions[1]]})
    assert vector == reference
    assert counts == {"fresh": 1, "saved": 1, "recomputed": 0}

    # A missing fragment is recomputed from its partition (and re-saved).
    first_fragment = fragment_path(store.partition_path(name, *partitions[0]))
    first_fragment.unlink()
    vector, counts = collect({}, workers=2)
    assert vector == reference
    assert counts == {"fresh": 0, "saved": 1, "recomputed": 1}
    assert first_fragment.exists()

    # A partition file that changed after its fragment was saved makes the
    # fragment stale.
    partition = store.partition_path(name, *partitions[1])
    stat = partition.stat()
    os.utime(partition, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    vector, counts = collect({})
    assert vector == reference
    assert counts == {"fresh": 0, "saved": 1, "recomputed": 1}
