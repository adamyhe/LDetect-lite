"""Tests for shrinkage LD covariance calculation."""

from __future__ import annotations

import gzip
from io import StringIO
from pathlib import Path

import numpy as np

from ldetect2.shrinkage import calc_covariance, partition_chromosome

_FULL_COVARIANCE_KEYS = {
    "i_pos",
    "j_pos",
    "i_gpos",
    "j_gpos",
    "naive_ld",
    "shrink_ld",
    "i_id",
    "j_id",
}
_COMPACT_COVARIANCE_KEYS = {"i_pos", "j_pos", "shrink_ld"}


def _write_map(path: Path) -> None:
    with gzip.open(path, "wt") as f:
        f.write("1 100 0.000\n")
        f.write("1 200 0.001\n")
        f.write("1 300 0.002\n")


def _write_individuals(path: Path) -> None:
    path.write_text("sample_a\nsample_b\n")


def _vcf_stream() -> StringIO:
    return StringIO(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
                "\tsample_a\tsample_b",
                "1\t100\trs_mono\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|0",
                "1\t200\trs_poly_a\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0",
                "1\t300\trs_poly_b\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|1",
                "",
            ]
        )
    )


def test_calc_covariance_skips_population_monomorphic_variant(tmp_path: Path) -> None:
    """Match reference ldetect: apply cutoff before adding diagonal shrinkage."""
    map_path = tmp_path / "map.gz"
    _write_map(map_path)

    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    out_path = tmp_path / "cov.npz"
    calc_covariance(
        vcf_stream=_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
    )

    cov = np.load(out_path)
    positions = set(cov["i_pos"]) | set(cov["j_pos"])

    assert 100 not in positions
    assert {200, 300}.issubset(positions)


def test_calc_covariance_default_writes_full_schema(tmp_path: Path) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    out_path = tmp_path / "full.npz"
    calc_covariance(
        vcf_stream=_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
    )

    with np.load(out_path) as cov:
        assert set(cov.files) == _FULL_COVARIANCE_KEYS


def test_calc_covariance_compact_output_writes_only_compact_schema(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    out_path = tmp_path / "compact.npz"
    calc_covariance(
        vcf_stream=_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
        compact_output=True,
    )

    with np.load(out_path) as cov:
        assert set(cov.files) == _COMPACT_COVARIANCE_KEYS
        assert cov["i_pos"].dtype == np.int32
        assert cov["j_pos"].dtype == np.int32
        assert cov["shrink_ld"].dtype == np.float64


def test_partition_chromosome_clamps_uncut_window_to_last_snp(
    tmp_path: Path,
) -> None:
    """A low-recombination tail should not index one past the final SNP."""
    map_path = tmp_path / "flat_map.gz"
    with gzip.open(map_path, "wt") as f:
        for i in range(1, 8):
            f.write(f"chr1 {i * 100} 0.0\n")

    output_path = tmp_path / "chr1_partitions.txt"
    partition_chromosome(
        genetic_map_path=map_path,
        n_individuals=4,
        output_path=output_path,
        window_size=3,
    )

    assert output_path.read_text().splitlines() == [
        "100 700",
        "400 700",
    ]
