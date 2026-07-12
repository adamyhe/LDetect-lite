"""Tests for shrinkage LD covariance calculation."""

from __future__ import annotations

import gzip
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ldetect_lite.io.covariance_hdf5 import (
    HDF5_DATASET_CHUNK_ROWS,
    open_covariance_reader,
)
from ldetect_lite.shrinkage import (
    _count_pairwise_ld_by_i_impl,
    _genetic_stop_bounds_impl,
    calc_covariance,
    partition_chromosome,
)

_FULL_HDF5_DATASETS = {
    "covariance/lo",
    "covariance/hi",
    "covariance/naive_ld",
    "covariance/shrink_ld",
    "metadata/i_gpos",
    "metadata/j_gpos",
    "metadata/i_id",
    "metadata/j_id",
    "index/diag_pos",
    "index/diag_val",
    "index/lo_values",
    "index/lo_offsets",
}
_COMPACT_HDF5_DATASETS = {
    "covariance/lo",
    "covariance/hi",
    "covariance/shrink_ld",
    "index/diag_pos",
    "index/diag_val",
    "index/lo_values",
    "index/lo_offsets",
}

_VCF_HEADER = [
    "##fileformat=VCFv4.2",
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    "##contig=<ID=1>",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample_a\tsample_b",
]

_TABIX_TOOLS_AVAILABLE = all(
    shutil.which(tool) is not None for tool in ("bgzip", "tabix")
)
_BCFTOOLS_AVAILABLE = shutil.which("bcftools") is not None
requires_htslib_tools = pytest.mark.skipif(
    not _TABIX_TOOLS_AVAILABLE, reason="bgzip/tabix not found on PATH"
)
requires_bcftools = pytest.mark.skipif(
    not _BCFTOOLS_AVAILABLE, reason="bcftools not found on PATH"
)


def _write_map(path: Path) -> None:
    with gzip.open(path, "wt") as f:
        f.write("1 100 0.000\n")
        f.write("1 200 0.001\n")
        f.write("1 300 0.002\n")


def _write_individuals(path: Path) -> None:
    path.write_text("sample_a\nsample_b\n")


def _write_indexed_vcf(tmp_path: Path, name: str, lines: list[str]) -> Path:
    """Write VCF text, bgzip it, and build a .tbi index. Returns the .vcf.gz path."""
    raw_path = tmp_path / f"{name}.vcf"
    raw_path.write_text("\n".join(lines) + "\n")
    subprocess.run(["bgzip", "-f", str(raw_path)], check=True)
    gz_path = tmp_path / f"{name}.vcf.gz"
    subprocess.run(["tabix", "-f", "-p", "vcf", str(gz_path)], check=True)
    return gz_path


def _write_indexed_bcf(tmp_path: Path, name: str, lines: list[str]) -> Path:
    """Write VCF text, convert to BCF, and build a .csi index. Returns the .bcf path."""
    raw_path = tmp_path / f"{name}.vcf"
    raw_path.write_text("\n".join(lines) + "\n")
    bcf_path = tmp_path / f"{name}.bcf"
    subprocess.run(
        ["bcftools", "view", "-O", "b", "-o", str(bcf_path), str(raw_path)],
        check=True,
    )
    subprocess.run(["bcftools", "index", "-f", str(bcf_path)], check=True)
    return bcf_path


def _vcf_stream(tmp_path: Path, name: str = "vcf") -> Path:
    return _write_indexed_vcf(
        tmp_path,
        name,
        [
            *_VCF_HEADER,
            "1\t100\trs_mono\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|0",
            "1\t200\trs_poly_a\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0",
            "1\t300\trs_poly_b\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|1",
        ],
    )


def _duplicate_position_vcf_stream(tmp_path: Path, name: str = "dup") -> Path:
    return _write_indexed_vcf(
        tmp_path,
        name,
        [
            *_VCF_HEADER,
            "1\t100\trs_a\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0",
            "1\t100\trs_b\tC\tT\t.\tPASS\t.\tGT\t0|0\t0|1",
            "1\t200\trs_c\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|1",
            "1\t300\trs_d\tA\tG\t.\tPASS\t.\tGT\t1|1\t0|1",
        ],
    )


@requires_htslib_tools
def test_calc_covariance_skips_population_monomorphic_variant(tmp_path: Path) -> None:
    """Match reference ldetect: apply cutoff before adding diagonal shrinkage."""
    map_path = tmp_path / "map.gz"
    _write_map(map_path)

    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    out_path = tmp_path / "cov.h5"
    calc_covariance(
        vcf_path=_vcf_stream(tmp_path),
        region=None,
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
    )

    with open_covariance_reader(out_path, 100, 300) as reader:
        rows = reader.read_all()
    positions = set(rows.lo) | set(rows.hi)

    assert 100 not in positions
    assert {200, 300}.issubset(positions)


@requires_htslib_tools
def test_calc_covariance_canonicalizes_duplicate_physical_positions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Duplicate VCF positions should be collapsed before pairwise LD."""
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    out_path = tmp_path / "duplicate-position.h5"
    calc_covariance(
        vcf_path=_duplicate_position_vcf_stream(tmp_path),
        region=None,
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
        compact_output=True,
    )
    captured = capsys.readouterr()
    assert "skipped 1 duplicate-position variant" in captured.err

    with open_covariance_reader(out_path, 100, 300) as reader:
        rows = reader.read_all()
        lo_values = reader.read_loci()

    assert np.all(rows.lo <= rows.hi)
    assert np.all(
        (rows.lo[1:] > rows.lo[:-1])
        | ((rows.lo[1:] == rows.lo[:-1]) & (rows.hi[1:] > rows.hi[:-1]))
    )
    np.testing.assert_array_equal(lo_values, np.unique(rows.lo))


def _vcf_stream_with_pos100_variant(tmp_path: Path, keep: str, name: str) -> Path:
    """Build a 2-locus VCF where POS=100 has 0, 1, or 2 candidate variants.

    ``keep`` selects which POS=100 record(s) are present: ``"a"`` (only the
    first-listed variant), ``"b"`` (only the second-listed variant), or
    ``"both"`` (both, exercising calc_covariance's duplicate-position dedup).
    The two variants have disjoint carrier haplotypes so their LD with the
    POS=200 neighbor differs in sign, making "which duplicate survives"
    directly observable in the output value.
    """
    rows_100 = {
        "a": "1\t100\trs_a\tA\tG\t.\tPASS\t.\tGT\t1|1\t0|0",
        "b": "1\t100\trs_b\tC\tT\t.\tPASS\t.\tGT\t0|0\t1|1",
    }
    body = [rows_100["a"], rows_100["b"]] if keep == "both" else [rows_100[keep]]
    return _write_indexed_vcf(
        tmp_path,
        name,
        [
            *_VCF_HEADER,
            *body,
            "1\t200\trs_c\tA\tG\t.\tPASS\t.\tGT\t1|0\t0|0",
        ],
    )


def _expected_shrink_ld(n11: int, n1x: int, nx1: int, n_haps: int, ne: float) -> float:
    """Independent reimplementation of the off-diagonal shrinkage formula.

    Deliberately re-derives ``ds2`` from the definition in
    ``_pairwise_ld_impl`` rather than importing/calling it, so the expected
    values below are an external check, not a circular one.
    """
    harmonic = sum(1.0 / i for i in range(1, n_haps))
    theta = (1.0 / harmonic) / (n_haps + 1.0 / harmonic)
    n_total = float(n_haps)
    f11, f1, f2 = n11 / n_total, n1x / n_total, nx1 / n_total
    d_naive = f11 - f1 * f2
    # gpos(200) - gpos(100) == 0.001 per _write_map; n_ind == n_haps / 2.
    ee = math.exp(-4.0 * ne * 0.001 / (2.0 * (n_haps / 2.0)))
    return (1.0 - theta) ** 2 * d_naive * ee


@requires_htslib_tools
def test_calc_covariance_duplicate_position_matches_first_encountered_variant(
    tmp_path: Path,
) -> None:
    """The duplicate-position dedup must behave like keeping only the first
    VCF-encountered variant, matching the legacy reference's effective
    first-write-wins matrix-insert semantics (see
    notes/logs/ldetect-original-main-pipeline-audit.md)."""
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    def _shrink_ld_100_200(keep: str, name: str) -> float:
        out_path = tmp_path / f"{name}.h5"
        calc_covariance(
            vcf_path=_vcf_stream_with_pos100_variant(tmp_path, keep, f"{name}_vcf"),
            region=None,
            genetic_map_path=map_path,
            individuals_path=individuals_path,
            output_path=out_path,
            ne=1.0,
            cutoff=1e-7,
            compact_output=True,
        )
        with open_covariance_reader(out_path, 100, 200) as reader:
            rows = reader.read_all()
        mask = (rows.lo == 100) & (rows.hi == 200)
        assert mask.sum() == 1
        return float(rows.shrink_ld[mask][0])

    # haps: rs_a=[1,1,0,0], rs_b=[0,0,1,1], rs_c=[1,0,0,0] (n_haps=4)
    expected_first = _expected_shrink_ld(n11=1, n1x=2, nx1=1, n_haps=4, ne=1.0)
    expected_second = _expected_shrink_ld(n11=0, n1x=2, nx1=1, n_haps=4, ne=1.0)
    assert expected_first == pytest.approx(0.09670324838, abs=1e-10)
    assert expected_second == pytest.approx(-0.09670324838, abs=1e-10)

    first_only = _shrink_ld_100_200("a", "first")
    second_only = _shrink_ld_100_200("b", "second")
    dup_value = _shrink_ld_100_200("both", "dup")

    assert first_only == pytest.approx(expected_first)
    assert second_only == pytest.approx(expected_second)

    # calc_covariance's dedup must reproduce "keep the first-encountered
    # duplicate", not the second one and not some blend of the two.
    assert dup_value == pytest.approx(first_only)
    assert dup_value != pytest.approx(second_only)


@requires_htslib_tools
def test_calc_covariance_default_writes_full_schema(tmp_path: Path) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    out_path = tmp_path / "full.h5"
    calc_covariance(
        vcf_path=_vcf_stream(tmp_path),
        region=None,
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
    )

    import h5py

    with h5py.File(out_path, "r") as cov:
        datasets = {
            path
            for path in _FULL_HDF5_DATASETS
            if path in cov
        }
        assert datasets == _FULL_HDF5_DATASETS


@requires_htslib_tools
def test_calc_covariance_compact_output_writes_only_compact_schema(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    out_path = tmp_path / "compact.h5"
    calc_covariance(
        vcf_path=_vcf_stream(tmp_path),
        region=None,
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
        compact_output=True,
    )

    import h5py

    with h5py.File(out_path, "r") as cov:
        datasets = {
            path
            for path in _COMPACT_HDF5_DATASETS
            if path in cov
        }
        assert datasets == _COMPACT_HDF5_DATASETS
        assert "metadata/i_id" not in cov
        assert cov["covariance/lo"].dtype == np.int32
        assert cov["covariance/hi"].dtype == np.int32
        assert cov["covariance/shrink_ld"].dtype == np.float64
        assert cov.attrs["dataset_chunk_rows"] == HDF5_DATASET_CHUNK_ROWS
        assert cov.attrs["write_chunk_rows"] == 1_000_000
        assert cov["covariance/lo"].chunks == (HDF5_DATASET_CHUNK_ROWS,)
        assert cov["covariance/hi"].chunks == (HDF5_DATASET_CHUNK_ROWS,)
        assert cov["covariance/shrink_ld"].chunks == (HDF5_DATASET_CHUNK_ROWS,)


@requires_htslib_tools
def test_calc_covariance_compact_chunked_writer_matches_full_rows(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    full_path = tmp_path / "full.h5"
    compact_path = tmp_path / "compact-chunked.h5"
    calc_covariance(
        vcf_path=_vcf_stream(tmp_path, "full_vcf"),
        region=None,
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=full_path,
        cutoff=1e-7,
    )
    calc_covariance(
        vcf_path=_vcf_stream(tmp_path, "compact_vcf"),
        region=None,
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=compact_path,
        cutoff=1e-7,
        compact_output=True,
        compact_chunk_rows=2,
    )

    with open_covariance_reader(full_path, 100, 300) as full_reader:
        full_rows = full_reader.read_all()
        full_diag = full_reader.read_diagonal()
        full_loci = full_reader.read_loci()
    with open_covariance_reader(compact_path, 100, 300) as compact_reader:
        compact_rows = compact_reader.read_all()
        compact_diag = compact_reader.read_diagonal()
        compact_loci = compact_reader.read_loci()

    np.testing.assert_array_equal(compact_rows.lo, full_rows.lo)
    np.testing.assert_array_equal(compact_rows.hi, full_rows.hi)
    np.testing.assert_allclose(compact_rows.shrink_ld, full_rows.shrink_ld)
    np.testing.assert_array_equal(compact_diag[0], full_diag[0])
    np.testing.assert_allclose(compact_diag[1], full_diag[1])
    np.testing.assert_array_equal(compact_loci, full_loci)

    import h5py

    with h5py.File(compact_path, "r") as cov:
        assert cov.attrs["dataset_chunk_rows"] == HDF5_DATASET_CHUNK_ROWS
        assert cov.attrs["write_chunk_rows"] == 2
        assert cov["covariance/lo"].chunks == (HDF5_DATASET_CHUNK_ROWS,)
        lo = cov["covariance/lo"][:]
        lo_values = cov["index/lo_values"][:]
        lo_offsets = cov["index/lo_offsets"][:]
        expected_values, expected_counts = np.unique(lo, return_counts=True)
        np.testing.assert_array_equal(lo_values, expected_values)
        np.testing.assert_array_equal(
            lo_offsets,
            np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.cumsum(expected_counts, dtype=np.int64),
                )
            ),
        )


@requires_htslib_tools
def test_calc_covariance_lzf_override_matches_zstd_default(tmp_path: Path) -> None:
    """compression="lzf" must thread through to identical values as the
    zstd default -- compression is lossless, so only the codec differs."""
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    zstd_path = tmp_path / "zstd.h5"
    lzf_path = tmp_path / "lzf.h5"
    calc_covariance(
        vcf_path=_vcf_stream(tmp_path, "zstd_vcf"),
        region=None,
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=zstd_path,
        cutoff=1e-7,
        compact_output=True,
    )
    calc_covariance(
        vcf_path=_vcf_stream(tmp_path, "lzf_vcf"),
        region=None,
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=lzf_path,
        cutoff=1e-7,
        compact_output=True,
        compression="lzf",
    )

    with open_covariance_reader(zstd_path, 100, 300) as reader:
        zstd_rows = reader.read_all()
    with open_covariance_reader(lzf_path, 100, 300) as reader:
        lzf_rows = reader.read_all()

    np.testing.assert_array_equal(zstd_rows.lo, lzf_rows.lo)
    np.testing.assert_array_equal(zstd_rows.hi, lzf_rows.hi)
    np.testing.assert_allclose(zstd_rows.shrink_ld, lzf_rows.shrink_ld)


def _write_map_positions(path: Path, positions: list[int]) -> None:
    with gzip.open(path, "wt") as f:
        for i, pos in enumerate(positions):
            f.write(f"1 {pos} {i * 0.001:.6f}\n")


@requires_htslib_tools
def test_calc_covariance_skips_unphased_and_missing_allele_genotypes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unphased and missing-allele rows are dropped and share one counter --
    cyvcf2 folds both conditions into a single skip test (see shrinkage.py),
    matching the naive parser's combined ``skipped_unphased`` warning."""
    map_path = tmp_path / "map.gz"
    _write_map_positions(map_path, [100, 200, 300, 400])
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    vcf_path = _write_indexed_vcf(
        tmp_path,
        "skip_vcf",
        [
            *_VCF_HEADER,
            "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0",
            "1\t200\trs2\tA\tG\t.\tPASS\t.\tGT\t0|1\t1/0",  # sample_b unphased
            "1\t300\trs3\tA\tG\t.\tPASS\t.\tGT\t.|1\t0|1",  # sample_a missing allele
            "1\t400\trs4\tA\tG\t.\tPASS\t.\tGT\t1|0\t0|1",
        ],
    )

    out_path = tmp_path / "cov.h5"
    calc_covariance(
        vcf_path=vcf_path,
        region=None,
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
        compact_output=True,
    )

    captured = capsys.readouterr()
    assert "skipped 2 variant(s) with unphased or missing genotypes" in captured.err

    with open_covariance_reader(out_path, 100, 400) as reader:
        diag_pos, _ = reader.read_diagonal()
    assert set(diag_pos.tolist()) == {100, 400}


@requires_htslib_tools
def test_calc_covariance_missing_individual_raises(tmp_path: Path) -> None:
    """A requested individual absent from the VCF header must fail loudly,
    naming the individual -- not silently produce empty output."""
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)  # requests sample_a AND sample_b

    vcf_path = _write_indexed_vcf(
        tmp_path,
        "missing_ind_vcf",
        [
            "##fileformat=VCFv4.2",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "##contig=<ID=1>",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample_a",
            "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1",
        ],
    )

    with pytest.raises(ValueError, match="sample_b"):
        calc_covariance(
            vcf_path=vcf_path,
            region=None,
            genetic_map_path=map_path,
            individuals_path=individuals_path,
            output_path=tmp_path / "cov.h5",
            cutoff=1e-7,
        )


@requires_htslib_tools
def test_calc_covariance_region_restricts_output(tmp_path: Path) -> None:
    """Passing *region* must restrict reads to that window, mirroring what
    the previous tabix-subprocess pre-slicing did outside calc_covariance."""
    positions = [100, 200, 300, 400, 500]
    map_path = tmp_path / "map.gz"
    _write_map_positions(map_path, positions)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    vcf_path = _write_indexed_vcf(
        tmp_path,
        "region_vcf",
        [
            *_VCF_HEADER,
            *(
                f"1\t{pos}\trs{pos}\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0"
                for pos in positions
            ),
        ],
    )

    out_path = tmp_path / "cov.h5"
    calc_covariance(
        vcf_path=vcf_path,
        region="1:1-250",
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
        compact_output=True,
    )

    with open_covariance_reader(out_path, 100, 500) as reader:
        diag_pos, _ = reader.read_diagonal()
    assert set(diag_pos.tolist()) == {100, 200}


def _boundary_spanning_indel_vcf_stream(tmp_path: Path, name: str = "spanning") -> Path:
    """A long-REF deletion at POS=100 whose reference span [100, 149]
    reaches into a later region query, plus two ordinary SNPs at 250/300
    that are unambiguously inside that region. Mirrors the real htslib
    behavior observed on 1000G chr22 (esv2670795): region overlap is
    span-based (POS to POS + len(REF) - 1), not POS-based, for ordinary
    long-REF records, not just symbolic SV alleles."""
    long_ref = "A" * 50
    return _write_indexed_vcf(
        tmp_path,
        name,
        [
            *_VCF_HEADER,
            f"1\t100\tdel_spanning\t{long_ref}\tA\t.\tPASS\t.\tGT\t0|1\t0|0",
            "1\t250\trs_a\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0",
            "1\t300\trs_b\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0",
        ],
    )


@requires_htslib_tools
def test_calc_covariance_default_includes_boundary_spanning_variant(
    tmp_path: Path,
) -> None:
    """Historical behavior (strict_region_bounds=False, the default): a
    record whose span crosses into a queried region is included even
    though its own POS lies outside that region -- matches the original
    ldetect's tabix -h behavior, see
    notes/logs/sv-boundary-diagnostics-investigation.md."""
    map_path = tmp_path / "map.gz"
    _write_map_positions(map_path, [100, 250, 300])
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)
    vcf_path = _boundary_spanning_indel_vcf_stream(tmp_path)

    out_path = tmp_path / "cov.h5"
    calc_covariance(
        vcf_path=vcf_path,
        region="1:120-300",
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
        compact_output=True,
    )

    with open_covariance_reader(out_path, 100, 300) as reader:
        rows = reader.read_all()
    positions = set(rows.lo.tolist()) | set(rows.hi.tolist())
    assert 100 in positions


@requires_htslib_tools
def test_calc_covariance_strict_region_bounds_excludes_boundary_spanning_variant(
    tmp_path: Path,
) -> None:
    """strict_region_bounds=True adds the explicit POS check the original
    never had: the same boundary-spanning record from the previous test is
    excluded once its own POS is required to fall inside the region."""
    map_path = tmp_path / "map.gz"
    _write_map_positions(map_path, [100, 250, 300])
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)
    vcf_path = _boundary_spanning_indel_vcf_stream(tmp_path)

    out_path = tmp_path / "cov.h5"
    calc_covariance(
        vcf_path=vcf_path,
        region="1:120-300",
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
        compact_output=True,
        strict_region_bounds=True,
    )

    with open_covariance_reader(out_path, 100, 300) as reader:
        rows = reader.read_all()
    positions = set(rows.lo.tolist()) | set(rows.hi.tolist())
    assert 100 not in positions
    assert positions == {250, 300}


@requires_htslib_tools
def test_calc_covariance_strict_region_bounds_noop_without_region(
    tmp_path: Path,
) -> None:
    """strict_region_bounds only means something when region is set; passing
    it with region=None (whole-file read) must not change output at all."""
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)
    vcf_path = _vcf_stream(tmp_path)

    default_out = tmp_path / "default.h5"
    strict_out = tmp_path / "strict.h5"
    for out_path, strict in ((default_out, False), (strict_out, True)):
        calc_covariance(
            vcf_path=vcf_path,
            region=None,
            genetic_map_path=map_path,
            individuals_path=individuals_path,
            output_path=out_path,
            cutoff=1e-7,
            compact_output=True,
            strict_region_bounds=strict,
        )

    with open_covariance_reader(default_out, 100, 300) as reader:
        default_rows = reader.read_all()
    with open_covariance_reader(strict_out, 100, 300) as reader:
        strict_rows = reader.read_all()
    np.testing.assert_array_equal(default_rows.lo, strict_rows.lo)
    np.testing.assert_array_equal(default_rows.hi, strict_rows.hi)
    np.testing.assert_array_equal(default_rows.shrink_ld, strict_rows.shrink_ld)


@requires_htslib_tools
@requires_bcftools
def test_calc_covariance_bcf_matches_vcf_gz_output(tmp_path: Path) -> None:
    """Identical content read as .bcf/.csi vs .vcf.gz/.tbi must produce
    byte-identical HDF5 output -- cyvcf2's region-fetch API is format-
    agnostic given the right index, this confirms it, not just assumes it."""
    lines = [
        *_VCF_HEADER,
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0",
        "1\t200\trs2\tA\tG\t.\tPASS\t.\tGT\t1|1\t0|1",
        "1\t300\trs3\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0",
    ]
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    vcf_path = _write_indexed_vcf(tmp_path, "cmp_vcf", lines)
    bcf_path = _write_indexed_bcf(tmp_path, "cmp_bcf", lines)

    vcf_out = tmp_path / "vcf_out.h5"
    bcf_out = tmp_path / "bcf_out.h5"
    for path, out in ((vcf_path, vcf_out), (bcf_path, bcf_out)):
        calc_covariance(
            vcf_path=path,
            region="1:1-1000",
            genetic_map_path=map_path,
            individuals_path=individuals_path,
            output_path=out,
            cutoff=1e-7,
        )

    with open_covariance_reader(vcf_out, 100, 300) as reader:
        vcf_rows = reader.read_all()
        vcf_diag = reader.read_diagonal()
    with open_covariance_reader(bcf_out, 100, 300) as reader:
        bcf_rows = reader.read_all()
        bcf_diag = reader.read_diagonal()

    np.testing.assert_array_equal(vcf_rows.lo, bcf_rows.lo)
    np.testing.assert_array_equal(vcf_rows.hi, bcf_rows.hi)
    np.testing.assert_array_equal(vcf_rows.shrink_ld, bcf_rows.shrink_ld)
    np.testing.assert_array_equal(vcf_diag[0], bcf_diag[0])
    np.testing.assert_array_equal(vcf_diag[1], bcf_diag[1])

    import h5py

    with h5py.File(vcf_out, "r") as cov:
        vcf_i_id = cov["metadata/i_id"][:]
        vcf_j_id = cov["metadata/j_id"][:]
    with h5py.File(bcf_out, "r") as cov:
        bcf_i_id = cov["metadata/i_id"][:]
        bcf_j_id = cov["metadata/j_id"][:]
    np.testing.assert_array_equal(vcf_i_id, bcf_i_id)
    np.testing.assert_array_equal(vcf_j_id, bcf_j_id)


def _naive_parse_positions(lines: list[str], individuals: list[str]) -> list[int]:
    """Independent reimplementation of the pre-cyvcf2 naive text parser's
    row-survival logic (VCF ingestion only, not the shrinkage math) -- lives
    only in this test module as an oracle to diff the real implementation
    against, and must never be imported by production code."""
    ind2col: dict[str, int] = {}
    positions: list[int] = []
    for raw in lines:
        if raw.startswith("##"):
            continue
        parts = raw.split("\t")
        if raw.startswith("#CHROM"):
            for col_idx in range(9, len(parts)):
                if parts[col_idx] in individuals:
                    ind2col[parts[col_idx]] = col_idx
            continue
        pos = int(parts[1])
        skip = False
        for ind in individuals:
            col = ind2col.get(ind)
            if col is None:
                skip = True
                break
            gt_field = parts[col].split(":")[0]
            if "|" not in gt_field:
                skip = True
                break
            alleles = gt_field.split("|")
            if "." in alleles:
                skip = True
                break
        if skip:
            continue
        positions.append(pos)

    seen: set[int] = set()
    unique_pos: list[int] = []
    for pos in positions:
        if pos in seen:
            continue
        seen.add(pos)
        unique_pos.append(pos)
    return unique_pos


@requires_htslib_tools
def test_calc_covariance_cyvcf2_matches_naive_reference_parser(
    tmp_path: Path,
) -> None:
    """Randomized mix of phased/unphased/missing/duplicate rows: the
    cyvcf2-backed real implementation must survive exactly the same rows as
    an independent, test-only reimplementation of the retired naive parser."""
    import random

    rng = random.Random(0)
    individuals = ["sample_a", "sample_b", "sample_c"]
    positions = list(range(100, 100 + 50 * 20, 20))  # 50 well-separated loci

    def _random_gt() -> str:
        choice = rng.random()
        if choice < 0.15:
            return rng.choice(["0/1", "1/0"])  # unphased
        if choice < 0.3:
            return rng.choice([".|0", "0|.", ".|."])  # missing allele
        return rng.choice(["0|0", "0|1", "1|0", "1|1"])  # valid

    lines = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "##contig=<ID=1>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(individuals),
    ]
    for pos in positions:
        gts = "\t".join(_random_gt() for _ in individuals)
        lines.append(f"1\t{pos}\trs{pos}\tA\tG\t.\tPASS\t.\tGT\t{gts}")
        if rng.random() < 0.2:
            # Duplicate-position row immediately after, with different GTs,
            # to also exercise first-wins dedup under randomization.
            gts_dup = "\t".join(_random_gt() for _ in individuals)
            lines.append(f"1\t{pos}\trs{pos}_dup\tA\tG\t.\tPASS\t.\tGT\t{gts_dup}")

    map_path = tmp_path / "map.gz"
    _write_map_positions(map_path, positions)
    individuals_path = tmp_path / "inds.txt"
    individuals_path.write_text("\n".join(individuals) + "\n")

    expected_positions = _naive_parse_positions(lines, individuals)

    vcf_path = _write_indexed_vcf(tmp_path, "random_vcf", lines)
    out_path = tmp_path / "cov.h5"
    calc_covariance(
        vcf_path=vcf_path,
        region=None,
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=0.0,  # keep every pair regardless of magnitude
        compact_output=True,
    )

    with open_covariance_reader(out_path, positions[0], positions[-1]) as reader:
        diag_pos, _ = reader.read_diagonal()

    assert sorted(diag_pos.tolist()) == sorted(expected_positions)


def test_genetic_stop_bounds_preserve_pair_count_cutoff() -> None:
    hap_mat = np.array(
        [
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    gpos_arr = np.array([0.0, 0.001, 0.003, 1.0], dtype=np.float64)
    hap_sums = np.asarray(hap_mat.sum(axis=1), dtype=np.float64)
    ne = 11418.0
    n_ind = 2.0
    theta = 0.01
    cutoff = 1e-7
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, ne, n_ind, cutoff)
    counts = _count_pairwise_ld_by_i_impl(
        hap_mat,
        gpos_arr,
        hap_sums,
        j_stop_by_i,
        ne,
        n_ind,
        theta,
        cutoff,
    )

    expected = np.zeros(hap_mat.shape[0], dtype=np.int64)
    n_total = float(hap_mat.shape[1])
    for i in range(hap_mat.shape[0]):
        for j in range(i, hap_mat.shape[0]):
            df = gpos_arr[j] - gpos_arr[i]
            ee = np.exp(-4.0 * ne * df / (2.0 * n_ind))
            if ee < cutoff:
                break
            f11 = np.sum(hap_mat[i] * hap_mat[j]) / n_total
            f1 = hap_sums[i] / n_total
            f2 = hap_sums[j] / n_total
            ds2 = (1.0 - theta) ** 2 * (f11 - f1 * f2) * ee
            if abs(ds2) >= cutoff:
                expected[i] += 1

    np.testing.assert_array_equal(counts, expected)


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
