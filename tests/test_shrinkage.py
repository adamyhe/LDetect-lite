"""Tests for shrinkage LD covariance calculation."""

from __future__ import annotations

import gzip
import math
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from ldetect2.io.covariance_hdf5 import (
    HDF5_DATASET_CHUNK_ROWS,
    open_covariance_reader,
)
from ldetect2.shrinkage import (
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


def _duplicate_position_vcf_stream() -> StringIO:
    return StringIO(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
                "\tsample_a\tsample_b",
                "1\t100\trs_a\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0",
                "1\t100\trs_b\tC\tT\t.\tPASS\t.\tGT\t0|0\t0|1",
                "1\t200\trs_c\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|1",
                "1\t300\trs_d\tA\tG\t.\tPASS\t.\tGT\t1|1\t0|1",
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

    out_path = tmp_path / "cov.h5"
    calc_covariance(
        vcf_stream=_vcf_stream(),
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
        vcf_stream=_duplicate_position_vcf_stream(),
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


def _vcf_stream_with_pos100_variant(keep: str) -> StringIO:
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
    return StringIO(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
                "\tsample_a\tsample_b",
                *body,
                "1\t200\trs_c\tA\tG\t.\tPASS\t.\tGT\t1|0\t0|0",
                "",
            ]
        )
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


def test_calc_covariance_duplicate_position_matches_first_encountered_variant(
    tmp_path: Path,
) -> None:
    """The duplicate-position dedup must behave like keeping only the first
    VCF-encountered variant, matching the legacy reference's effective
    first-write-wins matrix-insert semantics (see
    notes/ldetect-original-main-pipeline-audit.md)."""
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    def _shrink_ld_100_200(keep: str, name: str) -> float:
        out_path = tmp_path / f"{name}.h5"
        calc_covariance(
            vcf_stream=_vcf_stream_with_pos100_variant(keep),
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


def test_calc_covariance_default_writes_full_schema(tmp_path: Path) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    out_path = tmp_path / "full.h5"
    calc_covariance(
        vcf_stream=_vcf_stream(),
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


def test_calc_covariance_compact_output_writes_only_compact_schema(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    out_path = tmp_path / "compact.h5"
    calc_covariance(
        vcf_stream=_vcf_stream(),
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
        vcf_stream=_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=full_path,
        cutoff=1e-7,
    )
    calc_covariance(
        vcf_stream=_vcf_stream(),
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
