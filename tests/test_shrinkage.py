"""Tests for shrinkage LD covariance calculation."""

from __future__ import annotations

import gzip
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from ldetect2.io.covariance_hdf5 import (
    HDF5_DATASET_CHUNK_ROWS,
    open_covariance_reader,
)
from ldetect2.io.partitions import CovarianceStore
from ldetect2.io.r2_zarr import open_r2_zarr_reader, validate_r2_zarr_partition
from ldetect2.matrix_analysis import MatrixAnalysis
from ldetect2.shrinkage import (
    _count_pairwise_ld_by_i_impl,
    _genetic_stop_bounds_impl,
    calc_covariance,
    calc_covariance_vector,
    calc_r2_zarr_partition,
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


def _write_overlapping_map(path: Path) -> None:
    with gzip.open(path, "wt") as f:
        for idx, pos in enumerate([100, 200, 300, 400, 500]):
            f.write(f"1 {pos} {idx * 0.001:.3f}\n")


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


def _read_vector(path: Path) -> dict[int, float]:
    data: dict[int, float] = {}
    with gzip.open(path, "rt") as f:
        for raw in f:
            row = raw.strip().split()
            if row:
                data[int(row[0])] = float(row[1])
    return data


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


def _overlapping_vcf_stream(start: int = 100, end: int = 500) -> StringIO:
    records = [
        (100, "rs_a", "0|1", "0|0"),
        (200, "rs_b", "0|1", "0|1"),
        (300, "rs_c", "1|0", "0|1"),
        (400, "rs_d", "1|1", "0|1"),
        (500, "rs_e", "0|0", "1|0"),
    ]
    lines = [
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
        "\tsample_a\tsample_b",
    ]
    for pos, rs, gt_a, gt_b in records:
        if start <= pos <= end:
            lines.append(f"1\t{pos}\t{rs}\tA\tG\t.\tPASS\t.\tGT\t{gt_a}\t{gt_b}")
    lines.append("")
    return StringIO("\n".join(lines))


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


def test_calc_covariance_retains_legacy_duplicate_physical_positions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Duplicate VCF positions should keep legacy first-physical-pair precedence."""
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
    assert "retained 1 duplicate-position variant" in captured.err

    with open_covariance_reader(out_path, 100, 300) as reader:
        rows = reader.read_all()
        lo_values = reader.read_loci()

    assert np.all(rows.lo <= rows.hi)
    assert np.all(
        (rows.lo[1:] > rows.lo[:-1])
        | ((rows.lo[1:] == rows.lo[:-1]) & (rows.hi[1:] > rows.hi[:-1]))
    )
    np.testing.assert_array_equal(lo_values, np.unique(rows.lo))
    assert rows.lo.tolist().count(100) > 0


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


def test_calc_covariance_vector_matches_matrix_to_vector(tmp_path: Path) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    root = tmp_path / "store"
    chrom = "chr1"
    chrom_dir = root / chrom
    chrom_dir.mkdir(parents=True)
    (root / f"{chrom}_partitions.txt").write_text("100 300\n")

    covariance_path = chrom_dir / f"{chrom}.100.300.h5"
    direct_path = tmp_path / "direct-vector.txt.gz"
    matrix_path = tmp_path / "matrix-vector.txt.gz"
    calc_covariance(
        vcf_stream=_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=covariance_path,
        cutoff=1e-7,
        compact_output=True,
    )
    calc_covariance_vector(
        vcf_stream=_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=direct_path,
        cutoff=1e-7,
    )
    MatrixAnalysis(chrom, CovarianceStore(root=root)).calc_diag_lean(matrix_path)

    direct = _read_vector(direct_path)
    matrix = _read_vector(matrix_path)
    assert direct.keys() == matrix.keys()
    for pos, value in direct.items():
        assert value == pytest.approx(matrix[pos])


def test_calc_covariance_vector_duplicate_positions_match_matrix_to_vector(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    root = tmp_path / "store"
    chrom = "chr1"
    chrom_dir = root / chrom
    chrom_dir.mkdir(parents=True)
    (root / f"{chrom}_partitions.txt").write_text("100 300\n")

    covariance_path = chrom_dir / f"{chrom}.100.300.h5"
    direct_path = tmp_path / "direct-vector.txt.gz"
    matrix_path = tmp_path / "matrix-vector.txt.gz"
    calc_covariance(
        vcf_stream=_duplicate_position_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=covariance_path,
        cutoff=1e-7,
        compact_output=True,
    )
    calc_covariance_vector(
        vcf_stream=_duplicate_position_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=direct_path,
        cutoff=1e-7,
    )
    MatrixAnalysis(chrom, CovarianceStore(root=root)).calc_diag_lean(matrix_path)

    direct = _read_vector(direct_path)
    matrix = _read_vector(matrix_path)
    assert direct.keys() == matrix.keys()
    for pos, value in direct.items():
        assert value == pytest.approx(matrix[pos])


def test_calc_r2_zarr_partition_matches_covariance_normalization(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    hdf5_path = tmp_path / "cov.h5"
    root = tmp_path / "store"
    chrom = "chr1"
    calc_covariance(
        vcf_stream=_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=hdf5_path,
        cutoff=1e-7,
        compact_output=True,
    )
    calc_r2_zarr_partition(
        vcf_stream=_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_root=root,
        name=chrom,
        start=100,
        end=300,
        cutoff=1e-7,
        compact_chunk_rows=2,
    )

    assert validate_r2_zarr_partition(root, chrom, 100, 300)
    with open_covariance_reader(hdf5_path, 100, 300) as hdf5_reader:
        hdf5_rows = hdf5_reader.read_all()
        diag_pos, diag_val = hdf5_reader.read_diagonal()
    with open_r2_zarr_reader(root, chrom, 100, 300) as r2_reader:
        r2_chunks = list(r2_reader.iter_rows(100, 300, chunk_rows=2))

    lo = np.concatenate([chunk.lo for chunk in r2_chunks])
    hi = np.concatenate([chunk.hi for chunk in r2_chunks])
    r2 = np.concatenate([chunk.r2 for chunk in r2_chunks])
    expected: dict[tuple[int, int], float] = {}
    for lo_pos, hi_pos, shrink in zip(
        hdf5_rows.lo, hdf5_rows.hi, hdf5_rows.shrink_ld, strict=True
    ):
        lo_diag = diag_val[np.searchsorted(diag_pos, lo_pos)]
        hi_diag = diag_val[np.searchsorted(diag_pos, hi_pos)]
        expected[(int(lo_pos), int(hi_pos))] = (
            1.0 if lo_pos == hi_pos else float(shrink * shrink / (lo_diag * hi_diag))
        )

    assert set(zip(lo.tolist(), hi.tolist(), strict=True)) == set(expected)
    for lo_pos, hi_pos, value in zip(lo, hi, r2, strict=True):
        assert value == pytest.approx(expected[(int(lo_pos), int(hi_pos))])


def test_calc_r2_zarr_partition_duplicate_positions_match_covariance(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    hdf5_path = tmp_path / "cov.h5"
    root = tmp_path / "store"
    chrom = "chr1"
    calc_covariance(
        vcf_stream=_duplicate_position_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=hdf5_path,
        cutoff=1e-7,
        compact_output=True,
    )
    calc_r2_zarr_partition(
        vcf_stream=_duplicate_position_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_root=root,
        name=chrom,
        start=100,
        end=300,
        cutoff=1e-7,
        compact_chunk_rows=2,
    )

    assert validate_r2_zarr_partition(root, chrom, 100, 300)
    with open_covariance_reader(hdf5_path, 100, 300) as hdf5_reader:
        hdf5_rows = hdf5_reader.read_all()
        diag_pos, diag_val = hdf5_reader.read_diagonal()
    with open_r2_zarr_reader(root, chrom, 100, 300) as r2_reader:
        r2_rows = r2_reader.read_loci()
        r2_chunks = list(r2_reader.iter_rows(100, 300, chunk_rows=2))

    assert len(r2_rows) == len(set(r2_rows.tolist()))
    lo = np.concatenate([chunk.lo for chunk in r2_chunks])
    hi = np.concatenate([chunk.hi for chunk in r2_chunks])
    r2 = np.concatenate([chunk.r2 for chunk in r2_chunks])
    expected: dict[tuple[int, int], float] = {}
    for lo_pos, hi_pos, shrink in zip(
        hdf5_rows.lo, hdf5_rows.hi, hdf5_rows.shrink_ld, strict=True
    ):
        lo_diag = diag_val[np.searchsorted(diag_pos, lo_pos)]
        hi_diag = diag_val[np.searchsorted(diag_pos, hi_pos)]
        expected[(int(lo_pos), int(hi_pos))] = (
            1.0 if lo_pos == hi_pos else float(shrink * shrink / (lo_diag * hi_diag))
        )

    assert set(zip(lo.tolist(), hi.tolist(), strict=True)) == set(expected)
    for lo_pos, hi_pos, value in zip(lo, hi, r2, strict=True):
        assert value == pytest.approx(expected[(int(lo_pos), int(hi_pos))])


def test_calc_covariance_vector_partition_bounds_match_matrix_to_vector(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map.gz"
    _write_overlapping_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    root = tmp_path / "store"
    chrom = "chr1"
    chrom_dir = root / chrom
    chrom_dir.mkdir(parents=True)
    (root / f"{chrom}_partitions.txt").write_text("100 400\n300 500\n")

    calc_covariance(
        vcf_stream=_overlapping_vcf_stream(100, 400),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=chrom_dir / f"{chrom}.100.400.h5",
        cutoff=1e-7,
        compact_output=True,
    )
    calc_covariance(
        vcf_stream=_overlapping_vcf_stream(300, 500),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=chrom_dir / f"{chrom}.300.500.h5",
        cutoff=1e-7,
        compact_output=True,
    )

    direct_path = tmp_path / "direct-vector.txt.gz"
    matrix_path = tmp_path / "matrix-vector.txt.gz"
    calc_covariance_vector(
        vcf_stream=_overlapping_vcf_stream(100, 400),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=direct_path,
        cutoff=1e-7,
        center_lower_bound=100,
        center_upper_bound=350,
    )
    calc_covariance_vector(
        vcf_stream=_overlapping_vcf_stream(300, 500),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=direct_path,
        cutoff=1e-7,
        center_lower_bound=350,
        center_lower_inclusive=False,
        center_upper_bound=500,
        center_upper_inclusive=False,
        append_output=True,
    )
    MatrixAnalysis(chrom, CovarianceStore(root=root)).calc_diag_lean(matrix_path)

    direct = _read_vector(direct_path)
    matrix = _read_vector(matrix_path)
    assert direct.keys() == matrix.keys()
    for pos, value in direct.items():
        assert value == pytest.approx(matrix[pos])


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
