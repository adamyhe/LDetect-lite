"""Tests for ldetect2.io.covariance."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pytest

from ldetect2._util.covariance_array import load_chromosome_covariance
from ldetect2.io.covariance import (
    delete_loci_smaller_than,
    delete_loci_smaller_than_leanest,
    insert_into_matrix_lean,
    read_partition_into_matrix,
    read_partition_into_matrix_lean,
)
from ldetect2.io.covariance_hdf5 import (
    open_covariance_reader,
    write_covariance_partition_hdf5,
)
from ldetect2.io.partitions import CovarianceStore
from ldetect2.matrix_analysis import MatrixAnalysis
from tests._partition_fixtures import (
    divergent_overlap_partitions,
    make_custom_partitioned_store,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> tuple[CovarianceStore, list[tuple[int, int]]]:
    """Create a minimal CovarianceStore in *tmp_path* with a 3-SNP fixture."""
    root = tmp_path / "cov"
    root.mkdir(parents=True)
    (root / "testchr").mkdir()
    (root / "testchr_partitions.txt").write_text("100 300\n")
    h5_path = root / "testchr" / "testchr.100.300.h5"
    write_covariance_partition_hdf5(
        h5_path,
        i_pos=np.array([100, 100, 200, 200, 300], dtype=np.int32),
        j_pos=np.array([100, 200, 200, 300, 300], dtype=np.int32),
        shrink_ld=np.array([0.5, 0.3, 0.7, 0.1, 0.9]),
        i_gpos=np.array([1.0, 1.0, 2.0, 2.0, 3.0]),
        j_gpos=np.array([1.0, 2.0, 2.0, 3.0, 3.0]),
        naive_ld=np.array([0.5, 0.3, 0.7, 0.1, 0.9]),
        i_id=np.array(["snpA", "snpA", "snpB", "snpB", "snpC"]),
        j_id=np.array(["snpA", "snpB", "snpB", "snpC", "snpC"]),
    )
    store = CovarianceStore(root=root)
    partitions: list[tuple[int, int]] = [(100, 300)]
    return store, partitions


def _make_compact_store(
    tmp_path: Path,
) -> tuple[CovarianceStore, list[tuple[int, int]]]:
    """Create a minimal CovarianceStore with compact covariance fields only."""
    root = tmp_path / "cov"
    root.mkdir(parents=True)
    (root / "testchr").mkdir()
    (root / "testchr_partitions.txt").write_text("100 300\n")
    h5_path = root / "testchr" / "testchr.100.300.h5"
    write_covariance_partition_hdf5(
        h5_path,
        i_pos=np.array([100, 100, 200, 200, 300], dtype=np.int32),
        j_pos=np.array([100, 200, 200, 300, 300], dtype=np.int32),
        shrink_ld=np.array([0.5, 0.3, 0.7, 0.1, 0.9], dtype=np.float64),
    )
    store = CovarianceStore(root=root)
    partitions: list[tuple[int, int]] = [(100, 300)]
    return store, partitions


def test_hdf5_writer_trusted_canonical_path_matches_generic_path(
    tmp_path: Path,
) -> None:
    """Trusted writer mode should preserve canonical rows without reordering."""
    i_pos = np.array([100, 100, 200, 200, 300], dtype=np.int32)
    j_pos = np.array([100, 200, 200, 300, 300], dtype=np.int32)
    shrink_ld = np.array([0.5, 0.3, 0.7, 0.1, 0.9], dtype=np.float64)
    generic = tmp_path / "generic.h5"
    trusted = tmp_path / "trusted.h5"

    write_covariance_partition_hdf5(
        generic,
        i_pos=i_pos,
        j_pos=j_pos,
        shrink_ld=shrink_ld,
    )
    write_covariance_partition_hdf5(
        trusted,
        i_pos=i_pos,
        j_pos=j_pos,
        shrink_ld=shrink_ld,
        assume_canonical_sorted_unique=True,
    )

    import h5py

    with h5py.File(generic, "r") as generic_h5, h5py.File(trusted, "r") as trusted_h5:
        for dataset in (
            "covariance/lo",
            "covariance/hi",
            "covariance/shrink_ld",
            "index/diag_pos",
            "index/diag_val",
            "index/lo_values",
            "index/lo_offsets",
        ):
            np.testing.assert_array_equal(
                generic_h5[dataset][:], trusted_h5[dataset][:]
            )


def test_hdf5_reader_accepts_files_without_layout_attrs(tmp_path: Path) -> None:
    """Full-schema and legacy compact HDF5 files predate layout debug attrs."""
    path = tmp_path / "legacy.h5"
    write_covariance_partition_hdf5(
        path,
        i_pos=np.array([100, 100, 200], dtype=np.int32),
        j_pos=np.array([100, 200, 200], dtype=np.int32),
        shrink_ld=np.array([1.0, 0.5, 1.0], dtype=np.float64),
    )

    import h5py

    with h5py.File(path, "r") as h5:
        assert "dataset_chunk_rows" not in h5.attrs
        assert "write_chunk_rows" not in h5.attrs

    with open_covariance_reader(path, 100, 200) as reader:
        rows = reader.read_all()
        loci = reader.read_loci()

    np.testing.assert_array_equal(rows.lo, np.array([100, 100, 200]))
    np.testing.assert_array_equal(loci, np.array([100, 200]))


@pytest.mark.parametrize(
    ("i_pos", "j_pos", "message"),
    [
        (
            np.array([100, 300], dtype=np.int32),
            np.array([100, 200], dtype=np.int32),
            "canonical",
        ),
        (
            np.array([100, 300, 200], dtype=np.int32),
            np.array([100, 300, 200], dtype=np.int32),
            "sorted",
        ),
        (
            np.array([100, 200, 200], dtype=np.int32),
            np.array([100, 200, 200], dtype=np.int32),
            "duplicates",
        ),
    ],
)
def test_hdf5_writer_trusted_path_enforces_sorted_unique_rows(
    tmp_path: Path,
    i_pos: np.ndarray,
    j_pos: np.ndarray,
    message: str,
) -> None:
    """Trusted writer mode should fail instead of writing invalid indexes."""
    with pytest.raises(ValueError, match=message):
        write_covariance_partition_hdf5(
            tmp_path / "bad.h5",
            i_pos=i_pos,
            j_pos=j_pos,
            shrink_ld=np.ones(i_pos.size, dtype=np.float64),
            assume_canonical_sorted_unique=True,
        )


def _make_two_partition_store(
    tmp_path: Path,
    *,
    compact: bool = True,
) -> tuple[CovarianceStore, list[tuple[int, int]]]:
    root = tmp_path / "cov"
    chrom_dir = root / "testchr"
    chrom_dir.mkdir(parents=True)
    partitions = [(100, 500), (400, 800)]
    (root / "testchr_partitions.txt").write_text("100 500\n400 800\n")
    rows_by_partition = {
        (100, 500): [
            (100, 100, 1.0),
            (100, 200, 0.5),
            (200, 200, 1.0),
            (200, 400, 0.4),
            (400, 400, 1.0),
            (400, 500, 0.3),
            (500, 500, 1.0),
        ],
        (400, 800): [
            (400, 400, 1.0),
            (400, 500, 0.3),
            (500, 500, 1.0),
            (500, 700, 0.6),
            (700, 700, 1.0),
            (700, 800, 0.2),
            (800, 800, 1.0),
        ],
    }
    for start, end in partitions:
        rows = rows_by_partition[(start, end)]
        output = {
            "i_pos": np.array([row[0] for row in rows], dtype=np.int32),
            "j_pos": np.array([row[1] for row in rows], dtype=np.int32),
            "shrink_ld": np.array([row[2] for row in rows], dtype=np.float64),
        }
        if not compact:
            output.update(
                {
                    "i_gpos": np.zeros(len(rows)),
                    "j_gpos": np.zeros(len(rows)),
                    "naive_ld": np.array([row[2] for row in rows]),
                    "i_id": np.array([f"snp{row[0]}" for row in rows]),
                    "j_id": np.array([f"snp{row[1]}" for row in rows]),
                }
            )
        write_covariance_partition_hdf5(
            chrom_dir / f"testchr.{start}.{end}.h5",
            i_pos=output["i_pos"],
            j_pos=output["j_pos"],
            shrink_ld=output["shrink_ld"],
            naive_ld=output.get("naive_ld"),
            i_gpos=output.get("i_gpos"),
            j_gpos=output.get("j_gpos"),
            i_id=output.get("i_id"),
            j_id=output.get("j_id"),
        )
    return CovarianceStore(root=root), partitions


def _read_vector(path: Path) -> dict[int, float]:
    data: dict[int, float] = {}
    with gzip.open(path, "rt") as f:
        for raw in f:
            row = raw.strip().split()
            if row:
                data[int(row[0])] = float(row[1])
    return data


def _assert_vectors_close(left: Path, right: Path) -> None:
    left_data = _read_vector(left)
    right_data = _read_vector(right)
    assert left_data.keys() == right_data.keys()
    for locus, value in left_data.items():
        assert value == pytest.approx(right_data[locus])


# ---------------------------------------------------------------------------
# insert_into_matrix_lean
# ---------------------------------------------------------------------------

def test_insert_diagonal():
    matrix: dict = {}
    locus_list: list[int] = []
    row = ["snpA", "snpA", "100", "100", "1.0", "1.0", "0.5", "0.5"]
    insert_into_matrix_lean(row, matrix, locus_list)
    assert 100 in matrix
    assert matrix[100][100] == pytest.approx(0.5)
    assert locus_list == [100]


def test_insert_off_diagonal():
    matrix: dict = {}
    locus_list: list[int] = []
    row = ["snpA", "snpB", "100", "200", "1.0", "2.0", "0.3", "0.3"]
    insert_into_matrix_lean(row, matrix, locus_list)
    # lo=100, hi=200 — should create matrix[100][200] = 0.3
    assert matrix[100][200] == pytest.approx(0.3)


def test_insert_locus_list_sorted():
    matrix: dict = {}
    locus_list: list[int] = []
    rows = [
        ["snpA", "snpA", "100", "100", "1.0", "1.0", "0.5", "0.5"],
        ["snpA", "snpB", "100", "200", "1.0", "2.0", "0.3", "0.3"],
        ["snpB", "snpB", "200", "200", "2.0", "2.0", "0.7", "0.7"],
        ["snpB", "snpC", "200", "300", "2.0", "3.0", "0.1", "0.1"],
        ["snpC", "snpC", "300", "300", "3.0", "3.0", "0.9", "0.9"],
    ]
    for row in rows:
        insert_into_matrix_lean(row, matrix, locus_list)
    assert locus_list == [100, 200, 300]


def test_insert_no_duplicate():
    matrix: dict = {}
    locus_list: list[int] = []
    row = ["snpA", "snpA", "100", "100", "1.0", "1.0", "0.5", "0.5"]
    insert_into_matrix_lean(row, matrix, locus_list)
    insert_into_matrix_lean(row, matrix, locus_list)
    assert locus_list.count(100) == 1


# ---------------------------------------------------------------------------
# read_partition_into_matrix_lean
# ---------------------------------------------------------------------------

def test_read_partition_keys(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    assert set(locus_list) == {100, 200, 300}


def test_read_partition_diagonal_values(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    assert matrix[100][100] == pytest.approx(0.5)
    assert matrix[200][200] == pytest.approx(0.7)
    assert matrix[300][300] == pytest.approx(0.9)


def test_read_partition_off_diagonal_values(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    assert matrix[100][200] == pytest.approx(0.3)
    assert matrix[200][300] == pytest.approx(0.1)


def test_read_partition_lean_accepts_compact_hdf5(tmp_path):
    store, partitions = _make_compact_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []

    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )

    assert locus_list == [100, 200, 300]
    assert matrix[100][200] == pytest.approx(0.3)
    assert matrix[300][300] == pytest.approx(0.9)


def test_read_partition_full_rejects_compact_hdf5(tmp_path):
    store, partitions = _make_compact_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []

    with pytest.raises(ValueError, match="lacks full-metadata"):
        read_partition_into_matrix(
            partitions, 0, matrix, locus_list, "testchr", store, 100, 300
        )


def test_matrix_to_vector_lean_accepts_compact_hdf5(tmp_path):
    store, _ = _make_compact_store(tmp_path)
    output_path = tmp_path / "vector.txt.gz"

    MatrixAnalysis("testchr", store).calc_diag_lean(output_path)

    assert output_path.exists()


def test_matrix_to_vector_array_matches_legacy_single_partition(tmp_path):
    store, _ = _make_compact_store(tmp_path)
    array_path = tmp_path / "array.txt.gz"
    legacy_path = tmp_path / "legacy.txt.gz"

    MatrixAnalysis("testchr", store).calc_diag_array(array_path)
    MatrixAnalysis("testchr", store)._calc_diag_lean_legacy(legacy_path)

    _assert_vectors_close(array_path, legacy_path)


def test_matrix_to_vector_array_matches_legacy_overlapping_partitions(tmp_path):
    store, _ = _make_two_partition_store(tmp_path)
    array_path = tmp_path / "array.txt.gz"
    legacy_path = tmp_path / "legacy.txt.gz"

    MatrixAnalysis("testchr", store).calc_diag_array(array_path)
    MatrixAnalysis("testchr", store)._calc_diag_lean_legacy(legacy_path)

    _assert_vectors_close(array_path, legacy_path)


def test_matrix_to_vector_array_matches_legacy_with_divergent_overlap_pair(tmp_path):
    """`_make_two_partition_store`'s overlapping partitions always agree on
    shared pairs, so it never exercises actual overlap-resolution precedence.
    This uses a fixture where the redundant pair (200, 400) has genuinely
    different values in each partition (0.7 vs 0.2), forcing both the
    array-path (position-window ownership) and the legacy dict-path
    (read-order first-write-wins) to pick a specific winner."""
    store = make_custom_partitioned_store(tmp_path, divergent_overlap_partitions())
    array_path = tmp_path / "array.txt.gz"
    legacy_path = tmp_path / "legacy.txt.gz"

    MatrixAnalysis("chr1", store).calc_diag_array(array_path)
    MatrixAnalysis("chr1", store)._calc_diag_lean_legacy(legacy_path)

    _assert_vectors_close(array_path, legacy_path)


def _load_compare_partition_overlap_duplicates_module():
    """Dynamically load the example diagnostic script (not an importable package)."""
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "ldetect_original"
        / "scripts"
        / "compare_partition_overlap_duplicates.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_partition_overlap_duplicates", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_partition_overlap_duplicates_flags_mismatch(tmp_path):
    """Synthetic-fixture validation for the example diagnostic script, since
    real materialized covariance partitions aren't available to test against
    (see the script's module docstring)."""
    module = _load_compare_partition_overlap_duplicates_module()
    store = make_custom_partitioned_store(tmp_path, divergent_overlap_partitions())

    args = argparse.Namespace(
        population="EUR",
        chromosome="1",
        store_root=store.root,
        name="chr1",
        vcf_path=None,
    )
    result = module.compare(args)

    assert result["n_overlapping_partition_pairs"] == "1"
    assert int(result["n_redundant_pairs_checked"]) >= 1
    assert result["n_redundant_pairs_mismatched"] == "1"
    assert result["first_mismatch_lo"] == "200"
    assert result["first_mismatch_hi"] == "400"
    assert float(result["max_abs_shrink_diff"]) == pytest.approx(0.5)


@pytest.mark.parametrize("compact", [True, False])
def test_matrix_to_vector_chunked_hdf5_matches_materialized_cache(
    tmp_path,
    monkeypatch,
    compact,
):
    import ldetect2._util.vector_array as vector_array

    monkeypatch.setattr(vector_array, "MATRIX_TO_VECTOR_CHUNK_ROWS", 2)
    store, partitions = _make_two_partition_store(tmp_path, compact=compact)
    chunked_path = tmp_path / "chunked.txt.gz"
    cache_path = tmp_path / "cache.txt.gz"
    cache = load_chromosome_covariance("testchr", store, partitions, 100, 800)

    MatrixAnalysis("testchr", store).calc_diag_array(chunked_path)
    MatrixAnalysis("testchr", store).calc_diag_array(
        cache_path,
        covariance_cache=cache,
    )

    _assert_vectors_close(chunked_path, cache_path)


def test_matrix_to_vector_hdf5_workers_match_single_process(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import ldetect2._util.vector_array as vector_array

    monkeypatch.setattr(vector_array, "MATRIX_TO_VECTOR_CHUNK_ROWS", 2)
    monkeypatch.setattr(vector_array, "ProcessPoolExecutor", ThreadPoolExecutor)
    store, _ = _make_two_partition_store(tmp_path)
    worker_path = tmp_path / "worker.txt.gz"
    single_path = tmp_path / "single.txt.gz"

    MatrixAnalysis("testchr", store).calc_diag_array(worker_path, matrix_workers=2)
    MatrixAnalysis("testchr", store).calc_diag_array(single_path)

    _assert_vectors_close(worker_path, single_path)


def test_matrix_to_vector_array_accepts_chromosome_covariance_cache(tmp_path):
    store, partitions = _make_two_partition_store(tmp_path)
    cached_path = tmp_path / "cached.txt.gz"
    file_path = tmp_path / "file.txt.gz"
    cache = load_chromosome_covariance("testchr", store, partitions, 100, 800)

    MatrixAnalysis("testchr", store).calc_diag_array(
        cached_path,
        covariance_cache=cache,
    )
    MatrixAnalysis("testchr", store).calc_diag_array(file_path)

    _assert_vectors_close(cached_path, file_path)


def test_matrix_to_vector_array_accepts_full_hdf5(tmp_path):
    store, _ = _make_two_partition_store(tmp_path, compact=False)
    output_path = tmp_path / "vector.txt.gz"

    MatrixAnalysis("testchr", store).calc_diag_array(output_path)

    assert _read_vector(output_path)


def test_matrix_to_vector_array_respects_explicit_snp_range(tmp_path, monkeypatch):
    import ldetect2._util.vector_array as vector_array

    monkeypatch.setattr(vector_array, "MATRIX_TO_VECTOR_CHUNK_ROWS", 2)
    store, _ = _make_two_partition_store(tmp_path)
    array_path = tmp_path / "array.txt.gz"
    legacy_path = tmp_path / "legacy.txt.gz"

    MatrixAnalysis("testchr", store, snp_first=400, snp_last=800).calc_diag_array(
        array_path
    )
    MatrixAnalysis(
        "testchr", store, snp_first=400, snp_last=800
    )._calc_diag_lean_legacy(legacy_path)

    _assert_vectors_close(array_path, legacy_path)
    assert min(_read_vector(array_path)) >= 400


def test_matrix_to_vector_array_skips_rows_without_positive_diagonal(
    tmp_path,
):
    root = tmp_path / "cov"
    chrom_dir = root / "testchr"
    chrom_dir.mkdir(parents=True)
    (root / "testchr_partitions.txt").write_text("100 300\n")
    write_covariance_partition_hdf5(
        chrom_dir / "testchr.100.300.h5",
        i_pos=np.array([100, 100, 200, 200, 300], dtype=np.int32),
        j_pos=np.array([100, 200, 200, 300, 300], dtype=np.int32),
        shrink_ld=np.array([1.0, 0.5, 0.0, 0.2, 1.0], dtype=np.float64),
    )
    output_path = tmp_path / "vector.txt.gz"

    MatrixAnalysis("testchr", CovarianceStore(root=root)).calc_diag_array(output_path)

    assert _read_vector(output_path) == {100: pytest.approx(1.0)}


def test_matrix_to_vector_array_requires_hdf5_partition(tmp_path):
    root = tmp_path / "cov"
    chrom_dir = root / "testchr"
    chrom_dir.mkdir(parents=True)
    (root / "testchr_partitions.txt").write_text("100 300\n")

    with pytest.raises(FileNotFoundError, match="requires HDF5 covariance"):
        MatrixAnalysis("testchr", CovarianceStore(root=root)).calc_diag_array(
            tmp_path / "vector.txt.gz"
        )


# ---------------------------------------------------------------------------
# delete_loci_smaller_than_leanest
# ---------------------------------------------------------------------------

def test_delete_loci_removes_entries(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    delete_loci_smaller_than_leanest(250, matrix, locus_list)
    assert 100 not in matrix
    assert 200 not in matrix
    assert 300 in matrix


def test_delete_loci_updates_locus_list(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    delete_loci_smaller_than_leanest(250, matrix, locus_list)
    assert locus_list == [300]


def test_delete_loci_exact_boundary(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    # cutoff=200 → delete loci < 200 → only locus 100 removed
    delete_loci_smaller_than_leanest(200, matrix, locus_list)
    assert 100 not in matrix
    assert 200 in matrix
    assert locus_list == [200, 300]


# ---------------------------------------------------------------------------
# delete_loci_smaller_than (with deleted list)
# ---------------------------------------------------------------------------

def test_delete_with_deleted_list(tmp_path):
    store, partitions = _make_store(tmp_path)
    matrix: dict = {}
    locus_list: list[int] = []
    locus_list_deleted: list[int] = []
    read_partition_into_matrix_lean(
        partitions, 0, matrix, locus_list, "testchr", store, 100, 300
    )
    delete_loci_smaller_than(250, matrix, locus_list, locus_list_deleted)
    assert locus_list_deleted == [100, 200]
    assert locus_list == [300]
