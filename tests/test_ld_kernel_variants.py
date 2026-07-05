"""Correctness tests for prototype LD kernel variants against the reference kernel."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ldetect2._util.ld_kernel_variants import (
    _popcount_sum_rows,
    pack_haplotypes,
    pairwise_ld_bitpacked_popcount,
    pairwise_ld_chunked_matmul,
    pairwise_ld_row_vectorized,
)
from ldetect2.shrinkage import _genetic_stop_bounds_impl, _pairwise_ld_impl

NE = 11418.0
THETA = 0.01
CUTOFF = 1e-7

_VARIANTS = pytest.mark.parametrize(
    "kernel_fn",
    [
        pairwise_ld_row_vectorized,
        pairwise_ld_chunked_matmul,
        pairwise_ld_bitpacked_popcount,
    ],
    ids=["row_vectorized", "chunked_matmul", "bitpacked_popcount"],
)


def _hap_sums(hap_mat: np.ndarray) -> np.ndarray:
    return np.asarray(hap_mat.sum(axis=1), dtype=np.float64)


def _assert_kernel_matches_reference(
    kernel_fn,
    hap_mat: np.ndarray,
    gpos_arr: np.ndarray,
    n_ind: float,
    ne: float = NE,
    theta: float = THETA,
    cutoff: float = CUTOFF,
) -> None:
    """Run both kernels, sort each by (i, j), and assert set + value equality.

    The (i, j) index set is compared exactly: both candidate kernels compute
    ds2 from the same inputs before any summation-order divergence can flip
    the |ds2| >= cutoff decision. d_naive/ds2 values allow a small tolerance
    since matmul/BLAS may sum in a different order than elementwise np.sum.
    """
    hap_sums = _hap_sums(hap_mat)
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, ne, n_ind, cutoff)
    ref = _pairwise_ld_impl(
        hap_mat, gpos_arr, hap_sums, j_stop_by_i, ne, n_ind, theta, cutoff
    )
    cand = kernel_fn(hap_mat, gpos_arr, hap_sums, j_stop_by_i, ne, n_ind, theta, cutoff)

    ref_order = np.lexsort((ref[1], ref[0]))
    cand_order = np.lexsort((cand[1], cand[0]))
    ref_ii, ref_jj, ref_d, ref_ds2 = (a[ref_order] for a in ref)
    cand_ii, cand_jj, cand_d, cand_ds2 = (a[cand_order] for a in cand)

    np.testing.assert_array_equal(cand_ii, ref_ii)
    np.testing.assert_array_equal(cand_jj, ref_jj)
    np.testing.assert_allclose(cand_d, ref_d, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(cand_ds2, ref_ds2, atol=1e-9, rtol=1e-9)


@_VARIANTS
def test_hand_built_small_case(kernel_fn) -> None:
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
    _assert_kernel_matches_reference(kernel_fn, hap_mat, gpos_arr, n_ind=2.0)


@_VARIANTS
def test_randomized_medium_case(kernel_fn) -> None:
    rng = np.random.default_rng(7)
    hap_mat = rng.integers(0, 2, size=(60, 40), dtype=np.uint8)
    gpos_arr = np.cumsum(rng.uniform(0.0005, 0.02, size=60))
    _assert_kernel_matches_reference(kernel_fn, hap_mat, gpos_arr, n_ind=20.0)


def test_chunked_matmul_small_tile_size_matches_reference() -> None:
    rng = np.random.default_rng(7)
    hap_mat = rng.integers(0, 2, size=(60, 40), dtype=np.uint8)
    gpos_arr = np.cumsum(rng.uniform(0.0005, 0.02, size=60))
    n_ind = 20.0
    hap_sums = _hap_sums(hap_mat)
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, NE, n_ind, CUTOFF)

    ref = _pairwise_ld_impl(
        hap_mat, gpos_arr, hap_sums, j_stop_by_i, NE, n_ind, THETA, CUTOFF
    )
    cand = pairwise_ld_chunked_matmul(
        hap_mat, gpos_arr, hap_sums, j_stop_by_i, NE, n_ind, THETA, CUTOFF, tile_size=8
    )

    ref_order = np.lexsort((ref[1], ref[0]))
    cand_order = np.lexsort((cand[1], cand[0]))
    np.testing.assert_array_equal(cand[0][cand_order], ref[0][ref_order])
    np.testing.assert_array_equal(cand[1][cand_order], ref[1][ref_order])
    np.testing.assert_allclose(
        cand[3][cand_order], ref[3][ref_order], atol=1e-9, rtol=1e-9
    )


@_VARIANTS
def test_all_identical_haplotypes_no_overflow(kernel_fn) -> None:
    # n11 == n_haps == 300 for every pair; a naive uint8 accumulation would
    # wrap (300 % 256 == 44), so this guards against that regression.
    hap_mat = np.ones((50, 300), dtype=np.uint8)
    gpos_arr = np.arange(50, dtype=np.float64) * 0.001
    _assert_kernel_matches_reference(
        kernel_fn, hap_mat, gpos_arr, n_ind=150.0, ne=100.0
    )


@_VARIANTS
def test_all_zero_row_edge_case(kernel_fn) -> None:
    rng = np.random.default_rng(3)
    hap_mat = rng.integers(0, 2, size=(20, 30), dtype=np.uint8)
    hap_mat[5, :] = 0
    gpos_arr = np.cumsum(rng.uniform(0.0005, 0.02, size=20))
    _assert_kernel_matches_reference(kernel_fn, hap_mat, gpos_arr, n_ind=15.0)


@_VARIANTS
def test_single_snp_case(kernel_fn) -> None:
    hap_mat = np.array([[0, 1, 1, 0, 1, 0, 1, 1, 0, 1]], dtype=np.uint8)
    gpos_arr = np.array([0.0], dtype=np.float64)
    _assert_kernel_matches_reference(kernel_fn, hap_mat, gpos_arr, n_ind=5.0)


@_VARIANTS
def test_aggressive_cutoff_filters_most_pairs(kernel_fn) -> None:
    rng = np.random.default_rng(11)
    hap_mat = rng.integers(0, 2, size=(40, 50), dtype=np.uint8)
    gpos_arr = np.cumsum(rng.uniform(0.0005, 0.02, size=40))
    _assert_kernel_matches_reference(
        kernel_fn, hap_mat, gpos_arr, n_ind=25.0, cutoff=0.5
    )


@_VARIANTS
def test_extreme_cutoff_yields_empty_result(kernel_fn) -> None:
    rng = np.random.default_rng(11)
    hap_mat = rng.integers(0, 2, size=(40, 50), dtype=np.uint8)
    gpos_arr = np.cumsum(rng.uniform(0.0005, 0.02, size=40))
    n_ind = 25.0
    hap_sums = _hap_sums(hap_mat)
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, NE, n_ind, 10.0)

    cand = kernel_fn(hap_mat, gpos_arr, hap_sums, j_stop_by_i, NE, n_ind, THETA, 10.0)

    assert cand[0].size == 0
    assert cand[1].size == 0
    assert cand[2].size == 0
    assert cand[3].size == 0


@_VARIANTS
def test_diagonal_correction_applied_once_per_row(kernel_fn) -> None:
    rng = np.random.default_rng(5)
    hap_mat = rng.integers(0, 2, size=(15, 20), dtype=np.uint8)
    gpos_arr = np.cumsum(rng.uniform(0.0005, 0.005, size=15))
    n_ind = 10.0
    cutoff = 0.0  # nothing filtered, so every diagonal pair survives.
    hap_sums = _hap_sums(hap_mat)
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, NE, n_ind, cutoff)

    cand = kernel_fn(hap_mat, gpos_arr, hap_sums, j_stop_by_i, NE, n_ind, THETA, cutoff)
    ii, jj, d_naive, ds2 = cand

    diag_mask = ii == jj
    assert diag_mask.any()
    expected_pre_correction = (1.0 - THETA) ** 2 * d_naive[diag_mask]
    delta = ds2[diag_mask] - expected_pre_correction
    np.testing.assert_allclose(
        delta, (THETA / 2.0) * (1.0 - THETA / 2.0), atol=1e-9, rtol=1e-9
    )


@pytest.mark.parametrize("n_haps", [64, 128])
def test_bitpacked_popcount_n_haps_exact_multiple_of_64(n_haps: int) -> None:
    rng = np.random.default_rng(13)
    hap_mat = rng.integers(0, 2, size=(10, n_haps), dtype=np.uint8)
    gpos_arr = np.cumsum(rng.uniform(0.0005, 0.02, size=10))
    _assert_kernel_matches_reference(
        pairwise_ld_bitpacked_popcount, hap_mat, gpos_arr, n_ind=float(n_haps // 2)
    )


def test_bitpacked_popcount_n_haps_just_under_64_word_boundary() -> None:
    rng = np.random.default_rng(13)
    hap_mat = rng.integers(0, 2, size=(10, 63), dtype=np.uint8)
    gpos_arr = np.cumsum(rng.uniform(0.0005, 0.02, size=10))
    _assert_kernel_matches_reference(
        pairwise_ld_bitpacked_popcount, hap_mat, gpos_arr, n_ind=30.0
    )


def test_bitpacked_popcount_n_haps_just_over_64_word_boundary() -> None:
    rng = np.random.default_rng(13)
    hap_mat = rng.integers(0, 2, size=(10, 65), dtype=np.uint8)
    gpos_arr = np.cumsum(rng.uniform(0.0005, 0.02, size=10))
    _assert_kernel_matches_reference(
        pairwise_ld_bitpacked_popcount, hap_mat, gpos_arr, n_ind=32.0
    )


def test_pack_haplotypes_hand_computed_single_word_case() -> None:
    hap_mat = np.array([[1, 0, 1, 1, 0, 0, 0, 0]], dtype=np.uint8)
    packed = pack_haplotypes(hap_mat)
    # bits 0, 2, 3 set -> 1 + 4 + 8 = 13.
    np.testing.assert_array_equal(packed, np.array([[13]], dtype=np.uint64))
    assert packed.dtype == np.uint64


def test_pack_haplotypes_hand_computed_two_word_boundary_case() -> None:
    hap_mat = np.zeros((1, 65), dtype=np.uint8)
    hap_mat[0, 64] = 1
    packed = pack_haplotypes(hap_mat)
    # Haplotype 64 is bit 0 of word 1, pinning down the LSB/word convention.
    np.testing.assert_array_equal(packed, np.array([[0, 1]], dtype=np.uint64))


@pytest.mark.parametrize("n_haps", [1, 7, 63, 64, 65, 127, 128, 129, 300])
def test_pack_haplotypes_matches_naive_n11_across_word_boundaries(
    n_haps: int,
) -> None:
    rng = np.random.default_rng(17)
    hap_mat = rng.integers(0, 2, size=(8, n_haps), dtype=np.uint8)
    packed = pack_haplotypes(hap_mat)

    for i in range(hap_mat.shape[0]):
        for j in range(hap_mat.shape[0]):
            expected = int(np.sum(hap_mat[i].astype(np.int64) * hap_mat[j]))
            actual = sum(
                bin(int(w)).count("1") for w in (packed[i] & packed[j]).tolist()
            )
            assert actual == expected


def test_popcount_sum_rows_matches_python_bin_count() -> None:
    rng = np.random.default_rng(19)
    random_words = rng.integers(
        0, np.iinfo(np.uint64).max, size=(5, 3), dtype=np.uint64
    )
    edge_words = np.array(
        [
            [0, 0, 0],
            [0xFFFFFFFFFFFFFFFF, 0, 0],
            [0x5555555555555555, 0, 0],
            [0x8000000000000000, 0, 0],
        ],
        dtype=np.uint64,
    )
    bits = np.concatenate([random_words, edge_words])

    actual = _popcount_sum_rows(bits)
    expected = np.array(
        [sum(bin(int(w)).count("1") for w in row) for row in bits.tolist()]
    )
    np.testing.assert_array_equal(actual, expected)


# No adversarial padding-corruption test: pack_haplotypes zero-allocates
# padding via np.zeros and never writes past n_haps, so no code path can
# leave a padding bit set. The word-boundary tests above already fail loudly
# if that invariant were ever broken.


def test_no_production_import_of_variants_module() -> None:
    src_root = Path(__file__).resolve().parent.parent / "src" / "ldetect2"
    target = "ld_kernel_variants"
    for py_file in src_root.rglob("*.py"):
        if py_file.name == "ld_kernel_variants.py":
            continue
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and target in node.module
            ):
                pytest.fail(f"{py_file} imports research-only module {target}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if target in alias.name:
                        pytest.fail(f"{py_file} imports research-only module {target}")
