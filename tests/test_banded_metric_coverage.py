"""Correctness + size/query-time benchmark for the exact crossing-sum prototype.

Reuses the ``rich_store`` synthetic fixture from ``test_covariance_sidecars.py``
(same dataset where the flat 1D difference array was proven to overcount on
close breakpoints) to prove this decomposition gets the multi-crossing case
right, then benchmarks both variants against the real chr2 fixture (the same
one used in ``test_compact_schema_v2.py``) for the size/query-time comparison.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from ldetect_lite._util.banded_metric_coverage import (
    MergeSortRangeSumTree,
    normalize_v2_pairs,
    sum_crossing_linear_scan,
)
from ldetect_lite._util.compact_schema_v2 import (
    convert_v1_to_v2,
    read_v2_diagonal,
    read_v2_index_arrays,
)
from ldetect_lite._util.covariance_array import (
    load_chromosome_covariance,
    metric_from_arrays,
)
from ldetect_lite._util.covariance_sidecars import CovarianceSidecarAccumulator
from ldetect_lite.io.partitions import CovarianceStore
from tests.test_covariance_sidecars import _rich_positions, rich_store  # noqa: F401

_CHR2_START = 39967768
_CHR2_END = 40067768


def _v2_normalized_from_store(
    store: CovarianceStore, chrom: str, start: int, end: int, tmp_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v1_path = store.partition_path(chrom, start, end)
    v2_path = tmp_path / f"{chrom}.{start}.{end}.v2.h5"
    convert_v1_to_v2(v1_path, v2_path, start=start, end=end, chrom=chrom)
    positions, lo_rank_values, lo_offsets, hi_idx, shrink_ld = read_v2_index_arrays(
        v2_path
    )
    diag_pos, diag_val = read_v2_diagonal(v2_path)
    lo_rank_values, lo_offsets, hi_idx, r2 = normalize_v2_pairs(
        positions=positions,
        lo_rank_values=lo_rank_values,
        lo_offsets=lo_offsets,
        hi_idx=hi_idx,
        shrink_ld=shrink_ld,
        diag_pos=diag_pos,
        diag_val=diag_val,
    )
    return positions, lo_rank_values, lo_offsets, hi_idx, r2


def test_linear_scan_correct_including_close_breakpoints(
    rich_store: tuple[CovarianceStore, CovarianceSidecarAccumulator],  # noqa: F811
    tmp_path: Path,
) -> None:
    store, _sidecar = rich_store
    positions_snp = _rich_positions()
    snp_first, snp_last = positions_snp[0], positions_snp[-1]

    positions, lo_rank_values, lo_offsets, hi_idx, r2 = _v2_normalized_from_store(
        store, "chr1", snp_first, snp_last, tmp_path
    )

    cache = load_chromosome_covariance(
        "chr1", store, [(snp_first, snp_last)], snp_first, snp_last
    )
    # The exact case that made the flat 1D difference array overcount (see
    # the "violates single crossing assumption" test in
    # test_covariance_sidecars.py): a pair spans SNP indices 2..8, crossing
    # both breakpoints at once.
    breakpoints = [positions_snp[2], positions_snp[5], positions_snp[9]]
    expected = metric_from_arrays(cache, breakpoints)

    got = sum_crossing_linear_scan(
        positions=positions,
        lo_rank_values=lo_rank_values,
        lo_offsets=lo_offsets,
        hi_idx=hi_idx,
        r2=r2,
        breakpoints=np.asarray(breakpoints, dtype=np.int64),
    )
    assert got == pytest.approx(expected["sum"], rel=1e-9)


def test_merge_sort_tree_matches_linear_scan_and_metric_from_arrays(
    rich_store: tuple[CovarianceStore, CovarianceSidecarAccumulator],  # noqa: F811
    tmp_path: Path,
) -> None:
    store, _sidecar = rich_store
    positions_snp = _rich_positions()
    snp_first, snp_last = positions_snp[0], positions_snp[-1]

    positions, lo_rank_values, lo_offsets, hi_idx, r2 = _v2_normalized_from_store(
        store, "chr1", snp_first, snp_last, tmp_path
    )
    tree = MergeSortRangeSumTree.build(hi_idx, r2)
    total_mass = float(np.sum(r2))

    cache = load_chromosome_covariance(
        "chr1", store, [(snp_first, snp_last)], snp_first, snp_last
    )
    for breakpoints in (
        [positions_snp[3]],
        [positions_snp[1], positions_snp[9]],
        [positions_snp[2], positions_snp[5], positions_snp[9]],
        list(positions_snp[1:-1]),
    ):
        expected = metric_from_arrays(cache, breakpoints)
        bp_arr = np.asarray(breakpoints, dtype=np.int64)
        linear = sum_crossing_linear_scan(
            positions=positions,
            lo_rank_values=lo_rank_values,
            lo_offsets=lo_offsets,
            hi_idx=hi_idx,
            r2=r2,
            breakpoints=bp_arr,
        )
        tree_sum = tree.sum_crossing(
            positions=positions,
            lo_rank_values=lo_rank_values,
            lo_offsets=lo_offsets,
            total_mass=total_mass,
            breakpoints=bp_arr,
        )
        assert linear == pytest.approx(expected["sum"], rel=1e-9), breakpoints
        assert tree_sum == pytest.approx(expected["sum"], rel=1e-9), breakpoints


# ---------------------------------------------------------------------------
# Benchmark: banded persisted tree vs. the lo-less compact cache, real fixture
# ---------------------------------------------------------------------------


def test_banded_storage_and_query_time_vs_v2_compact_cache(
    example_data_dir: Path, example_store: CovarianceStore, tmp_path: Path
) -> None:
    positions, lo_rank_values, lo_offsets, hi_idx, r2 = _v2_normalized_from_store(
        example_store, "chr2", _CHR2_START, _CHR2_END, tmp_path
    )
    v2_path = tmp_path / "chr2.39967768.40067768.v2.h5"
    v2_size = v2_path.stat().st_size
    total_mass = float(np.sum(r2))

    build_start = time.perf_counter()
    tree = MergeSortRangeSumTree.build(hi_idx, r2)
    build_seconds = time.perf_counter() - build_start

    tree_path = tmp_path / "chr2.39967768.40067768.banded_tree.h5"
    tree.to_hdf5(tree_path)
    tree_size = tree_path.stat().st_size
    loaded_tree = MergeSortRangeSumTree.from_hdf5(tree_path)

    import pickle

    with open(
        example_data_dir / "minima/minima-EUR-chr2-50-39967768-40067768.pickle",
        "rb",
    ) as f:
        minima = pickle.load(f)
    breakpoint_sets = {
        "fourier_ls (12 bp)": np.asarray(
            sorted(int(x) for x in minima["fourier_ls"]["loci"]), dtype=np.int64
        ),
        "single breakpoint": np.asarray(
            [sorted(int(x) for x in minima["fourier_ls"]["loci"])[6]], dtype=np.int64
        ),
    }

    print(f"\n[banded benchmark] n_rows={r2.size}")
    print(
        f"[banded benchmark] v2_compact_cache_bytes={v2_size} "
        f"banded_tree_bytes={tree_size} "
        f"extra_storage_ratio={tree_size / v2_size:.2f}x "
        f"tree_build_seconds={build_seconds:.4f}"
    )

    for label, bp_arr in breakpoint_sets.items():
        n_reps = 20

        linear_start = time.perf_counter()
        for _ in range(n_reps):
            linear_sum = sum_crossing_linear_scan(
                positions=positions,
                lo_rank_values=lo_rank_values,
                lo_offsets=lo_offsets,
                hi_idx=hi_idx,
                r2=r2,
                breakpoints=bp_arr,
            )
        linear_seconds = (time.perf_counter() - linear_start) / n_reps

        tree_start = time.perf_counter()
        for _ in range(n_reps):
            tree_sum = loaded_tree.sum_crossing(
                positions=positions,
                lo_rank_values=lo_rank_values,
                lo_offsets=lo_offsets,
                total_mass=total_mass,
                breakpoints=bp_arr,
            )
        tree_seconds = (time.perf_counter() - tree_start) / n_reps

        assert linear_sum == pytest.approx(tree_sum, rel=1e-9)
        print(
            f"[banded benchmark] breakpoints={label} n_bp={bp_arr.size} "
            f"linear_scan_seconds={linear_seconds:.6f} "
            f"tree_query_seconds={tree_seconds:.6f} "
            f"speedup={linear_seconds / tree_seconds:.1f}x"
        )
