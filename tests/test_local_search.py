"""Tests for array-backed local search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ldetect2._util.covariance_array import (
    CovariancePartition,
    load_chromosome_covariance,
    local_search_partition,
)
from ldetect2.io.covariance_hdf5 import write_covariance_partition_hdf5
from ldetect2.io.partitions import CovarianceStore
from ldetect2.local_search import (
    DenseLocalSearchAccumulator,
    LocalSearch,
    _first_seen_pair_mask,
    _iter_hdf5_canonical_segment_rows,
    _materialize_canonical_row_stream,
    _open_hdf5_reader_pool,
    _segment_rows_from_hdf5_partitions,
    local_search_hdf5_partition,
)
from ldetect2.metric import Metric


def _make_store(
    tmp_path: Path,
    loci: list[int],
    r2_by_pair: dict[tuple[int, int], float],
    compact: bool = False,
) -> CovarianceStore:
    root = tmp_path / "cov"
    chrom_dir = root / "chr1"
    chrom_dir.mkdir(parents=True)
    (root / "chr1_partitions.txt").write_text(f"{loci[0]} {loci[-1]}\n")

    rows: list[tuple[int, int, float]] = []
    for i, pos_i in enumerate(loci):
        for pos_j in loci[i:]:
            r2 = 1.0 if pos_i == pos_j else r2_by_pair.get((pos_i, pos_j), 0.0)
            rows.append((pos_i, pos_j, float(np.sqrt(r2))))

    output = {
        "i_pos": np.array([r[0] for r in rows], dtype=np.int32),
        "j_pos": np.array([r[1] for r in rows], dtype=np.int32),
        "shrink_ld": np.array([r[2] for r in rows]),
    }
    if not compact:
        output.update(
            {
                "i_gpos": np.zeros(len(rows)),
                "j_gpos": np.zeros(len(rows)),
                "naive_ld": np.array([r[2] for r in rows]),
                "i_id": np.array([f"snp{r[0]}" for r in rows]),
                "j_id": np.array([f"snp{r[1]}" for r in rows]),
            }
        )
    write_covariance_partition_hdf5(
        chrom_dir / f"chr1.{loci[0]}.{loci[-1]}.h5",
        i_pos=output["i_pos"],
        j_pos=output["j_pos"],
        shrink_ld=output["shrink_ld"],
        naive_ld=output.get("naive_ld"),
        i_gpos=output.get("i_gpos"),
        j_gpos=output.get("j_gpos"),
        i_id=output.get("i_id"),
        j_id=output.get("j_id"),
    )
    return CovarianceStore(root=root)


def _make_partitioned_store(
    tmp_path: Path,
    loci: list[int],
    partitions: list[tuple[int, int]],
    r2_by_pair: dict[tuple[int, int], float],
    compact: bool = False,
) -> CovarianceStore:
    root = tmp_path / "cov"
    chrom_dir = root / "chr1"
    chrom_dir.mkdir(parents=True)
    with (root / "chr1_partitions.txt").open("w") as f:
        for start, end in partitions:
            f.write(f"{start} {end}\n")

    for start, end in partitions:
        partition_loci = [locus for locus in loci if start <= locus <= end]
        rows: list[tuple[int, int, float]] = []
        for i, pos_i in enumerate(partition_loci):
            for pos_j in partition_loci[i:]:
                r2 = 1.0 if pos_i == pos_j else r2_by_pair.get((pos_i, pos_j), 0.0)
                rows.append((pos_i, pos_j, float(np.sqrt(r2))))

        output = {
            "i_pos": np.array([r[0] for r in rows], dtype=np.int64),
            "j_pos": np.array([r[1] for r in rows], dtype=np.int64),
            "shrink_ld": np.array([r[2] for r in rows]),
        }
        if not compact:
            output.update(
                {
                    "i_gpos": np.zeros(len(rows)),
                    "j_gpos": np.zeros(len(rows)),
                    "naive_ld": np.array([r[2] for r in rows]),
                    "i_id": np.array([f"snp{r[0]}" for r in rows]),
                    "j_id": np.array([f"snp{r[1]}" for r in rows]),
                }
            )
        write_covariance_partition_hdf5(
            chrom_dir / f"chr1.{start}.{end}.h5",
            i_pos=output["i_pos"],
            j_pos=output["j_pos"],
            shrink_ld=output["shrink_ld"],
            naive_ld=output.get("naive_ld"),
            i_gpos=output.get("i_gpos"),
            j_gpos=output.get("j_gpos"),
            i_id=output.get("i_id"),
            j_id=output.get("j_id"),
        )
    return CovarianceStore(root=root)


def _make_custom_partitioned_store(
    tmp_path: Path,
    partitions: dict[tuple[int, int], list[tuple[int, int, float]]],
) -> CovarianceStore:
    root = tmp_path / "cov"
    chrom_dir = root / "chr1"
    chrom_dir.mkdir(parents=True)
    with (root / "chr1_partitions.txt").open("w") as f:
        for start, end in partitions:
            f.write(f"{start} {end}\n")

    for (start, end), rows in partitions.items():
        write_covariance_partition_hdf5(
            chrom_dir / f"chr1.{start}.{end}.h5",
            i_pos=np.array([row[0] for row in rows], dtype=np.int32),
            j_pos=np.array([row[1] for row in rows], dtype=np.int32),
            shrink_ld=np.array([row[2] for row in rows], dtype=np.float64),
        )
    return CovarianceStore(root=root)


def _metric(
    store: CovarianceStore,
    breakpoints: list[int],
    first: int = 100,
    last: int = 500,
) -> dict:
    return Metric("chr1", store, breakpoints, first, last).calc_metric()


def _search(
    store: CovarianceStore,
    breakpoints: list[int],
    idx: int,
    start: int,
    stop: int,
    *,
    use_decimal: bool,
    first: int = 100,
    last: int = 500,
) -> tuple[int | None, dict | None]:
    metric = _metric(store, breakpoints, first, last)
    search = LocalSearch(
        "chr1",
        start,
        stop,
        idx,
        breakpoints,
        metric["sum"],
        metric["N_zero"],
        store,
        use_decimal=use_decimal,
    )
    return search.search()


def _assert_searches_match(
    store: CovarianceStore,
    breakpoints: list[int],
    idx: int,
    start: int,
    stop: int,
    first: int = 100,
    last: int = 500,
) -> tuple[int | None, dict | None]:
    fast_bp, fast_metric = _search(
        store, breakpoints, idx, start, stop, use_decimal=False, first=first, last=last
    )
    legacy_bp, legacy_metric = _search(
        store, breakpoints, idx, start, stop, use_decimal=True, first=first, last=last
    )

    assert fast_bp == legacy_bp
    if legacy_metric is None:
        assert fast_metric is None
    else:
        assert fast_metric is not None
        assert fast_metric["sum"] == pytest.approx(float(legacy_metric["sum"]))
        assert fast_metric["N_zero"] == float(legacy_metric["N_zero"])
    return fast_bp, fast_metric


def _assert_precompute_matches_legacy(
    store: CovarianceStore,
    breakpoints: list[int],
    idx: int,
    start: int,
    stop: int,
    first: int,
    last: int,
) -> None:
    metric = _metric(store, breakpoints, first, last)
    fast = LocalSearch(
        "chr1",
        start,
        stop,
        idx,
        breakpoints,
        metric["sum"],
        metric["N_zero"],
        store,
        use_decimal=False,
    )
    legacy = LocalSearch(
        "chr1",
        start,
        stop,
        idx,
        breakpoints,
        metric["sum"],
        metric["N_zero"],
        store,
        use_decimal=True,
    )

    fast.init_search()
    legacy.init_search()

    assert fast._array_loci is not None
    assert fast._array_sum_vert is not None
    assert fast._array_sum_horiz is not None
    assert fast.precomputed["locus_list"] == legacy.precomputed["locus_list"]
    for offset, locus in enumerate(fast.precomputed["locus_list"]):
        legacy_data = legacy.precomputed["data"].get(
            locus, {"sum_vert": 0.0, "sum_horiz": 0.0}
        )
        assert fast._array_sum_vert[offset] == pytest.approx(
            float(legacy_data["sum_vert"])
        )
        assert fast._array_sum_horiz[offset] == pytest.approx(
            float(legacy_data["sum_horiz"])
        )


def test_local_search_partition_canonicalizes_rows_exactly() -> None:
    partition = CovariancePartition(
        start=100,
        end=400,
        i_pos=np.array([300, 100, 200, 100, 200, 400], dtype=np.int64),
        j_pos=np.array([100, 100, 100, 300, 200, 400], dtype=np.int64),
        shrink_ld=np.array([0.3, 1.0, 0.5, 0.9, 1.2, 0.0]),
    )

    canonical = local_search_partition(partition)

    np.testing.assert_array_equal(
        canonical.lo, np.array([100, 100, 100, 200, 400], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        canonical.hi, np.array([100, 200, 300, 200, 400], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        canonical.shrink_ld, np.array([1.0, 0.5, 0.3, 1.2, 0.0])
    )
    np.testing.assert_array_equal(
        canonical.diag_pos, np.array([100, 200, 400], dtype=np.int32)
    )
    np.testing.assert_array_equal(canonical.diag_val, np.array([1.0, 1.2, 0.0]))
    np.testing.assert_array_equal(
        canonical.loci, np.array([100, 200, 400], dtype=np.int64)
    )
    assert canonical.source_row_count == 6


def test_local_search_matches_legacy_with_duplicate_pairs(
    tmp_path: Path,
) -> None:
    rows = [
        (100, 100, 1.0),
        (200, 200, 1.0),
        (300, 300, 1.0),
        (400, 400, 1.0),
        (500, 500, 1.0),
        (300, 100, 0.6),
        (100, 300, 0.9),
        (200, 400, 0.7),
        (300, 500, 0.8),
    ]
    store = _make_custom_partitioned_store(tmp_path, {(100, 500): rows})

    bp, metric = _assert_searches_match(store, [200, 400], 0, 100, 400)

    assert bp is None or 100 <= bp <= 400
    assert metric is not None


def test_local_search_matches_legacy_with_cross_partition_duplicate_pairs(
    tmp_path: Path,
) -> None:
    partitions = {
        (100, 400): [
            (100, 100, 1.0),
            (200, 200, 1.0),
            (300, 300, 1.0),
            (400, 400, 1.0),
            (200, 400, 0.7),
        ],
        (200, 500): [
            (200, 200, 1.0),
            (300, 300, 1.0),
            (400, 400, 1.0),
            (500, 500, 1.0),
            (400, 200, 0.2),
            (300, 500, 0.8),
        ],
    }
    store = _make_custom_partitioned_store(tmp_path, partitions)

    _assert_precompute_matches_legacy(
        store,
        [200, 400],
        0,
        100,
        400,
        first=100,
        last=500,
    )


def test_hdf5_precompute_matches_legacy_with_duplicate_pairs(
    tmp_path: Path,
) -> None:
    partitions = {
        (100, 500): [
            (100, 100, 1.0),
            (200, 200, 1.0),
            (300, 300, 1.0),
            (400, 400, 1.0),
            (500, 500, 0.0),
            (100, 300, 0.8),
            (200, 400, 0.7),
            (300, 500, 0.6),
        ],
        (200, 600): [
            (200, 200, 1.0),
            (300, 300, 1.0),
            (400, 400, 1.0),
            (500, 500, 1.0),
            (600, 600, 1.0),
            (100, 300, 0.1),
            (400, 200, 0.2),
            (300, 500, 0.9),
        ],
        (400, 700): [
            (400, 400, 1.0),
            (600, 600, 1.0),
            (700, 700, 1.0),
            (400, 600, 0.5),
            (600, 700, 0.4),
        ],
    }
    store = _make_custom_partitioned_store(tmp_path, partitions)

    _assert_precompute_matches_legacy(
        store,
        [300, 500, 650],
        1,
        300,
        650,
        first=100,
        last=700,
    )
    bp, metric = _assert_searches_match(
        store,
        [300, 500, 650],
        1,
        300,
        650,
        first=100,
        last=700,
    )
    assert bp is None or 300 <= bp <= 650
    assert metric is not None


def test_hdf5_segment_stream_preserves_first_partition_duplicate_pair(
    tmp_path: Path,
) -> None:
    partitions = {
        (100, 400): [
            (100, 100, 1.0),
            (200, 200, 1.0),
            (300, 300, 1.0),
            (400, 400, 1.0),
            (200, 400, 0.7),
        ],
        (200, 500): [
            (200, 200, 1.0),
            (300, 300, 1.0),
            (400, 400, 1.0),
            (500, 500, 1.0),
            (400, 200, 0.2),
            (300, 500, 0.8),
        ],
    }
    store = _make_custom_partitioned_store(tmp_path, partitions)
    hdf5_partitions = [
        local_search_hdf5_partition("chr1", store, start, end)
        for start, end in partitions
    ]

    lo, hi, shrink = _segment_rows_from_hdf5_partitions(
        hdf5_partitions,
        active_min_lo=100,
        lo_min=100,
        lo_max=500,
        chunk_rows=2,
    )

    duplicate_idx = np.flatnonzero((lo == 200) & (hi == 400))
    assert duplicate_idx.size == 1
    assert shrink[int(duplicate_idx[0])] == pytest.approx(0.7)


def test_hdf5_segment_stream_matches_with_reader_pool_reuse(tmp_path: Path) -> None:
    partitions = {
        (100, 400): [
            (100, 100, 1.0),
            (200, 200, 1.0),
            (300, 300, 1.0),
            (400, 400, 1.0),
            (200, 400, 0.7),
        ],
        (200, 500): [
            (200, 200, 1.0),
            (300, 300, 1.0),
            (400, 400, 1.0),
            (500, 500, 1.0),
            (400, 200, 0.2),
            (300, 500, 0.8),
        ],
    }
    store = _make_custom_partitioned_store(tmp_path, partitions)
    hdf5_partitions = [
        local_search_hdf5_partition("chr1", store, start, end)
        for start, end in partitions
    ]

    baseline = _segment_rows_from_hdf5_partitions(
        hdf5_partitions,
        active_min_lo=100,
        lo_min=100,
        lo_max=500,
        chunk_rows=2,
    )
    with _open_hdf5_reader_pool(tuple(hdf5_partitions)) as readers_by_partition:
        pooled = _materialize_canonical_row_stream(
            _iter_hdf5_canonical_segment_rows(
                hdf5_partitions,
                active_min_lo=100,
                lo_min=100,
                lo_max=500,
                chunk_rows=2,
                readers_by_partition=readers_by_partition,
            )
        )

    for pooled_values, baseline_values in zip(pooled, baseline):
        np.testing.assert_array_equal(pooled_values, baseline_values)


def test_first_seen_pair_mask_uses_seen_rows_across_chunks() -> None:
    seen_hi_by_lo: dict[int, np.ndarray] = {}
    first_lo = np.array([100, 100, 100, 200], dtype=np.int32)
    first_hi = np.array([100, 200, 300, 200], dtype=np.int32)
    second_lo = np.array([100, 100, 100, 200, 200], dtype=np.int32)
    second_hi = np.array([200, 300, 400, 200, 500], dtype=np.int32)

    np.testing.assert_array_equal(
        _first_seen_pair_mask(first_lo, first_hi, seen_hi_by_lo),
        np.array([True, True, True, True]),
    )
    np.testing.assert_array_equal(
        _first_seen_pair_mask(second_lo, second_hi, seen_hi_by_lo),
        np.array([False, False, True, False, True]),
    )
    np.testing.assert_array_equal(
        seen_hi_by_lo[100],
        np.array([100, 200, 300, 400], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        seen_hi_by_lo[200],
        np.array([200, 500], dtype=np.int32),
    )


def test_dense_local_search_accumulator_sums_known_loci() -> None:
    accumulator = DenseLocalSearchAccumulator(np.array([100, 200, 300]))

    accumulator.add_vertical(
        np.array([100, 200, 200, 400]),
        np.array([1.0, 2.0, 3.0, 4.0]),
    )
    accumulator.add_horizontal(
        np.array([300, 100, 300, 500]),
        np.array([5.0, 7.0, 11.0, 13.0]),
    )

    np.testing.assert_allclose(accumulator.sum_vert, np.array([1.0, 5.0, 0.0]))
    np.testing.assert_allclose(accumulator.sum_horiz, np.array([7.0, 0.0, 16.0]))


def test_dense_local_search_accumulator_add_pairs_sums_known_loci() -> None:
    accumulator = DenseLocalSearchAccumulator(np.array([100, 200, 300]))

    accumulator.add_pairs(
        np.array([100, 200, 200, 400]),
        np.array([300, 100, 300, 500]),
        np.array([1.0, 2.0, 3.0, 4.0]),
    )

    np.testing.assert_allclose(accumulator.sum_vert, np.array([1.0, 5.0, 0.0]))
    np.testing.assert_allclose(accumulator.sum_horiz, np.array([2.0, 0.0, 4.0]))


def test_dense_local_search_accumulator_empty_chunks_are_noops() -> None:
    accumulator = DenseLocalSearchAccumulator(np.array([100, 200, 300]))

    accumulator.add_vertical(np.array([], dtype=np.int32), np.array([], dtype=float))
    accumulator.add_horizontal(np.array([400, 500]), np.array([1.0, 2.0]))

    np.testing.assert_allclose(accumulator.sum_vert, np.zeros(3))
    np.testing.assert_allclose(accumulator.sum_horiz, np.zeros(3))


def test_array_local_search_keeps_unchanged_breakpoint(tmp_path: Path) -> None:
    loci = [100, 200, 300, 400, 500]
    store = _make_store(tmp_path, loci, {})

    bp, metric = _assert_searches_match(store, [200, 400], 0, 100, 400)

    assert bp is None
    assert metric is not None


def test_array_local_search_matches_legacy_move_left(tmp_path: Path) -> None:
    loci = [100, 200, 300, 400, 500]
    r2 = {(300, 400): 1.0}
    store = _make_store(tmp_path, loci, r2)

    bp, _ = _assert_searches_match(store, [200, 400], 1, 200, 500)

    assert bp is not None
    assert bp < 400


def test_array_local_search_matches_legacy_move_right(tmp_path: Path) -> None:
    loci = [100, 200, 300, 400, 500]
    r2 = {
        (100, 200): 1.0,
        (100, 300): 1.0,
        (200, 300): 1.0,
        (400, 500): 1.0,
    }
    store = _make_store(tmp_path, loci, r2)

    bp, _ = _assert_searches_match(store, [200, 400], 0, 100, 400)

    assert bp is not None
    assert bp > 200


def test_array_local_search_matches_legacy_with_neighbor_bounds(
    tmp_path: Path,
) -> None:
    loci = [100, 200, 300, 400, 500]
    store = _make_store(tmp_path, loci, {})

    bp, _ = _assert_searches_match(store, [200, 300, 400], 1, 200, 400)

    assert bp is None or 200 <= bp <= 400


def test_local_search_matches_legacy_across_multiple_partitions(
    tmp_path: Path,
) -> None:
    loci = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    partitions = [(100, 400), (300, 700), (600, 900)]
    r2 = {
        (300, 500): 0.2,
        (400, 500): 0.7,
        (500, 600): 0.5,
        (600, 700): 0.3,
    }
    store = _make_partitioned_store(tmp_path, loci, partitions, r2)

    bp, metric = _assert_searches_match(
        store,
        [300, 600, 800],
        1,
        400,
        750,
        first=100,
        last=900,
    )

    assert bp is None or 400 <= bp <= 750
    assert metric is not None

    _assert_precompute_matches_legacy(
        store,
        [300, 600, 800],
        1,
        400,
        750,
        first=100,
        last=900,
    )


def test_float_local_search_uses_array_path_for_multiple_partitions(
    tmp_path: Path,
) -> None:
    loci = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    partitions = [(100, 400), (300, 700), (600, 900)]
    store = _make_partitioned_store(tmp_path, loci, partitions, {})
    metric = _metric(store, [300, 600, 800], 100, 900)
    search = LocalSearch(
        "chr1",
        400,
        750,
        1,
        [300, 600, 800],
        metric["sum"],
        metric["N_zero"],
        store,
        use_decimal=False,
    )

    search.init_search()

    assert search._array_loci is not None
    assert len(search.precomputed["locus_list"]) > 0
    assert search.precompute_stats.candidate_rows > 0
    assert search.precompute_stats.eligible_rows > 0
    assert search.precompute_stats.active_rows_peak > 0
    assert search.precompute_stats.segments > 0


def test_array_local_search_matches_legacy_across_compact_partitions(
    tmp_path: Path,
) -> None:
    loci = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    partitions = [(100, 400), (300, 700), (600, 900)]
    r2 = {
        (300, 500): 0.2,
        (400, 500): 0.7,
        (500, 600): 0.5,
        (600, 700): 0.3,
    }
    store = _make_partitioned_store(tmp_path, loci, partitions, r2, compact=True)

    bp, metric = _assert_searches_match(
        store,
        [300, 600, 800],
        1,
        400,
        750,
        first=100,
        last=900,
    )

    assert bp is None or 400 <= bp <= 750
    assert metric is not None


def test_cached_array_local_search_does_not_reload_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ldetect2.local_search as local_search_mod

    loci = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    partitions = [(100, 400), (300, 700), (600, 900)]
    store = _make_partitioned_store(tmp_path, loci, partitions, {(400, 500): 0.7})
    metric = _metric(store, [300, 600, 800], 100, 900)
    cache = load_chromosome_covariance("chr1", store, partitions, 100, 900)

    def fail_reload(*args, **kwargs):
        raise AssertionError("cached local search should not reload covariance arrays")

    monkeypatch.setattr(local_search_mod, "load_covariance_arrays", fail_reload)

    search = LocalSearch(
        "chr1",
        400,
        750,
        1,
        [300, 600, 800],
        metric["sum"],
        metric["N_zero"],
        store,
        covariance_cache=cache,
    )

    bp, search_metric = search.search()

    assert bp is None or 400 <= bp <= 750
    assert search_metric is not None


def test_local_search_fallback_accepts_compact_partition(tmp_path: Path) -> None:
    loci = [100, 200, 300, 400, 500]
    store = _make_store(tmp_path, loci, {}, compact=True)
    metric = _metric(store, [200, 400])
    search = LocalSearch(
        "chr1",
        100,
        400,
        0,
        [200, 400],
        metric["sum"],
        metric["N_zero"],
        store,
        use_decimal=True,
    )

    bp, search_metric = search.search()

    assert bp is None
    assert search_metric is not None
