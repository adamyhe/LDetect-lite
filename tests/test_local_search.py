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
from ldetect2.io.partitions import CovarianceStore
from ldetect2.local_search import LocalSearch
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
    np.savez_compressed(chrom_dir / f"chr1.{loci[0]}.{loci[-1]}.npz", **output)
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
        np.savez_compressed(chrom_dir / f"chr1.{start}.{end}.npz", **output)
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
        np.savez_compressed(
            chrom_dir / f"chr1.{start}.{end}.npz",
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
        legacy_data = legacy.precomputed["data"][locus]
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
