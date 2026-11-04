"""Tests for array-backed LD metric evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from ldetect_lite.io.covariance_hdf5 import write_covariance_partition_hdf5
from ldetect_lite.io.partitions import CovarianceStore, read_partitions
from ldetect_lite.metric import Metric
from tests._partition_fixtures import (
    divergent_overlap_partitions,
    first_write_wins_pair_value,
    make_custom_partitioned_store,
)


def _make_store(
    tmp_path: Path,
    loci: list[int],
    r2_by_pair: dict[tuple[int, int], float],
    partitions: list[tuple[int, int]] | None = None,
    compact: bool = False,
) -> CovarianceStore:
    root = tmp_path / "cov"
    chrom_dir = root / "chr1"
    chrom_dir.mkdir(parents=True)
    if partitions is None:
        partitions = [(loci[0], loci[-1])]
    (root / "chr1_partitions.txt").write_text(
        "\n".join(f"{start} {end}" for start, end in partitions) + "\n"
    )

    for start, end in partitions:
        rows: list[tuple[int, int, float]] = []
        for i, pos_i in enumerate(loci):
            for pos_j in loci[i:]:
                if start <= pos_i <= end and start <= pos_j <= end:
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


def _legacy_metric(store: CovarianceStore, breakpoints: list[int]) -> dict:
    metric = Metric("chr1", store, breakpoints, 100, 500)
    return metric._calc_metric_lean()


def _fast_metric(store: CovarianceStore, breakpoints: list[int]) -> dict:
    return Metric("chr1", store, breakpoints, 100, 500).calc_metric()


@pytest.mark.parametrize(
    "breakpoints",
    [
        [300],
        [200, 400],
        [100, 300],
        [200, 300, 400],
    ],
)
def test_array_metric_matches_legacy_for_breakpoint_shapes(
    tmp_path: Path,
    breakpoints: list[int],
) -> None:
    loci = [100, 200, 300, 400, 500]
    r2 = {
        (100, 200): 0.4,
        (100, 300): 0.2,
        (100, 400): 0.1,
        (200, 300): 0.3,
        (200, 500): 0.5,
        (300, 400): 0.7,
        (400, 500): 0.6,
    }
    store = _make_store(tmp_path, loci, r2)

    legacy = _legacy_metric(store, breakpoints)
    fast = _fast_metric(store, breakpoints)

    assert fast["sum"] == pytest.approx(legacy["sum"])
    assert fast["N_nonzero"] == legacy["N_nonzero"]
    assert fast["N_zero"] == pytest.approx(legacy["N_zero"])


def test_array_metric_deduplicates_overlapping_partitions(tmp_path: Path) -> None:
    loci = [100, 200, 300, 400, 500]
    r2 = {(100, 300): 0.5, (200, 400): 0.25, (300, 500): 0.75}
    store = _make_store(
        tmp_path,
        loci,
        r2,
        partitions=[(100, 400), (200, 500)],
    )

    legacy = _legacy_metric(store, [300])
    fast = _fast_metric(store, [300])

    assert fast["sum"] == pytest.approx(legacy["sum"])
    assert fast["N_nonzero"] == legacy["N_nonzero"]
    assert fast["N_zero"] == pytest.approx(legacy["N_zero"])


def test_array_metric_matches_legacy_with_divergent_overlap_pair(
    tmp_path: Path,
) -> None:
    """`_make_store`'s overlapping partitions always agree on shared pairs
    (they're derived from one shared r2_by_pair dict), so it never exercises
    actual overlap-resolution precedence. Use a fixture where the redundant
    pair (200, 400) has genuinely different values in each partition (0.7 vs
    0.2), and cross-check against an independent first-write-wins oracle."""
    partitions = divergent_overlap_partitions()
    store = make_custom_partitioned_store(tmp_path, partitions)

    legacy = Metric("chr1", store, [300], 100, 500)._calc_metric_lean()
    fast = Metric("chr1", store, [300], 100, 500).calc_metric()

    assert fast["sum"] == pytest.approx(legacy["sum"])
    assert fast["N_nonzero"] == legacy["N_nonzero"]
    assert fast["N_zero"] == pytest.approx(legacy["N_zero"])
    assert first_write_wins_pair_value(partitions, 200, 400) == pytest.approx(0.7)


def test_metric_covariance_loader_does_not_retain_raw_partitions(
    tmp_path: Path,
) -> None:
    from ldetect_lite._util.covariance_array import (
        load_chromosome_covariance,
        load_metric_covariance,
        metric_from_arrays,
        metric_from_files,
    )

    loci = [100, 200, 300, 400, 500]
    r2 = {(100, 300): 0.5, (200, 400): 0.25, (300, 500): 0.75}
    partitions = [(100, 400), (200, 500)]
    store = _make_store(tmp_path, loci, r2, partitions=partitions)

    full = load_chromosome_covariance("chr1", store, partitions, 100, 500)
    metric_only = load_metric_covariance("chr1", store, partitions, 100, 500)

    assert metric_only.partition_arrays == ()
    assert metric_from_arrays(metric_only, [300]) == pytest.approx(
        metric_from_arrays(full, [300])
    )
    assert metric_from_files(
        "chr1", store, partitions, 100, 500, [300]
    ) == pytest.approx(
        metric_from_arrays(full, [300])
    )


def test_streaming_metric_matches_materialized_array_for_multiple_breakpoints(
    tmp_path: Path,
) -> None:
    from ldetect_lite._util.covariance_array import (
        load_metric_covariance,
        metric_from_arrays,
        metric_from_files,
    )

    loci = [100, 200, 300, 400, 500, 600]
    r2 = {
        (100, 300): 0.5,
        (100, 500): 0.2,
        (200, 400): 0.25,
        (300, 500): 0.75,
        (400, 600): 0.125,
    }
    partitions = [(100, 400), (200, 600)]
    store = _make_store(tmp_path, loci, r2, partitions=partitions, compact=True)
    breakpoints = [250, 450]

    materialized = load_metric_covariance("chr1", store, partitions, 100, 600)

    assert metric_from_files(
        "chr1", store, partitions, 100, 600, breakpoints
    ) == pytest.approx(metric_from_arrays(materialized, breakpoints))


def test_streaming_metric_workers_match_single_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ldetect_lite._util.covariance_array as covariance_array
    from ldetect_lite._util.covariance_array import metric_from_files

    loci = [100, 200, 300, 400, 500, 600]
    r2 = {
        (100, 300): 0.5,
        (100, 500): 0.2,
        (200, 400): 0.25,
        (300, 500): 0.75,
        (400, 600): 0.125,
    }
    partitions = [(100, 400), (200, 600)]
    store = _make_store(tmp_path, loci, r2, partitions=partitions, compact=True)
    breakpoints = [250, 450]
    monkeypatch.setattr(covariance_array, "ProcessPoolExecutor", ThreadPoolExecutor)

    single = metric_from_files("chr1", store, partitions, 100, 600, breakpoints)
    parallel = metric_from_files(
        "chr1",
        store,
        partitions,
        100,
        600,
        breakpoints,
        workers=2,
    )

    assert parallel == pytest.approx(single)


def test_metric_accepts_compact_covariance_partition(tmp_path: Path) -> None:
    loci = [100, 200, 300, 400, 500]
    r2 = {(100, 300): 0.5, (200, 400): 0.25}
    store = _make_store(tmp_path, loci, r2, compact=True)

    metric = Metric("chr1", store, [300], 100, 500)

    assert metric.calc_metric()["sum"] == pytest.approx(0.25)


def test_high_precision_metric_ignores_supplied_covariance_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ldetect_lite.pipeline as pipeline_mod
    from ldetect_lite._util.covariance_array import load_chromosome_covariance

    loci = [100, 200, 300, 400, 500]
    store = _make_store(tmp_path, loci, {(200, 400): 0.25}, compact=True)
    partitions = [(100, 500)]
    cache = load_chromosome_covariance("chr1", store, partitions, 100, 500)

    def fail_array_metric(*args, **kwargs):
        raise AssertionError("high precision metrics must use the Decimal path")

    monkeypatch.setattr(pipeline_mod, "metric_from_arrays", fail_array_metric)

    result = pipeline_mod._apply_metric(
        "chr1",
        100,
        500,
        store,
        [300],
        use_decimal=True,
        covariance_arrays=cache,
    )

    assert result["sum"] == pytest.approx(0.25)


def _pre_reorder_metric_partition_result(
    *,
    name,
    store,
    partitions,
    p_index,
    start,
    end,
    snp_first,
    snp_last,
    breakpoints,
    unique_diag_pos,
    unique_diag_val,
):
    """Oracle: the per-partition metric as it was before the crossing-first
    reorder (diagonal lookups and masks on every pair row, crossing last)."""
    import ldetect_lite._util.covariance_array as ca

    lower_min, lower_max, include_lower_min = ca._owned_bounds(
        partitions, p_index, snp_first, snp_last
    )
    total_sum = 0.0
    n_nonzero = 0
    path = store.partition_path(name, start, end)
    with ca.open_covariance_reader(path, start, end) as reader:
        for chunk in reader.iter_owned_rows(
            lower_min,
            lower_max,
            snp_first,
            snp_last,
            ca._DEFAULT_CHUNK_ROWS,
            include_lower_min=include_lower_min,
        ):
            pair_mask = chunk.lo < chunk.hi
            if not np.any(pair_mask):
                continue
            pair_i = chunk.lo[pair_mask]
            pair_j = chunk.hi[pair_mask]
            pair_s = chunk.shrink_ld[pair_mask]
            diag_i_idx = np.searchsorted(unique_diag_pos, pair_i)
            diag_j_idx = np.searchsorted(unique_diag_pos, pair_j)
            has_diag = (diag_i_idx < unique_diag_pos.size) & (
                diag_j_idx < unique_diag_pos.size
            )
            safe_i_idx = np.minimum(diag_i_idx, unique_diag_pos.size - 1)
            safe_j_idx = np.minimum(diag_j_idx, unique_diag_pos.size - 1)
            has_diag &= (unique_diag_pos[safe_i_idx] == pair_i) & (
                unique_diag_pos[safe_j_idx] == pair_j
            )
            if not np.any(has_diag):
                continue
            pair_i = pair_i[has_diag]
            pair_j = pair_j[has_diag]
            pair_s = pair_s[has_diag]
            diag_i = unique_diag_val[diag_i_idx[has_diag]]
            diag_j = unique_diag_val[diag_j_idx[has_diag]]
            positive = (diag_i > 0.0) & (diag_j > 0.0)
            if not np.any(positive):
                continue
            pair_i = pair_i[positive]
            pair_j = pair_j[positive]
            pair_s = pair_s[positive]
            diag_i = diag_i[positive]
            diag_j = diag_j[positive]
            i_blocks = np.searchsorted(breakpoints, pair_i, side="left")
            j_blocks = np.searchsorted(breakpoints, pair_j, side="left")
            crossing = i_blocks != j_blocks
            crossing_count = int(np.count_nonzero(crossing))
            if crossing_count == 0:
                continue
            r2 = (
                pair_s[crossing]
                * pair_s[crossing]
                / (diag_i[crossing] * diag_j[crossing])
            )
            total_sum += float(np.sum(r2))
            n_nonzero += crossing_count
    return ca._MetricPartitionResult(
        p_index=p_index,
        total_sum=total_sum,
        n_nonzero=n_nonzero,
        rows_read=0,
        pair_rows=0,
        normalized_rows=0,
        crossing_rows=0,
        row_read_seconds=0.0,
        normalize_seconds=0.0,
        crossing_seconds=0.0,
    )


def _random_compact_store(tmp_path: Path, seed: int) -> tuple[CovarianceStore, list]:
    """Overlapping partitions with random LD, missing and zero diagonals."""
    rng = np.random.default_rng(seed)
    loci = np.sort(rng.choice(np.arange(100, 5000), size=120, replace=False))
    missing_diag = set(rng.choice(loci, size=8, replace=False).tolist())
    zero_diag = set(rng.choice(loci, size=8, replace=False).tolist())
    partitions = [
        (int(loci[0]), int(loci[59])),
        (int(loci[40]), int(loci[99])),
        (int(loci[80]), int(loci[-1])),
    ]
    root = tmp_path / f"cov{seed}"
    chrom_dir = root / "chr1"
    chrom_dir.mkdir(parents=True)
    (root / "chr1_partitions.txt").write_text(
        "\n".join(f"{a} {b}" for a, b in partitions) + "\n"
    )
    for a, b in partitions:
        part = [int(p) for p in loci if a <= p <= b]
        i_pos, j_pos, s = [], [], []
        for k, pi in enumerate(part):
            for pj in part[k : k + 25]:
                if pi == pj:
                    if pi in missing_diag:
                        continue
                    value = 0.0 if pi in zero_diag else rng.uniform(0.2, 1.0)
                elif rng.random() < 0.3:
                    continue
                else:
                    value = rng.normal(0.0, 0.2)
                i_pos.append(pi)
                j_pos.append(pj)
                s.append(value)
        write_covariance_partition_hdf5(
            chrom_dir / f"chr1.{a}.{b}.h5",
            i_pos=np.array(i_pos, dtype=np.int32),
            j_pos=np.array(j_pos, dtype=np.int32),
            shrink_ld=np.array(s),
        )
    return CovarianceStore(root=root), [int(p) for p in loci]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("chunk_rows", [7, 1_000_000])
def test_crossing_first_metric_is_bitwise_identical_to_previous_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    chunk_rows: int,
) -> None:
    import ldetect_lite._util.covariance_array as ca

    store, loci = _random_compact_store(tmp_path, seed)
    partitions = read_partitions("chr1", store)
    monkeypatch.setattr(ca, "_DEFAULT_CHUNK_ROWS", chunk_rows)
    rng = np.random.default_rng(100 + seed)
    breakpoint_sets = [
        [],
        [loci[30]],  # on a locus
        [loci[30] + 1, loci[70]],
        sorted(rng.choice(loci, size=6, replace=False).tolist()),
        [loci[0], loci[-1]],
    ]
    first, last = loci[0], loci[-1]
    new = [
        ca.metric_from_files("chr1", store, partitions, first, last, bp)
        for bp in breakpoint_sets
    ]
    monkeypatch.setattr(
        ca, "_metric_partition_result", _pre_reorder_metric_partition_result
    )
    old = [
        ca.metric_from_files("chr1", store, partitions, first, last, bp)
        for bp in breakpoint_sets
    ]
    assert new == old  # exact float equality, not approx
    assert any(result["N_nonzero"] > 0 for result in new)
