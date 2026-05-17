"""Tests for array-backed LD metric evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ldetect2.io.partitions import CovarianceStore
from ldetect2.metric import Metric


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
        np.savez_compressed(chrom_dir / f"chr1.{start}.{end}.npz", **output)
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


def test_metric_covariance_loader_does_not_retain_raw_partitions(
    tmp_path: Path,
) -> None:
    from ldetect2._util.covariance_array import (
        load_chromosome_covariance,
        load_metric_covariance,
        metric_from_arrays,
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
    import ldetect2.pipeline as pipeline_mod
    from ldetect2._util.covariance_array import load_chromosome_covariance

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
