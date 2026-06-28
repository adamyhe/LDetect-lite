"""Tests for the experimental normalized r2 Zarr cache."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ldetect2.io.covariance_hdf5 import write_covariance_partition_hdf5
from ldetect2.io.partitions import CovarianceStore
from ldetect2.io.r2_zarr import (
    R2RowChunk,
    open_r2_zarr_reader,
    validate_r2_zarr_partition,
    write_r2_zarr_partition_append,
)
from ldetect2.local_search import LocalSearch, local_search_r2_zarr_partition
from ldetect2.metric import Metric

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("zarr") is None,
    reason="zarr is not installed",
)


def _write_r2_partition(
    root: Path,
    chrom: str,
    start: int,
    end: int,
    rows: list[tuple[int, int, float]],
) -> None:
    positions = np.array(
        sorted({pos for row in rows for pos in row[:2]}), dtype=np.int32
    )
    write_r2_zarr_partition_append(
        root,
        chrom,
        start,
        end,
        positions=positions,
        row_chunks=iter(
            [
                R2RowChunk(
                    lo=np.array([row[0] for row in rows], dtype=np.int32),
                    hi=np.array([row[1] for row in rows], dtype=np.int32),
                    r2=np.array([row[2] for row in rows], dtype=np.float64),
                )
            ]
        ),
        ne=11418.0,
        cutoff=1e-7,
        chunk_rows=2,
        dataset_chunk_rows=2,
    )


def _make_dual_store(
    tmp_path: Path,
    loci: list[int],
    r2_by_pair: dict[tuple[int, int], float],
    partitions: list[tuple[int, int]],
) -> CovarianceStore:
    root = tmp_path / "cov"
    chrom_dir = root / "chr1"
    chrom_dir.mkdir(parents=True)
    (root / "chr1_partitions.txt").write_text(
        "\n".join(f"{start} {end}" for start, end in partitions) + "\n"
    )

    for start, end in partitions:
        rows: list[tuple[int, int, float]] = []
        for i, pos_i in enumerate(loci):
            if not (start <= pos_i <= end):
                continue
            for pos_j in loci[i:]:
                if not (start <= pos_j <= end):
                    continue
                r2 = 1.0 if pos_i == pos_j else r2_by_pair.get((pos_i, pos_j), 0.0)
                rows.append((pos_i, pos_j, r2))
        write_covariance_partition_hdf5(
            chrom_dir / f"chr1.{start}.{end}.h5",
            i_pos=np.array([row[0] for row in rows], dtype=np.int32),
            j_pos=np.array([row[1] for row in rows], dtype=np.int32),
            shrink_ld=np.sqrt(np.array([row[2] for row in rows], dtype=np.float64)),
        )
        _write_r2_partition(root, "chr1", start, end, rows)
    return CovarianceStore(root=root)


def test_r2_zarr_reader_streams_sorted_rows_and_preserves_diagonal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cov"
    rows = [
        (100, 100, 1.0),
        (100, 200, 0.25),
        (100, 300, 0.5),
        (200, 200, 1.0),
        (200, 300, 0.75),
        (300, 300, 1.0),
    ]
    _write_r2_partition(root, "chr1", 100, 300, rows)

    assert validate_r2_zarr_partition(root, "chr1", 100, 300)
    with open_r2_zarr_reader(root, "chr1", 100, 300) as reader:
        chunks = list(reader.iter_rows(100, 300, chunk_rows=2))

    lo = np.concatenate([chunk.lo for chunk in chunks])
    hi = np.concatenate([chunk.hi for chunk in chunks])
    r2 = np.concatenate([chunk.r2 for chunk in chunks])
    np.testing.assert_array_equal(lo, np.array([row[0] for row in rows]))
    np.testing.assert_array_equal(hi, np.array([row[1] for row in rows]))
    np.testing.assert_allclose(r2[lo == hi], np.ones(3))


def test_r2_zarr_owned_rows_filter_lower_bound(tmp_path: Path) -> None:
    root = tmp_path / "cov"
    rows = [
        (100, 100, 1.0),
        (100, 200, 0.25),
        (200, 200, 1.0),
        (200, 300, 0.75),
        (300, 300, 1.0),
    ]
    _write_r2_partition(root, "chr1", 100, 300, rows)

    with open_r2_zarr_reader(root, "chr1", 100, 300) as reader:
        chunks = list(
            reader.iter_owned_rows(
                100,
                300,
                100,
                300,
                chunk_rows=2,
                include_lower_min=False,
            )
        )

    lo = np.concatenate([chunk.lo for chunk in chunks])
    assert lo.tolist() == [200, 200, 300]


def test_r2_zarr_validation_rejects_missing_partition(tmp_path: Path) -> None:
    root = tmp_path / "cov"
    _write_r2_partition(root, "chr1", 100, 300, [(100, 100, 1.0)])

    assert not validate_r2_zarr_partition(root, "chr1", 200, 400)


def test_r2_zarr_empty_partition_validates(tmp_path: Path) -> None:
    root = tmp_path / "cov"
    write_r2_zarr_partition_append(
        root,
        "chr1",
        100,
        100,
        positions=np.array([], dtype=np.int32),
        row_chunks=iter(()),
        ne=11418.0,
        cutoff=1e-7,
    )

    assert validate_r2_zarr_partition(root, "chr1", 100, 100)
    with open_r2_zarr_reader(root, "chr1", 100, 100) as reader:
        assert reader.row_count == 0
        assert reader.read_loci().size == 0
        assert list(reader.iter_rows(0, 1, chunk_rows=1)) == []


def test_r2_zarr_metric_matches_hdf5_with_overlapping_partitions(
    tmp_path: Path,
) -> None:
    loci = [100, 200, 300, 400, 500, 600]
    partitions = [(100, 400), (200, 600)]
    r2 = {
        (100, 300): 0.5,
        (100, 500): 0.2,
        (200, 400): 0.25,
        (300, 500): 0.75,
        (400, 600): 0.125,
    }
    store = _make_dual_store(tmp_path, loci, r2, partitions)

    hdf5 = Metric("chr1", store, [250, 450], 100, 600).calc_metric()
    zarr = Metric(
        "chr1", store, [250, 450], 100, 600, pair_cache="r2-zarr"
    ).calc_metric()

    assert zarr == pytest.approx(hdf5)


def test_r2_zarr_local_search_matches_hdf5(tmp_path: Path) -> None:
    loci = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    partitions = [(100, 400), (300, 700), (600, 900)]
    r2 = {
        (300, 500): 0.2,
        (400, 500): 0.7,
        (500, 600): 0.5,
        (600, 700): 0.3,
    }
    store = _make_dual_store(tmp_path, loci, r2, partitions)
    breakpoints = [300, 600, 800]
    hdf5_metric = Metric("chr1", store, breakpoints, 100, 900).calc_metric()
    zarr_metric = Metric(
        "chr1", store, breakpoints, 100, 900, pair_cache="r2-zarr"
    ).calc_metric()
    r2_partitions = tuple(
        local_search_r2_zarr_partition("chr1", store, start, end)
        for start, end in partitions
    )

    hdf5_search = LocalSearch(
        "chr1",
        400,
        750,
        1,
        breakpoints,
        hdf5_metric["sum"],
        hdf5_metric["N_zero"],
        store,
    )
    zarr_search = LocalSearch(
        "chr1",
        400,
        750,
        1,
        breakpoints,
        zarr_metric["sum"],
        zarr_metric["N_zero"],
        store,
        local_search_r2_zarr_partitions=r2_partitions,
    )

    hdf5_bp, hdf5_out = hdf5_search.search()
    zarr_bp, zarr_out = zarr_search.search()
    assert zarr_bp == hdf5_bp
    if hdf5_out is None:
        assert zarr_out is None
    else:
        assert zarr_out is not None
        assert zarr_out["sum"] == pytest.approx(hdf5_out["sum"])
        assert zarr_out["N_zero"] == pytest.approx(hdf5_out["N_zero"])
