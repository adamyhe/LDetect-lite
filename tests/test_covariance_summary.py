"""Tests for covariance-summary diagnostics."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from ldetect2._cli.main import main
from ldetect2._util.covariance_summary import summarize_covariance
from ldetect2.io.covariance_hdf5 import write_covariance_partition_hdf5
from ldetect2.io.partitions import CovarianceStore


def _write_hdf5_partition(
    root: Path,
    name: str,
    start: int,
    end: int,
    rows: list[tuple[int, int]],
) -> None:
    chrom_dir = root / name
    chrom_dir.mkdir(parents=True, exist_ok=True)
    write_covariance_partition_hdf5(
        chrom_dir / f"{name}.{start}.{end}.h5",
        i_pos=np.array([row[0] for row in rows], dtype=np.int64),
        j_pos=np.array([row[1] for row in rows], dtype=np.int64),
        shrink_ld=np.ones(len(rows)),
        i_gpos=np.zeros(len(rows)),
        j_gpos=np.zeros(len(rows)),
        naive_ld=np.ones(len(rows)),
        i_id=np.array([f"snp{row[0]}" for row in rows]),
        j_id=np.array([f"snp{row[1]}" for row in rows]),
    )


def _make_hdf5_store(
    tmp_path: Path,
    partitions: list[tuple[int, int]],
    rows_by_partition: dict[tuple[int, int], list[tuple[int, int]]],
) -> CovarianceStore:
    root = tmp_path / "cov"
    root.mkdir()
    (root / "chr1_partitions.txt").write_text(
        "\n".join(f"{start} {end}" for start, end in partitions) + "\n"
    )
    for start, end in partitions:
        _write_hdf5_partition(
            root, "chr1", start, end, rows_by_partition[(start, end)]
        )
    return CovarianceStore(root=root)


def test_summarize_covariance_single_hdf5_partition(tmp_path: Path) -> None:
    store = _make_hdf5_store(
        tmp_path,
        [(100, 300)],
        {
            (100, 300): [
                (100, 100),
                (100, 200),
                (100, 300),
                (200, 200),
                (200, 300),
                (300, 300),
            ],
        },
    )

    parts, total = summarize_covariance("chr1", store)

    assert len(parts) == 1
    assert parts[0].rows == 6
    assert parts[0].diag_rows == 3
    assert parts[0].offdiag_rows == 3
    assert parts[0].owned_offdiag_rows == 3
    assert parts[0].unique_loci == 3
    assert total.as_dict()["partition_start"] == "TOTAL"
    assert total.owned_offdiag_rows == 3


def test_summarize_covariance_uses_overlap_ownership(tmp_path: Path) -> None:
    rows = {
        (100, 400): [
            (100, 100),
            (100, 200),
            (100, 300),
            (200, 200),
            (200, 300),
            (200, 400),
            (300, 300),
            (300, 400),
            (400, 400),
        ],
        (200, 500): [
            (200, 200),
            (200, 300),
            (300, 300),
            (300, 400),
            (300, 500),
            (400, 400),
            (400, 500),
            (500, 500),
        ],
    }
    store = _make_hdf5_store(tmp_path, [(100, 400), (200, 500)], rows)

    parts, total = summarize_covariance("chr1", store)

    assert [part.owned_offdiag_rows for part in parts] == [4, 3]
    assert total.owned_offdiag_rows == 7
    assert total.unique_loci == 5


def test_summarize_covariance_respects_snp_range(tmp_path: Path) -> None:
    store = _make_hdf5_store(
        tmp_path,
        [(100, 500)],
        {
            (100, 500): [
                (100, 100),
                (100, 200),
                (200, 200),
                (200, 300),
                (300, 300),
                (300, 400),
                (400, 400),
                (400, 500),
                (500, 500),
            ],
        },
    )

    parts, total = summarize_covariance("chr1", store, snp_first=200, snp_last=400)

    assert parts[0].rows == 9
    assert parts[0].diag_rows == 5
    assert parts[0].offdiag_rows == 4
    assert parts[0].owned_offdiag_rows == 2
    assert total.unique_loci == 3


def test_summarize_covariance_reads_legacy_gzip_partition(tmp_path: Path) -> None:
    root = tmp_path / "cov"
    chrom_dir = root / "chr1"
    chrom_dir.mkdir(parents=True)
    (root / "chr1_partitions.txt").write_text("100 300\n")
    with gzip.open(chrom_dir / "chr1.100.300.gz", "wt") as f:
        f.write("snp100 snp100 100 100 0 0 1 1\n")
        f.write("snp100 snp200 100 200 0 0 1 0.5\n")
        f.write("snp200 snp200 200 200 0 0 1 1\n")
        f.write("snp200 snp300 200 300 0 0 1 0.5\n")
        f.write("snp300 snp300 300 300 0 0 1 1\n")
    store = CovarianceStore(root=root)

    parts, total = summarize_covariance("chr1", store)

    assert parts[0].rows == 5
    assert parts[0].diag_rows == 3
    assert total.owned_offdiag_rows == 2
    assert total.unique_loci == 3


def test_covariance_summary_cli_writes_tsv(tmp_path: Path) -> None:
    store = _make_hdf5_store(
        tmp_path,
        [(100, 300)],
        {(100, 300): [(100, 100), (100, 200), (200, 200), (200, 300)]},
    )
    output = tmp_path / "summary.tsv"

    assert (
        main(
            [
                "covariance-summary",
                "--dataset-path",
                str(store.root),
                "--name",
                "chr1",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    lines = output.read_text().splitlines()
    assert lines[0].startswith("name\tpartition_start\tpartition_end")
    assert lines[-1].split("\t")[1] == "TOTAL"


def test_covariance_summary_cli_writes_json(tmp_path: Path) -> None:
    store = _make_hdf5_store(
        tmp_path,
        [(100, 300)],
        {(100, 300): [(100, 100), (100, 200), (200, 200), (200, 300)]},
    )
    output = tmp_path / "summary.json"

    assert (
        main(
            [
                "covariance-summary",
                "--dataset-path",
                str(store.root),
                "--name",
                "chr1",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    data = json.loads(output.read_text())
    assert len(data["partitions"]) == 1
    assert data["total"]["partition_start"] == "TOTAL"
    assert data["total"]["owned_offdiag_rows"] == 2
