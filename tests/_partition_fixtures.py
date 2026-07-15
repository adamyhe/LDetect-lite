"""Shared covariance-partition fixture builders for cross-module overlap tests.

Not a test module itself (leading underscore keeps pytest from collecting it,
mirroring the ``src/ldetect_lite/_util/`` naming convention). Centralizes fixture
builders that were previously duplicated or under-varied across
``test_local_search.py``, ``test_covariance_io.py``, and ``test_metric.py``, so
``matrix_analysis``, ``metric``, and ``local_search`` can all be exercised
against the exact same duplicate/cross-partition-overlap scenarios.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ldetect_lite.io.covariance_hdf5 import write_covariance_partition_hdf5
from ldetect_lite.io.partitions import CovarianceStore
from ldetect_lite.shrinkage import calc_covariance


def _write_indexed_vcf(path: Path, lines: list[str]) -> Path:
    """Write VCF text, bgzip it, and build a .tbi index. Returns the .vcf.gz path."""
    raw_path = path.with_suffix(".vcf")
    raw_path.write_text("\n".join(lines) + "\n")
    subprocess.run(["bgzip", "-f", str(raw_path)], check=True)
    gz_path = path.with_suffix(".vcf.gz")
    subprocess.run(["tabix", "-f", "-p", "vcf", str(gz_path)], check=True)
    return gz_path


def make_custom_partitioned_store(
    tmp_path: Path,
    partitions: dict[tuple[int, int], list[tuple[int, int, float]]],
    *,
    chrom: str = "chr1",
) -> CovarianceStore:
    """Write one compact HDF5 partition per ``(start, end) -> rows`` entry.

    Each row is ``(i_pos, j_pos, shrink_ld)``. Unlike the per-file ``_make_*``
    helpers in the individual test modules (which derive every partition's
    rows from one shared ``r2_by_pair`` dict, so overlapping partitions always
    agree on shared pairs), this lets each partition carry independently
    chosen values — required to test that overlap-resolution logic picks a
    specific winner rather than merely avoiding an accidental double-count.
    """
    root = tmp_path / "cov"
    chrom_dir = root / chrom
    chrom_dir.mkdir(parents=True)
    with (root / f"{chrom}_partitions.txt").open("w") as f:
        for start, end in partitions:
            f.write(f"{start} {end}\n")

    for (start, end), rows in partitions.items():
        write_covariance_partition_hdf5(
            chrom_dir / f"{chrom}.{start}.{end}.h5",
            i_pos=np.array([row[0] for row in rows], dtype=np.int32),
            j_pos=np.array([row[1] for row in rows], dtype=np.int32),
            shrink_ld=np.array([row[2] for row in rows], dtype=np.float64),
        )
    return CovarianceStore(root=root)


def divergent_overlap_partitions(
    first_value: float = 0.7,
    second_value: float = 0.2,
) -> dict[tuple[int, int], list[tuple[int, int, float]]]:
    """Two overlapping partitions that redundantly compute pair (200, 400)
    with genuinely different values (``first_value`` in the first-listed
    partition, ``second_value`` in the second).

    This is the "values actually differ" scenario: the legacy reference (and
    ``local_search.py``'s array/HDF5-streaming paths) resolve a duplicate
    ``(lo, hi)`` pair across overlapping partitions by first-write-wins in
    partition-list order, so any correct consumer must select ``first_value``
    here, not ``second_value`` and not an average of the two. Lifted from
    ``test_local_search.py::test_local_search_matches_legacy_with_cross_partition_duplicate_pairs``.
    """
    return {
        (100, 400): [
            (100, 100, 1.0),
            (200, 200, 1.0),
            (300, 300, 1.0),
            (400, 400, 1.0),
            (200, 400, first_value),
        ],
        (200, 500): [
            (200, 200, 1.0),
            (300, 300, 1.0),
            (400, 400, 1.0),
            (500, 500, 1.0),
            (400, 200, second_value),
            (300, 500, 0.8),
        ],
    }


def first_write_wins_pair_value(
    partitions: dict[tuple[int, int], list[tuple[int, int, float]]],
    lo: int,
    hi: int,
) -> float | None:
    """Independent, from-scratch oracle for legacy's cross-partition precedence.

    Mirrors ``insert_into_matrix``/``insert_into_matrix_lean``
    (mirroring the shipped legacy ``flat_file.py``): partitions
    are read in the given (ascending/list) order into one shared mapping, and
    a pair already present in that mapping is never replaced. Deliberately
    reimplemented with plain Python containers here rather than importing
    anything from ``ldetect_lite.io``/``ldetect_lite._util``, so it serves as ground
    truth independent of the code under test, not a self-comparison.
    """
    seen: dict[tuple[int, int], float] = {}
    for rows in partitions.values():
        for i_pos, j_pos, value in rows:
            key = (i_pos, j_pos) if i_pos <= j_pos else (j_pos, i_pos)
            if key not in seen:
                seen[key] = value
    return seen.get((lo, hi) if lo <= hi else (hi, lo))


def build_two_overlapping_partitions_with_duplicate_position(
    tmp_path: Path,
    *,
    ne: float = 1.0,
    chrom: str = "chr1",
) -> tuple[CovarianceStore, list[tuple[int, int]], int, int]:
    """Run ``calc_covariance`` twice on overlapping VCF slices sharing a
    duplicated physical position inside their overlap zone.

Mimics two region-sliced partitions of one shared source VCF (matching how
    ``_calc_partition`` slices one reference panel per partition): partition A
    covers ``[100, 400]`` and partition B covers ``[250, 600]``; both regions
    include the same two same-POS=300 records (mirroring
    ``_duplicate_position_vcf_stream`` in ``test_shrinkage.py``), so both
    independently exercise ``calc_covariance``'s duplicate-position dedup
    on real code (not hand-typed HDF5 rows). Returns
    ``(store, partitions, snp_first, snp_last)``.
    """
    map_path = tmp_path / "map.gz"
    individuals_path = tmp_path / "inds.txt"
    individuals_path.write_text("sample_a\nsample_b\n")
    import gzip

    with gzip.open(map_path, "wt") as f:
        for pos in (100, 200, 300, 400, 500, 600):
            f.write(f"1 {pos} {pos / 100_000:.6f}\n")

    header_lines = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "##contig=<ID=1>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample_a\tsample_b",
    ]
    # Loci: 100, 200, 300 (duplicate: rs_dup_a + rs_dup_b), 400, 500, 600.
    rows = {
        100: "1\t100\trs_100\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0",
        200: "1\t200\trs_200\tA\tG\t.\tPASS\t.\tGT\t1|0\t0|0",
        300: [
            "1\t300\trs_dup_a\tA\tG\t.\tPASS\t.\tGT\t1|1\t0|0",
            "1\t300\trs_dup_b\tC\tT\t.\tPASS\t.\tGT\t0|0\t1|1",
        ],
        400: "1\t400\trs_400\tA\tG\t.\tPASS\t.\tGT\t1|1\t0|1",
        500: "1\t500\trs_500\tA\tG\t.\tPASS\t.\tGT\t1|0\t1|0",
        600: "1\t600\trs_600\tA\tG\t.\tPASS\t.\tGT\t0|0\t1|1",
    }

    def _rows_for(positions: list[int]) -> list[str]:
        body: list[str] = []
        for pos in positions:
            value = rows[pos]
            body.extend(value if isinstance(value, list) else [value])
        return body

    partitions = [(100, 400), (250, 600)]

    root = tmp_path / "cov"
    chrom_dir = root / chrom
    chrom_dir.mkdir(parents=True)
    with (root / f"{chrom}_partitions.txt").open("w") as f:
        for start, end in partitions:
            f.write(f"{start} {end}\n")

    # One shared source VCF spanning every partition, region-sliced per
    # partition below -- mirrors production (`_calc_partition` slices one
    # reference panel via `region`, not separately pre-sliced streams).
    vcf_path = _write_indexed_vcf(
        tmp_path / "combined",
        [*header_lines, *_rows_for([100, 200, 300, 400, 500, 600])],
    )

    for start, end in partitions:
        calc_covariance(
            vcf_path=vcf_path,
            region=f"1:{start}-{end}",
            genetic_map_path=map_path,
            individuals_path=individuals_path,
            output_path=chrom_dir / f"{chrom}.{start}.{end}.h5",
            ne=ne,
            cutoff=1e-7,
            compact_output=True,
        )

    return CovarianceStore(root=root), partitions, 100, 600
