"""Partition file I/O and the CovarianceStore path abstraction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ldetect2._util.logging import log_debug


@dataclass(frozen=True)
class CovarianceStore:
    """Encapsulates the directory layout of a covariance matrix dataset.

    The expected layout::

        <root>/
            <name>_partitions.txt      # space-delimited start/end pairs
            <name>/
                <name>.<start>.<end>.h5  # HDF5 covariance partition files
    """

    root: Path

    @property
    def partitions_dir(self) -> Path:
        return self.root

    def partitions_path(self, name: str) -> Path:
        return self.root / f"{name}_partitions.txt"

    def partition_path(self, name: str, start: int, end: int) -> Path:
        return self.root / name / f"{name}.{start}.{end}.h5"


def read_partitions(name: str, store: CovarianceStore) -> list[tuple[int, int]]:
    """Return all (start, end) partition tuples for *name*."""
    path = store.partitions_path(name)
    partitions: list[tuple[int, int]] = []
    with open(path) as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            partitions.append((int(row[0]), int(row[1])))
    return partitions


def relevant_subpartitions(
    partitions: list[tuple[int, int]],
    snp_first: int,
    snp_last: int,
) -> list[tuple[int, int]]:
    """Return the subset of *partitions* that overlaps [snp_first, snp_last]."""
    p_first = -1
    p_last = -1
    found_first = False
    found_last = False

    for i, (start, end) in enumerate(partitions):
        if start <= snp_first <= end and not found_first:
            p_first = i
            found_first = True
        if start <= snp_last <= end:
            p_last = i
            found_last = True

    if not found_first or not found_last:
        raise ValueError(
            f"Partition covering snp_first={snp_first} or snp_last={snp_last} not found"
        )

    return partitions[p_first : p_last + 1]


def first_last(
    name: str,
    store: CovarianceStore,
    first: int = -1,
    last: int = -1,
) -> tuple[int, int]:
    """Return (first, last) SNP positions, auto-filling -1 from partition boundaries."""
    if first == -1 or last == -1:
        parts = read_partitions(name, store)
        if first == -1:
            first = parts[0][0]
        if last == -1:
            last = parts[-1][1]
    return first, last


def get_final_partitions(
    store: CovarianceStore,
    name: str,
    snp_first: int,
    snp_last: int,
) -> list[tuple[int, int]]:
    """Read partitions and filter to those overlapping [snp_first, snp_last]."""
    if snp_first > snp_last:
        raise ValueError(f"snp_first ({snp_first}) > snp_last ({snp_last})")

    log_debug("Reading partitions file")
    partitions = read_partitions(name, store)

    log_debug("Getting relevant partitions")
    partitions = relevant_subpartitions(partitions, snp_first, snp_last)

    if not partitions:
        raise ValueError("No relevant subpartitions found")

    return partitions
