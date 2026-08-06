"""Reference-panel loading helpers for covariance calculation."""

from __future__ import annotations

import gzip
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import cyvcf2
import numpy as np


@dataclass(frozen=True)
class ReferencePanel:
    """Phased reference haplotypes after map filtering and de-duplication.

    Positions and ``rs_ids`` are kept parallel to ``haplotypes``. Each
    haplotype row is a flat diploid expansion of the requested individuals:
    ``sample0_hap0, sample0_hap1, sample1_hap0, ...``.
    """

    positions: list[int]
    rs_ids: list[str]
    haplotypes: list[list[int]]
    skipped_unphased: int
    duplicate_positions: int


def read_individuals(individuals_path: Path) -> list[str]:
    """Read the requested sample IDs from a whitespace-delimited text file."""
    individuals: list[str] = []
    with open(individuals_path) as f:
        for line in f:
            line = line.strip()
            if line:
                individuals.append(line.split()[0])
    return individuals


@cache
def read_genetic_map(genetic_map_path: Path) -> dict[int, float]:
    """Return a physical-position to genetic-position lookup from a gzipped map.

    Memoized per process: ``calc_covariance`` calls this once per partition,
    but a single ``ldetect run`` invocation always passes the same
    chromosome-wide map path, so a worker that handles multiple partitions
    would otherwise reparse the whole (potentially large) map once per
    partition it processes. The returned dict is shared across calls with
    the same path -- callers must treat it as read-only.
    """
    with gzip.open(genetic_map_path, "rt") as gf:
        cols = np.loadtxt(gf, usecols=(1, 2), dtype=np.float64, ndmin=2)
    positions = cols[:, 0].astype(np.int64).tolist()
    return dict(zip(positions, cols[:, 1].tolist()))


def watterson_theta(n_haps: int) -> float:
    """Compute the Wen/Stephens shrinkage theta from haplotype count."""
    harmonic = sum(1.0 / i for i in range(1, n_haps))
    return (1.0 / harmonic) / (n_haps + 1.0 / harmonic)


def read_reference_panel(
    vcf_path: Path,
    region: str | None,
    individuals: list[str],
    pos2gpos: dict[int, float],
) -> ReferencePanel:
    """Load phased haplotypes for mapped variants in a VCF/BCF region.

    Only variants present in ``pos2gpos`` are retained, because covariance
    output is keyed to the interpolated genetic map. Missing or unphased
    genotypes are skipped. If multiple records share a physical position, the
    first record is retained to preserve legacy position-keyed semantics.
    """
    vcf = cyvcf2.VCF(str(vcf_path), samples=individuals)
    missing = [ind for ind in individuals if ind not in vcf.samples]
    if missing:
        vcf.close()
        raise ValueError(
            f"individuals not found in VCF/BCF header: {', '.join(missing)}"
        )

    # cyvcf2 subsets to the requested samples but does not guarantee it
    # preserves the caller's order.
    sample_index = {ind: idx for idx, ind in enumerate(vcf.samples)}
    order = np.array([sample_index[ind] for ind in individuals])

    all_pos: list[int] = []
    all_rs: list[str] = []
    haps: list[list[int]] = []
    skipped_unphased = 0

    variants: Iterator[Any] = vcf(region) if region is not None else vcf
    for variant in variants:
        pos = variant.POS
        if pos not in pos2gpos:
            continue

        # ``genotype.array()`` returns ``(n_samples, 3)``: allele1, allele2,
        # phased (1/0), matching ``variant.genotypes`` but as a single
        # vectorized array instead of a Python list of tuples -- avoids a
        # per-individual Python loop over every variant.
        gt = variant.genotype.array()[order]
        if np.any(gt[:, 2] == 0) or np.any(gt[:, :2] < 0):
            skipped_unphased += 1
            continue

        all_pos.append(pos)
        all_rs.append(variant.ID or ".")
        haps.append(gt[:, :2].reshape(-1).tolist())

    vcf.close()

    unique_pos, unique_rs, unique_haps, duplicate_positions = _dedupe_positions(
        all_pos, all_rs, haps
    )
    return ReferencePanel(
        positions=unique_pos,
        rs_ids=unique_rs,
        haplotypes=unique_haps,
        skipped_unphased=skipped_unphased,
        duplicate_positions=duplicate_positions,
    )


def warn_reference_panel_skips(panel: ReferencePanel) -> None:
    """Emit user-visible warnings for variants skipped during panel loading."""
    if panel.skipped_unphased:
        print(
            f"Warning: skipped {panel.skipped_unphased} variant(s) with unphased or "
            f"missing genotypes",
            file=sys.stderr,
        )

    if panel.duplicate_positions:
        print(
            f"Warning: skipped {panel.duplicate_positions} duplicate-position "
            f"variant(s); covariance partitions are keyed by physical position",
            file=sys.stderr,
        )


def _dedupe_positions(
    positions: list[int],
    rs_ids: list[str],
    haplotypes: list[list[int]],
) -> tuple[list[int], list[str], list[list[int]], int]:
    """Keep the first variant for each physical position."""
    if not positions:
        return positions, rs_ids, haplotypes, 0

    duplicate_positions = 0
    seen_positions: set[int] = set()
    unique_pos: list[int] = []
    unique_rs: list[str] = []
    unique_haps: list[list[int]] = []
    for pos, rs, row_haps in zip(positions, rs_ids, haplotypes, strict=True):
        if pos in seen_positions:
            duplicate_positions += 1
            continue
        seen_positions.add(pos)
        unique_pos.append(pos)
        unique_rs.append(rs)
        unique_haps.append(row_haps)
    return unique_pos, unique_rs, unique_haps, duplicate_positions
