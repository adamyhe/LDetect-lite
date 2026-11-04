"""Reference-panel loading helpers for covariance calculation."""

from __future__ import annotations

import gzip
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import cyvcf2
import numpy as np

HaplotypeLayout = Literal["uint8", "packed"]

#: Rows buffered as uint8 before each ``np.packbits`` call in packed layout.
#: Bounds the transient uint8 footprint to ``_PACK_BLOCK_ROWS * n_haps`` bytes.
_PACK_BLOCK_ROWS = 4096


@dataclass(frozen=True)
class ReferencePanel:
    """Phased reference haplotypes after map filtering and de-duplication.

    ``positions`` and ``rs_ids`` are parallel to the haplotype rows. Each row is
    a flat diploid expansion of the requested individuals:
    ``sample0_hap0, sample0_hap1, sample1_hap0, ...``.

    Exactly one haplotype representation is populated, chosen by
    ``read_reference_panel``'s ``layout``:

    * ``haplotypes`` -- ``uint8`` matrix ``(n_snps, n_haps)`` of allele indexes
      (``layout="uint8"``).
    * ``packed_haplotypes`` -- ``uint64`` matrix ``(n_snps, ceil(n_haps/64))``
      with haplotype ``k`` at word ``k // 64``, bit ``k % 64``, set for any
      non-reference allele (``layout="packed"``; identical to
      ``shrinkage._pack_haplotypes_impl`` applied to the uint8 matrix).

    ``hap_sums`` is the per-row sum of allele indexes, as the uint8 matrix's
    ``sum(axis=1)`` would give, in both layouts.
    """

    positions: list[int]
    rs_ids: list[str]
    n_haps: int
    hap_sums: np.ndarray
    skipped_unphased: int
    duplicate_positions: int
    haplotypes: np.ndarray | None = None
    packed_haplotypes: np.ndarray | None = None

    def haplotype_matrix(self) -> np.ndarray:
        """The ``uint8`` haplotype matrix, unpacking packed rows if needed.

        Unpacking recovers 0/1 values, so for the (unsupported) case of
        multi-allelic records with allele indexes above 1 it is not identical
        to what ``layout="uint8"`` would have returned.
        """
        if self.haplotypes is not None:
            return self.haplotypes
        assert self.packed_haplotypes is not None
        return unpack_haplotypes(self.packed_haplotypes, self.n_haps)


def pack_haplotype_rows(rows: np.ndarray) -> np.ndarray:
    """Pack ``(n, n_haps)`` allele rows into ``uint64`` words (nonzero -> bit)."""
    rows = np.asarray(rows)
    n_haps = rows.shape[1]
    n_words = (n_haps + 63) // 64
    if rows.shape[0] == 0:
        return np.zeros((0, n_words), dtype=np.uint64)
    bits = rows != 0
    pad = n_words * 64 - n_haps
    if pad:
        bits = np.concatenate(
            (bits, np.zeros((rows.shape[0], pad), dtype=bool)), axis=1
        )
    # little bit order + little-endian word view: bit k -> word k//64, bit k%64
    packed = np.packbits(bits, axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u8").astype(np.uint64, copy=False)


def unpack_haplotypes(packed: np.ndarray, n_haps: int) -> np.ndarray:
    """Inverse of ``pack_haplotype_rows`` for 0/1 rows."""
    packed = np.ascontiguousarray(packed, dtype="<u8")
    if packed.shape[0] == 0:
        return np.zeros((0, n_haps), dtype=np.uint8)
    bits = np.unpackbits(packed.view(np.uint8), axis=1, bitorder="little")
    return np.ascontiguousarray(bits[:, :n_haps])


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
    layout: HaplotypeLayout = "uint8",
) -> ReferencePanel:
    """Load phased haplotypes for mapped variants in a VCF/BCF region.

    Only variants present in ``pos2gpos`` are retained, because covariance
    output is keyed to the interpolated genetic map. Missing or unphased
    genotypes are skipped. If multiple records share a physical position, the
    first usable record is retained to preserve legacy position-keyed
    semantics.

    Rows are written straight into NumPy buffers -- a ``uint8`` matrix, or
    with ``layout="packed"`` bounded ``uint8`` blocks packed into ``uint64``
    words as they fill -- rather than round-tripping through per-variant
    Python lists.
    """
    if layout not in ("uint8", "packed"):
        raise ValueError("layout must be one of: uint8, packed")
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
    identity_order = bool(np.array_equal(order, np.arange(order.size)))
    n_haps = 2 * len(individuals)
    packed_layout = layout == "packed"

    positions: list[int] = []
    rs_ids: list[str] = []
    hap_sums: list[int] = []
    seen_positions: set[int] = set()
    skipped_unphased = 0
    duplicate_positions = 0
    # uint8 layout: growable matrix; packed layout: one reusable block.
    buf = np.empty((_PACK_BLOCK_ROWS, n_haps), dtype=np.uint8)
    n_buf = 0
    blocks: list[np.ndarray] = []

    variants: Iterator[Any] = vcf(region) if region is not None else vcf
    for variant in variants:
        pos = variant.POS
        if pos not in pos2gpos:
            continue

        # ``genotype.array()`` returns ``(n_samples, 3)``: allele1, allele2,
        # phased (1/0), matching ``variant.genotypes`` but as a single
        # vectorized array instead of a Python list of tuples.
        gt = variant.genotype.array()
        if not identity_order:
            gt = gt[order]
        if np.any(gt[:, 2] == 0) or np.any(gt[:, :2] < 0):
            skipped_unphased += 1
            continue
        if pos in seen_positions:
            duplicate_positions += 1
            continue
        seen_positions.add(pos)

        if n_buf == buf.shape[0]:
            if packed_layout:
                blocks.append(pack_haplotype_rows(buf))
                n_buf = 0
            else:
                grown = np.empty((2 * buf.shape[0], n_haps), dtype=np.uint8)
                grown[:n_buf] = buf
                buf = grown
        row = buf[n_buf]
        row[:] = gt[:, :2].reshape(-1)
        hap_sums.append(int(row.sum()))
        n_buf += 1
        positions.append(pos)
        rs_ids.append(variant.ID or ".")

    vcf.close()

    sums = np.asarray(hap_sums, dtype=np.float64)
    if packed_layout:
        blocks.append(pack_haplotype_rows(buf[:n_buf]))
        packed = blocks[0] if len(blocks) == 1 else np.concatenate(blocks)
        return ReferencePanel(
            positions=positions,
            rs_ids=rs_ids,
            n_haps=n_haps,
            hap_sums=sums,
            skipped_unphased=skipped_unphased,
            duplicate_positions=duplicate_positions,
            packed_haplotypes=packed,
        )
    return ReferencePanel(
        positions=positions,
        rs_ids=rs_ids,
        n_haps=n_haps,
        hap_sums=sums,
        skipped_unphased=skipped_unphased,
        duplicate_positions=duplicate_positions,
        haplotypes=np.ascontiguousarray(buf[:n_buf]),
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
