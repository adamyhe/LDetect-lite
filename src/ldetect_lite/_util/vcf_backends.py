"""VCF-region genotype readers: naive text parsing vs. pysam vs. cyvcf2.

Timing instrumentation to inform whether switching `calc_covariance`'s VCF
parsing (currently naive per-line `str.split`, `shrinkage.py:622-665`) to a
C-accelerated library is worth doing. All three readers below extract the
same thing -- phased biallelic genotypes for a set of individuals over one
genomic region, in the same column order, with the same skip/dedup
semantics -- so they're directly comparable, both for equivalence and for
speed. Deliberately excludes `calc_covariance`'s genetic-map filtering step
(`if pos not in pos2gpos: continue`): that's a cheap, backend-independent
dict lookup, not part of what a faster VCF library would change.

See `tests/test_vcf_backend_timing.py` for the equivalence check and the
benchmark. `pysam`/`cyvcf2` are optional (`vcf-benchmark` extra) -- neither
is a core dependency, and neither is used by any production code path.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from ldetect_lite._util.logging import log_debug

GenotypeRows = tuple[list[int], list[list[int]]]


def _dedup_first_wins(positions: list[int], haps: list[list[int]]) -> GenotypeRows:
    seen: set[int] = set()
    out_pos: list[int] = []
    out_haps: list[list[int]] = []
    for pos, row in zip(positions, haps, strict=True):
        if pos in seen:
            continue
        seen.add(pos)
        out_pos.append(pos)
        out_haps.append(row)
    return out_pos, out_haps


def read_genotypes_naive(
    vcf_path: Path, chrom: str, start: int, end: int, individuals: list[str]
) -> GenotypeRows:
    """Naive per-line text parsing, tabix-sliced -- mirrors calc_covariance."""
    region = f"{chrom}:{start}-{end}"
    read_start = time.perf_counter()
    proc = subprocess.Popen(
        ["tabix", "-h", str(vcf_path), region], stdout=subprocess.PIPE, text=True
    )
    stdout = proc.stdout
    assert stdout is not None

    ind2col: dict[str, int] = {}
    positions: list[int] = []
    haps: list[list[int]] = []
    with stdout:
        for raw in stdout:
            raw = raw.rstrip("\n")
            if raw.startswith("##"):
                continue
            parts = raw.split("\t")
            if raw.startswith("#CHROM"):
                for col_idx in range(9, len(parts)):
                    if parts[col_idx] in individuals:
                        ind2col[parts[col_idx]] = col_idx
                continue

            pos = int(parts[1])
            row_haps: list[int] = []
            skip = False
            for ind in individuals:
                col = ind2col.get(ind)
                if col is None:
                    skip = True
                    break
                gt_field = parts[col].split(":")[0]
                if "|" not in gt_field:
                    skip = True
                    break
                alleles = gt_field.split("|")
                if "." in alleles:
                    skip = True
                    break
                row_haps.append(int(alleles[0]))
                row_haps.append(int(alleles[1]))
            if skip:
                continue
            positions.append(pos)
            haps.append(row_haps)
    proc.wait()

    positions, haps = _dedup_first_wins(positions, haps)
    log_debug(
        "read_genotypes_naive "
        f"region={region} n_individuals={len(individuals)} n_rows={len(positions)} "
        f"seconds={time.perf_counter() - read_start:.3f}"
    )
    return positions, haps


def read_genotypes_pysam(
    vcf_path: Path, chrom: str, start: int, end: int, individuals: list[str]
) -> GenotypeRows:
    import pysam

    region = f"{chrom}:{start}-{end}"
    read_start = time.perf_counter()
    positions: list[int] = []
    haps: list[list[int]] = []
    with pysam.VariantFile(str(vcf_path)) as vcf:
        for record in vcf.fetch(region=region):
            row_haps: list[int] = []
            skip = False
            for ind in individuals:
                sample = record.samples.get(ind)
                if sample is None:
                    skip = True
                    break
                gt = sample["GT"]
                if not sample.phased or gt is None or any(a is None for a in gt):
                    skip = True
                    break
                row_haps.extend(int(a) for a in gt)
            if skip:
                continue
            positions.append(int(record.pos))
            haps.append(row_haps)

    positions, haps = _dedup_first_wins(positions, haps)
    log_debug(
        "read_genotypes_pysam "
        f"region={region} n_individuals={len(individuals)} n_rows={len(positions)} "
        f"seconds={time.perf_counter() - read_start:.3f}"
    )
    return positions, haps


def read_genotypes_cyvcf2(
    vcf_path: Path, chrom: str, start: int, end: int, individuals: list[str]
) -> GenotypeRows:
    import cyvcf2

    region = f"{chrom}:{start}-{end}"
    read_start = time.perf_counter()
    vcf = cyvcf2.VCF(str(vcf_path), samples=individuals)
    # cyvcf2 subsets to the requested samples but keeps *its own* internal
    # order, not the caller's -- remap columns back to `individuals` order
    # so output is directly comparable to the naive/pysam backends.
    order = [vcf.samples.index(ind) for ind in individuals]

    positions: list[int] = []
    haps: list[list[int]] = []
    for variant in vcf(region):
        genotypes = variant.genotypes
        row_haps: list[int] = []
        skip = False
        for col in order:
            allele1, allele2, phased = genotypes[col]
            if not phased or allele1 < 0 or allele2 < 0:
                skip = True
                break
            row_haps.append(int(allele1))
            row_haps.append(int(allele2))
        if skip:
            continue
        positions.append(int(variant.POS))
        haps.append(row_haps)
    vcf.close()

    positions, haps = _dedup_first_wins(positions, haps)
    log_debug(
        "read_genotypes_cyvcf2 "
        f"region={region} n_individuals={len(individuals)} n_rows={len(positions)} "
        f"seconds={time.perf_counter() - read_start:.3f}"
    )
    return positions, haps
