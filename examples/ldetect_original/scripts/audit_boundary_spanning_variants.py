"""Audit whether any SV/long-indel's span crosses a partition boundary.

Cheap, site-only precursor to the full `Snakefile.sv_boundary_diagnostics`
pipeline comparison: checks whether the SV-boundary-duplication mechanism
described in `notes/findings/ldetect-original-reproduction.md` ("New
candidate mechanism: SV/indel partition-boundary duplication") is even *in
play* for a given chromosome's actual partition boundaries, before spending
compute on the full `ldetect run` comparison. Uses `bcftools view -G`
(genotypes dropped) since span is a record-level property (`POS`, `REF`
length, `INFO/END`) independent of any individual's genotype — this makes
the scan seconds, not minutes, even on a full multi-sample VCF.

A "hit" is a record whose own `POS` falls outside a given partition
`[start, end]`, but whose span `[POS, end_of_record]` still overlaps that
partition — exactly the condition under which `calc_covariance`'s
htslib-backed region read (`cyvcf2`, see `docs/optimizations.md` #10) would
pull that record into the wrong partition's covariance calculation.

Usage:
    uv run python scripts/audit_boundary_spanning_variants.py \
        --vcf results/sv_boundary_diagnostics/AFR/22/all/input.vcf.gz \
        --partitions results/sv_boundary_diagnostics/AFR/22/all/22_partitions.txt \
        --population AFR --chrom 22 \
        --output results/sv_boundary_diagnostics/AFR/22/boundary_audit.tsv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def read_partitions(path: Path) -> list[tuple[int, int]]:
    partitions = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                partitions.append((int(parts[0]), int(parts[1])))
    return partitions


def record_span(pos: int, ref: str, info: str) -> tuple[int, int]:
    end = pos + len(ref) - 1
    for field in info.split(";"):
        if field.startswith("END="):
            try:
                end = max(end, int(field[4:]))
            except ValueError:
                pass
    return pos, end


def is_sv_like(ref: str, alt: str, info: str) -> bool:
    return (
        "SVTYPE=" in info
        or len(ref) > 50
        or any(len(a) > 50 for a in alt.split(","))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--partitions", required=True, type=Path)
    parser.add_argument("--population", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    partitions = read_partitions(args.partitions)

    proc = subprocess.run(
        ["bcftools", "view", "-G", "-H", str(args.vcf)],
        capture_output=True,
        text=True,
        check=True,
    )

    n_sv_like = 0
    hits: list[dict] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        parts = line.split("\t", 8)
        pos = int(parts[1])
        variant_id = parts[2]
        ref = parts[3]
        alt = parts[4]
        info = parts[7]
        if not is_sv_like(ref, alt, info):
            continue
        n_sv_like += 1
        start, end = record_span(pos, ref, info)
        for pstart, pend in partitions:
            pos_in_this_partition = pstart <= pos <= pend
            span_overlaps_this_partition = start <= pend and end >= pstart
            if span_overlaps_this_partition and not pos_in_this_partition:
                hits.append(
                    {
                        "population": args.population,
                        "chrom": args.chrom,
                        "variant_id": variant_id,
                        "pos": pos,
                        "end": end,
                        "partition_start": pstart,
                        "partition_end": pend,
                    }
                )

    print(
        f"{args.population} chr{args.chrom}: {len(partitions)} partitions, "
        f"{n_sv_like} SV-like/long-REF records scanned, "
        f"{len(hits)} boundary-spanning mismatches"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "population",
        "chrom",
        "n_partitions",
        "n_sv_like",
        "n_boundary_spanning",
        "variant_id",
        "pos",
        "end",
        "partition_start",
        "partition_end",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        if hits:
            for hit in hits:
                writer.writerow(
                    {
                        **hit,
                        "n_partitions": len(partitions),
                        "n_sv_like": n_sv_like,
                        "n_boundary_spanning": len(hits),
                    }
                )
        else:
            writer.writerow(
                {
                    "population": args.population,
                    "chrom": args.chrom,
                    "n_partitions": len(partitions),
                    "n_sv_like": n_sv_like,
                    "n_boundary_spanning": 0,
                    "variant_id": "",
                    "pos": "",
                    "end": "",
                    "partition_start": "",
                    "partition_end": "",
                }
            )
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
