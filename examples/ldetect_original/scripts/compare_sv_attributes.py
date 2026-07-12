"""Compare SV/long-indel span attributes across VCF releases.

`compare_vcf_positions.py` compares whole-file called-position sets and
found VCF release version doesn't distinguish divergent from control
chromosomes for `ldetect_original` (see
`notes/findings/ldetect-original-reproduction.md`, "Ruled out"). That check
is blind to a narrower possibility: an SV/indel's own `POS` staying stable
across releases while its *span* (`INFO/END`, or `REF` length) shifts --
exactly the attribute that determines whether
`calc_covariance`'s region-based read spuriously pulls it into a
neighboring partition (see "SV/indel partition-boundary duplication" in the
same findings doc). SV calling in 1000G Phase 1 was actively being refined
across releases; `END`/`CIEND` are expected to be far less stable than
ordinary SNP `POS` calls, which is exactly what this checks for.

Matches SV-like records (`INFO/SVTYPE` present, or `REF`/`ALT` longer than
50bp -- same heuristic as `audit_boundary_spanning_variants.py`) between a
baseline and candidate release by variant ID (1000G SV IDs like `esv...`
reference a shared DGV/1000G structural-variant catalog entry and are
expected to be stable identifiers across processing releases, unlike POS-
based matching which breaks if POS itself shifted). Reports, per ID present
in both: whether `POS` and/or span (`END`) changed.

Usage:
    uv run python scripts/compare_sv_attributes.py \
        --population AFR --chromosome 22 \
        --baseline-label v3/all --candidate-label v2/all \
        --baseline-vcf .../v3/all/AFR/chr22.AFR.all.vcf.gz \
        --candidate-vcf .../v2/all/AFR/chr22.AFR.all.vcf.gz \
        --output results/.../sv_attributes/v2_all_vs_v3_all.tsv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

FIELDNAMES = [
    "population",
    "chrom",
    "baseline_label",
    "candidate_label",
    "baseline_sv_like",
    "candidate_sv_like",
    "shared_ids",
    "baseline_only_ids",
    "candidate_only_ids",
    "pos_changed",
    "span_changed",
    "example_span_changed_ids",
]

_EXAMPLE_LIMIT = 10


def _record_span(pos: int, ref: str, info: str) -> tuple[int, int]:
    end = pos + len(ref) - 1
    for field in info.split(";"):
        if field.startswith("END="):
            try:
                end = max(end, int(field[4:]))
            except ValueError:
                pass
    return pos, end


def _is_sv_like(ref: str, alt: str, info: str) -> bool:
    return (
        "SVTYPE=" in info
        or len(ref) > 50
        or any(len(a) > 50 for a in alt.split(","))
    )


def read_sv_records(path: Path) -> dict[str, tuple[int, int]]:
    """Return {variant_id: (pos, end)} for every SV-like, ID-bearing record."""
    proc = subprocess.run(
        ["bcftools", "view", "-G", "-H", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    records: dict[str, tuple[int, int]] = {}
    for line in proc.stdout.splitlines():
        if not line:
            continue
        parts = line.split("\t", 8)
        pos = int(parts[1])
        variant_id = parts[2]
        ref = parts[3]
        alt = parts[4]
        info = parts[7]
        if variant_id in ("", ".") or not _is_sv_like(ref, alt, info):
            continue
        records[variant_id] = _record_span(pos, ref, info)
    return records


def compare(args: argparse.Namespace) -> dict[str, str]:
    baseline = read_sv_records(args.baseline_vcf)
    candidate = read_sv_records(args.candidate_vcf)
    shared_ids = set(baseline) & set(candidate)

    pos_changed = 0
    span_changed = 0
    example_span_changed: list[str] = []
    for variant_id in shared_ids:
        b_pos, b_end = baseline[variant_id]
        c_pos, c_end = candidate[variant_id]
        if b_pos != c_pos:
            pos_changed += 1
        if (b_pos, b_end) != (c_pos, c_end):
            span_changed += 1
            if len(example_span_changed) < _EXAMPLE_LIMIT:
                example_span_changed.append(
                    f"{variant_id}:({b_pos},{b_end})->({c_pos},{c_end})"
                )

    return {
        "population": args.population,
        "chrom": f"chr{args.chromosome.removeprefix('chr')}",
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "baseline_sv_like": str(len(baseline)),
        "candidate_sv_like": str(len(candidate)),
        "shared_ids": str(len(shared_ids)),
        "baseline_only_ids": str(len(set(baseline) - set(candidate))),
        "candidate_only_ids": str(len(set(candidate) - set(baseline))),
        "pos_changed": str(pos_changed),
        "span_changed": str(span_changed),
        "example_span_changed_ids": ";".join(example_span_changed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", required=True)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-vcf", required=True, type=Path)
    parser.add_argument("--candidate-vcf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    row = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
