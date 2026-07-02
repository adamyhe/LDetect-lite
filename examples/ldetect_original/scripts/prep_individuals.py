"""Build a per-population individual list intersected with VCF sample header.

Reads the 1000G panel annotation file to find samples belonging to the
requested subpopulations, then intersects with the samples actually present
in the VCF to produce a clean individual list.

Usage:
    uv run python scripts/prep_individuals.py \
        --panel resources/integrated_call_samples_v3.20130502.ALL.panel \
        --subpops CEU TSI FIN GBR IBS \
        --vcf data/filtered/chr22.vcf.gz \
        --output resources/eurinds.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ldetect2.io.vcf import read_vcf_samples


def read_panel(panel_path: Path, subpops: list[str]) -> set[str]:
    """Return sample IDs belonging to any of *subpops*."""
    samples: set[str] = set()
    with open(panel_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].lower() == "sample":
                continue
            sample_id, pop = parts[0], parts[1]
            if pop in subpops:
                samples.add(sample_id)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path,
                        help="1000G panel annotation file.")
    parser.add_argument("--subpops", nargs="+", required=True, metavar="POP",
                        help="Subpopulation codes to include (e.g. TSI IBS CEU GBR).")
    parser.add_argument("--vcf", required=True, type=Path,
                        help="Any VCF from the filtered set (to get sample list).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output file: one sample ID per line.")
    args = parser.parse_args()

    panel_samples = read_panel(args.panel, args.subpops)
    vcf_samples = read_vcf_samples(args.vcf)
    intersection = sorted(panel_samples & vcf_samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(intersection) + "\n")

    print(f"Panel samples in {args.subpops}: {len(panel_samples)}")
    print(f"VCF samples: {len(vcf_samples)}")
    print(f"Intersection: {len(intersection)} → {args.output}")

    # Expected Phase 1 counts: EUR ~379, AFR ~246, ASN ~286
    _EXPECTED = {"CEU TSI FIN GBR IBS": 379, "YRI LWK ASW": 246, "CHB JPT CHS": 286}
    for subpops_str, expected in _EXPECTED.items():
        if set(subpops_str.split()) == set(args.subpops):
            if len(intersection) != expected:
                import sys
                print(
                    f"WARNING: expected {expected} individuals for "
                    f"{args.subpops} (Phase 1), got {len(intersection)}.",
                    file=sys.stderr,
                )
            break


if __name__ == "__main__":
    main()
