"""Build a per-population individual list intersected with VCF sample header.

Reads the 1000G population directories used by the MacDonald et al. README,
then intersects those sample IDs with the samples actually present in the VCF
to produce a clean individual list.

Usage:
    uv run python scripts/prep_individuals.py \
        --sample-data-base-url ftp://.../1000_genomes_project/data \
        --subpops TSI IBS CEU GBR \
        --vcf data/filtered/chr22.vcf.gz \
        --output resources/eurinds.txt
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cyvcf2


def read_vcf_samples(vcf_path: Path) -> set[str]:
    """Return sample IDs present in a VCF header."""
    return set(cyvcf2.VCF(str(vcf_path)).samples)


def read_population_directory_samples(base_url: str, subpops: list[str]) -> set[str]:
    """Return sample IDs listed in the 1000G directory for each subpopulation."""
    samples: set[str] = set()
    root = base_url.rstrip("/")
    for subpop in subpops:
        with urllib.request.urlopen(f"{root}/{subpop}/") as response:
            for line in response.read().decode().splitlines():
                parts = line.split()
                if parts:
                    samples.add(parts[-1])
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-data-base-url",
        required=True,
        help="1000G data directory root containing one directory per subpopulation.",
    )
    parser.add_argument(
        "--subpops",
        nargs="+",
        required=True,
        metavar="POP",
        help="Subpopulation codes to include (e.g. TSI IBS CEU GBR).",
    )
    parser.add_argument(
        "--vcf",
        required=True,
        type=Path,
        help="Any VCF from the filtered set (to get sample list).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output file: one sample ID per line.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Optional expected intersection count; fail if it differs.",
    )
    args = parser.parse_args()

    candidate_samples = read_population_directory_samples(
        args.sample_data_base_url,
        args.subpops,
    )
    vcf_samples = read_vcf_samples(args.vcf)
    intersection = sorted(candidate_samples & vcf_samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(intersection) + "\n")

    print(f"1000G directory samples in {args.subpops}: {len(candidate_samples)}")
    print(f"VCF samples: {len(vcf_samples)}")
    print(f"Intersection: {len(intersection)} -> {args.output}")

    if args.expected_count is not None and len(intersection) != args.expected_count:
        print(
            f"ERROR: expected {args.expected_count} individuals, got "
            f"{len(intersection)}.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
