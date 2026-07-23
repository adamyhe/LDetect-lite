"""Compare filtered VCF positions with positions in an ldetect genetic map."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import cyvcf2


def _read_map_positions(path: Path) -> set[int]:
    opener = gzip.open if path.suffix == ".gz" else open
    positions: set[int] = set()
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                positions.add(int(parts[1]))
            except ValueError:
                continue
    return positions


def _read_vcf_positions(path: Path) -> set[int]:
    return {variant.POS for variant in cyvcf2.VCF(str(path))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--block-set", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    vcf_positions = _read_vcf_positions(args.vcf)
    map_positions = _read_map_positions(args.genetic_map)
    shared = vcf_positions & map_positions
    vcf_only = vcf_positions - map_positions
    map_only = map_positions - vcf_positions

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "block_set",
        "chrom",
        "vcf_positions",
        "map_positions",
        "shared_positions",
        "vcf_only_positions",
        "map_only_positions",
        "shared_fraction_of_vcf",
        "shared_fraction_of_map",
    ]
    row = {
        "block_set": args.block_set,
        "chrom": args.chromosome,
        "vcf_positions": len(vcf_positions),
        "map_positions": len(map_positions),
        "shared_positions": len(shared),
        "vcf_only_positions": len(vcf_only),
        "map_only_positions": len(map_only),
        "shared_fraction_of_vcf": len(shared) / len(vcf_positions)
        if vcf_positions
        else 0.0,
        "shared_fraction_of_map": len(shared) / len(map_positions)
        if map_positions
        else 0.0,
    }
    with args.output.open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        handle.write("\t".join(str(row[column]) for column in columns) + "\n")


if __name__ == "__main__":
    main()
