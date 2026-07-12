"""Fetch and filter the 1000G Phase 1 interval matching the LDetect toy example."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

THOUSAND_GENOMES_BASE = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20110521"
THOUSAND_GENOMES_CHR2 = (
    "ALL.chr2.phase1_release_v3.20101123.snps_indels_svs.genotypes.vcf.gz"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individuals", required=True, type=Path)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--output-vcf", required=True, type=Path)
    parser.add_argument("--output-bcf", required=True, type=Path)
    args = parser.parse_args()

    require_tool("bcftools")
    require_tool("tabix")

    args.output_vcf.parent.mkdir(parents=True, exist_ok=True)
    args.output_bcf.parent.mkdir(parents=True, exist_ok=True)
    for path in (
        args.output_vcf,
        Path(str(args.output_vcf) + ".tbi"),
        args.output_bcf,
        Path(str(args.output_bcf) + ".csi"),
    ):
        path.unlink(missing_ok=True)

    source_vcf = f"{THOUSAND_GENOMES_BASE}/{THOUSAND_GENOMES_CHR2}"
    region = f"{args.chrom}:{args.start}-{args.end}"
    run(
        [
            "bcftools",
            "view",
            "-S",
            str(args.individuals),
            "-r",
            region,
            "-i",
            "MAC[0]>=1",
            "-m2",
            "-M2",
            "-Oz",
            "-o",
            str(args.output_vcf),
            source_vcf,
        ]
    )
    run(["tabix", "-f", "-p", "vcf", str(args.output_vcf)])
    run(["bcftools", "view", "-Ob", "-o", str(args.output_bcf), str(args.output_vcf)])
    run(["bcftools", "index", "-f", str(args.output_bcf)])


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"{name} is required on PATH")


def run(argv: list[str]) -> None:
    print("Running " + " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
