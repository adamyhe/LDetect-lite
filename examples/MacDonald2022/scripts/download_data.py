"""Download raw input data for the MacDonald et al. GRCh38 LD block analysis.

Downloads:
  - 1000G GRCh38 Dec 2018 biallelic SNV VCFs (one per chromosome + index)
  - deCODE recombination map (Halldorsson et al. 2019)
  - UCSC GRCh38 centromere coordinates

Sample lists are built by the Snakemake workflow from the 1000G population
directories, following the MacDonald et al. README.

Usage:
    uv run python scripts/download_data.py --config config.yaml [--chromosomes 22]

The deCODE map is behind a paywall. If the download fails, place the file
manually at data/maps/decode_raw/aau1043_datas3.gz and re-run; the script
will skip files that already exist.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import yaml


def _download(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {label} → {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        print(f"  Download manually and place at: {dest}", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--chromosomes", nargs="+", type=int, metavar="N",
        help="Subset of chromosomes to download (default: all from config)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    base = args.config.parent
    chroms = args.chromosomes or cfg["chromosomes"]

    raw_dir = base / cfg["raw_vcf_dir"]
    maps_dir = base / cfg["maps_dir"]
    resources_dir = base / cfg["resources_dir"]

    # ------------------------------------------------------------------ #
    # VCFs                                                                 #
    # ------------------------------------------------------------------ #
    print(f"\nDownloading 1000G VCFs ({len(chroms)} chromosomes)...")
    for chrom in chroms:
        fname = cfg["vcf_filename_template"].format(chrom=chrom)
        url_base = cfg["vcf_base_url"].rstrip("/")
        _download(f"{url_base}/{fname}", raw_dir / fname, f"chr{chrom} VCF")
        _download(
            f"{url_base}/{fname}.tbi",
            raw_dir / f"{fname}.tbi",
            f"chr{chrom} index",
        )

    # ------------------------------------------------------------------ #
    # deCODE recombination map                                             #
    # ------------------------------------------------------------------ #
    print("\nDownloading deCODE recombination map...")
    decode_dest = maps_dir / "decode_raw" / "aau1043_datas3.gz"
    try:
        _download(cfg["decode_map_url"], decode_dest, "deCODE map")
    except Exception:
        print(
            "  NOTE: Place the deCODE supplementary file S3 at:\n"
            f"  {decode_dest}\n"
            "  then re-run this script.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------ #
    # Centromere coordinates                                               #
    # ------------------------------------------------------------------ #
    print("\nDownloading UCSC GRCh38 centromere coordinates...")
    centromere_dest = resources_dir / "hg38_centromeres.txt.gz"
    _download(cfg["centromere_url"], centromere_dest, "centromeres")

    print("\nDone.")


if __name__ == "__main__":
    main()
