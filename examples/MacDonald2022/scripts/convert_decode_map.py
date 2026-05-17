"""Convert the deCODE recombination map to the 3-column format expected by ldetect2.

The deCODE supplementary file S3 (aau1043_datas3.gz) has 5 columns:
    chr  interval_start  interval_end  cM_per_Mb  cumulative_cM

ldetect2 interpolate-maps expects a gzipped file with a header line, then:
    position  rate_cM_Mb  genetic_position_cM

This script splits the genome-wide deCODE file into one per-chromosome file,
selecting columns: interval_start → position, cM_per_Mb → rate, cumulative_cM → cM.

Usage:
    python scripts/convert_decode_map.py \
        --input data/maps/decode_raw/aau1043_datas3.gz \
        --output-dir data/maps/decode_raw/ \
        [--chromosomes 1 2 22]
"""

from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="deCODE genome-wide map (aau1043_datas3.gz).")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory to write per-chromosome output files.")
    parser.add_argument("--chromosomes", nargs="+", type=int, metavar="N",
                        help="Chromosomes to extract (default: 1–22).")
    args = parser.parse_args()

    target_chroms = set(args.chromosomes) if args.chromosomes else set(range(1, 23))
    # Normalise: accept both "1" and "chr1" as input chromosome names
    target_names = {f"chr{c}" for c in target_chroms} | {str(c) for c in target_chroms}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Buffer rows per chromosome
    rows: dict[str, list[str]] = defaultdict(list)

    print(f"Reading {args.input} ...")
    with gzip.open(args.input, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            chrom_raw = parts[0]
            # Normalise chromosome name to "chrN"
            chrom = chrom_raw if chrom_raw.startswith("chr") else f"chr{chrom_raw}"
            if chrom not in target_names and chrom_raw not in target_names:
                continue
            # Columns: interval_start (1), cM_per_Mb (3), cumulative_cM (4)
            rows[chrom].append(f"{parts[1]}\t{parts[3]}\t{parts[4]}\n")

    for chrom, lines in sorted(rows.items(), key=lambda x: _chrom_sort_key(x[0])):
        chrom_num = chrom.lstrip("chr")
        out_path = args.output_dir / f"chr{chrom_num}.tab.gz"
        with gzip.open(out_path, "wt") as out:
            out.write("position\tcM_per_Mb\tgenetic_position\n")
            out.writelines(lines)
        print(f"  Wrote {len(lines)} intervals → {out_path.name}")


def _chrom_sort_key(chrom: str) -> int:
    try:
        return int(chrom.lstrip("chr"))
    except ValueError:
        return 99


if __name__ == "__main__":
    main()
