"""Post-process ldetect-lite BED output for MacDonald2022 replication.

Operations:
  1. Optional centromere removal — drop blocks overlapping a centromeric region.
  2. Small block merging — merge any block with fewer than *min_snps* SNPs
     (counted from a filtered VCF) into its left neighbour.

Usage:
    uv run python scripts/postprocess.py \
        --bed results/EUR/chr2/chr2-ld-blocks.bed \
        --vcf data/filtered/chr2.vcf.gz \
        --centromeres resources/hg38_centromeres.txt.gz \
        [--remove-centromeres] \
        --min-snps 100 \
        --output results/EUR/chr2/chr2-ld-blocks.postprocessed.bed
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
from pathlib import Path

from ldetect_lite.io.bed import read_single_chrom_bed, write_block_bed

# ---------------------------------------------------------------------------
# Centromere removal
# ---------------------------------------------------------------------------

def load_centromeres(path: Path, chrom: str) -> list[tuple[int, int]]:
    """Return centromere intervals for *chrom* from a UCSC centromeres file.

    UCSC centromeres.txt.gz format (no header):
        bin  chrom  chromStart  chromEnd  name  ...
    or tab-delimited with: chrom chromStart chromEnd ...
    """
    intervals: list[tuple[int, int]] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:  # type: ignore[call-overload]
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            # Handle both 5-column UCSC format (with bin) and 3-column BED
            if len(parts) >= 4 and parts[1] == chrom:
                intervals.append((int(parts[2]), int(parts[3])))
            elif len(parts) >= 3 and parts[0] == chrom:
                intervals.append((int(parts[1]), int(parts[2])))
    return intervals


def overlaps_any(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    return any(s < end and start < e for s, e in intervals)


def remove_centromere_blocks(
    blocks: list[tuple[int, int]],
    centromeres: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    return [(s, e) for s, e in blocks if not overlaps_any(s, e, centromeres)]


# ---------------------------------------------------------------------------
# SNP counting
# ---------------------------------------------------------------------------

def count_snps_per_block(
    vcf_path: Path,
    chrom: str,
    blocks: list[tuple[int, int]],
) -> list[int]:
    """Count SNPs in each block using bcftools view."""
    counts: list[int] = []
    for start, end in blocks:
        region = f"{chrom}:{start + 1}-{end}"  # BED → 1-based closed
        result = subprocess.run(
            ["bcftools", "view", "--no-header", "-r", region, str(vcf_path)],
            capture_output=True, text=True, check=True,
        )
        counts.append(result.stdout.count("\n"))
    return counts


# ---------------------------------------------------------------------------
# Small block merging
# ---------------------------------------------------------------------------

def merge_small_blocks(
    blocks: list[tuple[int, int]],
    counts: list[int],
    min_snps: int,
) -> list[tuple[int, int]]:
    """Merge blocks with fewer than *min_snps* SNPs into their left neighbour."""
    if not blocks:
        return blocks

    merged = list(blocks)
    snp_counts = list(counts)

    changed = True
    while changed:
        changed = False
        new_blocks: list[tuple[int, int]] = []
        new_counts: list[int] = []
        i = 0
        while i < len(merged):
            if snp_counts[i] < min_snps and i > 0:
                # Merge into left neighbour
                prev_start, _ = new_blocks[-1]
                _, curr_end = merged[i]
                new_blocks[-1] = (prev_start, curr_end)
                new_counts[-1] += snp_counts[i]
                changed = True
            else:
                new_blocks.append(merged[i])
                new_counts.append(snp_counts[i])
            i += 1
        merged = new_blocks
        snp_counts = new_counts

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bed", required=True, type=Path)
    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--centromeres", required=True, type=Path)
    parser.add_argument(
        "--remove-centromeres",
        action="store_true",
        help=(
            "Drop blocks overlapping supplied centromere intervals. Disabled by "
            "default because the published MacDonald BEDs retain several "
            "centromere-spanning blocks relative to current UCSC intervals."
        ),
    )
    parser.add_argument("--min-snps", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    chrom, blocks = read_single_chrom_bed(args.bed)
    n_raw = len(blocks)
    print(f"Input: {n_raw} blocks on {chrom}")

    # Step 1: optional centromere removal
    if args.remove_centromeres:
        centromeres = load_centromeres(args.centromeres, chrom)
        if centromeres:
            blocks = remove_centromere_blocks(blocks, centromeres)
            print(
                f"After centromere removal: {len(blocks)} blocks "
                f"({n_raw - len(blocks)} removed)"
            )
        else:
            print(f"  No centromere intervals found for {chrom}; skipping")
    else:
        print("Centromere removal disabled")

    # Step 2: small block merging
    print(f"Counting SNPs per block (min_snps={args.min_snps})...")
    counts = count_snps_per_block(args.vcf, chrom, blocks)
    n_small = sum(1 for c in counts if c < args.min_snps)
    blocks = merge_small_blocks(blocks, counts, args.min_snps)
    print(f"After merging {n_small} small blocks: {len(blocks)} blocks")

    write_block_bed(chrom, blocks, args.output)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
