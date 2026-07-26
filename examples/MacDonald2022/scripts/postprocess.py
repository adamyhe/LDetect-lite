"""Post-process ldetect-lite BED output for MacDonald2022 replication.

Operations:
  1. Optional centromere removal — drop blocks overlapping a centromeric region.
  2. VCF-span trimming and empty block removal — drop edge/empty blocks outside
     the filtered VCF SNP support.
  3. Optional small block merging — merge any block with fewer than *min_snps*
     SNPs (counted from a filtered VCF) into its left neighbour. Disabled when
     *min_snps* is 0.

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


def drop_nonpositive_blocks(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop invalid zero/negative-width BED intervals."""
    return [(s, e) for s, e in blocks if s < e]


def drop_empty_blocks(
    blocks: list[tuple[int, int]],
    counts: list[int],
) -> tuple[list[tuple[int, int]], list[int]]:
    """Drop blocks with no SNPs in the filtered VCF."""
    kept = [
        (block, count)
        for block, count in zip(blocks, counts, strict=True)
        if count > 0
    ]
    return [block for block, _ in kept], [count for _, count in kept]


def trim_blocks_to_snp_span(
    blocks: list[tuple[int, int]],
    first_snp: int,
    last_snp: int,
) -> list[tuple[int, int]]:
    """Drop leading/trailing edge intervals outside the filtered VCF SNP span."""
    return [
        (start, end)
        for start, end in blocks
        if end > first_snp and start <= last_snp
    ]


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


def vcf_snp_span(vcf_path: Path, chrom: str) -> tuple[int, int]:
    """Return first and last SNP position in *chrom* from a filtered VCF."""
    result = subprocess.run(
        ["bcftools", "query", "-f", "%POS\n", "-r", chrom, str(vcf_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    first: int | None = None
    last: int | None = None
    for line in result.stdout.splitlines():
        if not line:
            continue
        position = int(line)
        if first is None:
            first = position
        last = position
    if first is None or last is None:
        raise RuntimeError(f"No SNPs found for {chrom} in {vcf_path}")
    return first, last


# ---------------------------------------------------------------------------
# Small block merging
# ---------------------------------------------------------------------------

def merge_small_blocks(
    blocks: list[tuple[int, int]],
    counts: list[int],
    min_snps: int,
) -> list[tuple[int, int]]:
    """Merge blocks with fewer than *min_snps* SNPs into a neighbour.

    Interior small blocks follow MacDonald's left-merge convention. A leading
    small block has no left neighbour, so merge it into the right block instead.
    """
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
            elif snp_counts[i] < min_snps and i + 1 < len(merged):
                # Leading small block: merge into right neighbour.
                curr_start, _ = merged[i]
                _, next_end = merged[i + 1]
                new_blocks.append((curr_start, next_end))
                new_counts.append(snp_counts[i] + snp_counts[i + 1])
                i += 1
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

    blocks = drop_nonpositive_blocks(blocks)
    n_nonpositive = n_raw - len(blocks)
    if n_nonpositive:
        print(f"After dropping nonpositive-width blocks: {len(blocks)} blocks")

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

    # Step 2: VCF-span trimming, empty block removal, and optional small block merging
    first_snp, last_snp = vcf_snp_span(args.vcf, chrom)
    n_before_span_trim = len(blocks)
    blocks = trim_blocks_to_snp_span(blocks, first_snp, last_snp)
    n_span_trimmed = n_before_span_trim - len(blocks)
    if n_span_trimmed:
        print(
            f"After trimming {n_span_trimmed} block(s) outside VCF SNP span "
            f"{first_snp}-{last_snp}: {len(blocks)} blocks"
        )

    print("Counting SNPs per block...")
    counts = count_snps_per_block(args.vcf, chrom, blocks)
    n_empty = sum(1 for c in counts if c == 0)
    blocks, counts = drop_empty_blocks(blocks, counts)
    if n_empty:
        print(f"After dropping {n_empty} empty blocks: {len(blocks)} blocks")

    if args.min_snps > 0:
        print(f"Merging blocks with fewer than {args.min_snps} SNPs...")
        n_small = sum(1 for c in counts if c < args.min_snps)
        blocks = merge_small_blocks(blocks, counts, args.min_snps)
        print(f"After merging {n_small} small blocks: {len(blocks)} blocks")
    else:
        print("Small-block merging disabled")

    write_block_bed(chrom, blocks, args.output)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
