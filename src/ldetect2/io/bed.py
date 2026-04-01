"""BED file output for LD block breakpoints."""

from __future__ import annotations

import sys
from pathlib import Path


def write_bed(
    name: str,
    loci: list[int],
    snp_first: int,
    snp_last: int,
    output: Path | None = None,
) -> None:
    """Write breakpoints as a BED file.

    Columns: chromosome_name  region_start  region_stop

    Args:
        name: Chromosome name (e.g. "chr2").
        loci: Sorted list of breakpoint positions.
        snp_first: Start of the first region.
        snp_last: End of the last region (inclusive; written as snp_last + 1).
        output: Output path. Writes to stdout if None.
    """
    lines: list[str] = ["chr\tstart\tstop"]
    lines.append(f"{name}\t{snp_first}\t{loci[0]}")
    for i in range(len(loci) - 1):
        lines.append(f"{name}\t{loci[i]}\t{loci[i + 1]}")
    lines.append(f"{name}\t{loci[-1]}\t{snp_last + 1}")

    text = "\n".join(lines) + "\n"

    if output is None:
        sys.stdout.write(text)
    else:
        output.write_text(text)
