"""BED file I/O for LD block breakpoints."""

from __future__ import annotations

import gzip
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

Block = tuple[int, int]


def _is_compressed_path(path: Path) -> bool:
    return path.suffix.lower() in {".gz", ".gzip", ".bgz"}


def _open_text(path: Path, mode: str) -> TextIO:
    if _is_compressed_path(path):
        return gzip.open(path, mode)  # type: ignore[return-value]
    return open(path, mode)  # type: ignore[return-value]


def _iter_bed_records(path: Path) -> Iterator[tuple[str, int, int]]:
    with _open_text(path, "rt") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            parts = line.split()
            if len(parts) < 3 or not parts[1].lstrip("-").isdigit():
                continue
            try:
                yield parts[0], int(parts[1]), int(parts[2])
            except ValueError:
                continue


def read_genome_bed(path: Path) -> dict[str, list[Block]]:
    """Return ``{chrom: [(start, end), ...]}`` from a BED-like file."""
    blocks: dict[str, list[Block]] = {}
    for chrom, start, end in _iter_bed_records(path):
        blocks.setdefault(chrom, []).append((start, end))
    return blocks


def read_single_chrom_bed(path: Path) -> tuple[str, list[Block]]:
    """Return ``(chrom, blocks)`` from a single-chromosome BED-like file."""
    chrom = ""
    blocks: list[Block] = []
    for record_chrom, start, end in _iter_bed_records(path):
        if not chrom:
            chrom = record_chrom
        blocks.append((start, end))
    return chrom, blocks


def write_block_bed(chrom: str, blocks: list[Block], output: Path) -> None:
    """Write already-materialized block intervals as a simple BED-like file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["chr\tstart\tstop"]
    lines.extend(f"{chrom}\t{start}\t{end}" for start, end in blocks)
    text = "\n".join(lines) + "\n"

    if not _is_compressed_path(output):
        output.write_text(text)
        return

    try:
        result = subprocess.run(
            ["bgzip", "-c"],
            input=text.encode(),
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Cannot write compressed BED output because bgzip was not found. "
            "Install htslib/bgzip or write an uncompressed .bed file."
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"bgzip failed while writing {output}: {message}") from exc

    output.write_bytes(result.stdout)


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
    lines: list[str] = ["#chr\tstart\tstop"]
    lines.append(f"{name}\t{snp_first}\t{loci[0]}")
    for i in range(len(loci) - 1):
        lines.append(f"{name}\t{loci[i]}\t{loci[i + 1]}")
    lines.append(f"{name}\t{loci[-1]}\t{snp_last + 1}")

    text = "\n".join(lines) + "\n"

    if output is None:
        sys.stdout.write(text)
    else:
        output.write_text(text)
