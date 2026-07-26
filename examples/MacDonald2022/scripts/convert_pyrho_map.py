"""Extract and convert pyrho's GRCh38 HapMap-format maps.

pyrho publishes per-population maps in a tarball. MacDonald et al.'s
``interpolate_pyhro.R`` expects files named like:

    hg38/IBS/IBS_recombination_map_hapmap_format_hg38_chr_21.txt

with four HapMap-style columns:

    Chromosome  Position(bp)  Rate(cM/Mb)  Map(cM)

The HapMap ``Map(cM)`` column is the cumulative genetic position at
``Position(bp)``. ``ldetect interpolate-maps --mode hapmap`` consumes this
coordinate convention directly from a gzipped three-column map:

    position  cM_per_Mb  genetic_position_cM

where ``genetic_position_cM`` is the cumulative cM at ``position``. This script
extracts one population/chromosome map from the archive, writes the unshifted
three-column HapMap-style form, and fails if the cumulative map is not
nondecreasing. Feeding this output through interval interpolation reproduces
MacDonald et al.'s pyrho interpolation convention; the corrected path uses
``ldetect interpolate-maps --mode hapmap``.
"""

from __future__ import annotations

import argparse
import gzip
import io
import re
import tarfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PyrhoRow:
    chrom: str
    position: int
    rate_cm_mb: float
    map_cm: float


def _normalise_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _column_indexes(header: list[str]) -> tuple[int, int, int, int]:
    names = {_normalise_header(name): idx for idx, name in enumerate(header)}

    chrom_idx = names.get("chromosome", names.get("chr", 0))
    pos_idx = (
        names.get("positionbp")
        if "positionbp" in names
        else names.get("position", names.get("begin", 1))
    )
    rate_idx = names.get("ratecmmb", names.get("cmpermb", 2))
    map_idx = names.get("mapcm", names.get("cm", 3))
    return chrom_idx, pos_idx, rate_idx, map_idx


def _open_text(path: Path) -> Iterator[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            yield from handle
    else:
        with open(path) as handle:
            yield from handle


def _read_rows(lines: Iterable[str]) -> list[PyrhoRow]:
    iterator = iter(lines)
    for first in iterator:
        first = first.strip()
        if first:
            break
    else:
        raise ValueError("pyrho map is empty")

    header = first.split()
    chrom_idx, pos_idx, rate_idx, map_idx = _column_indexes(header)
    rows: list[PyrhoRow] = []

    for line_no, line in enumerate(iterator, start=2):
        parts = line.strip().split()
        if not parts:
            continue
        try:
            rows.append(
                PyrhoRow(
                    chrom=parts[chrom_idx],
                    position=int(float(parts[pos_idx])),
                    rate_cm_mb=float(parts[rate_idx]),
                    map_cm=float(parts[map_idx]),
                )
            )
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"could not parse pyrho map line {line_no}: {line!r}"
            ) from exc

    if not rows:
        raise ValueError("pyrho map has no data rows")
    return rows


def _member_name(population: str, chromosome: str) -> str:
    chrom = chromosome.removeprefix("chr")
    return f"{population}_recombination_map_hapmap_format_hg38_chr_{chrom}.txt"


def _extract_from_archive(archive: Path, population: str, chromosome: str) -> list[str]:
    wanted = _member_name(population, chromosome)
    pop_segment = f"/{population}/"

    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            normalised = "/" + member.name.lstrip("/")
            if member.name.endswith(wanted) and pop_segment in normalised:
                handle = tar.extractfile(member)
                if handle is None:
                    raise FileNotFoundError(f"could not extract {member.name!r}")
                text = io.TextIOWrapper(handle)
                return text.read().splitlines()

    raise FileNotFoundError(
        f"archive has no member ending with {wanted!r} under population "
        f"{population!r}"
    )


def _validate_rows(rows: list[PyrhoRow], chromosome: str) -> None:
    chrom = chromosome.removeprefix("chr")
    prev_position = -1
    prev_cm = float("-inf")

    for idx, row in enumerate(rows, start=1):
        row_chrom = row.chrom.removeprefix("chr")
        if row_chrom != chrom:
            raise ValueError(
                f"row {idx} has chromosome {row.chrom!r}, expected chr{chrom}"
            )
        if row.position < prev_position:
            raise ValueError(
                f"positions are not sorted at row {idx}: "
                f"{row.position} < {prev_position}"
            )
        if row.rate_cm_mb < 0:
            raise ValueError(f"row {idx} has negative recombination rate")
        if row.map_cm < prev_cm:
            raise ValueError(
                f"cumulative cM decreases at row {idx}: {row.map_cm} < {prev_cm}"
            )
        prev_position = row.position
        prev_cm = row.map_cm


def convert(
    *,
    archive: Path | None,
    input_map: Path | None,
    population: str,
    chromosome: str,
    output: Path,
) -> None:
    if (archive is None) == (input_map is None):
        raise ValueError("provide exactly one of --archive or --input-map")

    if archive is not None:
        rows = _read_rows(_extract_from_archive(archive, population, chromosome))
    else:
        assert input_map is not None
        rows = _read_rows(_open_text(input_map))

    _validate_rows(rows, chromosome)

    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt") as out:
        out.write("position\tcM_per_Mb\tgenetic_position\n")
        for row in rows:
            out.write(f"{row.position}\t{row.rate_cm_mb:.17g}\t{row.map_cm:.17g}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="pyrho hg38_maps.tar.gz")
    source.add_argument(
        "--input-map",
        type=Path,
        help="One already-extracted pyrho HapMap-format map file.",
    )
    parser.add_argument("--population", required=True, help="pyrho population code")
    parser.add_argument("--chromosome", required=True, help="Chromosome, e.g. 21")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    convert(
        archive=args.archive,
        input_map=args.input_map,
        population=args.population,
        chromosome=args.chromosome,
        output=args.output,
    )


if __name__ == "__main__":
    main()
