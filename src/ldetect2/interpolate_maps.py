"""Interpolate genetic map positions onto a set of SNP physical positions.

Python 3 port of joepickrell/1000-genomes-genetic-maps/scripts/interpolate_maps.py.
"""

from __future__ import annotations

import gzip
from pathlib import Path


def interpolate(
    snp_file: Path,
    genetic_map: Path,
    output: Path,
) -> None:
    """Assign genetic positions to SNPs by linear interpolation.

    Reads SNP physical positions from a BED file and a reference recombination
    map, then writes a gzipped 3-column output file.

    **Boundary behaviour**:
    - Positions before the first map marker receive genetic position 0.
    - Positions after the last map marker receive the last marker's genetic
      position (no extrapolation).

    Args:
        snp_file: BED file with columns ``chrom start end rs_id ...``.
            Physical position is taken from column 2 (0-based, half-open end).
        genetic_map: Gzipped recombination map.  Expected columns:
            ``position  rate_cM_Mb  genetic_position_cM``
            (OMNI-style; column 0 = position, column 2 = cM).
            The first header line is skipped automatically.
        output: Gzipped output file with columns ``rs_id  position  genetic_pos``.
    """
    # Read SNP positions from BED
    snp_positions: list[int] = []
    snp_ids: list[str] = []
    with gzip.open(snp_file, "rt") if _is_gz(snp_file) else open(snp_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            snp_positions.append(int(parts[2]))  # BED end coord = physical position
            snp_ids.append(parts[3])

    # Read reference recombination map
    map_positions: list[int] = []
    map_gpos: list[float] = []
    with gzip.open(genetic_map, "rt") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            map_positions.append(int(parts[0]))
            map_gpos.append(float(parts[2]))

    # Interpolate
    with gzip.open(output, "wt") as out:
        map_idx = 0
        for snp_idx, (pos, rs) in enumerate(zip(snp_positions, snp_ids)):
            # Advance map pointer past positions < pos
            while map_idx < len(map_positions) - 1 and map_positions[map_idx] < pos:
                map_idx += 1

            if map_positions[map_idx] == pos:
                # Exact match
                gp = map_gpos[map_idx]
            elif pos < map_positions[map_idx]:
                if map_idx == 0:
                    # Before first map marker
                    gp = 0.0
                else:
                    # Interpolate between map_idx-1 and map_idx
                    frac = (
                        (pos - map_positions[map_idx - 1])
                        / (map_positions[map_idx] - map_positions[map_idx - 1])
                    )
                    gp = map_gpos[map_idx - 1] + frac * (
                        map_gpos[map_idx] - map_gpos[map_idx - 1]
                    )
            else:
                # pos > all map positions — clamp to last
                gp = map_gpos[-1]

            out.write(f"{rs} {pos} {gp}\n")


def _is_gz(path: Path) -> bool:
    return path.suffix.lower() in (".gz", ".gzip")
