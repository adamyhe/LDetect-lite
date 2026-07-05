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
    snp_positions, snp_ids = _read_snp_bed(snp_file)
    map_positions, _rates, map_gpos = _read_map_rows(genetic_map)

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
                    frac = (pos - map_positions[map_idx - 1]) / (
                        map_positions[map_idx] - map_positions[map_idx - 1]
                    )
                    gp = map_gpos[map_idx - 1] + frac * (
                        map_gpos[map_idx] - map_gpos[map_idx - 1]
                    )
            else:
                # pos > all map positions — clamp to last
                gp = map_gpos[-1]

            out.write(f"{rs} {pos} {gp}\n")


def interpolate_intervals(
    snp_file: Path,
    genetic_map: Path,
    output: Path,
) -> None:
    """Assign genetic positions to SNPs using interval-rate interpolation.

    Direct port of MacDonald et al.'s R interpolation scripts
    (https://github.com/jmacdon/LDblocks_GRCh38/blob/master/scripts/interpolate.R
    and
    https://github.com/jmacdon/LDblocks_GRCh38/blob/master/scripts/interpolate_pyhro.R),
    which is the algorithm that actually produced the published deCODE/pyrho
    reference maps. Unlike :func:`interpolate`,
    which treats the map as discrete points and interpolates *between* two
    bracketing points, this treats each map row as the start of a
    genomic interval with its own recombination rate: for a SNP falling in
    interval ``i`` (``Begin[i] <= pos``, and ``pos < Begin[i+1]`` when a next
    row exists), the genetic position is

        cM = (0 if i == 0 else cM[i-1]) + (pos - Begin[i]) * rate[i] / 1e6

    i.e. anchored at the *previous* interval's cumulative endpoint and
    advanced using *this* interval's own rate — not derived from the
    difference between two map rows' cM values (which is what
    :func:`interpolate` does, and which is incorrect for this data: the
    map's ``cM`` column is the cumulative genetic position at each
    interval's *end*, not at its start, so bracketing between rows ``i``
    and ``i+1`` uses interval ``i+1``'s rate for a SNP physically located
    in interval ``i``).

    **Boundary behaviour**:
    - Positions before the first interval's start receive genetic position 0
      (matches :func:`interpolate` and the R script's own convention).
    - Positions at or past the last interval's start continue extrapolating
      with that interval's own rate (matches the R script, which extends
      the last interval's end past the last SNP rather than clamping) —
      this differs from :func:`interpolate`'s clamp-to-last-value behavior.

    Args:
        snp_file: BED file with columns ``chrom start end rs_id ...``.
            Physical position is taken from column 2 (0-based, half-open end).
        genetic_map: Gzipped interval-rate recombination map, as produced by
            ``convert_decode_map.py``. Columns: ``position(=Begin)
            rate_cM_Mb  genetic_position_cM(=cumulative cM at End)``.
            The first header line is skipped automatically.
        output: Gzipped output file with columns ``rs_id  position  genetic_pos``.
    """
    snp_positions, snp_ids = _read_snp_bed(snp_file)
    begins, rates, cum_cm = _read_map_rows(genetic_map)
    n = len(begins)

    with gzip.open(output, "wt") as out:
        idx = 0
        for pos, rs in zip(snp_positions, snp_ids):
            while idx < n - 1 and begins[idx + 1] <= pos:
                idx += 1

            if pos < begins[0]:
                gp = 0.0
            else:
                startcm = 0.0 if idx == 0 else cum_cm[idx - 1]
                gp = startcm + (pos - begins[idx]) * rates[idx] / 1e6

            out.write(f"{rs} {pos} {gp}\n")


def _read_snp_bed(snp_file: Path) -> tuple[list[int], list[str]]:
    positions: list[int] = []
    ids: list[str] = []
    with gzip.open(snp_file, "rt") if _is_gz(snp_file) else open(snp_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            positions.append(int(parts[2]))  # BED end coord = physical position
            ids.append(parts[3])
    return positions, ids


def _read_map_rows(genetic_map: Path) -> tuple[list[int], list[float], list[float]]:
    positions: list[int] = []
    rates: list[float] = []
    gpos: list[float] = []
    with gzip.open(genetic_map, "rt") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            positions.append(int(parts[0]))
            rates.append(float(parts[1]))
            gpos.append(float(parts[2]))
    return positions, rates, gpos


def _is_gz(path: Path) -> bool:
    return path.suffix.lower() in (".gz", ".gzip")
