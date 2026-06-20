"""Check ldetect-data all-BED files against chromosome-specific BED files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDNAMES = [
    "population",
    "chrom",
    "match",
    "all_n",
    "chrom_n",
    "first_diff",
]


def read_bed(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open() as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.strip().split()
            if parts[:3] == ["chr", "start", "stop"]:
                continue
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def compare_population(
    population: str,
    all_bed: Path,
    chrom_beds: list[Path],
) -> list[dict[str, str]]:
    all_rows = read_bed(all_bed)
    output: list[dict[str, str]] = []
    for chrom_bed in chrom_beds:
        chrom = chrom_bed.stem.removeprefix("fourier_ls-")
        chrom_rows = read_bed(chrom_bed)
        all_chrom_rows = [row for row in all_rows if row[0] == chrom]
        first_diff = ""
        if chrom_rows != all_chrom_rows:
            common_len = min(len(chrom_rows), len(all_chrom_rows))
            for index in range(common_len):
                if chrom_rows[index] != all_chrom_rows[index]:
                    first_diff = (
                        f"{index}: all={all_chrom_rows[index]} "
                        f"chrom={chrom_rows[index]}"
                    )
                    break
            if not first_diff:
                first_diff = "length differs after common prefix"
        output.append(
            {
                "population": population,
                "chrom": chrom,
                "match": str(chrom_rows == all_chrom_rows),
                "all_n": str(len(all_chrom_rows)),
                "chrom_n": str(len(chrom_rows)),
                "first_diff": first_diff,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", action="append", required=True)
    parser.add_argument("--all-bed", action="append", required=True, type=Path)
    parser.add_argument(
        "--chrom-beds",
        action="append",
        nargs="+",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not (
        len(args.population) == len(args.all_bed) == len(args.chrom_beds)
    ):
        raise SystemExit(
            "--population, --all-bed, and --chrom-beds must be supplied "
            "the same number of times"
        )

    rows: list[dict[str, str]] = []
    for population, all_bed, chrom_beds in zip(
        args.population, args.all_bed, args.chrom_beds
    ):
        rows.extend(compare_population(population, all_bed, chrom_beds))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
