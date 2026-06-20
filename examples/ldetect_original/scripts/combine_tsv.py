"""Combine TSV files with identical headers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in args.input:
        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif fieldnames != list(reader.fieldnames or []):
                raise SystemExit(f"Header mismatch in {path}")
            rows.extend(reader)

    if fieldnames is None:
        raise SystemExit("No input files supplied")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
