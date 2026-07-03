"""Download a configured provenance diagnostic input file."""

from __future__ import annotations

import argparse
import subprocess
import re
from pathlib import Path

import yaml


def load_source(config_path: Path, dataset: str) -> dict:
    with config_path.open() as f:
        config = yaml.safe_load(f)
    sources = config["sources"]
    if dataset not in sources:
        known = ", ".join(sorted(sources))
        raise SystemExit(f"Unknown dataset {dataset!r}; known datasets: {known}")
    return sources[dataset]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--kind", required=True, choices=["vcf", "vcf-index", "panel"])
    parser.add_argument("--chromosome")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dataset = args.dataset
    chromosome = args.chromosome
    if dataset is None:
        parts = args.output.parts
        try:
            dataset = parts[parts.index("provenance") + 1]
        except (ValueError, IndexError) as exc:
            raise SystemExit(
                "--dataset is required when it cannot be inferred from output path"
            ) from exc
    if args.kind in {"vcf", "vcf-index"} and chromosome is None:
        match = re.search(r"chr([^/.]+)\.vcf\.gz$", args.output.name)
        if not match:
            match = re.search(r"chr([^/.]+)\.vcf\.gz\.tbi$", args.output.name)
        if not match:
            raise SystemExit(
                "--chromosome is required when it cannot be inferred from output path"
            )
        chromosome = match.group(1)

    source = load_source(args.config, dataset)
    if args.kind in {"vcf", "vcf-index"}:
        if not chromosome:
            raise SystemExit("--chromosome is required for --kind vcf")
        fname = source["filename_template"].format(chrom=chromosome)
        if args.kind == "vcf-index":
            fname = f"{fname}.tbi"
        url = f"{source['base_url']}/{fname}"
    else:
        url = source["panel_url"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["wget", "-q", "-O", str(args.output), url], check=True)


if __name__ == "__main__":
    main()
