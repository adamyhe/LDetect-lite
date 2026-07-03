"""Run targeted VCF provenance position-set diagnostics.

The Snakemake wrapper intentionally keeps this as one static rule because the
candidate inputs are multi-GB remote VCFs and the diagnostic matrix should stay
small and explicit.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import yaml

from compare_vcf_positions import FIELDNAMES, compare


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def raw_vcf_path(raw_root: Path, dataset: str, chrom: str) -> Path:
    return raw_root / dataset / f"chr{chrom}.vcf.gz"


def panel_path(res_root: Path, dataset: str) -> Path:
    return res_root / dataset / "panel.tsv"


def individuals_path(res_root: Path, dataset: str, population: str) -> Path:
    return res_root / dataset / f"{population}_inds.txt"


def filtered_vcf_path(
    filt_root: Path,
    dataset: str,
    variant_filter: str,
    population: str,
    chrom: str,
) -> Path:
    return (
        filt_root
        / dataset
        / variant_filter
        / population
        / f"chr{chrom}.{population}.{variant_filter}.vcf.gz"
    )


def ensure_download(
    diag_config: Path,
    dataset: str,
    kind: str,
    output: Path,
    chrom: str | None = None,
) -> None:
    if output.exists():
        return
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/download_provenance_input.py",
        "--config",
        str(diag_config),
        "--dataset",
        dataset,
        "--kind",
        kind,
        "--output",
        str(output),
    ]
    if chrom is not None:
        cmd.extend(["--chromosome", chrom])
    run(cmd)


def ensure_individuals(
    base: dict,
    res_root: Path,
    raw_root: Path,
    diag_config: Path,
    dataset: str,
    population: str,
) -> Path:
    output = individuals_path(res_root, dataset, population)
    if output.exists():
        return output
    source_panel = panel_path(res_root, dataset)
    source_chr22 = raw_vcf_path(raw_root, dataset, "22")
    ensure_download(diag_config, dataset, "panel", source_panel)
    ensure_download(diag_config, dataset, "vcf", source_chr22, "22")
    subpops = base["populations"][population]["subpops"]
    run(
        [
            "uv",
            "run",
            "python",
            "scripts/prep_individuals.py",
            "--panel",
            str(source_panel),
            "--subpops",
            *subpops,
            "--vcf",
            str(source_chr22),
            "--output",
            str(output),
        ]
    )
    return output


def ensure_filtered_vcf(
    base: dict,
    raw_root: Path,
    filt_root: Path,
    res_root: Path,
    diag_config: Path,
    dataset: str,
    variant_filter: str,
    population: str,
    chrom: str,
) -> Path:
    output = filtered_vcf_path(filt_root, dataset, variant_filter, population, chrom)
    if output.exists() and output.with_suffix(output.suffix + ".tbi").exists():
        return output

    raw_vcf = raw_vcf_path(raw_root, dataset, chrom)
    ensure_download(diag_config, dataset, "vcf", raw_vcf, chrom)
    individuals = ensure_individuals(
        base, res_root, raw_root, diag_config, dataset, population
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    type_filter = ["-v", "snps"] if variant_filter == "snps" else []
    shell_cmd = [
        "bash",
        "-lc",
        "set -euo pipefail; bcftools view -S \"$1\" -Ou \"$2\" | "
        "bcftools view $3 -i 'MAC[0]>=1' -m2 -M2 -Oz -o \"$4\"",
        "filter-vcf",
        str(individuals),
        str(raw_vcf),
        " ".join(type_filter),
        str(output),
    ]
    run(shell_cmd)
    run(["tabix", "-f", "-p", "vcf", str(output)])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--diagnostics-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    base = load_yaml(args.base_config)
    diag = load_yaml(args.diagnostics_config)
    raw_root = Path(base["raw_vcf_dir"]) / "provenance"
    res_root = Path(base["resources_dir"]) / "provenance"
    filt_root = Path(base["results_dir"]) / "provenance_diagnostics" / "filtered_vcf"
    detail_root = Path(base["results_dir"]) / "provenance_diagnostics"

    baseline_source = str(diag["baseline_source"])
    baseline_filter = str(diag["baseline_variant_filter"])
    rows: list[dict[str, str]] = []

    for population in diag["diagnostic_populations"]:
        chroms = [str(chrom) for chrom in diag["chromosomes_by_population"][population]]
        for chrom in chroms:
            baseline_vcf = ensure_filtered_vcf(
                base,
                raw_root,
                filt_root,
                res_root,
                args.diagnostics_config,
                baseline_source,
                baseline_filter,
                population,
                chrom,
            )
            for candidate in diag["comparison_candidates"]:
                candidate_source = str(candidate["source"])
                candidate_filter = str(candidate["variant_filter"])
                candidate_vcf = ensure_filtered_vcf(
                    base,
                    raw_root,
                    filt_root,
                    res_root,
                    args.diagnostics_config,
                    candidate_source,
                    candidate_filter,
                    population,
                    chrom,
                )
                out = (
                    detail_root
                    / population
                    / f"chr{chrom}"
                    / "position_sets"
                    / f"{candidate_source}_{candidate_filter}_vs_{baseline_source}_{baseline_filter}.tsv"
                )
                row = compare(
                    argparse.Namespace(
                        population=population,
                        chromosome=chrom,
                        baseline_label=f"{baseline_source}/{baseline_filter}",
                        candidate_label=f"{candidate_source}/{candidate_filter}",
                        baseline_vcf=baseline_vcf,
                        candidate_vcf=candidate_vcf,
                    )
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
                    writer.writeheader()
                    writer.writerow(row)
                rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
