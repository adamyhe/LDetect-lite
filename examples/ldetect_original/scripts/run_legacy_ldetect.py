"""Run vendored legacy ldetect on a staged covariance dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path


def _install_legacy_import_path() -> Path:
    legacy_root = Path(__file__).resolve().parent / "legacy_ldetect"
    sys.path.insert(0, str(legacy_root))
    return legacy_root


def _patch_legacy_scipy_window() -> None:
    import scipy.signal as signal

    original_get_window = signal.get_window

    def get_window(window, nx, fftbins=True, *, xp=None, device=None):
        if window == "hanning":
            window = "hann"
        try:
            return original_get_window(
                window, nx, fftbins=fftbins, xp=xp, device=device
            )
        except TypeError:
            return original_get_window(window, nx, fftbins=fftbins)

    signal.get_window = get_window


def _normalise_legacy_bed(raw_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open() as src, output_path.open("w", newline="") as dst:
        writer = csv.writer(dst, delimiter="\t")
        for line_num, raw in enumerate(src):
            parts = raw.split()
            if not parts:
                continue
            if line_num == 0 and parts[0] == "chr":
                writer.writerow(["#chr", "start", "stop"])
                continue
            chrom = parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
            writer.writerow([chrom, parts[1], parts[2]])


def _vector_summary(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    rows = 0
    first = ""
    last = ""
    with gzip.open(path, "rb") as f:
        for raw in f:
            digest.update(raw)
            parts = raw.decode().split()
            if not parts:
                continue
            if first == "":
                first = parts[0]
            last = parts[0]
            rows += 1
    return {
        "vector_rows": str(rows),
        "vector_first_locus": first,
        "vector_last_locus": last,
        "vector_sha256": digest.hexdigest(),
    }


def _write_summary(
    output_path: Path,
    *,
    population: str,
    chromosome: str,
    vector_path: Path,
    pickle_path: Path,
) -> None:
    with pickle_path.open("rb") as f:
        data = pickle.load(f)
    row = {
        "population": population,
        "chrom": f"chr{chromosome}",
        "n_bpoints": str(data["n_bpoints"]),
        "found_width": str(data["found_width"]),
        "fourier_n": str(len(data["fourier"]["loci"])),
        "fourier_ls_n": str(len(data["fourier_ls"]["loci"])),
        "uniform_n": str(len(data["uniform"]["loci"])),
        "uniform_ls_n": str(len(data["uniform_ls"]["loci"])),
    }
    row.update(_vector_summary(vector_path))
    cols = [
        "population",
        "chrom",
        "vector_rows",
        "vector_first_locus",
        "vector_last_locus",
        "vector_sha256",
        "n_bpoints",
        "found_width",
        "fourier_n",
        "fourier_ls_n",
        "uniform_n",
        "uniform_ls_n",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


def _write_json_breakpoints(pickle_path: Path, output_path: Path) -> None:
    with pickle_path.open("rb") as f:
        data = pickle.load(f)
    serialisable = {
        "n_bpoints": data["n_bpoints"],
        "found_width": data["found_width"],
        "fourier": {"loci": [int(x) for x in data["fourier"]["loci"]]},
        "fourier_ls": {"loci": [int(x) for x in data["fourier_ls"]["loci"]]},
        "uniform": {"loci": [int(x) for x in data["uniform"]["loci"]]},
        "uniform_ls": {"loci": [int(x) for x in data["uniform_ls"]["loci"]]},
    }
    output_path.write_text(json.dumps(serialisable, indent=2) + "\n")


def run_legacy(args: argparse.Namespace) -> None:
    _install_legacy_import_path()
    _patch_legacy_scipy_window()

    import P01_matrix_to_vector_pipeline as matrix_pipeline
    import P02_minima_pipeline as minima_pipeline
    import P03_extract_bpoints as extract_pipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = args.output_dir / f"vector-{args.chromosome}.txt.gz"
    pickle_path = args.output_dir / f"breakpoints-{args.chromosome}.pickle"
    raw_bed_path = args.output_dir / f"{args.chromosome}-ld-blocks.raw.bed"
    bed_path = args.output_dir / f"{args.chromosome}-ld-blocks.bed"
    summary_path = args.output_dir / "summary.tsv"
    json_path = args.output_dir / f"breakpoints-{args.chromosome}.json"

    if args.stage in ("all", "matrix-to-vector"):
        matrix_pipeline.pipeline_lean(
            str(args.dataset_path) + "/",
            args.chromosome,
            str(vector_path),
        )

    if args.stage in ("all", "find-minima"):
        minima_pipeline.pipeline(
            str(vector_path),
            args.chromosome,
            str(args.dataset_path) + "/",
            args.n_snps_bw_bpoints,
            str(pickle_path),
        )

    if args.stage not in ("all", "extract-bpoints"):
        return

    with raw_bed_path.open("w") as f:
        old_stdout = sys.stdout
        try:
            sys.stdout = f
            extract_pipeline.chr_bpoints_to_bed(
                args.chromosome,
                str(args.dataset_path) + "/",
                args.subset,
                str(pickle_path),
            )
        finally:
            sys.stdout = old_stdout

    _normalise_legacy_bed(raw_bed_path, bed_path)
    _write_json_breakpoints(pickle_path, json_path)
    if args.stage == "all":
        _write_summary(
            summary_path,
            population=args.population,
            chromosome=args.chromosome,
            vector_path=vector_path,
            pickle_path=pickle_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-snps-bw-bpoints", required=True, type=int)
    parser.add_argument("--subset", default="fourier_ls")
    parser.add_argument(
        "--stage",
        choices=("all", "matrix-to-vector", "find-minima", "extract-bpoints"),
        default="all",
    )
    args = parser.parse_args()
    run_legacy(args)


if __name__ == "__main__":
    main()
