"""Run vendored legacy ldetect end-to-end from a filtered VCF.

This is a MacDonald2022 diagnostic runner, not part of the default
reproduction. It uses the exact filtered VCF, individual list, and genetic map
that the normal ldetect-lite rule uses, writes legacy-compatible partitions,
then drives the shipped vendored legacy scripts from covariance through BED
extraction. The output is intentionally raw pre-postprocessing BED; the
Snakefile applies the same MacDonald postprocess rule afterward when comparing
against the published BEDs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _vendored_legacy_root() -> Path:
    return _repo_root() / "examples" / "ldetect_original" / "scripts" / "legacy_ldetect"


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


def _count_individuals(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


class MapStats(NamedTuple):
    rows: int
    decrease_rows: int
    max_drop_cm: float


def _validate_legacy_map(input_path: Path) -> MapStats:
    """Validate the monotone-map assumption made by original LDetect.

    Original ldetect assumes genetic positions are monotone in physical order.
    Some published MacDonald pyrho maps contain local cM decreases, which make
    the original P00_01 script overflow in ``math.exp``. Treating that as a
    preflight failure preserves the clean diagnostic: legacy is run only on the
    exact published map, never a repaired copy.
    """
    rows = 0
    decrease_rows = 0
    max_drop = 0.0
    running_max: float | None = None

    with gzip.open(input_path, "rt") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 3:
                continue
            cm = float(parts[2])
            if running_max is None or cm >= running_max:
                running_max = cm
            else:
                decrease_rows += 1
                max_drop = max(max_drop, running_max - cm)
            rows += 1

    if rows == 0:
        raise ValueError(f"No map positions found in {input_path}")
    if decrease_rows:
        raise RuntimeError(
            "Original LDetect requires nondecreasing genetic-map cM values, "
            f"but {input_path} has {decrease_rows} local decrease(s) "
            f"(max drop {max_drop:.17g} cM). Refusing to rewrite the map for "
            "this clean legacy-on-published-inputs diagnostic."
        )
    return MapStats(rows=rows, decrease_rows=decrease_rows, max_drop_cm=max_drop)


def _read_map_positions(path: Path) -> list[tuple[int, float]]:
    positions: list[tuple[int, float]] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            positions.append((int(parts[1]), float(parts[2])))
    if not positions:
        raise ValueError(f"No map positions found in {path}")
    return positions


def _write_legacy_partitions(
    genetic_map: Path,
    output: Path,
    n_individuals: int,
    chunk_size: int = 5000,
) -> list[tuple[int, int]]:
    """Port P00_00_partition_chromosome.py, including its hardcoded CEU Ne."""
    positions = _read_map_positions(genetic_map)
    pos2gpos = dict(positions)
    poss = [pos for pos, _gpos in positions]
    chunk_count = int(math.floor(float(len(poss)) / float(chunk_size)))
    partitions: list[tuple[int, int]] = []

    for i in range(chunk_count):
        start = i * chunk_size
        end = i * chunk_size + chunk_size
        if i == chunk_count - 1:
            startpos = poss[start]
            endpos = poss[len(poss) - 1]
            partitions.append((startpos, endpos))
            continue

        startpos = poss[start]
        endpos = poss[end - 1]
        endgpos = pos2gpos[endpos]
        test = end + 1
        while test < len(poss):
            testpos = poss[test]
            testgpos = pos2gpos[testpos]
            df = testgpos - endgpos
            tmp = math.exp(-4.0 * 11418.0 * df / (2.0 * float(n_individuals)))
            if tmp < 1.5e-8:
                break
            test += 1
        if test >= len(poss):
            testpos = poss[-1]
        else:
            testpos = poss[test]
        partitions.append((startpos, testpos))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for start, end in partitions:
            f.write(f"{start} {end}\n")
    return partitions


def _run_covariance_partitions(
    *,
    genetic_map: Path,
    individuals: Path,
    vcf: Path,
    chrom: str,
    partitions: list[tuple[int, int]],
    dataset_path: Path,
    ne: float,
    cutoff: float,
    log_dir: Path,
) -> None:
    p00_01 = (
        _vendored_legacy_root()
        / "ldetect"
        / "examples"
        / "P00_01_calc_covariance.py"
    )
    if not p00_01.exists():
        raise FileNotFoundError(f"Legacy covariance script not found: {p00_01}")

    chrom_dir = dataset_path / chrom
    chrom_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    legacy_paths = [str(_vendored_legacy_root())]
    env["PYTHONPATH"] = os.pathsep.join(
        legacy_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )

    for start, end in partitions:
        region = f"{chrom}:{start}-{end}"
        output = chrom_dir / f"{chrom}.{start}.{end}.gz"
        log_path = log_dir / f"covariance-{start}-{end}.log"
        tabix_cmd = ["tabix", "-h", str(vcf), region]
        legacy_cmd = [
            sys.executable,
            str(p00_01),
            str(genetic_map),
            str(individuals),
            str(ne),
            str(cutoff),
            str(output),
        ]
        with log_path.open("wb") as log:
            tabix = subprocess.Popen(tabix_cmd, stdout=subprocess.PIPE, stderr=log)
            assert tabix.stdout is not None
            legacy = subprocess.Popen(
                legacy_cmd,
                stdin=tabix.stdout,
                stdout=log,
                stderr=log,
                env=env,
            )
            tabix.stdout.close()
            legacy_rc = legacy.wait()
            tabix_rc = tabix.wait()
        if tabix_rc != 0:
            raise RuntimeError(f"tabix failed for {region}; see {log_path}")
        if legacy_rc != 0:
            raise RuntimeError(f"legacy covariance failed for {region}; see {log_path}")


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


def _write_summary(
    output_path: Path,
    *,
    block_set: str,
    chrom: str,
    vector_path: Path,
    pickle_path: Path,
    partition_count: int,
    map_stats: MapStats,
) -> None:
    with pickle_path.open("rb") as f:
        data = pickle.load(f)
    row = {
        "block_set": block_set,
        "chrom": chrom,
        "partition_count": str(partition_count),
        "n_bpoints": str(data["n_bpoints"]),
        "found_width": str(data["found_width"]),
        "fourier_n": str(len(data["fourier"]["loci"])),
        "fourier_ls_n": str(len(data["fourier_ls"]["loci"])),
        "uniform_n": str(len(data["uniform"]["loci"])),
        "uniform_ls_n": str(len(data["uniform_ls"]["loci"])),
        "map_rows": str(map_stats.rows),
        "map_decrease_rows": str(map_stats.decrease_rows),
        "map_max_drop_cm": f"{map_stats.max_drop_cm:.17g}",
    }
    row.update(_vector_summary(vector_path))
    cols = [
        "block_set",
        "chrom",
        "partition_count",
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
        "map_rows",
        "map_decrease_rows",
        "map_max_drop_cm",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


def _run_downstream(
    *,
    dataset_path: Path,
    chrom: str,
    output_dir: Path,
    n_snps_bw_bpoints: int,
    subset: str,
) -> tuple[Path, Path, Path, Path]:
    sys.path.insert(0, str(_vendored_legacy_root()))
    _patch_legacy_scipy_window()

    import P01_matrix_to_vector_pipeline as matrix_pipeline
    import P02_minima_pipeline as minima_pipeline
    import P03_extract_bpoints as extract_pipeline

    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / f"vector-{chrom}.txt.gz"
    pickle_path = output_dir / f"breakpoints-{chrom}.pickle"
    json_path = output_dir / f"breakpoints-{chrom}.json"
    raw_bed_path = output_dir / f"{chrom}-ld-blocks.raw.bed"
    bed_path = output_dir / f"{chrom}-ld-blocks.bed"

    matrix_pipeline.pipeline_lean(str(dataset_path) + "/", chrom, str(vector_path))
    minima_pipeline.pipeline(
        str(vector_path),
        chrom,
        str(dataset_path) + "/",
        n_snps_bw_bpoints,
        str(pickle_path),
    )

    with raw_bed_path.open("w") as f:
        old_stdout = sys.stdout
        try:
            sys.stdout = f
            extract_pipeline.chr_bpoints_to_bed(
                chrom,
                str(dataset_path) + "/",
                subset,
                str(pickle_path),
            )
        finally:
            sys.stdout = old_stdout

    _normalise_legacy_bed(raw_bed_path, bed_path)
    _write_json_breakpoints(pickle_path, json_path)
    return vector_path, pickle_path, json_path, bed_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--individuals", required=True, type=Path)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--block-set", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ne", required=True, type=float)
    parser.add_argument("--cov-cutoff", required=True, type=float)
    parser.add_argument("--n-snps-bw-bpoints", required=True, type=int)
    parser.add_argument("--subset", default="fourier_ls")
    args = parser.parse_args()

    dataset_path = args.output_dir / "legacy_dataset"
    partitions_path = dataset_path / "scripts" / f"{args.chromosome}_partitions"
    log_dir = args.output_dir / "logs"
    output_dir = args.output_dir / "legacy"

    n_individuals = _count_individuals(args.individuals)
    map_stats = _validate_legacy_map(args.genetic_map)
    partitions = _write_legacy_partitions(
        args.genetic_map,
        partitions_path,
        n_individuals=n_individuals,
    )
    _run_covariance_partitions(
        genetic_map=args.genetic_map,
        individuals=args.individuals,
        vcf=args.vcf,
        chrom=args.chromosome,
        partitions=partitions,
        dataset_path=dataset_path,
        ne=args.ne,
        cutoff=args.cov_cutoff,
        log_dir=log_dir,
    )
    vector_path, pickle_path, _json_path, _bed_path = _run_downstream(
        dataset_path=dataset_path,
        chrom=args.chromosome,
        output_dir=output_dir,
        n_snps_bw_bpoints=args.n_snps_bw_bpoints,
        subset=args.subset,
    )
    _write_summary(
        output_dir / "summary.tsv",
        block_set=args.block_set,
        chrom=args.chromosome,
        vector_path=vector_path,
        pickle_path=pickle_path,
        partition_count=len(partitions),
        map_stats=map_stats,
    )


if __name__ == "__main__":
    main()
