"""Function-level benchmark and exactness check for the VCF-start toy example.

Run the Snakemake workflow once first so that ``ref/`` and ``work/vcf/`` exist.
This script avoids CLI launch overhead by calling ldetect-lite functions
directly for the four measured stages:

* calc_covariance from the prepared 1000G VCF
* MatrixAnalysis.calc_diag_lean
* find_breakpoints
* BED extraction via write_bed
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import shutil
import tempfile
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ldetect_lite.io.bed import read_single_chrom_bed, write_bed
from ldetect_lite.io.covariance_hdf5 import open_covariance_reader
from ldetect_lite.io.partitions import CovarianceStore, read_partitions
from ldetect_lite.matrix_analysis import MatrixAnalysis
from ldetect_lite.pipeline import find_breakpoints
from ldetect_lite.shrinkage import calc_covariance


@dataclass(frozen=True)
class Config:
    chrom: str
    vcf_chrom: str
    start: int
    end: int
    ne: float
    cutoff: float
    n_snps_bw_bpoints: int
    subset: str
    reference_panel: Path
    genetic_map: Path
    individuals: Path
    partitions: Path
    ref_covariance: Path
    ref_vector: Path
    ref_bpoints: Path
    ref_bed: Path
    ld_kernel: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrom", default="chr2")
    parser.add_argument("--vcf-chrom", default="2")
    parser.add_argument("--start", type=int, default=39_967_768)
    parser.add_argument("--end", type=int, default=40_067_768)
    parser.add_argument("--ne", type=float, default=11418.0)
    parser.add_argument("--cutoff", type=float, default=1e-7)
    parser.add_argument("--n-snps-bw-bpoints", type=int, default=50)
    parser.add_argument("--subset", default="fourier_ls")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument(
        "--ld-kernel",
        choices=("bitpacked", "uint8"),
        default="bitpacked",
        help=(
            "Covariance backend to benchmark. bitpacked writes compact HDF5 "
            "and compares shrinkage rows only; uint8 keeps the reference "
            "backend available for comparison (default: bitpacked)."
        ),
    )
    parser.add_argument(
        "--plot-format",
        choices=("svg", "png"),
        default="svg",
        help="Format for benchmark plots (default: svg).",
    )
    parser.add_argument("--reference-panel", type=Path, default=None)
    parser.add_argument(
        "--genetic-map",
        type=Path,
        default=Path("ref/chr2.interpolated_genetic_map.gz"),
    )
    parser.add_argument("--individuals", type=Path, default=Path("ref/eurinds.txt"))
    parser.add_argument(
        "--partitions",
        type=Path,
        default=Path("ref/cov_matrix/scripts/chr2_partitions"),
    )
    parser.add_argument(
        "--ref-covariance",
        type=Path,
        default=Path("ref/cov_matrix/chr2/chr2.39967768.40067768.gz"),
    )
    parser.add_argument(
        "--ref-vector",
        type=Path,
        default=Path("ref/vector/vector-EUR-chr2-39967768-40067768.txt.gz"),
    )
    parser.add_argument(
        "--ref-bpoints",
        type=Path,
        default=Path("ref/minima/minima-EUR-chr2-50-39967768-40067768.pickle"),
    )
    parser.add_argument(
        "--ref-bed",
        type=Path,
        default=Path("ref/bed/EUR-chr2-50-39967768-40067768.bed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/function_benchmark"),
    )
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")

    reference_panel = args.reference_panel or Path(
        f"work/vcf/1000G.phase1.EUR.{args.vcf_chrom}.{args.start}-{args.end}.vcf.gz"
    )
    cfg = Config(
        chrom=args.chrom,
        vcf_chrom=args.vcf_chrom,
        start=args.start,
        end=args.end,
        ne=args.ne,
        cutoff=args.cutoff,
        n_snps_bw_bpoints=args.n_snps_bw_bpoints,
        subset=args.subset,
        reference_panel=reference_panel,
        genetic_map=args.genetic_map,
        individuals=args.individuals,
        partitions=args.partitions,
        ref_covariance=args.ref_covariance,
        ref_vector=args.ref_vector,
        ref_bpoints=args.ref_bpoints,
        ref_bed=args.ref_bed,
        ld_kernel=args.ld_kernel,
    )
    require_inputs(cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing_rows: list[dict[str, str]] = []
    exactness_rows: list[dict[str, str]] = []
    measured_last: Path | None = None

    total_runs = args.warmups + args.repeats
    for run_index in range(total_runs):
        measured = run_index >= args.warmups
        repeat = run_index - args.warmups if measured else -1
        work_root = args.output_dir / f"work-repeat-{repeat}" if measured else None
        if work_root is None or not args.keep_workdir:
            tmp = tempfile.TemporaryDirectory(prefix="ldetect-example-functions-")
            root = Path(tmp.name)
        else:
            tmp = None
            root = work_root
            if root.exists():
                shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        prepare_store(root, cfg)

        try:
            outputs = run_pipeline(root, cfg, repeat, measured, timing_rows)
            if measured:
                exactness_rows.extend(compare_outputs(outputs, cfg, repeat, args.atol))
                measured_last = root
        finally:
            if tmp is not None:
                tmp.cleanup()

    write_tsv(args.output_dir / "timings.tsv", timing_rows)
    write_tsv(args.output_dir / "exactness.tsv", exactness_rows)
    write_summary(args.output_dir / "summary.md", timing_rows, exactness_rows)
    write_timing_plot(args.output_dir / f"timings.{args.plot_format}", timing_rows)
    if measured_last is not None and not args.keep_workdir:
        # The path was temporary and has already been cleaned; keep this branch
        # explicit so summary.md does not imply durable outputs.
        pass


def require_inputs(cfg: Config) -> None:
    missing = [
        path
        for path in (
            cfg.reference_panel,
            Path(str(cfg.reference_panel) + ".tbi"),
            cfg.genetic_map,
            cfg.individuals,
            cfg.partitions,
            cfg.ref_covariance,
            cfg.ref_vector,
            cfg.ref_bpoints,
            cfg.ref_bed,
        )
        if not path.exists()
    ]
    if missing:
        lines = "\n".join(f"  {path}" for path in missing)
        raise SystemExit(
            "Missing input(s). Run `uv run snakemake --cores 1` in "
            f"examples/ldetect_example first:\n{lines}"
        )


def prepare_store(root: Path, cfg: Config) -> None:
    (root / cfg.chrom).mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg.partitions, root / f"{cfg.chrom}_partitions.txt")


@dataclass(frozen=True)
class Outputs:
    root: Path
    covariance: Path
    vector: Path
    bpoints: Path
    bed: Path


def run_pipeline(
    root: Path,
    cfg: Config,
    repeat: int,
    measured: bool,
    timing_rows: list[dict[str, str]],
) -> Outputs:
    covariance = root / cfg.chrom / f"{cfg.chrom}.{cfg.start}.{cfg.end}.h5"
    vector = root / f"vector-{cfg.chrom}.txt.gz"
    bpoints = root / f"breakpoints-{cfg.chrom}.json"
    bed = root / f"{cfg.chrom}-ld-blocks.bed"

    time_stage(
        "calc_covariance",
        repeat,
        measured,
        timing_rows,
        lambda: calc_covariance(
            vcf_path=cfg.reference_panel,
            region=f"{cfg.vcf_chrom}:{cfg.start}-{cfg.end}",
            genetic_map_path=cfg.genetic_map,
            individuals_path=cfg.individuals,
            output_path=covariance,
            ne=cfg.ne,
            cutoff=cfg.cutoff,
            compact_output=cfg.ld_kernel == "bitpacked",
            compression="zstd",
            ld_kernel=cfg.ld_kernel,
        ),
    )
    time_stage(
        "matrix_to_vector",
        repeat,
        measured,
        timing_rows,
        lambda: MatrixAnalysis(cfg.chrom, CovarianceStore(root=root)).calc_diag_lean(
            vector,
            matrix_workers=1,
        ),
    )
    time_stage(
        "find_minima",
        repeat,
        measured,
        timing_rows,
        lambda: find_breakpoints(
            input_path=vector,
            chr_name=cfg.chrom,
            store=CovarianceStore(root=root),
            n_snps_bw_bpoints=cfg.n_snps_bw_bpoints,
            output_path=bpoints,
            workers=1,
            metric_workers=1,
        ),
    )
    time_stage(
        "extract_bpoints",
        repeat,
        measured,
        timing_rows,
        lambda: extract_bed(root, cfg, bpoints, bed),
    )
    return Outputs(
        root=root,
        covariance=covariance,
        vector=vector,
        bpoints=bpoints,
        bed=bed,
    )


def time_stage(
    stage: str,
    repeat: int,
    measured: bool,
    rows: list[dict[str, str]],
    fn: Callable[[], None],
) -> None:
    start = time.perf_counter()
    fn()
    seconds = time.perf_counter() - start
    label = f"repeat={repeat}" if measured else "warmup"
    print(f"{stage} {label} seconds={seconds:.4f}", flush=True)
    if measured:
        rows.append(
            {"stage": stage, "repeat": str(repeat), "seconds": f"{seconds:.6f}"}
        )


def extract_bed(root: Path, cfg: Config, bpoints: Path, bed: Path) -> None:
    partitions = read_partitions(cfg.chrom, CovarianceStore(root=root))
    data = json.loads(bpoints.read_text())
    write_bed(
        name=cfg.chrom,
        loci=data[cfg.subset]["loci"],
        snp_first=partitions[0][0],
        snp_last=partitions[-1][1],
        output=bed,
    )


def compare_outputs(
    outputs: Outputs,
    cfg: Config,
    repeat: int,
    atol: float,
) -> list[dict[str, str]]:
    rows = [
        compare_covariance(outputs.covariance, cfg.ref_covariance, cfg, repeat, atol),
        compare_vector(outputs.vector, cfg.ref_vector, repeat, atol),
        compare_bpoints(outputs.bpoints, cfg.ref_bpoints, repeat),
        compare_bed(outputs.bed, cfg.ref_bed, repeat),
    ]
    for row in rows:
        print(
            f"exactness {row['stage']} repeat={repeat} "
            f"status={row['status']} max_abs_diff={row.get('max_abs_diff', '')}",
            flush=True,
        )
    return rows


def compare_covariance(
    ours_path: Path,
    ref_path: Path,
    cfg: Config,
    repeat: int,
    atol: float,
) -> dict[str, str]:
    ours = read_hdf5_covariance(ours_path, cfg.start, cfg.end)
    ref = read_reference_covariance(ref_path)
    keys_exact = np.array_equal(ours["lo"], ref["lo"]) and np.array_equal(
        ours["hi"], ref["hi"]
    )
    max_abs = max(
        max_abs_diff(ours["shrink_ld"], ref["shrink_ld"]),
        max_abs_diff(ours["naive_ld"], ref["naive_ld"])
        if "naive_ld" in ours
        else 0.0,
    )
    metadata_fields = ("i_gpos", "j_gpos", "i_id", "j_id")
    metadata_exact = all(
        field not in ours or np.array_equal(ours[field], ref[field])
        for field in metadata_fields
    )
    equivalent = keys_exact and metadata_exact and max_abs <= atol
    return {
        "stage": "calc_covariance",
        "repeat": str(repeat),
        "status": "pass" if equivalent else "fail",
        "n_ours": str(ours["lo"].size),
        "n_ref": str(ref["lo"].size),
        "keys_exact": str(keys_exact),
        "metadata_exact": str(metadata_exact),
        "max_abs_diff": f"{max_abs:.6e}",
    }


def read_hdf5_covariance(path: Path, start: int, end: int) -> dict[str, np.ndarray]:
    import h5py
    import hdf5plugin  # noqa: F401

    with open_covariance_reader(path, start, end) as reader:
        rows = reader.read_all()
    with h5py.File(path, "r") as h5:
        data = {
            "lo": rows.lo.astype(np.int64, copy=False),
            "hi": rows.hi.astype(np.int64, copy=False),
            "shrink_ld": rows.shrink_ld,
        }
        if "covariance/naive_ld" in h5:
            data["naive_ld"] = np.asarray(
                h5["covariance/naive_ld"][:], dtype=np.float64
            )
        if "metadata/i_gpos" in h5:
            data["i_gpos"] = np.asarray(h5["metadata/i_gpos"][:], dtype=np.float64)
        if "metadata/j_gpos" in h5:
            data["j_gpos"] = np.asarray(h5["metadata/j_gpos"][:], dtype=np.float64)
        if "metadata/i_id" in h5:
            data["i_id"] = decode_strings(h5["metadata/i_id"][:])
        if "metadata/j_id" in h5:
            data["j_id"] = decode_strings(h5["metadata/j_id"][:])
        return canonicalize_covariance(data)


def read_reference_covariance(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list] = {
        "i_id": [],
        "j_id": [],
        "i_pos": [],
        "j_pos": [],
        "i_gpos": [],
        "j_gpos": [],
        "naive_ld": [],
        "shrink_ld": [],
    }
    with gzip.open(path, "rt") as f:
        for row in csv.reader(f, delimiter=" "):
            if not row or row[0].startswith("#"):
                continue
            values["i_id"].append(row[0])
            values["j_id"].append(row[1])
            values["i_pos"].append(int(row[2]))
            values["j_pos"].append(int(row[3]))
            values["i_gpos"].append(float(row[4]))
            values["j_gpos"].append(float(row[5]))
            values["naive_ld"].append(float(row[6]))
            values["shrink_ld"].append(float(row[7]))
    return canonicalize_covariance(
        {
            "lo": np.minimum(values["i_pos"], values["j_pos"]).astype(np.int64),
            "hi": np.maximum(values["i_pos"], values["j_pos"]).astype(np.int64),
            "naive_ld": np.asarray(values["naive_ld"], dtype=np.float64),
            "shrink_ld": np.asarray(values["shrink_ld"], dtype=np.float64),
            "i_gpos": np.asarray(values["i_gpos"], dtype=np.float64),
            "j_gpos": np.asarray(values["j_gpos"], dtype=np.float64),
            "i_id": np.asarray(values["i_id"], dtype=str),
            "j_id": np.asarray(values["j_id"], dtype=str),
        }
    )


def canonicalize_covariance(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = np.lexsort((data["hi"], data["lo"]))
    return {key: value[order] for key, value in data.items()}


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values
        ],
        dtype=str,
    )


def compare_vector(
    ours_path: Path,
    ref_path: Path,
    repeat: int,
    atol: float,
) -> dict[str, str]:
    ours = read_vector(ours_path)
    ref = read_vector(ref_path)
    shared = sorted(set(ours) & set(ref))
    max_abs = max((abs(ours[pos] - ref[pos]) for pos in shared), default=0.0)
    equivalent = set(ours) == set(ref) and max_abs <= atol
    return {
        "stage": "matrix_to_vector",
        "repeat": str(repeat),
        "status": "pass" if equivalent else "fail",
        "n_ours": str(len(ours)),
        "n_ref": str(len(ref)),
        "keys_exact": str(set(ours) == set(ref)),
        "metadata_exact": "",
        "max_abs_diff": f"{max_abs:.6e}",
    }


def read_vector(path: Path) -> dict[int, float]:
    opener = gzip.open if path.suffix == ".gz" else open
    values: dict[int, float] = {}
    with opener(path, "rt") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                values[int(row[0])] = float(row[1])
    return values


def compare_bpoints(ours_path: Path, ref_path: Path, repeat: int) -> dict[str, str]:
    ours = json.loads(ours_path.read_text())
    with open(ref_path, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = pickle.load(f)
    subsets = ("fourier", "fourier_ls", "uniform", "uniform_ls")
    exact = all(
        ours.get(subset, {}).get("loci", []) == ref.get(subset, {}).get("loci", [])
        for subset in subsets
    )
    exact = exact and ours.get("n_bpoints") == ref.get("n_bpoints")
    exact = exact and ours.get("found_width") == ref.get("found_width")
    return {
        "stage": "find_minima",
        "repeat": str(repeat),
        "status": "pass" if exact else "fail",
        "n_ours": str(
            sum(len(ours.get(subset, {}).get("loci", [])) for subset in subsets)
        ),
        "n_ref": str(
            sum(len(ref.get(subset, {}).get("loci", [])) for subset in subsets)
        ),
        "keys_exact": str(exact),
        "metadata_exact": str(exact),
        "max_abs_diff": "0.000000e+00" if exact else "nan",
    }


def compare_bed(ours_path: Path, ref_path: Path, repeat: int) -> dict[str, str]:
    _, ours = read_single_chrom_bed(ours_path)
    _, ref = read_single_chrom_bed(ref_path)
    exact = ours == ref
    return {
        "stage": "extract_bpoints",
        "repeat": str(repeat),
        "status": "pass" if exact else "fail",
        "n_ours": str(len(ours)),
        "n_ref": str(len(ref)),
        "keys_exact": str(exact),
        "metadata_exact": str(exact),
        "max_abs_diff": "0.000000e+00" if exact else "nan",
    }


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    timing_rows: list[dict[str, str]],
    exactness_rows: list[dict[str, str]],
) -> None:
    by_stage: dict[str, list[float]] = {}
    for row in timing_rows:
        by_stage.setdefault(row["stage"], []).append(float(row["seconds"]))
    lines = ["# Function Benchmark Summary", ""]
    lines.append("| stage | repeats | median seconds | min seconds | max seconds |")
    lines.append("|---|---:|---:|---:|---:|")
    for stage in sorted(by_stage):
        values = sorted(by_stage[stage])
        median = values[len(values) // 2]
        lines.append(
            f"| {stage} | {len(values)} | {median:.6f} | "
            f"{values[0]:.6f} | {values[-1]:.6f} |"
        )
    lines.extend(["", "## Exactness", ""])
    lines.append("| stage | status | max_abs_diff |")
    lines.append("|---|---|---:|")
    seen: set[str] = set()
    for row in exactness_rows:
        if row["stage"] in seen:
            continue
        seen.add(row["stage"])
        lines.append(f"| {row['stage']} | {row['status']} | {row['max_abs_diff']} |")
    path.write_text("\n".join(lines) + "\n")


def write_timing_plot(path: Path, rows: list[dict[str, str]]) -> None:
    mpl_config = path.parent / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache = path.parent / ".cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = sorted({row["stage"] for row in rows})
    values = [
        [float(row["seconds"]) for row in rows if row["stage"] == stage]
        for stage in stages
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    ax.boxplot(values, tick_labels=stages, showmeans=True)
    ax.set_ylabel("seconds")
    ax.set_title("Function-level ldetect_example timings")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
