"""Benchmark legacy LDetect stage commands against ldetect-lite CLIs.

The legacy scripts are not called through their disabled ``commanderline``
entry points. Instead, this benchmark times the repository's compatibility
wrapper, ``examples/ldetect_original/scripts/run_legacy_ldetect.py``, once per
stage. The ldetect-lite side is timed through the installed ``ldetect`` CLI.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STAGES = ("matrix_to_vector", "find_minima", "extract_bpoints")
LEGACY_STAGE_COMMANDS = {
    "matrix_to_vector": "matrix-to-vector",
    "find_minima": "find-minima",
    "extract_bpoints": "extract-bpoints",
}
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = EXAMPLES_ROOT / "ldetect_original" / "scripts" / "legacy_ldetect"
LEGACY_WRAPPER = (
    EXAMPLES_ROOT / "ldetect_original" / "scripts" / "run_legacy_ldetect.py"
)


@dataclass(frozen=True)
class Config:
    chrom: str
    start: int
    end: int
    n_snps_bw_bpoints: int
    subset: str
    legacy_dataset: Path
    lite_h5: Path
    partitions: Path
    ldetect_bin: Path
    python_bin: Path
    plot_format: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrom", default="chr2")
    parser.add_argument("--start", type=int, default=39_967_768)
    parser.add_argument("--end", type=int, default=40_067_768)
    parser.add_argument("--n-snps-bw-bpoints", type=int, default=50)
    parser.add_argument("--subset", default="fourier_ls")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--legacy-dataset", type=Path, default=Path("ref/cov_matrix"))
    parser.add_argument(
        "--lite-h5",
        type=Path,
        default=Path("work/chr2/chr2.39967768.40067768.h5"),
    )
    parser.add_argument(
        "--partitions",
        type=Path,
        default=Path("work/chr2_partitions.txt"),
    )
    parser.add_argument(
        "--ldetect-bin",
        type=Path,
        default=default_ldetect_bin(),
        help="ldetect-lite CLI executable used for timed ldetect-lite calls.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=default_python_bin(),
        help="Python executable used for timed legacy wrapper calls.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/legacy_pipeline_benchmark"),
    )
    parser.add_argument("--plot-format", choices=("svg", "png", "pdf"), default="svg")
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")

    cfg = Config(
        chrom=args.chrom,
        start=args.start,
        end=args.end,
        n_snps_bw_bpoints=args.n_snps_bw_bpoints,
        subset=args.subset,
        legacy_dataset=args.legacy_dataset,
        lite_h5=args.lite_h5,
        partitions=args.partitions,
        ldetect_bin=args.ldetect_bin,
        python_bin=args.python_bin,
        plot_format=args.plot_format,
    )
    require_inputs(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for run_index in range(args.warmups + args.repeats):
        measured = run_index >= args.warmups
        repeat = run_index - args.warmups
        with tempfile.TemporaryDirectory(prefix="legacy-pipeline-benchmark-") as tmp:
            root = Path(tmp)
            legacy_root = root / "legacy"
            lite_root = root / "lite"
            legacy_root.mkdir()
            lite_root.mkdir()
            prepare_lite_store(lite_root, cfg)
            run_legacy(cfg, legacy_root, repeat, measured, rows)
            run_lite(cfg, lite_root, repeat, measured, rows)

    write_tsv(args.output_dir / "timings.tsv", rows)
    write_summary(args.output_dir / "summary.tsv", rows)
    write_timing_plots(args.output_dir, cfg.plot_format, rows)
    print_summary(rows)
    print(f"output_dir: {args.output_dir.resolve()}")


def default_ldetect_bin() -> Path:
    venv_bin = REPO_ROOT / ".venv" / "bin" / "ldetect"
    if venv_bin.exists():
        return venv_bin
    found = shutil.which("ldetect")
    if found is not None:
        return Path(found)
    return Path("ldetect")


def default_python_bin() -> Path:
    venv_bin = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_bin.exists():
        return venv_bin
    return Path(sys.executable)


def require_inputs(cfg: Config) -> None:
    missing = [
        path
        for path in (
            LEGACY_ROOT,
            LEGACY_WRAPPER,
            cfg.legacy_dataset / cfg.chrom / f"{cfg.chrom}.{cfg.start}.{cfg.end}.gz",
            cfg.legacy_dataset / "scripts" / f"{cfg.chrom}_partitions",
            cfg.lite_h5,
            cfg.partitions,
            cfg.ldetect_bin,
            cfg.python_bin,
        )
        if not path.exists()
    ]
    if missing:
        lines = "\n".join(f"  {path}" for path in missing)
        raise SystemExit(
            "Missing input(s). Run `uv run snakemake --cores 1` in "
            f"examples/ldetect_example first:\n{lines}"
        )


def prepare_lite_store(root: Path, cfg: Config) -> None:
    chrom_dir = root / cfg.chrom
    chrom_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg.lite_h5, chrom_dir / f"{cfg.chrom}.{cfg.start}.{cfg.end}.h5")
    shutil.copy2(cfg.partitions, root / f"{cfg.chrom}_partitions.txt")


def run_legacy(
    cfg: Config,
    root: Path,
    repeat: int,
    measured: bool,
    rows: list[dict[str, str]],
) -> None:
    for stage in STAGES:
        time_stage(
            "legacy",
            stage,
            repeat,
            measured,
            rows,
            lambda stage=stage: run_legacy_stage(cfg, root, stage),
        )


def run_legacy_stage(cfg: Config, root: Path, stage: str) -> None:
    subprocess.run(
        [
            str(cfg.python_bin),
            str(LEGACY_WRAPPER),
            "--dataset-path",
            str(cfg.legacy_dataset),
            "--chromosome",
            cfg.chrom,
            "--population",
            "EUR",
            "--output-dir",
            str(root),
            "--n-snps-bw-bpoints",
            str(cfg.n_snps_bw_bpoints),
            "--subset",
            cfg.subset,
            "--stage",
            LEGACY_STAGE_COMMANDS[stage],
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_lite(
    cfg: Config,
    root: Path,
    repeat: int,
    measured: bool,
    rows: list[dict[str, str]],
) -> None:
    vector = root / f"vector-{cfg.chrom}.txt.gz"
    breakpoints = root / f"breakpoints-{cfg.chrom}.json"
    bed = root / f"{cfg.chrom}-ld-blocks.bed"
    commands = {
        "matrix_to_vector": [
            "matrix-to-vector",
            "--dataset-path",
            str(root),
            "--name",
            cfg.chrom,
            "--output",
            str(vector),
        ],
        "find_minima": [
            "find-minima",
            "--input",
            str(vector),
            "--chr-name",
            cfg.chrom,
            "--dataset-path",
            str(root),
            "--n-snps-bw-bpoints",
            str(cfg.n_snps_bw_bpoints),
            "--output",
            str(breakpoints),
        ],
        "extract_bpoints": [
            "extract-bpoints",
            "--name",
            cfg.chrom,
            "--dataset-path",
            str(root),
            "--breakpoints",
            str(breakpoints),
            "--subset",
            cfg.subset,
            "--output",
            str(bed),
        ],
    }
    for stage in STAGES:
        time_stage(
            "lite",
            stage,
            repeat,
            measured,
            rows,
            lambda stage=stage: run_lite_stage(cfg, commands[stage]),
        )


def run_lite_stage(cfg: Config, args: list[str]) -> None:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
    )
    subprocess.run(
        [str(cfg.ldetect_bin), *args],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def time_stage(
    backend: str,
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
    print(f"{backend} {stage} {label} seconds={seconds:.6f}", flush=True)
    if measured:
        rows.append(
            {
                "backend": backend,
                "stage": stage,
                "repeat": str(repeat),
                "seconds": f"{seconds:.6f}",
            }
        )


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ("backend", "stage", "repeat", "seconds")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ("stage", "legacy_mean_seconds", "lite_mean_seconds", "speedup")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for stage in STAGES:
            legacy = stage_values(rows, "legacy", stage)
            lite = stage_values(rows, "lite", stage)
            writer.writerow(
                {
                    "stage": stage,
                    "legacy_mean_seconds": f"{np.mean(legacy):.6f}",
                    "lite_mean_seconds": f"{np.mean(lite):.6f}",
                    "speedup": f"{np.mean(legacy) / np.mean(lite):.6f}",
                }
            )


def write_timing_plots(
    output_dir: Path,
    plot_format: str,
    rows: list[dict[str, str]],
) -> None:
    for stage in STAGES:
        write_stage_timing_plot(
            output_dir / f"timings-{stage}.{plot_format}", stage, rows
        )


def write_stage_timing_plot(path: Path, stage: str, rows: list[dict[str, str]]) -> None:
    plt = setup_matplotlib(path.parent)
    legacy = stage_values(rows, "legacy", stage)
    lite = stage_values(rows, "lite", stage)
    means = [float(np.mean(legacy)), float(np.mean(lite))]

    fig, ax = plt.subplots(figsize=(3.2, 2.7), constrained_layout=True)
    ax.bar(
        ["Original LDetect", "LDetect-lite"],
        means,
        color=["#0057b8", "#d62728"],
        edgecolor="#222222",
        linewidth=0.6,
    )
    ax.set_ylabel("mean seconds")
    ax.set_title(stage)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def setup_matplotlib(output_dir: Path):
    mpl_config = output_dir / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache = output_dir / ".cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def stage_values(rows: list[dict[str, str]], backend: str, stage: str) -> list[float]:
    return [
        float(row["seconds"])
        for row in rows
        if row["backend"] == backend and row["stage"] == stage
    ]


def print_summary(rows: list[dict[str, str]]) -> None:
    print("Summary")
    for stage in STAGES:
        legacy = stage_values(rows, "legacy", stage)
        lite = stage_values(rows, "lite", stage)
        print(
            f"  {stage}: legacy_mean={np.mean(legacy):.6f}s "
            f"lite_mean={np.mean(lite):.6f}s "
            f"speedup={np.mean(legacy) / np.mean(lite):.3f}x"
        )


if __name__ == "__main__":
    main()
