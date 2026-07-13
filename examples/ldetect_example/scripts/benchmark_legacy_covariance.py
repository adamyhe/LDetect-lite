"""Benchmark original LDetect covariance generation against ldetect-lite.

This script reproduces the covariance-only timing comparison for the toy chr2
example. It runs the vendored original ``P00_01_calc_covariance.py`` script as
a subprocess, feeding it the prepared VCF through ``gzip -dc``, and compares it
against the ldetect-lite ``ldetect calc-covariance`` CLI on the same VCF
interval.

Run the Snakemake workflow once first so that ``ref/`` and ``work/vcf/`` exist.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from benchmark_functions import read_hdf5_covariance, read_reference_covariance

TIMING_FIGSIZE = (3.1, 1.55)


@dataclass(frozen=True)
class Config:
    chrom: str
    vcf_chrom: str
    start: int
    end: int
    ne: float
    cutoff: float
    compression: str | None
    reference_panel: Path
    genetic_map: Path
    individuals: Path
    ref_covariance: Path
    original_script: Path
    python: Path
    ldetect_bin: Path
    poll_seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrom", default="chr2")
    parser.add_argument("--vcf-chrom", default="2")
    parser.add_argument("--start", type=int, default=39_967_768)
    parser.add_argument("--end", type=int, default=40_067_768)
    parser.add_argument("--ne", type=float, default=11418.0)
    parser.add_argument("--cutoff", type=float, default=1e-7)
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=("zstd", "lzf"),
        help="HDF5 compression for ldetect-lite output.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--reference-panel", type=Path, default=None)
    parser.add_argument(
        "--genetic-map",
        type=Path,
        default=Path("ref/chr2.interpolated_genetic_map.gz"),
    )
    parser.add_argument("--individuals", type=Path, default=Path("ref/eurinds.txt"))
    parser.add_argument(
        "--ref-covariance",
        type=Path,
        default=Path("ref/cov_matrix/chr2/chr2.39967768.40067768.gz"),
    )
    parser.add_argument(
        "--original-script",
        type=Path,
        default=Path(
            "../ldetect_original/scripts/legacy_ldetect/ldetect/examples/"
            "P00_01_calc_covariance.py"
        ),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--ldetect-bin",
        type=Path,
        default=default_ldetect_bin(),
        help="ldetect-lite CLI executable used for the timed CLI call.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/legacy_covariance_benchmark"),
    )
    parser.add_argument(
        "--plot-format",
        choices=("svg", "png", "pdf"),
        default="svg",
        help="Plot format for timing and profile summaries.",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    reference_panel = args.reference_panel or Path(
        f"work/vcf/1000G.phase1.EUR.{args.vcf_chrom}."
        f"{args.start}-{args.end}.vcf.gz"
    )
    cfg = Config(
        chrom=args.chrom,
        vcf_chrom=args.vcf_chrom,
        start=args.start,
        end=args.end,
        ne=args.ne,
        cutoff=args.cutoff,
        compression=args.compression,
        reference_panel=reference_panel,
        genetic_map=args.genetic_map,
        individuals=args.individuals,
        ref_covariance=args.ref_covariance,
        original_script=args.original_script,
        python=args.python,
        ldetect_bin=args.ldetect_bin,
        poll_seconds=args.poll_seconds,
    )
    require_inputs(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    last_paths: dict[str, Path] = {}
    for backend in ("legacy", "lite"):
        for run_index in range(args.warmups + args.repeats):
            measured = run_index >= args.warmups
            repeat = run_index - args.warmups
            label = f"{backend}-" + (f"repeat-{repeat}" if measured else "warmup")
            suffix = "gz" if backend == "legacy" else "h5"
            output_path = args.output_dir / f"{label}.{suffix}"
            result = (
                run_legacy(cfg, output_path)
                if backend == "legacy"
                else run_lite(cfg, output_path)
            )
            exactness = compare_covariance(
                output_path,
                cfg,
                kind="gz" if backend == "legacy" else "h5",
            )
            print_result(label, result, exactness)
            if measured:
                row = make_row(backend, repeat, result, exactness)
                rows.append(row)
                last_paths[backend] = output_path

    write_tsv(args.output_dir / "timings.tsv", rows)
    summary = summarize(rows, last_paths)
    write_summary(args.output_dir / "summary.tsv", summary)
    write_timing_plot(args.output_dir / f"timings.{args.plot_format}", rows)
    write_covariance_timing_plot(
        args.output_dir / f"timings-calc-covariance.{args.plot_format}", rows
    )
    print("Summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  output_dir: {args.output_dir.resolve()}")


def require_inputs(cfg: Config) -> None:
    missing = [
        path
        for path in (
            cfg.reference_panel,
            cfg.genetic_map,
            cfg.individuals,
            cfg.ref_covariance,
            cfg.original_script,
            cfg.python,
            cfg.ldetect_bin,
        )
        if not path.exists()
    ]
    if missing:
        lines = "\n".join(f"  {path}" for path in missing)
        raise SystemExit(
            "Missing input(s). Run `uv run snakemake --cores 1` in "
            f"examples/ldetect_example first:\n{lines}"
        )


def run_legacy(cfg: Config, output_path: Path) -> dict[str, object]:
    output_path.unlink(missing_ok=True)
    gzip_proc = subprocess.Popen(
        ["gzip", "-dc", str(cfg.reference_panel)],
        stdout=subprocess.PIPE,
    )
    legacy_proc = subprocess.Popen(
        [
            str(cfg.python),
            str(cfg.original_script),
            str(cfg.genetic_map),
            str(cfg.individuals),
            str(cfg.ne),
            str(cfg.cutoff),
            str(output_path),
        ],
        stdin=gzip_proc.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=False,
    )
    if gzip_proc.stdout is None:
        raise RuntimeError("failed to open gzip stdout pipe")
    gzip_proc.stdout.close()

    stop = threading.Event()
    peaks: list[float] = []
    sampler = threading.Thread(
        target=sample_rss,
        args=(
            lambda: [
                proc.pid for proc in (gzip_proc, legacy_proc) if proc.poll() is None
            ],
            stop,
            peaks,
            cfg.poll_seconds,
        ),
        daemon=True,
    )
    start = time.perf_counter()
    sampler.start()
    _, stderr = legacy_proc.communicate()
    gzip_rc = gzip_proc.wait()
    seconds = time.perf_counter() - start
    stop.set()
    sampler.join(timeout=1.0)

    if legacy_proc.returncode != 0 or gzip_rc != 0:
        raise RuntimeError(
            "legacy covariance failed "
            f"legacy_rc={legacy_proc.returncode} gzip_rc={gzip_rc}\n"
            + stderr.decode(errors="replace")
        )
    return {
        "seconds": seconds,
        "peak_rss_mib": max(peaks, default=0.0),
        "profile": {},
    }


def run_lite(cfg: Config, output_path: Path) -> dict[str, object]:
    output_path.unlink(missing_ok=True)
    stop = threading.Event()
    peaks: list[float] = []
    lite_proc: subprocess.Popen | None = None
    sampler = threading.Thread(
        target=sample_rss,
        args=(
            lambda: live_pid(lite_proc),
            stop,
            peaks,
            cfg.poll_seconds,
        ),
        daemon=True,
    )
    start = time.perf_counter()
    lite_proc = subprocess.Popen(
        lite_command(cfg, output_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=False,
    )
    sampler.start()
    _, stderr = lite_proc.communicate()
    seconds = time.perf_counter() - start
    stop.set()
    sampler.join(timeout=1.0)
    if lite_proc.returncode != 0:
        raise RuntimeError(
            "ldetect-lite covariance failed "
            f"returncode={lite_proc.returncode}\n"
            + stderr.decode(errors="replace")
        )
    return {
        "seconds": seconds,
        "peak_rss_mib": max(peaks, default=0.0),
        "profile": {},
    }


def live_pid(proc: subprocess.Popen | None) -> list[int]:
    if proc is not None and proc.poll() is None:
        return [proc.pid]
    return []


def lite_command(cfg: Config, output_path: Path) -> list[str]:
    cmd = [
        str(cfg.ldetect_bin),
        "calc-covariance",
        "--reference-panel",
        str(cfg.reference_panel),
        "--region",
        f"{cfg.vcf_chrom}:{cfg.start}-{cfg.end}",
        "--genetic-map",
        str(cfg.genetic_map),
        "--individuals",
        str(cfg.individuals),
        "--output",
        str(output_path),
        "--ne",
        str(cfg.ne),
        "--cutoff",
        str(cfg.cutoff),
    ]
    if cfg.compression is not None:
        cmd.extend(["--covariance-compression", cfg.compression])
    return cmd


def default_ldetect_bin() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    venv_bin = repo_root / ".venv" / "bin" / "ldetect"
    if venv_bin.exists():
        return venv_bin
    found = shutil.which("ldetect")
    if found is not None:
        return Path(found)
    return Path("ldetect")


def sample_rss(
    pid_getter,
    stop: threading.Event,
    peaks: list[float],
    poll_seconds: float,
) -> None:
    while not stop.is_set():
        peaks.append(sum(rss_mib(pid) for pid in pid_getter()))
        time.sleep(poll_seconds)


def rss_mib(pid: int) -> float:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, PermissionError):
        return 0.0
    text = out.strip()
    return float(text.split()[0]) / 1024.0 if text else 0.0


def compare_covariance(
    path: Path,
    cfg: Config,
    kind: str,
) -> dict[str, object]:
    ref = read_reference_covariance(cfg.ref_covariance)
    obs = (
        read_hdf5_covariance(path, cfg.start, cfg.end)
        if kind == "h5"
        else read_reference_covariance(path)
    )
    rows = int(len(obs["lo"]))
    ref_rows = int(len(ref["lo"]))
    exact_positions = bool(
        rows == ref_rows
        and np.array_equal(obs["lo"], ref["lo"])
        and np.array_equal(obs["hi"], ref["hi"])
    )
    if rows == ref_rows:
        max_abs_diff = float(np.max(np.abs(obs["shrink_ld"] - ref["shrink_ld"])))
    else:
        max_abs_diff = float("inf")
    return {
        "rows": rows,
        "ref_rows": ref_rows,
        "exact_positions": exact_positions,
        "max_abs_diff": max_abs_diff,
    }


def print_result(
    label: str,
    result: dict[str, object],
    exactness: dict[str, object],
) -> None:
    print(
        f"{label}\tseconds={float(result['seconds']):.6f}"
        f"\tpeak_rss_mib={float(result['peak_rss_mib']):.2f}"
        f"\trows={exactness['rows']}"
        f"\tmax_abs_diff={float(exactness['max_abs_diff']):.3g}"
        f"\texact_positions={exactness['exact_positions']}",
        flush=True,
    )


def make_row(
    backend: str,
    repeat: int,
    result: dict[str, object],
    exactness: dict[str, object],
) -> dict[str, str]:
    row = {
        "backend": backend,
        "repeat": str(repeat),
        "seconds": f"{float(result['seconds']):.6f}",
        "peak_rss_mib": f"{float(result['peak_rss_mib']):.6f}",
        "rows": str(exactness["rows"]),
        "ref_rows": str(exactness["ref_rows"]),
        "exact_positions": str(exactness["exact_positions"]),
        "max_abs_diff": f"{float(exactness['max_abs_diff']):.17g}",
    }
    profile = result["profile"]
    if isinstance(profile, dict):
        for key, value in sorted(profile.items()):
            row[f"profile_{key}"] = f"{float(value):.6f}"
    return row


def summarize(
    rows: list[dict[str, str]], last_paths: dict[str, Path]
) -> dict[str, str]:
    lite_backend = optimized_backend(rows)
    legacy_seconds = values(rows, "legacy", "seconds")
    lite_seconds = values(rows, lite_backend, "seconds")
    legacy_rss = values(rows, "legacy", "peak_rss_mib")
    lite_rss = values(rows, lite_backend, "peak_rss_mib")
    summary = {
        "legacy_mean_seconds": f"{np.mean(legacy_seconds):.6f}",
        "legacy_min_seconds": f"{np.min(legacy_seconds):.6f}",
        "lite_mean_seconds": f"{np.mean(lite_seconds):.6f}",
        "lite_min_seconds": f"{np.min(lite_seconds):.6f}",
        "mean_speedup": f"{np.mean(legacy_seconds) / np.mean(lite_seconds):.6f}",
        "min_time_speedup": f"{np.min(legacy_seconds) / np.min(lite_seconds):.6f}",
        "legacy_peak_rss_mib_max": f"{np.max(legacy_rss):.6f}",
        "lite_peak_rss_mib_max": f"{np.max(lite_rss):.6f}",
        "legacy_output_bytes": str(last_paths["legacy"].stat().st_size),
        "lite_output_bytes": str(last_paths["lite"].stat().st_size),
    }
    ratio = last_paths["lite"].stat().st_size / last_paths["legacy"].stat().st_size
    summary["output_size_ratio_lite_vs_legacy"] = f"{ratio:.6f}"
    return summary


def values(rows: list[dict[str, str]], backend: str, key: str) -> list[float]:
    return [float(row[key]) for row in rows if row["backend"] == backend]


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w") as f:
        f.write("\t".join(fieldnames) + "\n")
        for row in rows:
            f.write("\t".join(row.get(key, "") for key in fieldnames) + "\n")


def write_summary(path: Path, summary: dict[str, str]) -> None:
    with path.open("w") as f:
        for key, value in summary.items():
            f.write(f"{key}\t{value}\n")


def write_timing_plot(path: Path, rows: list[dict[str, str]]) -> None:
    write_covariance_timing_plot(path, rows)


def write_covariance_timing_plot(path: Path, rows: list[dict[str, str]]) -> None:
    plt = setup_matplotlib(path.parent)
    backends = ordered_backends(rows)
    labels = [backend_label(backend) for backend in backends]
    means = [float(np.mean(values(rows, backend, "seconds"))) for backend in backends]

    y = [0.0, 0.36]
    fig, ax = plt.subplots(figsize=TIMING_FIGSIZE, constrained_layout=True)
    ax.barh(
        y,
        means,
        height=0.26,
        color=["#0057b8", "#d62728"][: len(labels)],
        edgecolor="#222222",
        linewidth=0.6,
    )
    ax.set_yticks(y, labels)
    ax.set_ylim(0.58, -0.24)
    ax.set_xlabel("mean seconds")
    ax.set_title("calc_covariance")
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def setup_matplotlib(output_dir: Path):
    import os

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


def ordered_backends(rows: list[dict[str, str]]) -> list[str]:
    seen = {row["backend"] for row in rows}
    ordered: list[str] = []
    if "legacy" in seen:
        ordered.append("legacy")
    ordered.extend(sorted(seen - {"legacy"}))
    return ordered


def optimized_backend(rows: list[dict[str, str]]) -> str:
    backends = [backend for backend in ordered_backends(rows) if backend != "legacy"]
    if "lite" in backends:
        return "lite"
    if backends:
        return backends[0]
    return "lite"


def backend_label(backend: str) -> str:
    if backend == "legacy":
        return "LDetect"
    if backend == "lite":
        return "LDetect-lite"
    return backend


if __name__ == "__main__":
    main()
