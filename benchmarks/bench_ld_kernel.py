"""Benchmark: Numba JIT vs pure Python for _pairwise_ld_impl.

Usage:
    uv run python benchmarks/bench_ld_kernel.py
    uv run python benchmarks/bench_ld_kernel.py --n-snps 100 200 500
        --n-haps 200 400 800
    uv run python benchmarks/bench_ld_kernel.py --plot speedup.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ldetect2.shrinkage import _pairwise_ld_impl

NE = 11418.0
THETA = 0.01
CUTOFF = 1e-7

_DEFAULT_N_SNPS = [100, 200, 400, 800, 1600]
_DEFAULT_N_HAPS = [200, 400, 800, 1600]


def _make_inputs(
    n_snps: int, n_haps: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    hap_mat = rng.integers(0, 2, size=(n_snps, n_haps), dtype=np.uint8)
    gpos_arr = np.cumsum(rng.uniform(0.001, 0.01, size=n_snps))
    n_ind = float(n_haps // 2)
    return hap_mat, gpos_arr, n_ind


def _time(fn, args: tuple, n_reps: int) -> float:
    """Return mean wall time in milliseconds over n_reps calls."""
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn(*args)
    return (time.perf_counter() - t0) / n_reps * 1000


def _run_one(n_snps: int, n_haps: int, reps_jit: int, reps_py: int) -> dict:
    hap_mat, gpos_arr, n_ind = _make_inputs(n_snps, n_haps)
    call_args = (hap_mat, gpos_arr, NE, n_ind, THETA, CUTOFF)

    _pairwise_ld_impl(*call_args)  # warm up / compile
    jit_ms = _time(_pairwise_ld_impl, call_args, reps_jit)

    py_fn = _pairwise_ld_impl.py_func
    py_ms = _time(py_fn, call_args, reps_py)

    return {"n_snps": n_snps, "n_haps": n_haps, "jit_ms": jit_ms, "py_ms": py_ms}


def _plot(results: list[dict], n_snps_vals: list[int], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))

    for n_snps in n_snps_vals:
        rows = [r for r in results if r["n_snps"] == n_snps]
        xs = [r["n_haps"] for r in rows]
        ys = [r["py_ms"] / r["jit_ms"] for r in rows]
        ax.plot(xs, ys, marker="o", label=f"{n_snps} SNPs")

    ax.set_xlabel("n_haps")
    ax.set_ylabel("Speedup (pure Python / JIT)")
    ax.set_title("Numba JIT speedup vs haplotype count")
    ax.legend(title="n_snps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Plot saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-snps",
        type=int,
        nargs="+",
        default=_DEFAULT_N_SNPS,
        metavar="N",
        help="SNP counts to benchmark (default: 100 200 400 800 1600)",
    )
    parser.add_argument(
        "--n-haps",
        type=int,
        nargs="+",
        default=_DEFAULT_N_HAPS,
        metavar="N",
        help="Haplotype counts to benchmark (default: 200 400 800 1600)",
    )
    parser.add_argument(
        "--reps-jit",
        type=int,
        default=20,
        help="Timed repetitions for JIT version (default: 20)",
    )
    parser.add_argument(
        "--reps-py",
        type=int,
        default=3,
        help="Timed repetitions for Python version (default: 3)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save speedup plot to this file (requires matplotlib).",
    )
    cli = parser.parse_args()

    print("Warming up Numba JIT (first call triggers compilation)...")
    hap_mat, gpos_arr, n_ind = _make_inputs(cli.n_snps[0], cli.n_haps[0])
    t0 = time.perf_counter()
    _pairwise_ld_impl(hap_mat, gpos_arr, NE, n_ind, THETA, CUTOFF)
    print(f"  Compile + first call: {(time.perf_counter() - t0) * 1000:.0f} ms\n")

    col_w = 10
    header = (
        f"{'n_snps':>{col_w}}  {'n_haps':>{col_w}}  {'pairs':>{col_w}}"
        f"  {'jit_ms':>{col_w}}  {'py_ms':>{col_w}}  {'speedup':>{col_w}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    results = []
    for n_snps in cli.n_snps:
        for n_haps in cli.n_haps:
            r = _run_one(n_snps, n_haps, cli.reps_jit, cli.reps_py)
            results.append(r)
            pairs = n_snps * (n_snps + 1) // 2
            speedup = r["py_ms"] / r["jit_ms"]
            print(
                f"{n_snps:>{col_w}}  {n_haps:>{col_w}}  {pairs:>{col_w},}"
                f"  {r['jit_ms']:>{col_w}.3f}"
                f"  {r['py_ms']:>{col_w}.1f}"
                f"  {speedup:>{col_w}.0f}x"
            )

    print(sep)

    if cli.plot:
        _plot(results, cli.n_snps, cli.plot)


if __name__ == "__main__":
    main()
