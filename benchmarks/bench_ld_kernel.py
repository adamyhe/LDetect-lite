"""Benchmark: Numba JIT pair loop vs pure Python vs NumPy vectorized variants.

Usage:
    uv run python benchmarks/bench_ld_kernel.py
    uv run python benchmarks/bench_ld_kernel.py \\
        --n-snps 100 200 500 --n-haps 200 400 800
    uv run python benchmarks/bench_ld_kernel.py --plot speedup.png

For a larger, slower grid point to see how chunked_matmul scales (not part of
the default grid, since it would make the pure-Python fallback impractically
slow at every invocation):
    uv run python benchmarks/bench_ld_kernel.py \\
        --n-snps 3200 --n-haps 800 1600 --reps-py 1
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ldetect2._util.ld_kernel_variants import (
    LdKernelResult,
    pairwise_ld_bitpacked_popcount,
    pairwise_ld_chunked_matmul,
    pairwise_ld_row_vectorized,
)
from ldetect2.shrinkage import _genetic_stop_bounds_impl, _pairwise_ld_impl

NE = 11418.0
THETA = 0.01
CUTOFF = 1e-7

_DEFAULT_N_SNPS = [100, 200, 400, 800, 1600]
_DEFAULT_N_HAPS = [200, 400, 800, 1600]

KernelFn = Callable[..., LdKernelResult]

_PY_FUNC = getattr(_pairwise_ld_impl, "py_func", _pairwise_ld_impl)

# bitpacked_popcount's timed region includes pack_haplotypes on every call
# (no cross-call caching in this prototype) -- this faithfully mirrors what
# production would pay today; caching the packed representation across calls
# is a Phase 2 recompute-cache question, out of scope here.
_VARIANTS: dict[str, KernelFn] = {
    "numba_jit": _pairwise_ld_impl,
    "pure_python": _PY_FUNC,
    "row_vectorized": pairwise_ld_row_vectorized,
    "chunked_matmul": pairwise_ld_chunked_matmul,
    "bitpacked_popcount": pairwise_ld_bitpacked_popcount,
}


def _make_inputs(
    n_snps: int, n_haps: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    hap_mat = rng.integers(0, 2, size=(n_snps, n_haps), dtype=np.uint8)
    gpos_arr = np.cumsum(rng.uniform(0.001, 0.01, size=n_snps))
    n_ind = float(n_haps // 2)
    hap_sums = np.asarray(hap_mat.sum(axis=1), dtype=np.float64)
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, NE, n_ind, CUTOFF)
    return hap_mat, gpos_arr, hap_sums, j_stop_by_i, n_ind


def _time(fn: Callable[[], object], n_reps: int) -> float:
    """Return mean wall time in milliseconds over n_reps calls."""
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn()
    return (time.perf_counter() - t0) / n_reps * 1000


def _run_one(
    n_snps: int,
    n_haps: int,
    reps_by_variant: dict[str, int],
    tile_size: int,
) -> dict[str, float | int]:
    hap_mat, gpos_arr, hap_sums, j_stop_by_i, n_ind = _make_inputs(n_snps, n_haps)
    call_args = (hap_mat, gpos_arr, hap_sums, j_stop_by_i, NE, n_ind, THETA, CUTOFF)

    result: dict[str, float | int] = {"n_snps": n_snps, "n_haps": n_haps}
    for name, fn in _VARIANTS.items():
        if name == "chunked_matmul":
            warm = fn(*call_args, tile_size)
            ms = _time(lambda: fn(*call_args, tile_size), reps_by_variant[name])
        else:
            warm = fn(*call_args)
            ms = _time(lambda: fn(*call_args), reps_by_variant[name])
        if name == "numba_jit":
            result["pairs"] = int(warm[0].shape[0])
        result[f"{name}_ms"] = ms
    return result


def _plot(
    results: list[dict[str, float | int]], n_snps_vals: list[int], output: Path
) -> None:
    import matplotlib.pyplot as plt

    variants = ["pure_python", "row_vectorized", "chunked_matmul", "bitpacked_popcount"]
    fig, axes = plt.subplots(
        1, len(variants), figsize=(7 * len(variants), 4), sharey=True
    )

    for ax, variant in zip(axes, variants):
        for n_snps in n_snps_vals:
            rows = [r for r in results if r["n_snps"] == n_snps]
            xs = [r["n_haps"] for r in rows]
            ys = [r[f"{variant}_ms"] / r["numba_jit_ms"] for r in rows]
            ax.plot(xs, ys, marker="o", label=f"{n_snps} SNPs")
        ax.set_xlabel("n_haps")
        ax.set_title(f"{variant} vs numba_jit")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Slowdown factor (variant / numba_jit)")
    axes[-1].legend(title="n_snps")
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
        help="Timed repetitions for the Numba JIT kernel (default: 20)",
    )
    parser.add_argument(
        "--reps-py",
        type=int,
        default=3,
        help="Timed repetitions for the pure Python fallback (default: 3)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Timed repetitions for the NumPy vectorized variants (default: 10)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size for the chunked_matmul variant (default: 1024)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save speedup plot to this file (requires matplotlib).",
    )
    cli = parser.parse_args()

    reps_by_variant = {
        "numba_jit": cli.reps_jit,
        "pure_python": cli.reps_py,
        "row_vectorized": cli.reps,
        "chunked_matmul": cli.reps,
        "bitpacked_popcount": cli.reps,
    }

    print("Warming up Numba JIT (first call triggers compilation)...")
    hap_mat, gpos_arr, hap_sums, j_stop_by_i, n_ind = _make_inputs(
        cli.n_snps[0], cli.n_haps[0]
    )
    t0 = time.perf_counter()
    _pairwise_ld_impl(
        hap_mat, gpos_arr, hap_sums, j_stop_by_i, NE, n_ind, THETA, CUTOFF
    )
    print(f"  Compile + first call: {(time.perf_counter() - t0) * 1000:.0f} ms\n")

    col_w = 8
    header = (
        f"{'n_snps':>{col_w}}  {'n_haps':>{col_w}}  {'pairs':>{col_w}}"
        f"  {'jit_ms':>{col_w}}  {'py_ms':>{col_w}}"
        f"  {'rowvec_ms':>{col_w}}  {'matmul_ms':>{col_w}}  {'bitpack_ms':>{col_w}}"
        f"  {'py_x':>{col_w}}  {'rowvec_x':>{col_w}}  {'matmul_x':>{col_w}}"
        f"  {'bitpack_x':>{col_w}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    results = []
    for n_snps in cli.n_snps:
        for n_haps in cli.n_haps:
            r = _run_one(n_snps, n_haps, reps_by_variant, cli.tile_size)
            results.append(r)
            jit_ms = r["numba_jit_ms"]
            print(
                f"{n_snps:>{col_w}}  {n_haps:>{col_w}}  {r['pairs']:>{col_w},}"
                f"  {r['numba_jit_ms']:>{col_w}.3f}"
                f"  {r['pure_python_ms']:>{col_w}.1f}"
                f"  {r['row_vectorized_ms']:>{col_w}.3f}"
                f"  {r['chunked_matmul_ms']:>{col_w}.3f}"
                f"  {r['bitpacked_popcount_ms']:>{col_w}.3f}"
                f"  {r['pure_python_ms'] / jit_ms:>{col_w}.0f}x"
                f"  {r['row_vectorized_ms'] / jit_ms:>{col_w}.1f}x"
                f"  {r['chunked_matmul_ms'] / jit_ms:>{col_w}.1f}x"
                f"  {r['bitpacked_popcount_ms'] / jit_ms:>{col_w}.1f}x"
            )

    print(sep)

    if cli.plot:
        _plot(results, cli.n_snps, cli.plot)


if __name__ == "__main__":
    main()
