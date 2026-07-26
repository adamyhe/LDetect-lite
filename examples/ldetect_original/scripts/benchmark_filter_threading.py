"""Synthetic benchmark for the Step 4 filter's Numba `prange` threading.

Isolates `apply_filter`'s convolution kernel from the rest of the pipeline
(no reference panel, genetic map, or real covariance data needed) to measure
whether `--filter-workers > 1` actually speeds up a single convolution call,
independent of everything else the full pipeline does around it.

Sweeps thread count at each width to show where the parallel speedup curve
flattens. A curve that saturates well below the machine's core count points
at a shared-resource ceiling (last-level cache/memory bandwidth, SMT/
hyperthreading giving fewer real execution units than logical threads) rather
than a fixable software overhead -- confirmed separately: per-call
`_numba_thread_limit` reset overhead and threading-layer choice
(`workqueue` vs `tbb`) were both ruled out as the cause on real hardware
(`notes/logs/multicore-utilization-filter-width-search.md`).

Usage:
    uv run python scripts/benchmark_filter_threading.py
    uv run python scripts/benchmark_filter_threading.py --n 230000 --widths 9000 --calls 13 --threads 1,2,4,8
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ldetect_lite.filters import (
    _convolve1d_reflect,
    _convolve1d_reflect_parallel,
    set_num_threads,
)


def _kernel_for(width: int) -> np.ndarray:
    window = np.hanning(2 * width + 1)
    return window / window.sum()


def _time_calls(fn, arr: np.ndarray, kernel: np.ndarray, calls: int) -> float:
    start = time.perf_counter()
    for _ in range(calls):
        fn(arr, kernel)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=230_000,
        help="Synthetic vector length (default: 230000, ~ a real chr21 vector).",
    )
    parser.add_argument(
        "--widths",
        default="1000,4500,9000",
        help="Comma-separated filter half-widths to test (default: 1000,4500,9000).",
    )
    parser.add_argument(
        "--calls",
        type=int,
        default=13,
        help="Repeated calls per width, matching real binary-search call counts (default: 13).",
    )
    parser.add_argument(
        "--threads",
        default="1,2,4,8",
        help="Comma-separated thread counts to sweep (default: 1,2,4,8).",
    )
    args = parser.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    thread_counts = [int(t) for t in args.threads.split(",")]

    rng = np.random.default_rng(0)
    arr = rng.normal(size=args.n)

    warmup_kernel = _kernel_for(widths[0])
    _convolve1d_reflect(arr, warmup_kernel)
    _convolve1d_reflect_parallel(arr, warmup_kernel)

    import os

    import numba

    print(f"numba {numba.__version__}, threading layer: {numba.threading_layer()}")
    print(f"os.cpu_count()={os.cpu_count()}")
    print(f"N={args.n}, calls per width={args.calls}\n")

    for width in widths:
        kernel = _kernel_for(width)
        t_serial = _time_calls(_convolve1d_reflect, arr, kernel, args.calls)
        print(f"width={width} klen={2 * width + 1}")
        print(f"  serial (1 thread, non-prange path): {t_serial:.3f}s  ({t_serial / args.calls * 1000:.1f} ms/call)")

        for threads in thread_counts:
            set_num_threads(threads)
            t_parallel = _time_calls(_convolve1d_reflect_parallel, arr, kernel, args.calls)
            print(
                f"  prange threads={threads:<2d}  {t_parallel:.3f}s  "
                f"({t_parallel / args.calls * 1000:.1f} ms/call)  speedup={t_serial / t_parallel:.2f}x"
            )
        set_num_threads(1)
        print()


if __name__ == "__main__":
    main()
