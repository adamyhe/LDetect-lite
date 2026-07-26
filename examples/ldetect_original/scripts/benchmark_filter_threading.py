"""Synthetic benchmark for the Step 4 filter's threaded convolution.

Isolates `apply_filter`'s convolution kernel from the rest of the pipeline
(no reference panel, genetic map, or real covariance data needed) to measure
whether `--filter-workers > 1` actually speeds up a single convolution call,
independent of everything else the full pipeline does around it.

Sweeps thread count at each width to show the parallel speedup curve.
`_convolve1d_reflect_threaded` splits output rows into contiguous chunks and
runs each chunk concurrently via a Python `ThreadPoolExecutor` calling the
same plain (non-`parallel=True`) nogil kernel `_convolve1d_reflect` uses --
this was a deliberate design choice, not the default Numba `prange`/
`parallel=True` path: on real chr21-scale data, `prange`'s per-thread compute
throughput measured ~3.4x worse than the plain kernel doing identical work
(Numba's parallel accelerator did not get the same LLVM auto-vectorization
of the reduction loop), which fully explained why 4 `prange` threads only
recovered ~1.2x over serial regardless of thread count, threading layer
(`workqueue` vs `tbb`), or core availability. See
`notes/logs/multicore-utilization-filter-width-search.md`.

Usage:
    uv run python scripts/benchmark_filter_threading.py
    uv run python scripts/benchmark_filter_threading.py --n 230000 --widths 9000 --calls 13 --threads 1,2,4,8
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ldetect_lite.filters import _convolve1d_reflect, _convolve1d_reflect_threaded


def _kernel_for(width: int) -> np.ndarray:
    window = np.hanning(2 * width + 1)
    return window / window.sum()


def _time_calls(fn, arr: np.ndarray, kernel: np.ndarray, calls: int, *extra) -> float:
    start = time.perf_counter()
    for _ in range(calls):
        fn(arr, kernel, *extra)
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
    arr = np.ascontiguousarray(rng.normal(size=args.n), dtype=np.float64)

    warmup_kernel = _kernel_for(widths[0])
    _convolve1d_reflect(arr, warmup_kernel)
    _convolve1d_reflect_threaded(arr, warmup_kernel, 2)

    import os

    print(f"os.cpu_count()={os.cpu_count()}")
    print(f"N={args.n}, calls per width={args.calls}\n")

    for width in widths:
        kernel = _kernel_for(width)
        t_serial = _time_calls(_convolve1d_reflect, arr, kernel, args.calls)
        print(f"width={width} klen={2 * width + 1}")
        print(f"  serial:               {t_serial:.3f}s  ({t_serial / args.calls * 1000:.1f} ms/call)")

        for threads in thread_counts:
            t_threaded = _time_calls(
                _convolve1d_reflect_threaded, arr, kernel, args.calls, threads
            )
            print(
                f"  threaded, workers={threads:<2d}  {t_threaded:.3f}s  "
                f"({t_threaded / args.calls * 1000:.1f} ms/call)  speedup={t_serial / t_threaded:.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
