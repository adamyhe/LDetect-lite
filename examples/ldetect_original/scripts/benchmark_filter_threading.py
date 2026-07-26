"""Synthetic benchmark for the Step 4 filter's Numba `prange` threading.

Isolates `apply_filter`'s convolution kernel from the rest of the pipeline
(no reference panel, genetic map, or real covariance data needed) to measure
whether `--filter-workers > 1` actually speeds up a single convolution call,
independent of everything else the full pipeline does around it.

Reports three numbers per width: serial time, "warm" parallel time (thread
count set once, left running across calls), and "reset-per-call" parallel
time (thread count reset to 1 after every call, matching production
`_numba_thread_limit` behavior in `filters.py`). Comparing the latter two
isolates whether per-call thread-pool reset overhead -- not the underlying
parallel kernel -- is capping the observed speedup.

Also prints `numba.threading_layer()` after warm-up: Numba's `workqueue`
threading layer is known to scale worse than `tbb`/`omp` for short parallel
regions. If `tbb` is installed (`pip install tbb` / `uv add tbb` --
prebuilt wheels exist for manylinux x86_64 and Windows, not macOS arm64),
Numba prefers it automatically with no code change.

Usage:
    uv run python scripts/benchmark_filter_threading.py
    uv run python scripts/benchmark_filter_threading.py --n 230000 --widths 1000,4500,9000 --calls 13 --threads 4
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ldetect_lite.filters import (
    _convolve1d_reflect,
    _convolve1d_reflect_parallel,
    _numba_thread_limit,
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
        type=int,
        default=4,
        help="Thread count for the parallel kernel (default: 4).",
    )
    args = parser.parse_args()
    widths = [int(w) for w in args.widths.split(",")]

    rng = np.random.default_rng(0)
    arr = rng.normal(size=args.n)

    warmup_kernel = _kernel_for(widths[0])
    _convolve1d_reflect(arr, warmup_kernel)
    _convolve1d_reflect_parallel(arr, warmup_kernel)

    import numba

    print(f"numba {numba.__version__}, threading layer: {numba.threading_layer()}")
    print(f"N={args.n}, calls per width={args.calls}, threads={args.threads}\n")

    for width in widths:
        kernel = _kernel_for(width)

        t_serial = _time_calls(_convolve1d_reflect, arr, kernel, args.calls)

        set_num_threads(args.threads)
        t_warm = _time_calls(_convolve1d_reflect_parallel, arr, kernel, args.calls)
        set_num_threads(1)

        def _reset_per_call(a: np.ndarray, k: np.ndarray) -> None:
            with _numba_thread_limit(args.threads):
                _convolve1d_reflect_parallel(a, k)

        t_reset = _time_calls(_reset_per_call, arr, kernel, args.calls)

        print(f"width={width} klen={2 * width + 1}")
        print(f"  serial:              {t_serial:.3f}s  ({t_serial / args.calls * 1000:.1f} ms/call)")
        print(
            f"  parallel, warm:      {t_warm:.3f}s  ({t_warm / args.calls * 1000:.1f} ms/call)  "
            f"speedup={t_serial / t_warm:.2f}x"
        )
        print(
            f"  parallel, per-call reset: {t_reset:.3f}s  ({t_reset / args.calls * 1000:.1f} ms/call)  "
            f"speedup={t_serial / t_reset:.2f}x"
        )
        print()


if __name__ == "__main__":
    main()
