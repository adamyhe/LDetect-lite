"""Benchmark pairwise LD kernel variants.

Usage:
    uv run python benchmarks/bench_ld_kernel.py
    uv run python benchmarks/bench_ld_kernel.py --n-snps 400 1600 --n-haps 800 1600

The compact bitpacked backend is the default covariance backend. This benchmark
keeps a small synthetic harness for comparing it against the older uint8
reference backend across SNP and haplotype counts.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import numpy as np

from ldetect_lite.shrinkage import (
    _genetic_stop_bounds_impl,
    _pack_haplotypes_impl,
    _pairwise_ld_compact_chunk_bitpacked_impl,
    _pairwise_ld_compact_chunk_impl,
    _pairwise_ld_impl,
)

NE = 11418.0
THETA = 0.01
CUTOFF = 1e-7

_DEFAULT_N_SNPS = [100, 400, 800, 1600, 3200]
_DEFAULT_N_HAPS = [400, 800, 1000, 1600, 2500]


def _make_inputs(
    n_snps: int, n_haps: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    hap_mat = rng.integers(0, 2, size=(n_snps, n_haps), dtype=np.uint8)
    gpos_arr = np.cumsum(rng.uniform(0.001, 0.01, size=n_snps))
    hap_sums = np.asarray(hap_mat.sum(axis=1), dtype=np.float64)
    n_ind = float(n_haps // 2)
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, NE, n_ind, CUTOFF)
    return hap_mat, gpos_arr, hap_sums, j_stop_by_i, n_ind


def _time(fn: Callable[[], object], n_reps: int) -> float:
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _run_one(
    n_snps: int,
    n_haps: int,
    repeats: int,
    chunk_rows: int,
) -> dict[str, float | int]:
    hap_mat, gpos_arr, hap_sums, j_stop_by_i, n_ind = _make_inputs(n_snps, n_haps)
    n_pairs_capacity = chunk_rows + n_snps
    full_args = (hap_mat, gpos_arr, hap_sums, j_stop_by_i, NE, n_ind, THETA, CUTOFF)
    compact_args = (
        hap_mat,
        gpos_arr,
        hap_sums,
        j_stop_by_i,
        NE,
        n_ind,
        THETA,
        CUTOFF,
        0,
        chunk_rows,
        n_pairs_capacity,
    )

    # Warm up/compile.
    full_warm = _pairwise_ld_impl(*full_args)
    compact_warm = _pairwise_ld_compact_chunk_impl(*compact_args)
    packed = _pack_haplotypes_impl(hap_mat)
    bitpack_args = (
        packed,
        gpos_arr,
        hap_sums,
        j_stop_by_i,
        n_haps,
        NE,
        n_ind,
        THETA,
        CUTOFF,
        0,
        chunk_rows,
        n_pairs_capacity,
    )
    bitpack_warm = _pairwise_ld_compact_chunk_bitpacked_impl(*bitpack_args)

    full_ms = _time(lambda: _pairwise_ld_impl(*full_args), repeats)
    compact_ms = _time(lambda: _pairwise_ld_compact_chunk_impl(*compact_args), repeats)
    pack_ms = _time(lambda: _pack_haplotypes_impl(hap_mat), repeats)
    bitpack_ms = _time(
        lambda: _pairwise_ld_compact_chunk_bitpacked_impl(*bitpack_args), repeats
    )

    return {
        "n_snps": n_snps,
        "n_haps": n_haps,
        "pairs": int(full_warm[0].shape[0]),
        "compact_rows": int(compact_warm[1].shape[0]),
        "bitpack_rows": int(bitpack_warm[1].shape[0]),
        "full_ms": full_ms,
        "compact_ms": compact_ms,
        "pack_ms": pack_ms,
        "bitpack_ms": bitpack_ms,
        "bitpack_with_pack_ms": bitpack_ms + pack_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-snps",
        type=int,
        nargs="+",
        default=_DEFAULT_N_SNPS,
        metavar="N",
        help="SNP counts to benchmark.",
    )
    parser.add_argument(
        "--n-haps",
        type=int,
        nargs="+",
        default=_DEFAULT_N_HAPS,
        metavar="N",
        help="Haplotype counts to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timed repetitions per variant (default: 5).",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=1_000_000,
        help="Target rows for compact chunk kernels (default: 1,000,000).",
    )
    cli = parser.parse_args()

    col_w = 10
    header = (
        f"{'n_snps':>{col_w}}  {'n_haps':>{col_w}}  {'pairs':>{col_w}}"
        f"  {'full_ms':>{col_w}}  {'compact_ms':>{col_w}}"
        f"  {'pack_ms':>{col_w}}  {'bit_ms':>{col_w}}  {'bit+pack':>{col_w}}"
        f"  {'bit/compact':>{col_w}}  {'all/compact':>{col_w}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for n_snps in cli.n_snps:
        for n_haps in cli.n_haps:
            r = _run_one(n_snps, n_haps, cli.repeats, cli.chunk_rows)
            compact_ms = float(r["compact_ms"])
            bitpack_ms = float(r["bitpack_ms"])
            bitpack_with_pack_ms = float(r["bitpack_with_pack_ms"])
            print(
                f"{n_snps:>{col_w}}  {n_haps:>{col_w}}  {r['pairs']:>{col_w},}"
                f"  {r['full_ms']:>{col_w}.3f}"
                f"  {compact_ms:>{col_w}.3f}"
                f"  {r['pack_ms']:>{col_w}.3f}"
                f"  {bitpack_ms:>{col_w}.3f}"
                f"  {bitpack_with_pack_ms:>{col_w}.3f}"
                f"  {bitpack_ms / compact_ms:>{col_w}.2f}x"
                f"  {bitpack_with_pack_ms / compact_ms:>{col_w}.2f}x"
            )

    print(sep)


if __name__ == "__main__":
    main()
