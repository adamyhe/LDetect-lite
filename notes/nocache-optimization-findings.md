# r2-nocache Optimization Findings

Date: 2026-06-28

## Summary

The latest `examples/r2_zarr_exactness` profiling runs strongly suggest that
`--pair-cache r2-nocache` is unlikely to match `r2-zarr` wall time while
remaining a true no-pair-cache mode. It succeeds at the disk goal, but it pays
for repeated LD recomputation during both metric passes and local search.

`r2-nocache` remains useful as a low-disk benchmark and as an escape hatch for
environments where writing multi-GB pair caches is unacceptable. It should not
be treated as the likely fastest path.

## Latest Profile

Source files:

```text
examples/r2_zarr_exactness/results/runtime/EUR.13.runtime.tsv
examples/r2_zarr_exactness/results/runtime/EUR.21.runtime.tsv
examples/r2_zarr_exactness/results/runtime/EUR.22.runtime.tsv
examples/r2_zarr_exactness/results/logs/r2_nocache/EUR/*.ldetect2.log
examples/r2_zarr_exactness/results/logs/r2_zarr/EUR/*.ldetect2.log
```

End-to-end runtime:

| Chrom | `r2_zarr` | `r2_nocache` | Slowdown |
| --- | ---: | ---: | ---: |
| 13 | 568.48 s | 6726.81 s | 11.8x |
| 21 | 134.37 s | 1219.80 s | 9.1x |
| 22 | 160.00 s | 1826.47 s | 11.4x |

No-cache phase timing from the logs:

| Chrom | First metric | Local search | Final metric |
| --- | ---: | ---: | ---: |
| 13 | ~859 s | 4662.56 s | ~864 s |
| 21 | ~179 s | 773.81 s | ~178 s |
| 22 | ~215 s | 1295.49 s | ~216 s |

The cached `r2_zarr` path did the same metric/local-search phases much faster:

| Chrom | First metric | Local search | Final metric |
| --- | ---: | ---: | ---: |
| 13 | ~23 s | 127.55 s | ~23 s |
| 21 | ~5 s | 25.78 s | ~6 s |
| 22 | ~6 s | 34.64 s | ~7 s |

## Local-Search Row Pressure

No-cache local search repeatedly recomputes large active windows. Group-level
load lines show the scale of row work:

| Chrom | Groups | Total group rows | Max group rows | Total load seconds | Max compact payload estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 13 | 61 | 3.21B | 180.9M | 1986.3 s | ~1.7 GiB at 10 B/pair |
| 21 | 23 | 601.9M | 57.9M | 380.2 s | ~552 MiB at 10 B/pair |
| 22 | 23 | 918.5M | 139.6M | 497.3 s | ~1.3 GiB at 10 B/pair |

The compact payload estimate assumes an ideal in-memory layout like
`hi_delta:uint16` plus `r2:float64`. A simple row layout with
`lo:int64`, `hi:int64`, and `r2:float64` would be about 2.4x larger before
temporary masks and dedup arrays.

## Interpretation

The bottleneck is repeated pairwise LD work, not disk output. `r2_nocache`
writes almost nothing compared with cached modes, but the saved I/O is much
smaller than the cost of recomputing the same normalized `r2` rows for:

- the initial Fourier metric;
- each local-search active window;
- the final Fourier-LS metric.

The logs also contain many duplicate-position warnings. Current exactness
policy sends duplicate-position partitions through the canonical row fallback
instead of the fused no-cache fast path. That likely contributes to both wall
time and RSS, especially on chr13 where max RSS reached about 8.4 GiB.

## Optimization Options

These patches could improve no-cache, but they are unlikely to close the full
gap to `r2_zarr` on their own:

1. Add no-cache profiling counters.
   Split metric/local-search timing into VCF decode, array prep, row generation,
   duplicate fallback, filtering, deduplication, and accumulation. This should
   come before deeper no-cache work.
2. Parallelize no-cache metric.
   `metric_from_r2_nocache()` currently ignores `workers`. Partition-level
   workers should be exact and could reduce the two metric passes substantially.
3. Add a prepared-partition LRU.
   Cache decoded partition inputs (`hap_mat`, positions, allele sums,
   `diag_shrink`, `j_stop_by_i`) across local-search groups. This avoids
   repeated VCF/BCF decoding without storing pair rows.
4. Improve duplicate-position fast paths.
   Duplicate-position partitions are common in the profiling logs. A
   duplicate-aware fast path could avoid whole-partition canonical row
   materialization when physical duplicate groups are small and exact
   first-pair precedence can be preserved.
5. Add an opt-in bounded row/window cache.
   A compact in-memory cache can target repeated local-search row
   recomputation directly. This is no longer pure no-cache in spirit, but it can
   avoid disk writes while capping memory, e.g. 512 MiB to 2 GiB.
6. Fuse no-cache local-search accumulation.
   A Numba kernel could compute eligible pair `r2` and update dense
   vertical/horizontal accumulators directly. This should wait until duplicate
   fallback behavior is better understood, because duplicates currently limit
   the fast-path coverage.

## Recommendation

Do not prioritize no-cache as the primary fast path. Keep it experimental and
low-disk. If we invest further, use this order:

1. instrumentation;
2. no-cache metric workers;
3. prepared-partition LRU;
4. duplicate-aware fast path;
5. optional bounded row/window cache;
6. fused local-search kernel.

The main performance track should move back to `r2-zarr`, where cached
normalized pairs already deliver the best runtime in the downloaded profiles.
