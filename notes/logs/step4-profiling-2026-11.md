# Step 4 (find-minima) profiling log — 2026-11-04

Same data, flags and node as `notes/logs/step2-covariance-profiling-2026-11.md`
(EUR chr21, 1000G Phase 1, 379 individuals, GT-only BCF input, 103 partitions,
281M cached covariance rows, n_bpoints=23). Scripts are out-of-tree in
`/home2/ayh8/ldetect-bench/`: `prof_step4.sh`/`prof_step4.py` (cProfile) and
`ab_step4.sh` (A/B). Human summary: `docs/optimizations.md` #21.

## 1. Phase timings from `ldetect run` logs (code at 0daf460)

| Phase | 1 worker | 8 workers (`-c 8`, 4 cores + HT) |
|---|---:|---:|
| Filter-width search | 13 s | 4 s |
| Fourier metric | 16 s | 4 s |
| Local search | 33 s | 8 s |
| Final metric | 16 s | 3 s |
| **Total** | **78 s** | **19 s** |

## 2. cProfile, serial `find-minima` (83.6 s; JSON identical to the run)

- **Metric, 31.0 s over 2 passes.** `searchsorted` 17.5 s, mostly the two
  diagonal-index lookups (~233k entries) on every pair row.
  `_metric_partition_result` self time 7.3 s (mask copies). Reads
  (`iter_owned_rows`) ~5.4 s.
- **Local search, 36.0 s.** `_add_row_chunk_values` 19.5 s (of which `lookup`
  7.0 s, `_add` 10.8 s); `_first_seen_pair_mask` 8.9 s; `iter_rows`/h5py ~9.8 s.
  23 windows of 7-16 partitions load ~600M rows in total, against 281M rows in
  the cache, because windows overlap.
- **Filter-width search, 13.4 s.** 46 calls of `apply_filter_serial` at ~0.29 s
  each (Numba direct O(N·W) convolution).

## 3. Crossing-first metric (b8550bd) A/B (`ab_step4.sh`, `-c 8`, CPUs 35-38,163-166)

Steps 1-3 and the baseline JSON were rebuilt in the job with the old code
(0daf460). Every `find-minima` JSON was compared with `cmp` to that baseline.

| Run | old (0daf460) | new (b8550bd) |
|---|---:|---:|
| 1 worker, warm-up | 80.77 s | 61.08 s |
| 1 worker, rep 1 | 80.67 s | 59.72 s |
| 1 worker, rep 2 | 80.87 s | 59.61 s |
| 8 workers, rep 1 | 20.85 s | 16.66 s |
| 8 workers, rep 2 | 20.80 s | 16.50 s |

All 10 JSONs were identical. The Fourier metric took 6 s serial in the new log
(previously ~16 s).

## Incident: node scratch does not persist between jobs

`/workdir/ayh8/ldetect-chr21-data` (data plus the 1.9 GB Step 4 cache) was
found empty when the A/B job started, although its name is not tied to a job
ID. The Step 2 log's statement that such a directory "persisted" only held
while jobs were back to back. Benchmark scripts must now rebuild their inputs
inside the job; `ab_step4.sh` does.
