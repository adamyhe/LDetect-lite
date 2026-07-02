# ldetect2 Optimization Handoff

Date: 2026-06-27

## Purpose

This is the agent-facing working note for runtime/RSS optimization. Keep it
practical: what changed, what is validated, what is only local, what to profile
next, and what not to reintroduce. Human-readable write-up material belongs in
`notes/optimizations.md`.

The production target remains:

```text
ldetect2 run --subset fourier_ls --covariance-cache compact
```

The guiding constraint is unchanged: do not trade runtime for chromosome-scale
resident covariance arrays. New speedups should either keep RSS bounded by a
partition/chunk/window or stay opt-in until remote profiles prove otherwise.

## Current State Snapshot

| Area | State | Notes |
| --- | --- | --- |
| Compact HDF5 covariance | Implemented and remotely validated | Single-pass append writer ran on chr10/chr11/chr13/chr21/chr22 with no fallback and no compact pair-count pass. |
| Step 2 RSS | Remotely validated | chr10/chr11/chr13/chr21/chr22 remain below 1 GiB whole-run RSS after the single-pass writer. |
| Step 3 matrix-to-vector | Bounded and remotely validated | `matrix_workers=4` plus compact-index loci discovery cuts cached chr19-22 Step 3 to 14-28 s with low RSS. |
| Streaming metrics | Worker path implemented and remotely profiled | `metric_workers=4` cuts chr19-22 metric passes to roughly 6-13 s each with bounded RSS. |
| HDF5 read layout/reuse | Remotely validated | 65,536-row storage chunks plus per-precompute reader reuse recovered most local-search HDF5 read regression. |
| Local-search dense accumulator | Implemented and remotely profiled | Net local-search result is mixed but mostly positive; local search is now the largest cached chr19-22 phase. |
| Horizontal aggregation | Implemented and remotely profiled | Dense path keeps dictionary growth out of the hot path, but vertical/horizontal buckets include new dense accumulation detail. |
| Duplicate merge path | Reverted and remotely profiled | `dedup_merge_seconds` is back to zero; do not reintroduce the Python sorted merge. |
| Multiprocessing in Step 3/4 | Step 3 and metric workers remotely profiled | `--matrix-workers=4` gives the useful runtime win without material RSS inflation; `--metric-workers=4` is now useful for metric passes; `local-search-workers=4` inflates RSS and should stay off by default. |

## Working Change Notes

### Bounded Compact Covariance

Previous remote baseline before the single-pass writer:

- chr11 whole-run max RSS: 0.837 GiB.
- Step 2 used `workers=4` and generated 378 compact HDF5 partitions.
- Retained compact rows: about 8.81B total on chr11.
- Largest chr11 compact partition: about 677.9M retained rows.
- Compact pair counting summed to about 1877 s across workers.
- Compact generation/HDF5 writing summed to about 2593 s across workers.

Current implemented path:

- `calc_covariance()` now precomputes per-SNP allele counts and genetic cutoff
  stop bounds for full and compact kernels.
- Compact production path now uses a single-pass appendable HDF5 writer.
- The old count-then-generate compact writer remains as a fallback if an
  invariant fails.
- Compact dataset storage chunks remain `HDF5_DATASET_CHUNK_ROWS = 65_536`;
  write batches remain `COVARIANCE_WRITE_CHUNK_ROWS = 1_000_000`.

Latest remote validation:

- `compact_hdf5_written ... single_pass=true` appeared for every compact
  partition in the downloaded profiles.
- No fallback was used and `compact_pair_counts` did not appear:

| Chrom | Compact partitions | Single-pass writes | Fallbacks | Pair-count profiles |
| --- | ---: | ---: | ---: | ---: |
| chr10 | 376 | 376 | 0 | 0 |
| chr11 | 378 | 378 | 0 | 0 |
| chr13 | 274 | 274 | 0 | 0 |
| chr21 | 103 | 103 | 0 | 0 |
| chr22 | 98 | 98 | 0 | 0 |

The chr11 Step 2 wall split improved from about 1338 s to about 707 s. Whole
run wall improved from 3922.00 s to 3236.81 s, while max RSS moved from
0.837 GiB to 0.811 GiB.

### HDF5 Storage and Reader Layout

Keep this setup unless remote profiles clearly regress:

- compact HDF5 files store `/covariance/lo`, `/covariance/hi`, and
  `/covariance/shrink_ld`;
- `/index/diag_pos`, `/index/diag_val`, `/index/lo_values`, and
  `/index/lo_offsets` are required;
- compact datasets use 65,536-row HDF5 chunks;
- compact write batching is separate from storage chunking;
- local search reuses HDF5 readers within each breakpoint precompute.

Do not add flattened-log layout support. Keep chromosome-prefixed filenames.

### Duplicate Position and Duplicate Pair Semantics

Current production policy:

- duplicate physical VCF positions are collapsed before pairwise LD;
- compact covariance rows are canonical sorted `(lo, hi)`;
- HDF5 local search keeps a duplicate-safe row-stream boundary and preserves
  first-retained-pair precedence across partitions/chunks.

Known divergence from original `ldetect` on duplicate physical positions is
still acknowledged. The current changes do not make it substantially harder to
fix later because duplicate policy is centralized at covariance generation and
the local-search row stream boundary.

Avoid:

- reintroducing Python-level sorted merge loops for local-search duplicate
  tracking;
- scattering duplicate-position compatibility logic into aggregation kernels.

### Local Search

Latest downloaded remote profile includes dense accumulators and the duplicate
merge revert.

| Chrom | Local search | HDF5 read | Dedup | Dense lookup | Dense accumulate | Normalize |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr10 | 327.89 s | 145.42 s | 69.93 s | 34.68 s | 18.28 s | 44.95 s |
| chr11 | 844.41 s | 563.38 s | 134.84 s | 42.22 s | 21.35 s | 52.98 s |
| chr13 | 200.44 s | 83.94 s | 43.20 s | 23.93 s | 12.54 s | 28.22 s |
| chr21 | 48.63 s | 19.16 s | 7.93 s | 6.80 s | 4.09 s | 8.03 s |
| chr22 | 60.06 s | 25.36 s | 10.83 s | 7.44 s | 4.39 s | 8.99 s |

Compared with the previous sweep, the duplicate merge revert did what we
wanted: `dedup_merge_seconds` is zero and chr11 dedup fell from 243.58 s to
134.84 s. Dense accumulation did not create a dramatic standalone speedup;
chr11 dense lookup plus dense accumulation now accounts for about 63.6 s. Net
local-search wall still improved on the large downloaded chromosomes except for
minor small-chromosome noise.

Do not prioritize `_search_array()` or JIT candidate scoring. Search time is
effectively zero compared with precompute.

Latest matrix-worker profile with `matrix_workers=4` and
`local_search_workers=1`:

| Chrom | Local search | HDF5 read | Dedup | Dense lookup+accum | Read calls | Segment-partition reads | Max RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chr19 | 121.37 s | 56.50 s | 23.42 s | 20.49 s | 2172 | 773 | 480.8 MiB |
| chr20 | 90.96 s | 38.14 s | 18.23 s | 16.90 s | 1658 | 806 | 435.0 MiB |
| chr21 | 44.39 s | 17.17 s | 7.02 s | 9.77 s | 765 | 435 | 395.3 MiB |
| chr22 | 58.09 s | 24.35 s | 10.54 s | 11.15 s | 976 | 428 | 416.2 MiB |

This confirms the local-search worker conclusion: keep
`local_search_workers=1`. Additional local-search speed should come from less
repeated HDF5 work or cheaper dense accumulation, not from per-breakpoint
process fan-out.

Rejected local-search micro-optimizations:

- paired vertical/horizontal dense endpoint lookup was remotely profiled on
  chr19-22 and regressed dense lookup plus accumulation;
- adjacent segment coalescing reduced the reported segment count but did not
  reduce HDF5 read calls or segment-partition reads enough to help wall time;
- both changes have been reverted locally. Keep the original separate
  `add_vertical()` / `add_horizontal()` path unless a lower-allocation paired
  approach is designed and tested in isolation.

### Step 3 Matrix-To-Vector

Validated remote baseline:

| Chrom | Step 3 seconds | Step 3 max RSS |
| --- | ---: | ---: |
| chr11 | ~1025 s | ~465 MiB |
| chr21 | ~58 s | ~372 MiB |
| chr22 | ~76 s | ~372 MiB |

The current chunked helper solved the memory problem, and the latest
matrix-worker profile shows the Step 3 wall-time problem is largely addressed
for cached chr19-22. Local changes add:

- compact-index locus discovery instead of a row-streaming loci pass;
- per-partition debug timing for HDF5 open, locus index read, diagonal read,
  row read, normalization, and center accumulation;
- parent-level `matrix_to_vector_array profile` timing for wall, merge, flush,
  and worker wait;
- opt-in `--matrix-workers` for `ldetect2 run` and `ldetect2 matrix-to-vector`.

The worker path is bounded by at most `matrix_workers` in-flight partition
results. Workers compute partition sums; the parent merges and flushes output in
partition order.

Remote validation target:

- `matrix_workers=4` has been tested and gives largely the same runtime as
  setting both `matrix_workers=4` and `local_search_workers=4`, without the
  local-search RSS inflation.
- Prefer `matrix_workers=4`, `local_search_workers=1` for the next full remote
  profile.
- Verify vector/BED/JSON validation unchanged and keep watching Step 3 parent
  `worker_wait_seconds`.

Latest cached-run Step 3 results:

| Chrom | Partitions | Step 3 wall | Worker wait | Merge | Flush | Step 3 max RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr19 | 163 | 28.48 s | 26.45 s | 0.09 s | 1.88 s | 279.1 MiB |
| chr20 | 171 | 24.47 s | 22.08 s | 0.10 s | 2.06 s | 282.0 MiB |
| chr21 | 103 | 14.27 s | 12.67 s | 0.08 s | 1.48 s | 222.2 MiB |
| chr22 | 98 | 17.84 s | 16.37 s | 0.07 s | 1.37 s | 223.3 MiB |

The parent merge/write path is not the bottleneck. Step 3 no longer looks like
the most impactful near-term target unless larger chromosomes show different
behavior.

### Streaming Metrics

Current path avoids resident chromosome-wide metric arrays. The single-process
streaming metric path was still expensive, but `metric_workers=4` now gives a
clear chr19-22 runtime win.

Implemented and remotely profiled changes:

- metric denominator/loci discovery now uses compact HDF5 `lo_values` indexes
  instead of streaming all owned rows in the metadata pass;
- metric debug profile now splits `loci_index_seconds` and
  `diag_read_seconds` from broader `index_read_seconds`.
- `--metric-workers` is available on `ldetect2 run` and `ldetect2 find-minima`;
- streaming metric row passes can run partition-level workers and parent
  reduction keeps deterministic partition order;
- diagnostic Snakemake supports `metric_workers`; the active diagnostic config
  is set to test `metric_workers=4` with `matrix_workers=4` and
  `local_search_workers=1`.

Do not fuse Fourier and Fourier-LS metric passes until a larger pipeline
restructure makes both breakpoint sets available at once.

Previous cached-run metric baseline:

| Chrom | Metric passes | Rows read per pass | First pass | Final pass |
| --- | ---: | ---: | ---: | ---: |
| chr19 | 2 | 398.19M | ~87 s | ~41 s |
| chr20 | 2 | 323.32M | ~88 s | ~34 s |
| chr21 | 2 | 181.88M | ~53 s | ~18 s |
| chr22 | 2 | 205.12M | ~55 s | ~21 s |

Latest `metric_workers=4` profile:

| Chrom | Rows read per pass | First pass | Final pass | Worker wait, first/final |
| --- | ---: | ---: | ---: | ---: |
| chr19 | 398.19M | ~13 s | ~12 s | 12.75 / 11.84 s |
| chr20 | 323.32M | ~11 s | ~10 s | 10.48 / 9.67 s |
| chr21 | 181.88M | ~8 s | ~6 s | 6.97 / 5.65 s |
| chr22 | 205.12M | ~8 s | ~8 s | 8.16 / 7.58 s |

Metric workers are therefore worth keeping. The latest full-run wall time did
not improve dramatically because covariance was regenerated in the same run and
the local-search paired/coalesced micro-optimizations regressed slightly.

The profile fields show metadata/index time is tiny after compact-index
discovery. Remaining metric time is row read, normalization, and crossing
classification; the bounded partition-worker path is implemented and is the
right default diagnostic scaling knob alongside `matrix_workers=4`.

## Latest Downloaded Remote Profile

Files:

```text
examples/ldetect_original/results/diagnostics/EUR/logs/
examples/ldetect_original/results/diagnostics/EUR/profiling/
```

Earlier run summary after dense accumulation, duplicate-merge revert, and
single-pass compact covariance:

| Chrom | Wall time | Max RSS | Local search | HDF5 read | Dedup | Horizontal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr10 | 1505.05 s | 0.598 GiB | 327.89 s | 145.42 s | 69.93 s | 28.19 s |
| chr11 | 3236.81 s | 0.811 GiB | 844.41 s | 563.38 s | 134.84 s | 33.68 s |
| chr13 | 997.88 s | 0.493 GiB | 200.44 s | 83.94 s | 43.20 s | 19.70 s |
| chr21 | 267.31 s | 0.410 GiB | 48.63 s | 19.16 s | 7.93 s | 6.22 s |
| chr22 | 308.48 s | 0.412 GiB | 60.06 s | 25.36 s | 10.83 s | 6.57 s |

chr11 raw-log wall split:

| Phase | Seconds | Notes |
| --- | ---: | --- |
| Step 1 partitioning | 2 | negligible |
| Step 2 covariance | 707 | single-pass compact HDF5, `workers=4` |
| Step 3 matrix-to-vector | 1032 | bounded RSS, now larger than Step 2 |
| Filter/minima before metric | 126 | lower priority |
| Fourier metric | 261 | streaming |
| Fourier local search | 844 | dense accumulator plus duplicate merge revert |
| Final Fourier-LS metric | 262 | streaming |

Latest cached-run profile with `matrix_workers=4`, `local_search_workers=1`
skipped Step 2 because covariance partitions were already complete. It is still
the best profile for Step 3/4 optimization decisions:

| Chrom | Wall time | Max RSS | Step 3 | Fourier metric | Local search | Final metric |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr19 | 279.32 s | 0.507 GiB | 28.48 s | ~87 s | 121.37 s | ~41 s |
| chr20 | 239.38 s | 0.458 GiB | 24.47 s | ~88 s | 90.96 s | ~34 s |
| chr21 | 132.20 s | 0.400 GiB | 14.27 s | ~53 s | 44.39 s | ~18 s |
| chr22 | 154.43 s | 0.424 GiB | 17.84 s | ~55 s | 58.09 s | ~21 s |

Latest full run with `matrix_workers=4`, `metric_workers=4`, and
`local_search_workers=1` regenerated Step 2:

| Chrom | Wall time | Max RSS | Step 2 | Step 3 | Fourier metric | Local search | Final metric |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chr19 | 344.93 s | 0.522 GiB | 114 s | 25 s | 13 s | 132 s | 12 s |
| chr20 | 299.66 s | 0.475 GiB | 102 s | 23 s | 11 s | 97 s | 10 s |
| chr21 | 165.96 s | 0.445 GiB | 54 s | 13 s | 8 s | 49 s | 6 s |
| chr22 | 193.04 s | 0.456 GiB | 62 s | 16 s | 8 s | 62 s | 8 s |

## Remaining Optimization Posture

There probably are not major low-risk runtime wins left. The current bottleneck
shape is dominated by real row work:

- local search repeatedly reads/decompresses large HDF5 row ranges, then pays
  duplicate filtering and dense accumulation;
- fresh end-to-end runs still pay covariance generation, even after removing
  the compact count pass;
- Step 3 and metric passes now have bounded worker paths and are much less
  compelling targets.

Treat future runtime work as rewrite-scale unless a new profile exposes a clear
counter regression. Prefer validation, output parity, and clean instrumentation
over speculative micro-optimizations.

### 1. Output Parity and Profile Hygiene

The newest profiles validate the main runtime/RSS expectations, but keep the
next remote run focused on correctness and clean counters:

- confirm BED/JSON/HDF5 validation unchanged;
- keep whole-run max RSS near the current sub-1 GiB profile;
- keep `single_pass=true`, zero compact fallbacks, and zero compact pair-count
  profiles;
- keep `dedup_merge_seconds` at zero;
- watch new local-search read amplification counters:
  `hdf5_read_calls`, `hdf5_segment_partition_reads`, and `hdf5_segment_loci`.
  Only optimize if these counters move in the intended direction.

### 2. Worker Configuration

Status: validated on chr19-22. Keep `metric_workers=4` in the diagnostic config
unless larger chromosomes show RSS or I/O contention.

Why it is impactful:

- chr19-22 spend about 39-129 s total in the two metric passes, comparable to or
  above Step 3.
- Metadata time is already negligible, so the remaining work is partition-row
  streaming and vectorized classification, which should parallelize similarly
  to Step 3.

Implemented:

- opt-in `--metric-workers`, default `1`;
- each worker streams assigned partitions and returns partial metric sums and
  counters;
- parent reduces partials deterministically in partition order;
- in-flight futures are bounded to worker count, as in Step 3.

Follow-up validation target:

- repeat on chr10/chr11 and any higher-row chromosomes;
- compare metric pass wall time and whole-run RSS;
- if I/O contention appears when Step 2 is also running in fresh end-to-end
  runs, keep `metric_workers` documented as most useful for cached or
  post-covariance reruns.

### 3. Local-Search Read-Amplification Reduction

Goal: reduce local-search HDF5 read/decompression and duplicate/dense overhead
without reintroducing the RSS inflation seen with `local_search_workers=4`.

Why it is likely impactful:

- chr19 local search reads 1.71B rows across 2172 HDF5 read calls to retain
  397M candidate rows;
- chr22 reads 711M rows to retain 205M candidate rows;
- dense lookup+accumulate is also visible, about 10-20 s on chr19-22.

Only pursue if profiling justifies a deeper rewrite:

- evaluate a small per-precompute row-window cache keyed by partition and row
  range, capped by bytes/rows;
- if HDF5 reads remain dominant, consider larger read coalescing only with
  explicit row/read amplification counters.
- do not retry paired endpoint lookup or adjacent segment coalescing in their
  current form; both were remotely profiled and reverted.

### 4. Matrix/Metric Worker Validation On Large Chromosomes

Goal: confirm the current best worker profile across larger high-row
chromosomes.

Run order:

- use `matrix_workers=4`, `metric_workers=4`, `local_search_workers=1`;
- run chr10/chr11 and any remaining high-row chromosomes;
- compare Step 3/metric wall, whole-run RSS, output validation, and parent
  worker wait fields.

### 5. Local-Search Grouped Multiprocessing

Do not restore naive per-breakpoint multiprocessing by default. Remote profiling
showed `local_search_workers=4` causes significant RSS inflation, while
`matrix_workers=4` alone gives largely the same runtime as setting both to 4.
Local-search workers duplicate enough HDF5/precompute state that they are not
the right default scaling knob.

Only revisit if later profiles show enough remaining local-search CPU work:

- submit partition groups, not individual breakpoints;
- each worker opens group metadata/readers once and processes breakpoints
  sequentially;
- keep default and diagnostic baseline at `--local-search-workers=1`.

## Validation Commands

Use focused tests for touched areas, then full non-integration:

```text
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest tests/test_shrinkage.py tests/test_covariance_io.py -q
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest tests/test_local_search.py tests/test_profile_ldetect2.py -q
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest -m "not integration"
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run ruff check src/ldetect2 tests examples/ldetect_original/scripts
git diff --check
```

Remote validation order:

1. chr21/chr22 for iteration;
2. chr10/chr11 for dense/high-row stress;
3. full all-chromosome run only after output parity and RSS stay stable.
