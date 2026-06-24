# Local Search Memory and Runtime Handoff

Date: 2026-06-23

## Context

Recent profiling and `ldetect_original` runs show that local search dominates
runtime on small chromosomes, but larger chromosomes expose a second memory
risk: whole-chromosome metric materialization before and after local search.
The current production target is still normal `ldetect2 run --subset
fourier_ls`, where the first-order fix is to avoid unrequested local-search
subsets. After that, the remaining cost is split between local-search
precompute work and bounded metric/covariance I/O.

This handoff now has two tracks:

1. Completed non-storage optimizations that reduced unrequested work and
   repeated local-search precompute.
2. The current HDF5 storage path, which replaced `.npz` intermediate
   covariance partitions and gives metric, local-search, and matrix-to-vector
   code a shared chunked reader.

The guiding constraint is that peak RSS must not increase for whole-chromosome
runs. Any optimization that trades runtime for larger resident arrays should be
opt-in or guarded by instrumentation.

## Audited Status Summary

This table is based on the current code and tests, not just the historical
plan text.

| Area | Status | Evidence |
| --- | --- | --- |
| Selective subset computation | Implemented | `find_breakpoints(subsets=...)`, `ldetect2 run --all-breakpoint-subsets`, integration subset tests |
| Local-search phase instrumentation | Implemented | `LocalSearchPrecomputeStats`, per-breakpoint/group debug logs, `profile_ldetect2.py` parser tests |
| Canonical local-search partitions | Implemented | `LocalSearchPartition`, `canonical_local_search_rows()`, local-search canonicalization tests |
| Diagonal precompute per partition | Implemented | `diag_pos`/`diag_val` in local-search partitions and HDF5 indexes |
| Sorted range slicing | Implemented | `np.searchsorted()` in `_add_array_segment_values()` and HDF5 segment helpers |
| Vertical grouped reduction | Implemented | `np.add.reduceat()` vertical aggregation in `_add_array_segment_values()` |
| Horizontal aggregation rewrite | To-do | Still uses conservative `np.unique(..., return_inverse=True)` plus `np.bincount()` |
| Sequential breakpoint grouping | Implemented | `_group_local_search_tasks()` and grouped single-worker path in `_run_local_search()` |
| Multiprocessing-aware grouping | To-do | `workers > 1` still uses per-breakpoint process-pool fallback |
| Append/canonicalize reduction | Implemented | partition-slice precompute paths for canonical and HDF5 partitions |
| Streaming HDF5 segment assembly | Reverted | Improved chr11 local-search time but regressed wall time and did not reduce RSS |
| Streaming metric calculation | Implemented | `metric_from_files()` default path in `Metric.calc_metric()` |
| HDF5 chunked covariance reader | Implemented | `HDF5CovariancePartitionReader`, `iter_rows()`, `iter_owned_rows()` |
| HDF5 writer invariant and duplicate-position handling | Implemented | validated writer fast path, duplicate-position collapse in `calc_covariance()` |
| Dense local accumulators | To-do | local search still uses `sum_vert_by_locus`/`sum_horiz_by_locus` dictionaries |
| JIT for local-search numerics | Deferred | Candidate scoring is not hot in current profiles; revisit only after remote profiling |
| Remote real-data validation | To-do | Must be run remotely; do not profile real data from local checkout |

## Non-Storage Optimization Status

These changes were implemented before or alongside the HDF5 migration and
focus on reducing repeated local-search work. The current production
intermediate format is HDF5, not `.npz`.

### Completed

#### Sort and Deduplicate Rows Once Per Partition Load

**Affected code:** `src/ldetect2/local_search.py`,
`src/ldetect2/_util/covariance_array.py`

Implemented `LocalSearchPartition` plus `local_search_partition()` and
`canonical_local_search_rows()`. Each local-search partition now exposes
canonical `lo`, `hi`, `shrink_ld`, `diag_pos`, and `diag_val` arrays.

The canonical row helper:

- converts endpoints to lower/upper form;
- preserves `int32` positions when possible;
- sorts by `(lo, hi)`;
- deduplicates duplicate endpoint pairs with first-input-row-wins behavior;
- keeps compact diagonal arrays for local-search normalization.

Expected benefit carried forward:

- Less repeated `np.unique` work inside local-search windows.
- More predictable memory because deduplication happens once per loaded
  partition rather than repeatedly on growing active arrays.

Memory risk status:

- Sorting still allocates temporary index arrays, but the scope is one loaded
  partition/window rather than a chromosome-wide cache.

#### Precompute Diagonal Lookup Arrays Per Loaded Partition

**Affected code:** `src/ldetect2/local_search.py`,
`src/ldetect2/_util/covariance_array.py`

Implemented compact diagonal arrays in `LocalSearchPartition`:

```text
diag_pos int32/int64
diag_val float64
```

Active local-search diagonals are now derived from canonical active rows rather
than repeatedly sorting and uniquing diagonal positions in
`_add_array_segment_values()`.

Expected benefit:

- Lower CPU in `_add_array_segment_values()`.
- Lower temporary object churn from repeated dictionary-style diagonal lookup.

Memory risk status:

- Low. Diagonal arrays scale with number of loci, not covariance row count.

#### Replace Boolean Full-Row Scans With Sorted Range Slices

**Affected code:** `_add_array_segment_values()` in
`src/ldetect2/local_search.py`

Implemented sorted `lo` range slicing with `np.searchsorted()`. The function
now slices candidate rows for the current segment before applying remaining
eligibility masks, instead of building the first mask across the full active
row set.

Expected benefit:

- Less work per segment when active partitions contain many rows outside the
  local search interval.
- Smaller temporary masks.

Memory risk status:

- Low if slices are views and chunk processing remains bounded.

#### Use Grouped Reductions for Vertical Sums

**Affected code:** `_add_array_segment_values()` in
`src/ldetect2/local_search.py`

Implemented grouped vertical aggregation with `np.add.reduceat()` over chunks
that are already sorted by `lo`. Horizontal aggregation intentionally remains
on `np.unique(..., return_inverse=True)` plus `np.bincount()` because `hi` is
not globally sorted within the candidate `lo` range.

Expected benefit:

- Lower allocation and CPU in one of the hottest local-search loops.

Memory risk status:

- Low. This change does not add dense chromosome-wide arrays.

#### Process Breakpoints by Partition Range

**Affected code:** `_run_local_search()` in `src/ldetect2/pipeline.py`

Implemented sequential grouping for the normal float, single-worker, uncached
path. `_group_local_search_tasks()` groups tasks by required covariance
partition bounds. In the current HDF5 path, each group opens lightweight HDF5
partition metadata with `local_search_hdf5_partition()`, streams segment rows
through `HDF5CovariancePartitionReader`, processes that group's breakpoints,
then releases the group state.

This is intentionally not used for:

- Decimal local search;
- caller-supplied chromosome covariance caches;
- multiprocessing runs.

Expected benefit:

- Fewer repeated HDF5 partition opens/index reads for adjacent breakpoint
  windows that touch the same partition range.
- No additional memory multiplication across process workers.

Memory risk status:

- Low to moderate and bounded. One partition group's HDF5 metadata is retained
  at a time in the sequential path; full partition row arrays are not retained.

#### Keep Selective Subset Computation as the Default Hot Path

**Affected code:** `find_breakpoints()` in `src/ldetect2/pipeline.py`,
`src/ldetect2/_cli/cmd_run.py`

Already implemented before this pass. `ldetect2 run --subset fourier_ls`
computes raw Fourier plus Fourier local search and skips uniform local search
unless requested or `--all-breakpoint-subsets` is passed.

Expected benefit:

- Roughly halves local-search work for the default production subset.

Memory risk status:

- Lower than previous behavior because fewer local-search windows are loaded.

#### Add Precompute Phase Instrumentation

**Affected code:** `src/ldetect2/local_search.py`,
`src/ldetect2/pipeline.py`,
`examples/ldetect_original/scripts/profile_ldetect2.py`

Implemented debug-level phase timing and row-count diagnostics for the normal
array local-search precompute path. Each per-breakpoint debug row can now
include:

- partition load seconds;
- canonicalization seconds;
- active-row append/dedup seconds;
- diagonal extraction seconds;
- range-slice eligibility seconds;
- `r²` normalization seconds;
- vertical and horizontal aggregation seconds;
- HDF5 read, chunk-filter, deduplication, and accumulator seconds;
- candidate, eligible, and normalized row counts;
- rows read, rows after filter, rows after deduplication, and duplicate rows
  skipped;
- chunk count, segment count, peak active rows, and peak chunk rows.

The profiling parser now preserves these fields in
`local_search_breakpoints.tsv`, records partition-group load/canonicalization
timing in `local_search_groups.tsv`, aggregates both breakpoint and group
timing in `local_search_by_chrom.tsv`, and emits a phase-breakdown plot when
matplotlib is available.

Expected benefit:

- Identifies whether the next runtime lever is storage inflation,
  canonicalization, row filtering, normalization, or aggregation.
- Gives row-volume counters needed to validate row-reduction changes.

Memory risk status:

- Low. The change records scalar counters/timers only and does not retain
  additional row arrays.

#### Bounded HDF5 Segment Assembly Experiment

**Affected code:** `src/ldetect2/local_search.py`,
`src/ldetect2/io/covariance_hdf5.py`,
`examples/ldetect_original/scripts/profile_ldetect2.py`

Implemented and remotely profiled, then reverted from the active path. The
experiment split each local-search segment into bounded HDF5 `lo` windows and
aggregated each window separately.

Remote result:

- chr11 local-search time improved from 1730.3 s to 1477.5 s, about 14.6%.
- Whole-run chr11 wall time regressed from 5209 s to 5415 s.
- Max RSS was unchanged: about 102.35 GB before versus 102.39 GB after.
- System time and major page faults increased substantially.

Conclusion:

- Do not keep bounded-window segment assembly as the default.
- The experiment moved cost into repeated HDF5 reads and did not address the
  process RSS high-water mark.
- Keep the extra parser/stat fields for now because they are useful when
  comparing experimental profiles, but focus next on locating the current-RSS
  chokepoint.

#### Cache Canonical Partitions and Slice Active Rows by Segment

**Affected code:** `src/ldetect2/local_search.py`,
`src/ldetect2/_util/covariance_array.py`, `src/ldetect2/pipeline.py`

Implemented the next append/canonicalize reduction pass for the normal float,
single-worker grouped path.

Key changes:

- `LocalSearchPartition` now carries sorted unique `loci` plus
  `source_row_count`.
- Grouped local search canonicalizes each loaded covariance partition once per
  partition group and passes those canonical partitions into each breakpoint
  search.
- Array precompute no longer rebuilds and recanonicalizes one full active row
  array after every partition append. It maintains active canonical partitions,
  builds active loci from partition-level `loci`, and canonicalizes only the
  current segment row slice.
- Active diagonals are built from partition-level diagonal arrays with the same
  partition-order first-wins semantics as the legacy path.

Expected benefit:

- Lower per-breakpoint `canonicalize_seconds` because raw partitions are
  canonicalized once per group instead of once per breakpoint.
- Lower `append_seconds` because the active full-row array is not repeatedly
  concatenated, sorted, and deduplicated.

Memory risk status:

- Low to moderate. The implementation uses bounded segment row temporaries
  instead of chromosome-wide caches or full active-array recanonicalization.

#### Stream Metric Calculation From Partition Files

**Affected code:** `src/ldetect2/_util/covariance_array.py`,
`src/ldetect2/metric.py`, `src/ldetect2/pipeline.py`

Implemented `metric_from_files()` and routed the default float metric path
through it. Normal `find_breakpoints()` runs no longer load a full-chromosome
metric covariance cache before local search and no longer reload that cache
after local search just to score the final subset.

The streaming metric path reads covariance partitions in bounded passes:

- first pass collects loci and diagonal values needed for `N_zero` and
  normalization;
- second pass normalizes pair rows partition by partition;
- only crossing pairs for the requested breakpoint set contribute to the
  accumulated metric sum;
- caller-supplied covariance caches still use the existing `metric_from_arrays`
  path.

Expected benefit:

- Removes a major chromosome-wide allocation from normal `ldetect2 run
  --subset fourier_ls`.
- Avoids peak-memory stacking between metric arrays and local-search windows.
- Directly targets the chr10/chr11 failure mode, where earlier successful logs
  showed whole-run RSS already near 57 GiB for chr10 and 103 GiB for chr11.

Memory risk status:

- Low. The tradeoff is additional partition rereads and some repeated
  per-partition masks, but resident arrays are bounded by partition size plus
  diagonal/locus arrays rather than full-chromosome pair arrays.

Exactness coverage:

- Added tests comparing streaming metrics against the previous materialized
  array metric path, including overlapping partitions and multiple
  breakpoints.
- Existing metric/local-search/pipeline tests pass.

### Completed Exactness Coverage

**Affected code:** `tests/test_local_search.py`,
`tests/test_metric.py`, `tests/integration/test_pipeline.py`

Added/strengthened tests for:

- canonical local-search partition rows, including reversed endpoints,
  duplicates, `int32` preservation, and zero diagonal values;
- duplicate-pair local search versus the Decimal legacy path;
- exact selected breakpoint matching against Decimal local search;
- exact `N_zero` matching against Decimal local search;
- precompute parity for `loci`, `sum_vert`, and `sum_horiz` against the
  Decimal legacy path on multi-partition fixtures, including cross-partition
  duplicate pairs;
- HDF5 streaming precompute parity with tiny forced chunks, duplicate pairs
  across active partitions, and zero diagonal values;
- streaming metric parity against the previous materialized array metric path.

Current local validation run after the streaming HDF5 segment assembly patch:

```text
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest -q
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run ruff check src/ldetect2 tests examples/ldetect_original/scripts
git diff --check
```

Result: all checks passed (`168 passed`).

Current audit note:

- This section was rechecked against the implementation, not only the docs.
- Key code paths inspected: `src/ldetect2/pipeline.py`,
  `src/ldetect2/local_search.py`, `src/ldetect2/_util/covariance_array.py`,
  `src/ldetect2/io/covariance_hdf5.py`, `src/ldetect2/metric.py`, and
  `src/ldetect2/shrinkage.py`.
- Key tests inspected: `tests/test_local_search.py`, `tests/test_metric.py`,
  `tests/test_covariance_io.py`, `tests/test_shrinkage.py`,
  `tests/test_cmd_run.py`, and `tests/integration/test_pipeline.py`.
- Recent git history also matches this sequence: selective subset work,
  instrumentation, append/canonicalize optimization, streaming metrics, HDF5
  migration, stale example updates, duplicate-position handling, and streaming
  HDF5 segment assembly.

## Latest Downloaded Profiling: EUR chr10/chr11/chr21/chr22

The latest downloaded remote profiling outputs are under:

```text
examples/ldetect_original/results/diagnostics/EUR/profiling/
```

The current remote logs include pipeline memory checkpoints. They confirm that
the chr11 RSS high-water mark is not caused by the local-search segment
assembly path. The high-water mark first appears during Step 3
matrix-to-vector conversion.

Current run summary:

| Chrom | Wall time | Max RSS | Local search | LS % wall | Precompute | Search |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr21 | 441.29 s | 1.79 GiB | 80.71 s | 18.3% | 80.37 s | 0.056 s |
| chr22 | 453.00 s | 4.89 GiB | 81.16 s | 17.9% | 80.91 s | 0.030 s |
| chr10 | 1899.67 s | 19.42 GiB | 465.19 s | 24.5% | 464.16 s | 0.117 s |
| chr11 | 5401.00 s | 97.60 GiB | 1769.81 s | 32.8% | 1768.59 s | 0.126 s |

The bounded HDF5 segment assembly experiment was run before this profile. It
improved chr11 local-search time but regressed whole-run wall time and did not
reduce max RSS, so the active code path has been reverted to full segment
assembly.

The key runtime finding is still that local-search candidate scoring is not the
bottleneck. The remaining local-search cost is overwhelmingly precompute,
driven by the number of covariance rows loaded and aggregated per breakpoint.
The key memory finding has changed: the chr11 97.6 GiB lifetime RSS peak occurs
before local search.

Rows loaded and precompute time scale closely:

- chr21 loaded 601.8M rows across 23 breakpoint searches and filtered them to
  327.3M candidate rows and 181.9M eligible rows.
- chr22 loaded 918.3M rows across 23 breakpoint searches and filtered them to
  375.8M candidate rows and 205.1M eligible rows.
- chr10 loaded 6.02B rows across 84 breakpoint searches and filtered them to
  1.94B candidate rows and 999.3M eligible rows.
- chr11 loaded 43.03B rows across 83 breakpoint searches and filtered them to
  3.01B candidate rows and 1.22B eligible rows.

Phase timing still identifies segment row assembly as the dominant local-search
runtime bucket:

| Chrom | Append/assembly | Horizontal | Normalize | Active rows peak |
| --- | ---: | ---: | ---: | ---: |
| chr21 | 51.54 s (64.1%) | 18.29 s (22.8%) | 8.56 s (10.7%) | 27.9M |
| chr22 | 57.40 s (70.9%) | 13.88 s (17.2%) | 7.99 s (9.9%) | 120.2M |
| chr10 | 351.03 s (75.6%) | 66.82 s (14.4%) | 38.46 s (8.3%) | 635.7M |
| chr11 | 1623.00 s (91.8%) | 83.53 s (4.7%) | 49.39 s (2.8%) | 7.15B |

Worst chr11 breakpoint windows in the latest profile:

| Index | Window | Rows | Candidate | Eligible | Partitions | Precompute | Assembly |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 28 | 49.87-52.62 Mb | 6.80B | 326.6M | 82.5M | 15 | 366.94 s | 356.48 s |
| 27 | 47.01-49.87 Mb | 6.26B | 480.1M | 186.2M | 17 | 347.72 s | 324.35 s |
| 29 | 52.62-55.09 Mb | 7.09B | 193.7M | 15.7M | 18 | 265.28 s | 262.95 s |
| 30 | 55.09-56.41 Mb | 7.25B | 130.6M | 81.5M | 25 | 208.68 s | 199.05 s |
| 26 | 42.91-47.01 Mb | 4.90B | 295.0M | 37.5M | 24 | 138.63 s | 133.50 s |

Additional chr11 timing from the raw log:

- covariance calculation ran from `[22:21:46]` to `[22:44:49]`, about
  23m03s;
- matrix-to-vector ran from `[22:44:49]` to `[23:09:15]`, about 24m26s;
- breakpoint finding before local search ran from `[23:09:15]` to
  `[23:16:07]`, about 6m52s;
- local search ran from `[23:16:07]` to `[23:45:37]`, about 29m30s;
- final metric/write ran from `[23:45:37]` to `[23:50:09]`, about 4m32s.

Important memory caveat:

- `max_rss_mib` in breakpoint logs is a process lifetime high-water mark from
  `resource.getrusage`, not current RSS.
- chr11 reports about 99.94 GiB at breakpoint index 0 because the lifetime peak
  already occurred in Step 3.
- Current RSS stayed small around local search: about 513 MiB at
  `fourier_local_search_start`, about 549 MiB after breakpoint 0, and about
  429 MiB at `fourier_local_search_end`.

Step-level chr11 memory checkpoints:

| Checkpoint | Current RSS | Max RSS |
| --- | ---: | ---: |
| `step2_start` | 143.9 MiB | 328.5 MiB |
| `step2_end` | 144.3 MiB | 328.5 MiB |
| `step3_start` | 144.3 MiB | 328.5 MiB |
| `step3_end` | 430.5 MiB | 99,942.0 MiB |
| `fourier_metric_start` | 480.0 MiB | 99,942.0 MiB |
| `fourier_local_search_start` | 513.0 MiB | 99,942.0 MiB |
| `fourier_local_search_end` | 428.6 MiB | 99,942.0 MiB |
| `run_end` | 411.9 MiB | 99,942.0 MiB |

Covariance worker checkpoints did not show the 100 GiB peak. The largest
worker high-water marks in this run were about 22.6 GiB, with current RSS near
661 MiB at partition completion. That makes Step 3
`MatrixAnalysis.calc_diag_array()` / `write_diag_vector_array()` the primary
RSS chokepoint to investigate next.

Implications:

- Do not prioritize `_search_array()` candidate scoring. It is effectively
  free at this scale.
- Do not add JIT yet.
- The highest leverage memory task is now fixing the Step 3 matrix-to-vector
  transient allocation. The lifetime max RSS is already present by the first
  local-search breakpoint, so local-search segment assembly is not the chr11
  RSS chokepoint in this profile.
- Horizontal aggregation and normalization remain possible runtime targets, but
  should wait until the RSS chokepoint is understood.

### Next Profiling Targets

Review the next remote profile in this order:

1. Matrix-to-vector Step 3.
   Add subphase memory checkpoints inside `write_diag_vector_array()`,
   especially around HDF5 partition reads, in-partition filtering,
   `np.unique(np.concatenate((lo, hi)))`, `_r2_rows()`, `np.bincount()`, and
   vector-row flushing. The goal is to identify the exact temporary allocation
   that raises chr11 from a 328 MiB high-water mark to about 99.94 GiB.
2. Chunked matrix-to-vector replacement.
   If `_r2_rows()` or `np.bincount()` is the source, replace the one-partition
   materialized path with bounded HDF5 row-chunk accumulation. Preserve the
   current center-locus and pending-sum semantics exactly, and compare output
   vectors byte-for-byte or with existing float tolerance before enabling.
3. Segment assembly after RSS is localized.
   The reverted bounded-window experiment showed that naively reducing
   segment temporaries can trade runtime for HDF5 read overhead. Revisit only
   if current RSS actually spikes inside local search.
4. Horizontal aggregation.
   This was still material in the latest chr21/chr22 profile: about 13.2
   seconds for chr21 and 13.9 seconds for chr22. The current path uses
   `np.unique(row_hi, return_inverse=True)` plus `np.bincount()` per chunk,
   which is exact but allocation-heavy. Possible follow-ups are grouped
   reduction after sorting `hi` within the chunk, dense local accumulators
   indexed by the local locus window, or processing partition slices directly
   into accumulators.
5. Normalization.
   This was moderate: about 7.5 seconds for chr21 and 8.0 seconds for chr22.
   The current path does two `np.searchsorted()` calls into diagonal arrays per
   chunk, filters positive diagonals, then computes `r²`. Possible follow-ups
   are dense or dictionary-style diagonal lookup scoped to active segment loci,
   carrying per-partition diagonal lookup state, and combining eligibility plus
   diagonal filtering to shrink arrays earlier.
6. Group load and canonicalization outside breakpoint rows.
   Per-breakpoint phase totals still may not explain all local-search elapsed
   time. Use `local_search_groups.tsv`, `group_total_seconds`, and
   `local_search_unaccounted_seconds` to check whether HDF5 open/read overhead
   or group metadata setup has become significant. If it has, reduce redundant
   partition group loads, merge adjacent groups only when RSS allows it, or
   delay work for partitions that contribute only tiny row slices.
7. Metric recomputation around local search.
   This sits outside the per-breakpoint local-search rows but can affect wall
   time. The current plan is to validate streaming metrics first. If metric
   time becomes visible after the memory fix, instrument metric time
   separately before considering any cache reuse.
8. Filter-width search.
   For larger profiles, repeated width evaluations can be visible during
   exponential search, binary search, and trackback. A low-memory optimization
   is to cache `{width: minima_count}` only. Do not cache smoothed arrays.
Watch these ratios in each remote profiling run:

- `append_seconds / precompute_seconds`;
- `hdf5_read_seconds / precompute_seconds`;
- `chunk_filter_seconds / precompute_seconds`;
- `dedup_seconds / precompute_seconds`;
- `accumulator_seconds / precompute_seconds`;
- `horizontal_seconds / precompute_seconds`;
- `normalize_seconds / precompute_seconds`;
- `rows_after_dedup / rows_read`;
- `peak_chunk_rows`;
- `current_rss_mib` checkpoints;
- `group_total_seconds`;
- `local_search_unaccounted_seconds`;
- wall time outside local search: `elapsed_seconds - set_elapsed_seconds`.

If current RSS peaks before local search, prioritize that earlier stage. If it
peaks during local search, revisit segment assembly with a design that reduces
repeated HDF5 reads instead of forcing small bounded windows.

## Non-Storage To-Dos

### 1. Instrument Step 3 Matrix-to-Vector Memory

The latest chr11 memory checkpoints identify Step 3 as the RSS chokepoint:
`step3_start` had a 328.5 MiB lifetime max, while `step3_end` had a
99,942.0 MiB lifetime max. Add finer checkpoints inside
`write_diag_vector_array()` before changing local-search memory behavior again.

Acceptance criteria:

- logs identify the first Step 3 subphase where max RSS approaches 100 GiB;
- checkpoints cover HDF5 `read_all()`, partition filtering, locus list
  construction, `_r2_rows()`, center-locus `bincount`, pending-sum flushing,
  and loop cleanup;
- instrumentation remains scalar-only and does not retain extra arrays;
- output vector and final `fourier_ls` BED remain identical.

Run this remotely only; do not run real-data profiling from a local checkout.

### 2. Replace Step 3 Materialized Partition Work With Chunked Accumulation

After the subphase checkpoint identifies the allocation, replace the offending
one-partition materialization with bounded HDF5 row-chunk processing.

Acceptance criteria:

- diagonal normalization and duplicate-pair first-wins semantics match the
  existing array path;
- vector output matches the existing array path on focused tests;
- final `fourier_ls` BED remains byte-identical on remote validation;
- chr11 max RSS drops substantially from the current 97.6 GiB profile;
- wall time does not regress enough to offset the memory fix.

### 3. Remote Validation of Streaming Metric and Step 3 Fixes

Validate the default `fourier_ls` run on remote chr21/chr22 first, then chr10
and chr11 one at a time. This should happen before deeper local-search numeric
work because it verifies both the metric streaming path and the Step 3 memory
fix under realistic HDF5 data volume.

Acceptance criteria:

- final `fourier_ls` BED remains byte-identical;
- JSON metric values match the previous materialized path within existing
  float tolerance;
- max RSS drops on chr11 and does not increase on chr10/chr21/chr22;
- logs no longer contain `Loading metric covariance arrays` or `Reloading
  metric covariance arrays for final metric reuse` in the normal uncached path;
- wall time is acceptable despite rereading partitions for metrics.

Run this remotely only; do not run real-data profiling from a local checkout.

### 4. Representative chr10/chr11 Local-Search Runtime Validation

Acceptance criteria:

- final `fourier_ls` BED is byte-identical to the current branch baseline;
- max RSS does not increase;
- local-search precompute runtime improves or remains neutral;
- per-breakpoint diagnostics show lower repeated partition loading for grouped
  sequential runs.

Run this remotely only; do not run real-data profiling from a local checkout.

### 5. Dense Local Accumulators

The implementation still uses `sum_vert_by_locus` and `sum_horiz_by_locus`
dictionaries during precompute, then materializes arrays at the end. A future
pass can replace those dictionaries with dense local arrays aligned to the
current precomputed locus window.

Constraints:

- keep accumulators scoped to the local search window;
- preserve exact locus list semantics;
- compare `sum_vert` and `sum_horiz` against the Decimal legacy path before
  enabling by default.

Priority after chr21/chr22 profiling: medium. This should follow precompute
substep instrumentation so we know dictionary accumulation is material.

### 6. Horizontal Grouped Reduction

Horizontal aggregation still uses `np.unique(..., return_inverse=True)` and
`np.bincount()` per chunk. This is conservative because `hi` is not globally
sorted within each `lo` range.

Possible future approach:

- benchmark whether horizontal aggregation still dominates after the current
  changes;
- if it does, sort only the chunk's `(hi, r2)` view or use a bounded grouped
  reduction strategy;
- accept only if runtime improves without increasing peak RSS.

Priority after chr21/chr22 profiling: unknown. It may reduce wall time but can
increase memory pressure; profile single-worker precompute first.

### 7. Multiprocessing-Aware Grouping

Current grouping is sequential only. A future version could assign whole
partition groups to process workers.

Constraints:

- do not send a chromosome-wide covariance cache to every worker;
- group tasks must remain bounded by partition range;
- worker count should be documented as a memory multiplier.

Priority after chr21/chr22 profiling: low until precompute substeps identify a
pure numeric kernel as dominant.

### 8. JIT Review After Profiling

Do not add JIT until the representative chromosome run identifies remaining
hot pure-array kernels.

Likely candidates if still hot:

- chunk normalization and aggregation in `_add_array_segment_values()`;
- `_search_array()` candidate scoring to reduce temporary arrays;
- streaming metric calculation only if partition rereads become a material
  runtime regression after the memory fix is validated.

### 9. HDF5 Validation and Tuning

HDF5 has been promoted and implemented. Keep it as a bounded reader path, not
a large resident cache.

Recommended next checks:

- Re-profile remote chr21/chr22 first to confirm byte-identical BED output,
  lower or neutral RSS, and explainable HDF5 reader overhead.
- Re-run chr10/chr11 remotely one at a time after chr21/chr22 are clean.
- Tune chunk size only from remote profiles; avoid local real-data profiling.

## HDF5 Status

HDF5 is the production covariance partition format. Existing `.npz`
intermediates are not read by production paths; regenerate covariance outputs
with `calc-covariance` or `run`.

- `h5py` is now a normal project dependency.
- `CovarianceStore.partition_path()` returns `.h5` partition paths.
- `calc_covariance()` writes canonical, indexed HDF5 partitions and collapses
  duplicate physical VCF positions before pairwise LD, keeping the first
  variant at each position.
- The HDF5 writer enforces canonical sorted unique row order for every write:
  generic callers are canonicalized/sorted/deduplicated before validation, and
  the `calc_covariance()` fast path skips redundant sort/dedup only after
  validating that rows are already `lo <= hi`, sorted by `(lo, hi)`, and
  duplicate-free.
- CLI partition validation checks HDF5 attrs/datasets and regenerates invalid
  caches.
- Metric calculation streams partition row chunks through the HDF5 reader.
- Single-worker grouped local search caches only HDF5 partition metadata and
  reads HDF5 row chunks for each full segment range before aggregating.
- Matrix-to-vector reads HDF5 partitions through the reader; it is still
  one-partition-at-a-time rather than fully segment-chunked.
- Example workflows and diagnostics have been updated for HDF5.

### Known Behavior Divergence: Duplicate Positions

This is a known, intentional behavior difference to revisit if future
chromosomes diverge. It has not appeared to change EUR chr21/chr22 outputs so
far, but it is not perfectly legacy-equivalent in cutoff-sensitive duplicate
position cases.

Current behavior:

- `calc_covariance()` collapses duplicate physical VCF positions before
  pairwise LD, keeping the first variant at each position.
- This keeps retained covariance rows sorted by physical `(lo, hi)` for the
  HDF5 fast writer path and avoids the memory-heavy generic writer on
  duplicate-heavy partitions.
- Current EUR chr21/chr22 results do not appear to diverge from this behavior.

Legacy nuance:

- Legacy covariance generation wrote duplicate-position variants as separate
  rows, but downstream matrix readers keyed data by physical position.
- The legacy reader therefore kept the first retained covariance row for each
  physical `(lo, hi)` pair.
- Pre-LD duplicate collapse is not perfectly equivalent in cutoff-sensitive
  edge cases: if the first duplicate-position variant produces no retained row
  for a pair but a later duplicate variant would have survived the cutoff,
  legacy could keep the later retained row while the current path drops it.

Current decision:

- Keep pre-LD duplicate-position collapse because it preserves the HDF5 writer
  memory/speed path and has not shown observed chr21/chr22 divergence.
- Treat it as correctness-sensitive. If future profiles or chromosomes show
  breakpoint divergence, revisit this before changing local-search numerics.

Possible future writer path:

- Keep duplicate-position variants through pairwise LD.
- Exploit the pairwise kernel's SNP-index output order and non-decreasing
  physical positions to avoid a full partition-wide `lexsort`.
- Map variant indexes to physical-position ranks, then chunk through retained
  rows and drop adjacent duplicate physical `(lo_rank, hi_rank)` keys,
  preserving first-retained-pair semantics.
- Validate that `(lo_rank, hi_rank)` is non-decreasing; fall back to the
  generic writer only if that invariant fails.
- This should change extra writer work from `O(n_pairs log n_pairs)` sorting
  plus large temporaries to a chunked `O(n_pairs)` scan with
  `O(n_snps + chunk_rows)` extra memory.

### HDF5 Contract

One HDF5 file is written per covariance partition:

```text
{chrom}.{start}.{end}.h5
  /covariance/lo          int32 or int64, sorted
  /covariance/hi          int32 or int64, sorted with lo
  /covariance/shrink_ld   float64
  /covariance/naive_ld    float64, optional full-output dataset
  /metadata/*             optional full-output metadata
  /index/diag_pos         int32 or int64
  /index/diag_val         float64
  /index/lo_values        int32 or int64
  /index/lo_offsets       int64
  attrs:
    format = "ldetect2-covariance-h5"
    version = 1
    position_dtype = "int32" or "int64"
    sorted_by = "lo_hi"
    deduplicated = true
    compact = true or false
```

`lo_offsets` stores row-group offsets for each `lo_values` entry, so local
search can map a genomic `lo` range to row slices quickly.

Writer requirements and invariants:

- Convert `(i_pos, j_pos)` to sorted canonical `(lo, hi)` before writing.
- Deduplicate duplicate `(lo, hi)` pairs with the same first-pair-wins
  semantics used by `canonical_local_search_rows()`.
- Validate the final row order before writing. HDF5 `/index/lo_offsets`
  assumes rows are canonical, sorted by `(lo, hi)`, and duplicate-free.
- `calc_covariance()` may use the trusted fast path because its pairwise LD
  kernel emits unique rows in sorted SNP-index order. That fast path must still
  validate the invariant; it only skips the memory-heavy defensive sort/dedup.
- Duplicate physical positions are removed before pairwise LD, keeping the
  first variant for each position. Without this, common duplicate positions can
  make index-sorted pairwise output fail the `(lo, hi)` sorted-row invariant.
- Store positions as `int32` whenever all values fit.
- Store diagonal rows in both `/covariance/*` and `/index/diag_*`.
- Make `/index/lo_offsets` length `len(lo_values) + 1`, so
  `lo_offsets[k]:lo_offsets[k + 1]` gives the row slice for `lo_values[k]`.

### Chunked Reader Flows

`HDF5CovariancePartitionReader` is the only production reader:

- `iter_rows()` uses `/index/lo_values` and `/index/lo_offsets` to map
  `lo_min..lo_max` to contiguous row slices, then yields bounded HDF5 dataset
  reads.
- `iter_owned_rows()` applies the partition ownership rules used by metric and
  matrix-to-vector paths while streaming chunks.
- `read_diagonal()` and `read_loci()` are small enough to load eagerly per
  group or per chromosome pass.

Local search, metric calculation, and matrix-to-vector all stream HDF5 rows in
bounded chunks and discard chunk temporaries after aggregation. Local search
currently assembles and canonicalizes each full segment row range before
aggregation; a bounded-window replacement was tested and reverted because it
increased whole-run wall time and did not reduce process max RSS.

Correctness requirements remain strict: breakpoint loci, `N_zero`, final BED,
and selected local-search breakpoint positions must remain exact. Metric sums
may differ only at insignificant floating last-bit levels caused by chunk
aggregation order.

### Validation Status and Remaining Work

Local checks after the bounded-window local-search revert and RSS checkpoint
update:

```text
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest -q
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run ruff check src/ldetect2 tests examples/ldetect_original/scripts
git diff --check
```

Real-data profiling remains remote-only. Next remote checks:

1. Re-run chr11 with current-RSS checkpoints and identify where current RSS
   first approaches the high-water mark.
2. Re-test chr21/chr22 only if the RSS checkpoint change needs smaller
   validation before chr11.
3. Re-run chr10 after the chr11 RSS chokepoint is understood.
4. Tune `chunk_rows` only from remote profiles.
5. Add HDF5 reader I/O or current-RSS diagnostics only if remote profiles show
   unexplained elapsed time or RSS.
6. Consider true chunked matrix-to-vector accumulation if that becomes the next
   wall-time or RSS bottleneck.

## Open Questions

- Should full metadata arrays remain supported at all, or should debug
  metadata be regenerated from upstream inputs when needed?
- What chunk size best balances local-search window reads against compression
  efficiency on real 10-100 MB partition files?
- Should chunk size be fixed in code for reproducibility or exposed as an
  advanced CLI/config option for profiling?
