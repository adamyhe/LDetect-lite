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
| Streaming HDF5 segment assembly | Implemented locally | HDF5 local search now splits segments into bounded `lo` windows before aggregation; remote profiling pending |
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

#### Stream HDF5 Segment Assembly in Bounded Windows

**Affected code:** `src/ldetect2/local_search.py`,
`src/ldetect2/io/covariance_hdf5.py`,
`examples/ldetect_original/scripts/profile_ldetect2.py`

Implemented locally after the latest downloaded remote profile. The HDF5
local-search path no longer materializes one full segment row array before
aggregation. Instead, it:

- reads `/index/lo_values` plus `/index/lo_offsets` for each lightweight HDF5
  partition handle;
- splits each segment's loci into bounded `lo` windows using the indexed row
  counts;
- reads HDF5 rows chunk by chunk for each bounded window;
- applies search-window eligibility filters before deduplication and
  normalization;
- canonicalizes only the bounded window, preserving partition-order
  first-pair-wins duplicate semantics;
- aggregates the bounded window and releases its row arrays before moving to
  the next window.

Expected benefit:

- Reduces the large `append_seconds` bucket by replacing full segment
  concatenate/canonicalize work with bounded window assembly.
- Reduces local-search peak temporaries for large chr10/chr11 windows, where a
  single segment can otherwise involve hundreds of millions of candidate rows.
- Keeps exactness semantics unchanged for duplicate pairs and zero/missing
  diagonal values.

Memory risk status:

- Low to moderate and bounded by `HDF5_LOCAL_SEARCH_CHUNK_ROWS` plus one
  bounded `lo` window. The current implementation does not add chromosome-wide
  caches.

Remote validation status:

- Local unit/integration tests pass, but this change still needs remote
  profiling on chr21/chr22 and then chr10/chr11.

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

These results were generated before the bounded HDF5 streaming segment
assembly patch in this branch, so use them as the baseline for the next remote
comparison.

The key finding is that local-search candidate scoring is not the bottleneck.
The remaining cost is overwhelmingly local-search precompute, driven by the
number of covariance rows loaded and aggregated per breakpoint.

| Chrom | Wall time | Max RSS | Local search | LS % wall | Precompute | Search |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr21 | 349.74 s | 1.76 GiB | 62.97 s | 18.0% | 62.70 s | 0.039 s |
| chr22 | 385.07 s | 4.87 GiB | 81.31 s | 21.1% | 81.06 s | 0.032 s |
| chr10 | 1841.74 s | 19.33 GiB | 461.60 s | 25.1% | 460.58 s | 0.118 s |
| chr11 | 5209.00 s | 97.61 GiB | 1730.30 s | 33.2% | 1729.10 s | 0.121 s |

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
| chr21 | 40.36 s (64.4%) | 13.23 s (21.1%) | 7.54 s (12.0%) | 27.9M |
| chr22 | 57.44 s (70.9%) | 13.93 s (17.2%) | 8.03 s (9.9%) | 120.2M |
| chr10 | 347.65 s (75.5%) | 66.69 s (14.5%) | 38.43 s (8.3%) | 635.7M |
| chr11 | 1585.92 s (91.7%) | 82.32 s (4.8%) | 48.49 s (2.8%) | 7.15B |

Worst chr11 breakpoint windows in the latest profile:

| Index | Window | Rows | Candidate | Eligible | Partitions | Precompute | Assembly |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 28 | 49.87-52.62 Mb | 6.80B | 326.6M | 82.5M | 15 | 366.94 s | 356.48 s |
| 27 | 47.01-49.87 Mb | 6.26B | 480.1M | 186.2M | 17 | 347.72 s | 324.35 s |
| 29 | 52.62-55.09 Mb | 7.09B | 193.7M | 15.7M | 18 | 265.28 s | 262.95 s |
| 30 | 55.09-56.41 Mb | 7.25B | 130.6M | 81.5M | 25 | 208.68 s | 199.05 s |
| 26 | 42.91-47.01 Mb | 4.90B | 295.0M | 37.5M | 24 | 138.63 s | 133.50 s |

Additional chr11 timing from the raw log:

- covariance calculation ran from `[02:17:55]` to `[02:40:22]`, about
  22m27s;
- matrix-to-vector ran from `[02:40:22]` to `[03:04:19]`, about 23m57s;
- breakpoint finding before local search ran from `[03:04:19]` to
  `[03:11:07]`, about 6m48s;
- local search ran from `[03:11:07]` to `[03:39:57]`, about 28m50s;
- final metric/write ran from `[03:39:57]` to `[03:44:20]`, about 4m23s.

Important memory caveat:

- `max_rss_mib` in breakpoint logs is a process lifetime high-water mark from
  `resource.getrusage`, not current RSS.
- chr11 already reported about 99.95 GiB at breakpoint index 0, so the peak
  likely occurred before local search or during the first breakpoint rather
  than in the later giant 47-56 Mb local-search windows.
- Add current-RSS instrumentation around pipeline stages before attributing
  the 97.61 GiB chr11 peak solely to local search.

Implications:

- Do not prioritize `_search_array()` candidate scoring. It is effectively
  free at this scale.
- Do not add JIT yet.
- The highest leverage local-search runtime change is reducing HDF5 segment
  row assembly, which the bounded streaming patch now targets.
- Horizontal aggregation and normalization are the next numeric targets after
  streaming segment assembly is remotely validated.

### Post-Streaming-Segment Profiling Targets

After bounded HDF5 segment assembly is validated remotely, review the next
profile in this order:

1. Horizontal aggregation.
   This was still material in the latest chr21/chr22 profile: about 13.2
   seconds for chr21 and 13.9 seconds for chr22. The current path uses
   `np.unique(row_hi, return_inverse=True)` plus `np.bincount()` per chunk,
   which is exact but allocation-heavy. Possible follow-ups are grouped
   reduction after sorting `hi` within the chunk, dense local accumulators
   indexed by the local locus window, or processing partition slices directly
   into accumulators.
2. Normalization.
   This was moderate: about 7.5 seconds for chr21 and 8.0 seconds for chr22.
   The current path does two `np.searchsorted()` calls into diagonal arrays per
   chunk, filters positive diagonals, then computes `r²`. Possible follow-ups
   are dense or dictionary-style diagonal lookup scoped to active segment loci,
   carrying per-partition diagonal lookup state, and combining eligibility plus
   diagonal filtering to shrink arrays earlier.
3. Group load and canonicalization outside breakpoint rows.
   Per-breakpoint phase totals still may not explain all local-search elapsed
   time. Use `local_search_groups.tsv`, `group_total_seconds`, and
   `local_search_unaccounted_seconds` to check whether HDF5 open/read overhead
   or group metadata setup has become significant. If it has, reduce redundant
   partition group loads, merge adjacent groups only when RSS allows it, or
   delay work for partitions that contribute only tiny row slices.
4. Metric recomputation around local search.
   This sits outside the per-breakpoint local-search rows but can affect wall
   time. The current plan is to validate streaming metrics first. If metric
   time becomes visible after the memory fix, instrument metric time
   separately before considering any cache reuse.
5. Filter-width search.
   For larger profiles, repeated width evaluations can be visible during
   exponential search, binary search, and trackback. A low-memory optimization
   is to cache `{width: minima_count}` only. Do not cache smoothed arrays.
6. Matrix-to-vector.
   For larger chromosomes, matrix-to-vector conversion can be a major wall-time
   block outside local search. If profiling shows it dominates, add
   partition-level vector-conversion instrumentation, check for repeated reads,
   and consider bounded per-partition vector accumulation.

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
- `group_total_seconds`;
- `local_search_unaccounted_seconds`;
- wall time outside local search: `elapsed_seconds - set_elapsed_seconds`.

If append/assembly drops as expected, horizontal aggregation and normalization
are the next practical local-search targets. If `hdf5_read_seconds` dominates,
review chunk sizing and HDF5 compression/read behavior before touching numeric
kernels.

## Non-Storage To-Dos

### 1. Remote Validation of Streaming Metric Memory Fix

Validate the default `fourier_ls` run on remote chr21/chr22 first, then chr10
and chr11 one at a time. This should happen before deeper local-search numeric
work because it removes a known full-chromosome resident array.

Acceptance criteria:

- final `fourier_ls` BED remains byte-identical;
- JSON metric values match the previous materialized path within existing
  float tolerance;
- max RSS drops or at least no longer fails on chr10/chr11;
- logs no longer contain `Loading metric covariance arrays` or `Reloading
  metric covariance arrays for final metric reuse` in the normal uncached path;
- wall time is acceptable despite rereading partitions for metrics.

Run this remotely only; do not run real-data profiling from a local checkout.

### 2. Remote Validation of Streaming HDF5 Segment Assembly

Validate the bounded HDF5 segment assembly path on remote chr21/chr22 before
pursuing deeper numeric changes. Then run chr10/chr11 one at a time.

Acceptance criteria:

- final `fourier_ls` BED remains byte-identical;
- max RSS does not increase;
- `append_seconds` drops substantially because full segment row arrays are no
  longer materialized;
- `peak_chunk_rows` remains bounded and is much smaller than the old active
  row peaks on chr10/chr11;
- new HDF5 fields explain the assembly bucket:
  `hdf5_read_seconds`, `chunk_filter_seconds`, `dedup_seconds`, and
  `accumulator_seconds`;
- `rows_after_dedup` and `duplicate_rows_skipped` confirm duplicate handling
  remains active without full segment canonicalization.

### 3. Representative chr10/chr11 Local-Search Validation

Acceptance criteria:

- final `fourier_ls` BED is byte-identical to the current branch baseline;
- max RSS does not increase;
- local-search precompute runtime improves or remains neutral;
- per-breakpoint diagnostics show lower repeated partition loading for grouped
  sequential runs.

Run this remotely only; do not run real-data profiling from a local checkout.

### 4. Dense Local Accumulators

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

### 5. Horizontal Grouped Reduction

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

### 6. Multiprocessing-Aware Grouping

Current grouping is sequential only. A future version could assign whole
partition groups to process workers.

Constraints:

- do not send a chromosome-wide covariance cache to every worker;
- group tasks must remain bounded by partition range;
- worker count should be documented as a memory multiplier.

Priority after chr21/chr22 profiling: low until precompute substeps identify a
pure numeric kernel as dominant.

### 7. JIT Review After Profiling

Do not add JIT until the representative chromosome run identifies remaining
hot pure-array kernels.

Likely candidates if still hot:

- chunk normalization and aggregation in `_add_array_segment_values()`;
- `_search_array()` candidate scoring to reduce temporary arrays;
- streaming metric calculation only if partition rereads become a material
  runtime regression after the memory fix is validated.

### 8. HDF5 Validation and Tuning

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
  splits segment loci into bounded `lo` windows before reading and aggregating
  HDF5 chunks.
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
- `read_lo_index()` returns both `lo_values` and `lo_offsets` so local search
  can estimate row counts for bounded `lo` windows before reading row data.
- `read_diagonal()` and `read_loci()` are small enough to load eagerly per
  group or per chromosome pass.

Local search, metric calculation, and matrix-to-vector all stream HDF5 rows in
bounded chunks and discard chunk temporaries after aggregation. Local search
additionally bounds segment assembly by indexed `lo` windows, so it no longer
materializes and canonicalizes one full segment row array before aggregation.
This replaced the earlier memory-heavy pattern of inflating full compressed
arrays, then canonicalizing and slicing them in memory.

Correctness requirements remain strict: breakpoint loci, `N_zero`, final BED,
and selected local-search breakpoint positions must remain exact. Metric sums
may differ only at insignificant floating last-bit levels caused by chunk
aggregation order.

### Validation Status and Remaining Work

Local checks that passed after the latest streaming segment assembly update:

```text
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest -q
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run ruff check src/ldetect2 tests examples/ldetect_original/scripts
git diff --check
```

Real-data profiling remains remote-only. Next remote checks:

1. Re-test chr21/chr22 for byte-identical `fourier_ls` BED output, neutral or
   lower RSS, and lower/explainable segment assembly time.
2. Re-run chr10/chr11 one at a time after chr21/chr22 are clean.
3. Add current-RSS checkpoints around Step 2, Step 3, Step 4, local-search
   start, first breakpoint start/end, and final metric if chr11 RSS remains
   near 100 GiB.
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
