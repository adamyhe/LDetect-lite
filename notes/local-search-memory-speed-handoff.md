# Local Search Memory and Runtime Handoff

Date: 2026-06-23

## Context

Recent profiling and `ldetect_original` runs show that local search dominates
runtime on small chromosomes and is again the peak RSS source after chunked
matrix-to-vector reduced Step 3 memory. Earlier whole-chromosome metric and
matrix-to-vector materialization risks have been addressed with streaming HDF5
paths. The current production target is still normal `ldetect2 run --subset
fourier_ls`, and the remaining cost is split between dense local-search
precompute work, streaming metric passes, and bounded HDF5 covariance I/O.

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
| Streaming HDF5 segment assembly | Implemented pending remote validation | HDF5 local search now streams segment row chunks into accumulators instead of materializing full active row ranges |
| Streaming metric calculation | Implemented | `metric_from_files()` default path in `Metric.calc_metric()` |
| HDF5 chunked covariance reader | Implemented | `HDF5CovariancePartitionReader`, `iter_rows()`, `iter_owned_rows()` |
| HDF5 writer invariant and duplicate-position handling | Implemented | validated writer fast path, duplicate-position collapse in `calc_covariance()` |
| Duplicate-safe local-search row-stream boundary | Implemented | HDF5 local search routes segment rows through one canonical stream boundary preserving first-retained-pair precedence |
| Dense local accumulators | To-do | local search still uses `sum_vert_by_locus`/`sum_horiz_by_locus` dictionaries |
| JIT for local-search numerics | Deferred | Candidate scoring is not hot in current profiles; revisit only after remote profiling |
| Remote real-data validation | Partial | chr11/chr21/chr22 profiles downloaded; chr10 still needs current chunked Step 3 validation |

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

## Latest Downloaded Profiling: EUR chr11/chr21/chr22

The latest downloaded remote profiling outputs are under:

```text
examples/ldetect_original/results/diagnostics/EUR/profiling/
```

The current remote logs include chr11 plus chr21/chr22 validation after the
chunked HDF5 matrix-to-vector change. They show that the Step 3 memory fix
worked: matrix-to-vector no longer sets the run high-water mark. The run-level
RSS peak now occurs during local search, with the dense chr11 43-59 Mb region
still the main memory and runtime risk.

Current run summary:

| Chrom | Wall time | Max RSS | Local search | LS % wall | Precompute | Search |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr11 | 4888.00 s | 35.33 GiB | 1789.46 s | 36.6% | 1788.11 s | 0.127 s |
| chr21 | 317.41 s | 1.39 GiB | 66.79 s | 21.0% | 66.51 s | 0.036 s |
| chr22 | 389.07 s | 2.55 GiB | 92.73 s | 23.8% | 92.45 s | 0.037 s |

Step 3 validation:

| Chrom | Step 3 seconds | Step 3 max RSS | Notes |
| --- | ---: | ---: | --- |
| chr11 | 1079 s | 467.3 MiB | Was 63.55 GiB after helper-scope cleanup and 97.58 GiB before cleanup |
| chr21 | 60 s | 355.9 MiB | Smaller-chromosome validation |
| chr22 | 73 s | 361.9 MiB | Smaller-chromosome validation |

Current chr11 walltime split from memory checkpoints:

| Phase | Seconds | Minutes | Notes |
| --- | ---: | ---: | --- |
| Step 3 matrix-to-vector | 1079 s | 18.0 m | chunked HDF5 path |
| Step 4 total | 2640 s | 44.0 m | includes minima, metrics, local search |
| Filter-width/minima before metric | 205 s | 3.4 m | still lower priority than local search and metrics |
| Fourier metric | 268 s | 4.5 m | streaming metric pass |
| Fourier local search | 1790 s | 29.8 m | largest walltime phase |
| Final Fourier-LS metric | 268 s | 4.5 m | second streaming metric pass |

The bounded HDF5 segment assembly experiment was run before this profile. It
improved chr11 local-search time but regressed whole-run wall time and did not
reduce max RSS, so the active code path has been reverted to full segment
assembly.

The key runtime finding is still that local-search candidate scoring is not the
bottleneck. The remaining local-search cost is overwhelmingly precompute,
driven by the number of covariance rows loaded and aggregated per breakpoint.
The key memory finding has changed again: after chunked Step 3, chr11 max RSS
is set by local search rather than matrix-to-vector.

Rows loaded and precompute time scale closely:

- chr21 loaded 601.8M rows across 23 breakpoint searches and filtered them to
  327.3M candidate rows and 181.9M eligible rows.
- chr22 loaded 918.3M rows across 23 breakpoint searches and filtered them to
  375.8M candidate rows and 205.1M eligible rows.
- chr11 loaded 43.03B rows across 83 breakpoint searches and filtered them to
  3.01B candidate rows and 1.22B eligible rows.

Phase timing still identifies segment row assembly as the dominant local-search
runtime bucket:

| Chrom | Append/assembly | Horizontal | Normalize | Active rows peak |
| --- | ---: | ---: | ---: | ---: |
| chr11 | 1641.54 s (91.8%) | 83.26 s (4.7%) | 50.28 s (2.8%) | 7.15B |
| chr21 | 42.99 s (64.6%) | 13.39 s (20.1%) | 8.35 s (12.5%) | 27.9M |
| chr22 | 65.64 s (71.0%) | 15.23 s (16.5%) | 9.51 s (10.3%) | 120.2M |

Current chr11 local-search phase timing:

| Phase | Seconds | Share of LS precompute |
| --- | ---: | ---: |
| Append/segment assembly | 1641.54 s | 91.8% |
| Horizontal aggregation | 83.26 s | 4.7% |
| Normalization | 50.28 s | 2.8% |
| Slice/filter bookkeeping | 6.77 s | 0.4% |
| Vertical aggregation | 0.99 s | 0.1% |

The top six local-search windows dominate runtime:

| Window | Partitions | Rows loaded | Candidate rows | Eligible rows | Precompute | Assembly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 49.87-52.62 Mb | 15 | 6.80B | 326.6M | 82.5M | 382.12 s | 371.39 s |
| 47.01-49.87 Mb | 17 | 6.26B | 480.1M | 186.2M | 361.47 s | 337.61 s |
| 52.62-55.09 Mb | 18 | 7.09B | 193.7M | 15.7M | 275.35 s | 272.92 s |
| 55.09-56.41 Mb | 25 | 7.25B | 130.6M | 81.5M | 217.11 s | 207.30 s |
| 42.91-47.01 Mb | 24 | 4.90B | 295.0M | 37.5M | 141.54 s | 136.29 s |
| 56.41-59.45 Mb | 35 | 7.30B | 92.0M | 34.5M | 102.64 s | 98.53 s |

These six windows account for about 1480 seconds of local-search precompute,
or roughly 83% of all chr11 local-search precompute time. The key walltime
lever is reducing repeated segment row assembly in the dense 43-59 Mb region,
not candidate scoring.

Worst chr11 breakpoint windows in the current profile:

| Index | Window | Rows | Candidate | Eligible | Partitions | Precompute | Assembly |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 28 | 49.87-52.62 Mb | 6.80B | 326.6M | 82.5M | 15 | 382.12 s | 371.39 s |
| 27 | 47.01-49.87 Mb | 6.26B | 480.1M | 186.2M | 17 | 361.47 s | 337.61 s |
| 29 | 52.62-55.09 Mb | 7.09B | 193.7M | 15.7M | 18 | 275.35 s | 272.92 s |
| 30 | 55.09-56.41 Mb | 7.25B | 130.6M | 81.5M | 25 | 217.11 s | 207.30 s |
| 26 | 42.91-47.01 Mb | 4.90B | 295.0M | 37.5M | 24 | 141.54 s | 136.29 s |
| 31 | 56.41-59.45 Mb | 7.30B | 92.0M | 34.5M | 35 | 102.64 s | 98.53 s |

Additional chr11 timing from the raw log:

- matrix-to-vector took about 18 minutes, down from about 24.8 minutes in the
  helper-scope-only profile;
- local search was about 29.8 minutes in the aggregated profile;
- the two metric passes together were about 8.9 minutes.

Important memory caveat:

- `max_rss_mib` in breakpoint logs is a process lifetime high-water mark from
  `resource.getrusage`, not current RSS.
- In this profile the high-water mark is reached during local search, so
  breakpoint rows after index 28 all report the same lifetime max of about
  36,179 MiB.
- Current-RSS checkpoints still matter when deciding whether a future local
  search change reduces active memory or only shifts the lifetime mark.

Step-level chr11 memory checkpoints:

| Checkpoint | Current RSS | Max RSS |
| --- | ---: | ---: |
| `step3_start` | 193.5 MiB | 387.8 MiB |
| `matrix_to_vector_array_start` | 193.5 MiB | 387.8 MiB |
| `matrix_to_vector_array_end` | 317.2 MiB | 467.3 MiB |
| `step3_end` | 317.2 MiB | 467.3 MiB |
| `fourier_metric_start` | 405.1 MiB | 467.3 MiB |
| `fourier_metric_end` | 420.3 MiB | 603.5 MiB |
| `fourier_local_search_start` | 420.3 MiB | 603.5 MiB |
| `fourier_local_search_end` | 371.7 MiB | 36,179.3 MiB |
| `run_end` | 363.0 MiB | 36,179.3 MiB |

Implications:

- Do not prioritize `_search_array()` candidate scoring. It is effectively
  free at this scale.
- Do not add JIT yet.
- Keep the chunked Step 3 implementation. It reduced chr11 Step 3 peak memory
  from 63.55 GiB after helper-scope cleanup to under 0.5 GiB, and reduced Step
  3 runtime from about 1485 s to about 1079 s in the downloaded profiles.
- The highest leverage memory task is now local-search active-window assembly
  in the dense chr11 43-59 Mb region.
- The highest leverage walltime task is local-search append/segment assembly:
  1642 s out of 1788 s precompute in the current chr11 profile.
- Step 3 matrix-to-vector remains a major walltime phase at about 1079 s, but
  it is no longer the top memory risk.
- Horizontal aggregation and normalization are real but secondary walltime
  targets: about 84 s and 50 s respectively on chr11.

### Next Profiling Targets

Review the next remote profile in this order:

1. Local-search segment assembly for dense windows.
   Runtime is now dominated by append/segment assembly in six chr11 windows.
   Revisit row assembly with a runtime-first design that reduces repeated
   multi-billion-row scans without reintroducing the bounded-window HDF5 read
   regression. Candidate directions include caching reusable per-partition row
   slices for a small dense window group, processing adjacent breakpoints as a
   shared sweep, or building segment-local accumulators directly from
   partition slices.
2. Horizontal aggregation.
   This is material but secondary in the latest chr11 profile: about 84 seconds
   total, led by the 47.01-49.87 Mb window at 13.3 seconds. The current path uses
   `np.unique(row_hi, return_inverse=True)` plus `np.bincount()` per chunk,
   which is exact but allocation-heavy. Possible follow-ups are grouped
   reduction after sorting `hi` within the chunk, dense local accumulators
   indexed by the local locus window, or processing partition slices directly
   into accumulators.
3. Normalization.
   This is about 50 seconds total on chr11, again secondary to assembly. The
   current path does two `np.searchsorted()` calls into diagonal arrays per
   chunk, filters positive diagonals, then computes `r²`. Possible follow-ups
   are dense or dictionary-style diagonal lookup scoped to active segment loci,
   carrying per-partition diagonal lookup state, and combining eligibility plus
   diagonal filtering to shrink arrays earlier.
4. Group load and canonicalization outside breakpoint rows.
   This is not currently material: `group_total_seconds` is about 1.0 s and
   `local_search_unaccounted_seconds` is about 0.06 s on chr11. Keep monitoring
   but do not prioritize.
5. Metric recomputation around local search.
   The two streaming metric passes are each about 268 s, around 9 minutes
   together. This is a meaningful walltime target after Step 3 and local-search
   assembly. Avoid adding chromosome-wide covariance caches; look for reusable
   diagonal/locus metadata or a way to combine/reuse streaming passes.
6. Filter-width search.
   This is about 205 s before the first metric in the current chr11 profile. A
   low-memory optimization is to cache `{width: minima_count}` only. Do not
   cache smoothed arrays. This is lower priority than Step 3, local search, and
   metric passes.
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

Current RSS now peaks during local search, and walltime is dominated by local
search after Step 3. Treat local-search active-window assembly as the next
memory and runtime priority.

## Non-Storage To-Dos

### 1. Step 3 Per-Partition Array Lifetime Cleanup

Implemented in `write_diag_vector_array()` by moving per-partition
matrix-to-vector work into `_process_diag_vector_partition()`. The outer loop
now retains only `current_locus` and `pending_sums`, and logs
`helper_return` after the helper scope exits.

Remote chr11 result:

- max RSS dropped from 97.58 GiB to 63.55 GiB;
- wall time dropped from 5204 s to 3967 s in the downloaded profile;
- helper-return checkpoints show current RSS falling back below 0.5 GiB after
  each large partition;
- final command exited successfully.

Keep this change. Continue to avoid local real-data profiling.

### 2. Replace Step 3 `_r2_rows()` With Chunked Accumulation

After lifetime cleanup, the pre-chunking peak came from one-partition
materialized `_r2_rows()` and center-index arrays. The HDF5 path now replaces
that with bounded row-chunk processing. This targets both memory and runtime:
Step 3 took about 1485 s on chr11 before this change.

Implementation status:

- HDF5-backed matrix-to-vector now uses `MATRIX_TO_VECTOR_CHUNK_ROWS` and
  two bounded passes per partition.
- The first pass builds sorted owned loci from chunk-local unique arrays.
- The second pass normalizes rows and accumulates center-locus sums in chunks.
- The in-memory `ChromosomeCovariance` cache path keeps the materialized
  `_r2_rows()` implementation as a compatibility/reference path.

Remote validation:

- chr11 Step 3 max RSS dropped from 63.55 GiB after helper-scope cleanup to
  about 467 MiB.
- chr11 Step 3 walltime improved from about 1485 s to about 1079 s.
- chr21 and chr22 Step 3 max RSS stayed below 0.4 GiB.
- Whole-run max RSS is now set by local search, not matrix-to-vector.

Keep this change. Continue to use the in-memory path only as a compatibility
and test reference path.

### 3. Optimize Local-Search Segment Assembly in Dense Windows

Local-search precompute is the largest walltime phase after chunked Step 3:
1788 s on chr11, with 1642 s in append/segment assembly. Six dense
windows between about 42.9 and 59.5 Mb account for about 1486 s of local-search
precompute.

Candidate approaches:

- keep local-search aggregation behind a canonical covariance row-stream
  boundary. The HDF5 stream preserves partition-order first-retained-pair
  semantics with a per-locus duplicate tracker. If duplicate-position handling
  becomes a major compatibility issue, update this boundary rather than
  scattering pre-LD-collapse assumptions through local-search aggregation;
- HDF5-backed local search now streams segment row chunks directly into
  vertical/horizontal accumulators. This removes per-segment full-row
  concatenation and canonicalization from the hot path while keeping the
  in-memory path as the reference implementation;
- process adjacent dense breakpoints as a shared sweep over partition slices;
- cache reusable per-partition row slices only within a small dense-window
  group, with strict memory bounds;
- keep the previous bounded-window HDF5 experiment as a caution: do not trade
  assembly time for repeated HDF5 read inflation.

Acceptance criteria:

- final `fourier_ls` BED remains byte-identical;
- max RSS does not increase versus the current 35.33 GiB chr11 profile;
- local-search duplicate/cross-partition duplicate tests continue to match the
  Decimal legacy path and first-retained physical-pair precedence;
- append/segment assembly time drops materially in the six dense chr11
  windows;
- local-search precompute improves on chr11 and does not regress on chr21/22.

Run this remotely only; do not run real-data profiling from a local checkout.

### 4. Remote Validation of Streaming Metric and Step 3 Fixes

Remote chr11/chr21/chr22 validation is now available for the chunked Step 3
path. Keep using the same validation sequence for future local-search changes:
chr21/chr22 first when a smaller iteration run is useful, then chr10/chr11 one
at a time.

Acceptance criteria:

- final `fourier_ls` BED remains byte-identical;
- JSON metric values match the previous materialized path within existing
  float tolerance;
- max RSS does not increase on chr10/chr11/chr21/chr22;
- logs no longer contain `Loading metric covariance arrays` or `Reloading
  metric covariance arrays for final metric reuse` in the normal uncached path;
- wall time is acceptable despite rereading partitions for metrics.

Run this remotely only; do not run real-data profiling from a local checkout.

### 5. Metric Pass Runtime Review

The two streaming metric passes each take about 268 s on chr11, or about
9 minutes together. This is a meaningful walltime target after Step 3 and
local-search segment assembly.

Candidate approaches:

- instrument metric pass substeps before optimizing;
- reuse small metadata such as diagonal/locus information where exact and
  bounded;
- explore whether the Fourier and Fourier-LS metric passes can share safe
  streaming setup without retaining chromosome-wide covariance arrays.

Constraints:

- do not reintroduce full-chromosome covariance caches;
- preserve exact `N_zero` and metric behavior within existing tolerance.

### 6. Representative chr10/chr11 Local-Search Runtime Validation

Acceptance criteria:

- final `fourier_ls` BED is byte-identical to the current branch baseline;
- max RSS does not increase;
- local-search precompute runtime improves or remains neutral;
- per-breakpoint diagnostics show lower repeated partition loading for grouped
  sequential runs.

Run this remotely only; do not run real-data profiling from a local checkout.

### 7. Dense Local Accumulators

The implementation still uses `sum_vert_by_locus` and `sum_horiz_by_locus`
dictionaries during precompute, then materializes arrays at the end. A future
pass can replace those dictionaries with dense local arrays aligned to the
current precomputed locus window.

Constraints:

- keep accumulators scoped to the local search window;
- preserve exact locus list semantics;
- compare `sum_vert` and `sum_horiz` against the Decimal legacy path before
  enabling by default.

Priority: medium-low until segment assembly improves. Accumulator dictionaries
are not the dominant chr11 runtime bucket in the latest profile.

### 8. Horizontal Grouped Reduction

Horizontal aggregation still uses `np.unique(..., return_inverse=True)` and
`np.bincount()` per chunk. This is conservative because `hi` is not globally
sorted within each `lo` range.

Possible future approach:

- benchmark whether horizontal aggregation still dominates after the current
  changes;
- if it does, sort only the chunk's `(hi, r2)` view or use a bounded grouped
  reduction strategy;
- accept only if runtime improves without increasing peak RSS.

Priority: secondary. It is about 84 s on chr11, compared with 1649 s for
append/segment assembly.

### 9. Normalization Lookup Optimization

Normalization is about 50 s on chr11. Possible optimizations include dense or
dictionary-style diagonal lookup scoped to active segment loci, carrying
per-partition diagonal lookup state, and combining eligibility plus diagonal
filtering earlier.

Priority: secondary. Optimize only after Step 3 chunking and segment assembly.

### 10. Multiprocessing-Aware Grouping

Current grouping is sequential only. A future version could assign whole
partition groups to process workers.

Constraints:

- do not send a chromosome-wide covariance cache to every worker;
- group tasks must remain bounded by partition range;
- worker count should be documented as a memory multiplier.

Priority: low. Group loading is about 1 s and unaccounted local-search time is
near zero in the latest chr11 profile. Multiprocessing can also multiply memory.

### 11. JIT Review After Profiling

Do not add JIT until the representative chromosome run identifies remaining
hot pure-array kernels.

Likely candidates if still hot:

- chunk normalization and aggregation in `_add_array_segment_values()`;
- `_search_array()` candidate scoring to reduce temporary arrays;
- streaming metric calculation only if partition rereads become a material
  runtime regression after the memory fix is validated.

### 12. HDF5 Validation and Tuning

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
- Matrix-to-vector reads HDF5 partitions through the reader, processes each
  partition in a helper scope, and chunks HDF5 normalization/center
  accumulation with `MATRIX_TO_VECTOR_CHUNK_ROWS`; caller-supplied in-memory
  covariance caches still use the materialized reference path.
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

Local checks after the bounded-window local-search revert, RSS checkpoint
update, and chunked matrix-to-vector implementation:

```text
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest -q
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run ruff check src/ldetect2 tests examples/ldetect_original/scripts
git diff --check
```

Real-data profiling remains remote-only. Next remote checks:

1. Re-run chr10 with the current chunked Step 3 path to confirm the chr11
   memory improvement generalizes to the other problematic chromosome.
2. Design the next local-search active-window assembly optimization around the
   chr11 43-59 Mb dense region.
3. Re-test chr21/chr22 after local-search assembly changes as smaller
   iteration targets.
4. Tune matrix-to-vector `chunk_rows` only if remote profiles show a clear
   Step 3 walltime or RSS regression.
5. Add HDF5 reader I/O or current-RSS diagnostics only if remote profiles show
   unexplained elapsed time or RSS.

## Open Questions

- Should full metadata arrays remain supported at all, or should debug
  metadata be regenerated from upstream inputs when needed?
- What chunk size best balances local-search window reads against compression
  efficiency on real 10-100 MB partition files?
- Should chunk size be fixed in code for reproducibility or exposed as an
  advanced CLI/config option for profiling?
