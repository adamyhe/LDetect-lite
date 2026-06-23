# Local Search Memory and Runtime Handoff

Date: 2026-06-21

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
partition bounds, loads each group once with `load_covariance_partitions()`,
processes that group's breakpoints, then releases the group cache.

This is intentionally not used for:

- Decimal local search;
- caller-supplied chromosome covariance caches;
- multiprocessing runs.

Expected benefit:

- Fewer repeated partition loads for adjacent breakpoint windows that touch
  the same partition range.
- No additional memory multiplication across process workers.

Memory risk status:

- Moderate but bounded. One partition group is retained at a time in the
  sequential path.

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
- candidate, eligible, and normalized row counts;
- chunk count, segment count, and peak active rows.

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
- streaming metric parity against the previous materialized array metric path.

Validation run after implementation:

```text
uv run pytest -q
uv run ruff check src/ldetect2 tests
git diff --check
```

Result after the latest streaming-metric pass: all checks passed (`162 passed`).

## Profiling Findings: EUR chr21/chr22

Remote diagnostics were run for EUR chr21 and chr22 with profiling outputs
under:

```text
examples/ldetect_original/results/diagnostics/EUR/profiling/
```

The key finding is that local-search candidate scoring is not the bottleneck.
The remaining cost is overwhelmingly local-search precompute, driven by the
number of covariance rows loaded and aggregated per breakpoint.

| Chrom | Wall time | Max RSS | Local search | LS % wall | Precompute | Search |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr21 | 347.91 s | 10.76 GiB | 93.87 s | 27.0% | 71.09 s | 0.035 s |
| chr22 | 479.42 s | 11.60 GiB | 158.71 s | 33.1% | 123.07 s | 0.031 s |

Rows loaded and precompute time scale closely:

- chr21 loaded 603.0M rows across 23 breakpoint searches.
- chr22 loaded 920.1M rows across 23 breakpoint searches.
- chr21 filtered 603.0M loaded rows to 327.3M candidate rows and 181.9M
  eligible rows.
- chr22 filtered 920.1M loaded rows to 375.9M candidate rows and 205.1M
  eligible rows.

Phase timing identified active-array append/recanonicalization as the dominant
remaining cost:

| Chrom | Append | Canonicalize | Horizontal | Normalize |
| --- | ---: | ---: | ---: | ---: |
| chr21 | 30.27 s (42.6%) | 18.89 s (26.6%) | 11.84 s (16.6%) | 7.19 s (10.1%) |
| chr22 | 62.47 s (50.8%) | 34.47 s (28.0%) | 13.54 s (11.0%) | 8.16 s (6.6%) |

Worst breakpoint windows in the chr21/chr22 profile:

| Chrom | Index | Rows | Candidate | Eligible | Partitions | Precompute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr22 | 15 | 134.4M | 39.3M | 30.3M | 9 | 22.592 s |
| chr22 | 16 | 139.9M | 29.5M | 11.3M | 10 | 19.772 s |
| chr22 | 14 | 108.4M | 20.9M | 7.4M | 11 | 15.311 s |
| chr22 | 9 | 76.2M | 32.4M | 18.2M | 11 | 10.700 s |
| chr21 | 10 | 58.0M | 29.9M | 18.6M | 16 | 7.747 s |

Implications:

- Do not prioritize `_search_array()` candidate scoring. It is effectively
  free at this scale.
- Do not add JIT yet.
- The highest leverage changes are reducing repeated active-row append and
  canonicalization work.
- Horizontal aggregation is the next numeric target after append/canonicalize
  improvements are remotely validated.

### Post-Append Profiling Targets

After the append/segment row assembly changes are validated, review the next
profile in this order:

1. Horizontal aggregation.
   This was still material in the chr21/chr22 profile: about 11.7 seconds for
   chr21 and 13.4 seconds for chr22. The current path uses
   `np.unique(row_hi, return_inverse=True)` plus `np.bincount()` per chunk,
   which is exact but allocation-heavy. Possible follow-ups are grouped
   reduction after sorting `hi` within the chunk, dense local accumulators
   indexed by the local locus window, or processing partition slices directly
   into accumulators.
2. Normalization.
   This was moderate: about 6.9 seconds for chr21 and 7.8 seconds for chr22.
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
- `horizontal_seconds / precompute_seconds`;
- `normalize_seconds / precompute_seconds`;
- `group_total_seconds`;
- `local_search_unaccounted_seconds`;
- wall time outside local search: `elapsed_seconds - set_elapsed_seconds`.

If append drops as expected, horizontal aggregation and group
load/canonicalization are the next practical targets.

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

### 2. Remote Validation of Append/Canonicalize Reduction

Validate the canonical partition cache and segment-slice precompute path on
remote chr21/chr22 before pursuing deeper numeric changes.

Acceptance criteria:

- final `fourier_ls` BED remains byte-identical;
- max RSS does not increase;
- `canonicalize_seconds` drops substantially because group partitions are
  canonicalized once;
- `append_seconds` drops substantially because full active arrays are not
  repeatedly recanonicalized.

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
  reads segment row ranges from HDF5 chunks instead of preloading and
  canonicalizing full partition groups.
- Matrix-to-vector reads HDF5 partitions through the reader; it is still
  one-partition-at-a-time rather than fully segment-chunked.
- Example workflows and diagnostics have been updated for HDF5.

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
bounded chunks and discard chunk temporaries after aggregation. This replaced
the earlier memory-heavy pattern of inflating full compressed arrays, then
canonicalizing and slicing them in memory.

Correctness requirements remain strict: breakpoint loci, `N_zero`, final BED,
and selected local-search breakpoint positions must remain exact. Metric sums
may differ only at insignificant floating last-bit levels caused by chunk
aggregation order.

### Validation Status and Remaining Work

Local checks that passed after the HDF5/example updates:

```text
snakemake -s examples/ldetect_example/Snakefile -n
snakemake -s examples/ldetect_original/Snakefile -n --config chromosomes='[21]'
snakemake -s examples/ldetect_original/Snakefile.diagnostics -n --config chromosomes='[21]' case_chromosome=21 control_chromosome=21
snakemake -s examples/MacDonald2022/Snakefile -n
uv run ruff check src/ldetect2 tests examples/ldetect_example/scripts examples/ldetect_original/scripts examples/MacDonald2022/scripts
uv run pytest -q tests/test_covariance_io.py tests/test_covariance_array.py tests/test_covariance_summary.py tests/test_shrinkage.py tests/test_metric.py tests/test_local_search.py tests/test_cmd_run.py tests/test_partitions.py tests/integration/test_pipeline.py
git diff --check
```

Real-data profiling remains remote-only. Next remote checks:

1. Re-test chr21/chr22 for byte-identical `fourier_ls` BED output, neutral or
   lower RSS, and explainable HDF5 reader overhead.
2. Re-test chr11 covariance generation after the writer fast path and
   duplicate-position collapse.
3. Re-run chr10/chr11 one at a time after chr21/chr22 are clean.
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
