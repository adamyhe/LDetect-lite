# Local Search Memory and Runtime Handoff

Date: 2026-06-21

## Context

Recent profiling and `ldetect_original` runs show that local search dominates
both runtime and memory. The current production target is still normal
`ldetect2 run --subset fourier_ls`, where the first-order fix is to avoid
unrequested local-search subsets. After that, the remaining cost is mostly in
per-breakpoint covariance loading, row filtering, normalization, and repeated
array aggregation.

This handoff separates two tracks:

1. Non-storage optimizations that can be implemented against the current `.npz`
   covariance partitions.
2. A later migration path to HDF5 partition files with local-search-friendly
   chunking and indexes.

The guiding constraint is that peak RSS must not increase for whole-chromosome
runs. Any optimization that trades runtime for larger resident arrays should be
opt-in or guarded by instrumentation.

## Non-Storage Optimization Status

These changes keep the existing `.npz` covariance artifacts and focus on
reducing repeated local-search work.

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

- Fewer repeated `.npz` loads for adjacent breakpoint windows that touch the
  same partition range.
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

### Completed Exactness Coverage

**Affected code:** `tests/test_local_search.py`,
`tests/integration/test_pipeline.py`

Added/strengthened tests for:

- canonical local-search partition rows, including reversed endpoints,
  duplicates, `int32` preservation, and zero diagonal values;
- duplicate-pair local search versus the Decimal legacy path;
- exact selected breakpoint matching against Decimal local search;
- exact `N_zero` matching against Decimal local search;
- precompute parity for `loci`, `sum_vert`, and `sum_horiz` against the
  Decimal legacy path on multi-partition fixtures, including cross-partition
  duplicate pairs.

Validation run after implementation:

```text
uv run pytest -q
uv run ruff check src/ldetect2 tests
git diff --check
```

Result: all checks passed (`160 passed` after the append/canonicalize pass).

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

## Non-Storage To-Dos

### 1. Remote Validation of Append/Canonicalize Reduction

Validate the canonical partition cache and segment-slice precompute path on
remote chr21/chr22 before pursuing deeper numeric changes.

Acceptance criteria:

- final `fourier_ls` BED remains byte-identical;
- max RSS does not increase;
- `canonicalize_seconds` drops substantially because group partitions are
  canonicalized once;
- `append_seconds` drops substantially because full active arrays are not
  repeatedly recanonicalized.

### 2. Representative chr10/chr11 Validation

Acceptance criteria:

- final `fourier_ls` BED is byte-identical to the current branch baseline;
- max RSS does not increase;
- local-search precompute runtime improves or remains neutral;
- per-breakpoint diagnostics show lower repeated partition loading for grouped
  sequential runs.

Run this remotely only; do not run real-data profiling from a local checkout.

### 3. Dense Local Accumulators

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

### 4. Horizontal Grouped Reduction

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

### 5. Multiprocessing-Aware Grouping

Current grouping is sequential only. A future version could assign whole
partition groups to process workers.

Constraints:

- do not send a chromosome-wide covariance cache to every worker;
- group tasks must remain bounded by partition range;
- worker count should be documented as a memory multiplier.

Priority after chr21/chr22 profiling: low until precompute substeps identify a
pure numeric kernel as dominant.

### 6. JIT Review After Profiling

Do not add JIT until the representative chromosome run identifies remaining
hot pure-array kernels.

Likely candidates if still hot:

- chunk normalization and aggregation in `_add_array_segment_values()`;
- `_search_array()` candidate scoring to reduce temporary arrays;
- streaming metric calculation only if metric memory remains material.

## HDF5 Migration Plan

The current compressed `.npz` archives are storage-efficient but not
local-search-friendly. `np.load(..., mmap_mode=...)` does not give true partial
array access for compressed `.npz` members; each touched array member must be
inflated before it can be sliced. HDF5 gives us one partition artifact with
chunked, compressed datasets that can be read in bounded slices.

The goal is to replace or optionally supplement `.npz` partitions with HDF5
partition files that support local-search chunk reads without requiring raw
`.npy` caches.

### Target HDF5 Layout

One HDF5 file per covariance partition:

```text
{chrom}.{start}.{end}.h5
  /covariance/lo          int32 or int64, sorted
  /covariance/hi          int32 or int64, sorted with lo
  /covariance/shrink_ld   float64
  /index/diag_pos         int32 or int64
  /index/diag_val         float64
  /index/lo_values        int32 or int64
  /index/lo_offsets       int64
  attrs:
    format = "ldetect2-covariance-h5"
    version = 1
    chrom = ...
    start = ...
    end = ...
    position_dtype = "int32" or "int64"
    sorted_by = "lo_hi"
    deduplicated = true
```

`lo_offsets` stores row-group offsets for each `lo_values` entry, so local
search can map a genomic `lo` range to row slices quickly.

### Compression and Chunking

Initial recommendation:

- Use `shuffle=True`.
- Benchmark `lzf` for speed-first workflows.
- Benchmark `gzip` level 1 or 2 for space-first workflows.
- Chunk rows in fixed row-count chunks, for example 64K to 1M rows depending
  on observed partition sizes.

The cache should optimize for bounded local-search reads, not maximum
compression ratio. A slightly larger file is acceptable if it avoids inflating
100 MB archive members into much larger temporary arrays.

### CLI and Store Migration

Add a covariance format option without breaking current `.npz` users:

```text
ldetect2 calc-covariance --covariance-format npz|h5|both
ldetect2 run --covariance-format npz|h5|both
```

Recommended default during migration:

- `npz`: default compatibility mode.
- `h5`: write only HDF5 partition files.
- `both`: write both formats for validation and transition runs.

Update `CovarianceStore.partition_path()` or add a format-aware companion so
loaders can find either `.npz` or `.h5` without hardcoding suffix decisions in
algorithm code.

### Loader Abstraction

Introduce a small partition reader abstraction:

```python
class CovariancePartitionReader(Protocol):
    start: int
    end: int

    def read_all(self) -> CovariancePartition: ...
    def iter_rows(
        self,
        lo_min: int,
        lo_max: int,
        chunk_rows: int,
    ) -> Iterator[CovarianceRowChunk]: ...
    def read_diagonal(self) -> tuple[np.ndarray, np.ndarray]: ...
```

`.npz` readers can initially implement `read_all()` and emulate `iter_rows()`
from loaded arrays. HDF5 readers can implement true chunked reads.

This keeps `LocalSearch` focused on row aggregation and avoids scattering
storage-format checks through the algorithm.

### Local Search HDF5 Flow

For each local-search segment:

1. Determine `lo_min`, `lo_max`, and `hi` constraints from the search window.
2. Use `/index/lo_values` and `/index/lo_offsets` to locate candidate row
   ranges.
3. Read bounded chunks from `/covariance/lo`, `/covariance/hi`, and
   `/covariance/shrink_ld`.
4. Normalize chunks to `r²` using `/index/diag_pos` and `/index/diag_val`.
5. Aggregate into local `sum_vert` and `sum_horiz`.
6. Discard the chunk before reading the next one.

This should make peak memory depend on chunk size plus local accumulators,
rather than full partition array size.

### Validation Plan

Correctness:

- Add tests that write equivalent `.npz` and `.h5` partitions and compare
  loaded arrays.
- Compare HDF5 local-search output with current `.npz` output on synthetic
  single-partition and multi-partition fixtures.
- Run the existing metric and local-search test suite.
- Run the toy integration pipeline.

Performance:

- Re-run one representative `ldetect_original` chromosome, preferably EUR chr10
  or chr11.
- Compare local-search elapsed time and max RSS against current `.npz`.
- Confirm `--subset fourier_ls` output BED is identical to current output.
- Track HDF5 file size versus current `.npz` for 10 MB, 50 MB, and 100 MB
  partitions.

Acceptance criteria:

- No RSS increase relative to current `.npz` path.
- Identical final `fourier_ls` BED for the same inputs.
- HDF5 partition storage does not require keeping `.npz` intermediates unless
  `--covariance-format both` is requested.
- Local-search chunk size is configurable or at least centralized as a single
  tuning constant.

### Implementation Order

1. Add HDF5 dependency behind an optional extra, or detect `h5py` at runtime
   and emit a clear error for `--covariance-format h5` when unavailable.
2. Add format-aware covariance partition path handling.
3. Implement HDF5 writer from the arrays already produced by
   `calc_covariance()`.
4. Implement `.npz` and HDF5 readers behind a shared loader interface.
5. Teach metric and matrix-to-vector paths to read through the interface while
   preserving existing `.npz` behavior.
6. Teach local search to use HDF5 `iter_rows()` for bounded chunk processing.
7. Add `both` mode and equivalence tests.
8. Run representative chromosome validation before changing any defaults.

## Open Questions

- Should HDF5 be a required dependency or an optional `storage` extra?
- Should the first HDF5 implementation store only compact arrays, or also the
  full metadata arrays used by non-compact output?
- What chunk size best balances local-search window reads against compression
  efficiency on real 10-100 MB partition files?
- Should HDF5 become the default only for `ldetect2 run`, or also for
  standalone `calc-covariance`?
