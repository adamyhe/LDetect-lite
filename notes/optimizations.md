# Performance Optimizations

This document summarises the performance improvements applied to `ldetect2` since the initial implementation.

---

## 1. Numba JIT for pairwise LD kernel (`shrinkage.py`)

**Affected code:** `_pairwise_ld_impl` in `src/ldetect2/shrinkage.py`

The inner pairwise LD kernel was decorated with `@_jit` (`numba.njit(cache=True)` when Numba is available, no-op otherwise). The vectorised inner loop uses `np.sum(a * b)` instead of an explicit Python loop (Numba does not support `np.dot` on `uint8` arrays via BLAS).

**Measured speedup:** ~50× over pure Python on a 200-SNP × 400-haplotype matrix. Numba compilation (~300 ms first run) is disk-cached via `cache=True`, so subsequent calls pay no compile cost.

---

## 2. Parallel covariance calculation (`_cli/cmd_run.py`)

**Affected code:** `src/ldetect2/_cli/cmd_run.py`

Covariance partitions are fully independent (each writes to its own `{name}.{start}.{end}.gz` file). The sequential loop was replaced with `concurrent.futures.ProcessPoolExecutor`. The tabix spawn + `calc_covariance` call was extracted into a module-level `_calc_partition(...)` function so it is picklable.

**CLI:** `ldetect2 run --workers N`

**Speedup:** Linear with core count up to the number of partitions (~40 per chromosome).

---

## 3. O(log n) locus index lookup (`local_search.py`, `metric.py`, `matrix_analysis.py`)

**Affected code:** all three modules above

`list.index(value)` performs a linear O(n) scan. Every partition boundary triggers one such lookup to relocate `curr_locus` after the list is updated. Replaced with `bisect.bisect_left` (O(log n)):

```python
# before
curr_locus_index = self.locus_list.index(curr_locus)

# after
i = bisect_left(self.locus_list, curr_locus)
if i < len(self.locus_list) and self.locus_list[i] == curr_locus:
    curr_locus_index = i
else:
    ...  # fallback
```

For a chromosome with ~4 000 loci per partition, the worst case drops from ~4 000 comparisons to ~12.

---

## 4. Float arithmetic by default for local search and metric (`local_search.py`, `metric.py`, `pipeline.py`)

**Affected code:** `LocalSearch`, `Metric`, `find_breakpoints`, CLI flags

The original implementation used `decimal.Decimal` at 50-digit precision for all breakpoint metric comparisons and accumulation. This is ~10–30× slower than native `float` arithmetic with no practical difference in results for typical LD data.

`float` is now the default. Decimal precision is opt-in via `--high-precision`:

```
ldetect2 find-minima --high-precision ...
ldetect2 run --high-precision ...
```

Internally controlled by `use_decimal: bool = False` on `LocalSearch.__init__`, `Metric.__init__`, and `find_breakpoints`.

---

## 5. Parallel local search (`pipeline.py`)

**Affected code:** `_run_local_search`, `_local_search_worker` in `src/ldetect2/pipeline.py`

Each breakpoint's local search is independent. The sequential loop over breakpoints was replaced with `ProcessPoolExecutor`. The inner `_run_single` closure was extracted to a module-level `_local_search_worker(...)` function for picklability.

**CLI:** `ldetect2 find-minima --workers N` and `ldetect2 run --workers N` (shared flag; applies to both covariance and local search).

**Speedup:** Linear with core count up to the number of breakpoints (~50–100 per chromosome).

---

## 6. Eliminate `math.sqrt` per pair in matrix-to-vector conversion (`matrix_analysis.py`)

**Affected code:** `calc_diag_lean` in `src/ldetect2/matrix_analysis.py`

The correlation coefficient was computed as `corr = cov / sqrt(diag_x * diag_y)` and then squared in `_add_corr_coeff`. Since only `r²` is needed, the sqrt is unnecessary:

```python
# before
corr = matrix[x][y] / math.sqrt(diag_x * diag_y)
self._add_corr_coeff(corr, curr_locus)  # squared inside

# after
cov = matrix[x][y]
self._add_r2(cov * cov / (diag_x * diag_y), curr_locus)
```

`_add_corr_coeff` was renamed `_add_r2` to reflect that it receives `r²` directly. The `calc_diag` (heatmap) path still computes `corr` to store `corr_coeff`, so `math.sqrt` is retained there.

For a partition with N loci, this eliminates one `math.sqrt` call per off-diagonal pair processed (~N²/2 calls total per partition).

---

## 8. Historical: NumPy binary covariance partitions (`shrinkage.py`, `io/covariance.py`, `io/partitions.py`)

**Affected code:** `calc_covariance` in `src/ldetect2/shrinkage.py`; `read_partition_into_matrix_lean`, `read_partition_into_matrix` in `src/ldetect2/io/covariance.py`; `CovarianceStore.partition_path` in `src/ldetect2/io/partitions.py`

Partition files were previously moved from gzipped 8-column text (`.gz`) to
compressed NumPy archives (`.npz`). That change was useful for replacing Python
row parsing with typed arrays, but it has since been superseded by the HDF5
partition format described below.

**Write side:** `calc_covariance` previously iterated over all pairs in Python with an f-string per row. Since `ii`, `jj`, `all_pos`, `ds2_arr` etc. are already NumPy arrays at write time, a single `np.savez_compressed` replaces the loop entirely:

```python
np.savez_compressed(
    output_path,
    i_pos=pos_arr[ii], j_pos=pos_arr[jj],
    i_gpos=gpos_flat[ii], j_gpos=gpos_flat[jj],
    naive_ld=d_naive_arr, shrink_ld=ds2_arr,
    i_id=rs_arr[ii], j_id=rs_arr[jj],
)
```

**Read side:** `read_partition_into_matrix_lean` replaces `gzip.open` + `csv.reader` + per-row `int`/`float` parsing with `np.load` + iteration over pre-typed arrays. A private `_insert_lean_values` helper separates value-level insertion from string parsing, keeping `insert_into_matrix_lean(row: list[str])` as a public API for tests and external callers.

**Estimated speedups** (based on 226K-row test partition, 5.9 MB compressed text):

| Operation | Before | After | Speedup |
|---|---|---|---|
| Write | ~1–3 s (Python loop) | ~0.01–0.05 s (`np.savez_compressed`) | ~50–200× |
| Read (I/O + parse) | ~0.7–1.3 s | ~0.05–0.25 s | ~4–6× |
| File size | 5.9 MB (.gz text) | ~2–3 MB (.npz) | ~2× smaller |

**Superseded:** Current production intermediates are `.h5` covariance
partitions. Existing `.gz` and `.npz` partition files should be regenerated by
re-running `calc-covariance` or `ldetect2 run`.

---

## 7. Local variable caching for hot inner loop (`matrix_analysis.py`)

**Affected code:** `calc_diag_lean` and `calc_diag` in `src/ldetect2/matrix_analysis.py`

`self.matrix` and `self.locus_list` are attribute lookups that Python resolves via `__getattribute__` on every access. In the innermost loop, these are accessed dozens of times per locus. Binding them to local variables before the outer `while` loop reduces attribute lookup overhead:

```python
matrix = self.matrix
locus_list = self.locus_list

while curr_locus <= end_locus:
    ...
    x = locus_list[curr_locus_index]
    ...
    if x in matrix and y in matrix[x]:
```

---

## Future Work: Optimization Opportunities and Memory Risk

The remaining pipeline bottlenecks are mostly data-flow and repeated
precomputation issues. Several possible optimizations would improve runtime but
could raise peak memory, which has been a previous operational risk for
whole-chromosome runs.

### Updated remote profiling result

Remote EUR chr10/chr11/chr13/chr21/chr22 profiling after the compact HDF5
layout/read-cache follow-up shows that the Step 2 memory fix still holds and
the local-search read regression is mostly recovered. Whole-run RSS remains
below 1 GiB on all downloaded chromosomes, including chr11. The 65,536-row
dataset chunks plus per-precompute HDF5 reader reuse cut chr11 local-search
HDF5 read time from about 996 s to about 563 s.

| Chrom | Wall time | Max RSS | Local search | Precompute | Search |
|---|---:|---:|---:|---:|---:|
| chr10 | 1714.45 s | 0.596 GiB | 334.47 s | 333.46 s | 0.117 s |
| chr11 | 3868.00 s | 0.825 GiB | 856.66 s | 855.49 s | 0.118 s |
| chr13 | 1166.70 s | 0.476 GiB | 208.00 s | 207.29 s | 0.120 s |
| chr21 | 338.73 s | 0.397 GiB | 64.42 s | 64.08 s | 0.036 s |
| chr22 | 394.59 s | 0.412 GiB | 68.59 s | 68.31 s | 0.047 s |

The next refreshed logs use the previous optimization sweep, including the
duplicate-merge path and horizontal `add.reduceat()` grouping, but not the
dense local-search accumulator migration. They should therefore be read as a
validation profile for the sweep before dense arrays. RSS remains bounded, and
the horizontal rewrite is a clear win, but duplicate tracking regressed enough
to offset much of that gain on the larger chromosomes.

| Chrom | Wall time | Max RSS | Local search | HDF5 read | Dedup | Horizontal |
|---|---:|---:|---:|---:|---:|---:|
| chr10 | 1872.15 s | 0.587 GiB | 342.27 s | 144.85 s | 127.02 s | 12.69 s |
| chr11 | 3922.00 s | 0.837 GiB | 904.80 s | 563.47 s | 243.58 s | 16.44 s |
| chr13 | 1400.58 s | 0.488 GiB | 304.53 s | 116.04 s | 124.58 s | 13.59 s |
| chr21 | 320.68 s | 0.405 GiB | 45.94 s | 18.99 s | 13.60 s | 2.92 s |
| chr22 | 405.62 s | 0.412 GiB | 82.09 s | 33.14 s | 30.44 s | 4.22 s |

Compared with the compact-layout/read-cache baseline above, HDF5 read time is
essentially unchanged on chr10/chr11, horizontal aggregation improved to about
0.17x-0.31x of the prior cost, and deduplication worsened to about 1.2x-2.9x
of the prior cost. On chr11, `dedup_merge_seconds` accounts for 239.39 s of
243.58 s total dedup time, so the sorted merge implementation is the next
local-search target unless dense-array validation changes the phase balance.
The dense accumulator fields are absent from this profile by design; remote
dense validation remains pending.

Against the previous bounded compact profile, local-search wall time improved
to 0.76x on chr10, 0.67x on chr11, 0.55x on chr13, 1.01x on chr21, and 0.70x
on chr22. HDF5 read time improved to 0.56x, 0.57x, 0.41x, 0.70x, and 0.53x
respectively. RSS was effectively unchanged and still comfortably bounded.

Refreshed chr11 raw-log split:

| Phase | Seconds | Memory finding |
|---|---:|---|
| Step 2 covariance generation | 1338 s | whole-run max RSS now 0.837 GiB |
| Step 3 matrix-to-vector | 1025 s | parent max RSS 465.3 MiB |
| Fourier metric/minima setup | 126 s | filter/minima work before first metric |
| Fourier metric | 260 s | bounded streaming pass |
| Fourier local search | 905 s | previous-sweep profile; duplicate tracking regressed |
| Final Fourier-LS metric | 264 s | bounded streaming pass |

Step 2 generated 378 compact HDF5 partitions for chr11 with `workers=4`.
The bounded writer retained about 8.81B compact rows total; the largest
partition retained about 677.9M rows. Worker debug checkpoints peaked around
846 MiB current RSS and 845 MiB max RSS, validating that full-partition pair
index and mapped-position arrays are gone from the compact path. Compact
datasets used `dataset_chunk_rows=65536` with `write_chunk_rows=1000000`.

Step 3 memory and runtime improved substantially:

| Chrom | Step 3 seconds | Step 3 max RSS |
|---|---:|---:|
| chr11 | 1023 s | 452.6 MiB |
| chr21 | 58 s | 371.7 MiB |
| chr22 | 76 s | 372.2 MiB |

Local search logically requested 43.03B rows on chr11, but the streamed HDF5
path read 16.62B rows, filtered them to 5.30B rows, and deduplicated to 1.22B
candidate rows. HDF5 reads/decompression remain the largest chr11 local-search
bucket, but duplicate tracking, horizontal aggregation, and normalization are
now more visible:

| Chrom | HDF5 read | Dedup | Horizontal | Normalize |
|---|---:|---:|---:|---:|
| chr11 | 563.17 s | 134.21 s | 77.46 s | 52.14 s |
| chr21 | 23.55 s | 10.99 s | 17.54 s | 9.08 s |
| chr22 | 27.86 s | 13.24 s | 14.94 s | 9.26 s |

The current local-search optimization targets have shifted. HDF5-backed local
search now streams segment row chunks directly into vertical/horizontal
accumulators instead of materializing one full active row range per segment.
Keep the compact HDF5 layout/read-cache change. The previous sweep shows that
horizontal grouping should be kept, while duplicate tracking needs either a
faster merge implementation, a guarded fallback to the earlier membership path,
or a profile-driven hybrid. Continue deferring `_search_array()` optimization
and JIT. See `notes/local-search-memory-speed-handoff.md` for the detailed
handoff.

Local-search assembly should continue to consume covariance rows through a
canonical physical-pair stream boundary. The HDF5 implementation uses a
per-locus duplicate tracker to preserve first-retained-pair precedence. The
point is compatibility insurance: if the known duplicate-position divergence
becomes important, the duplicate-aware fix should live at that stream boundary
rather than inside every aggregation kernel.

### chr10/chr11 memory failure update

Earlier chr10/chr11 remote failures, even with one chromosome at a time on
128 GiB nodes, showed that memory pressure was not only local-search
precompute. The normal float metric path and Step 3 matrix-to-vector path also
materialized large normalized pair arrays.

Implemented follow-up:

- `metric_from_files()` streams metric calculation from covariance partitions;
- `Metric.calc_metric()` uses the streaming array path by default;
- normal `find_breakpoints()` no longer eagerly loads/reloads metric covariance
  arrays unless an explicit caller-supplied covariance cache is provided.
- HDF5-backed matrix-to-vector now chunks `_r2_rows()` normalization and
  center-locus accumulation instead of materializing full-partition arrays.

Expected effect:

- lower peak RSS for whole chromosomes, especially chr10/chr11;
- no change to selected breakpoint loci or BED output;
- possible runtime penalty from rereading partitions, accepted when the
  alternative is memory exhaustion.

Validation priority:

1. keep remote chr21/chr22 as small iteration targets;
2. rerun chr10 and chr11 one at a time after Step 2 memory or local-search
   runtime changes;
3. compare RSS and wall time against the current chunked HDF5 baseline.

HDF5 is now the production intermediate covariance format. Existing `.npz`
intermediates are not supported by production paths; regenerate covariance
outputs after this migration. Metric calculation, local search, and
matrix-to-vector now share an HDF5 chunked reader.

The HDF5 writer enforces sorted canonical `(lo, hi)` rows and `lo_offsets`
indexes. `calc_covariance()` uses a validated fast path that skips redundant
write-time sort/dedup allocations, and duplicate physical VCF positions are
collapsed before pairwise LD so common duplicate-position input does not force
the memory-heavy writer fallback.

Step 2 covariance generation is now bounded in the compact production path.
The writer counts retained pairs by lower SNP and writes compact HDF5 rows in
bounded sorted chunks, avoiding full-partition pair-index and mapped-position
arrays. The full/debug covariance schema still uses the materialized writer.
The compact writer now separates write batching from HDF5 storage layout:
bounded pair generation keeps 1,000,000-row write batches, while compact
`lo`/`hi`/`shrink_ld` datasets default to 65,536-row storage chunks and record
both values in HDF5 attrs/debug logs.
The latest chr11 whole-run max RSS is 0.837 GiB. See
`notes/local-search-memory-speed-handoff.md` for the HDF5 contract and current
read-performance checklist.

### Post-segment-assembly optimization order

After bounded compact writes, the next targets should be chosen from the phase
ratios, not guessed. The overall optimization order from the refreshed chr11
logs is:

1. final-output parity checks for BED/JSON/HDF5 against the previous compact
   baseline;
2. duplicate tracking validation after reverting the sorted merge path. The
   previous-sweep profile showed dedup was worse than the read-cache baseline
   because `dedup_merge_seconds` dominated the bucket;
3. dense-window local-search accumulation validation. The dense accumulator
   migration is implemented locally but was not included in the latest remote
   logs;
4. normalization, especially repeated diagonal lookup via `np.searchsorted()`;
5. metric recomputation around local search, now routed through streaming
   metrics and worth timing separately if wall time shifts there;
6. Step 3 matrix-to-vector runtime, while preserving the bounded helper-scope
   HDF5 path;
7. group load/canonicalization outside breakpoint rows, using
   `group_total_seconds` and `local_search_unaccounted_seconds`;
8. filter-width search count caching with `{width: minima_count}` only;
9. HDF5 compression or chunk-size tuning only if future remote profiles show
   clear walltime or RSS regressions.

For each profile, inspect HDF5 chunk layout/compression, local-search
`hdf5_read_seconds / precompute_seconds`, `hdf5_reader_open_count`,
`hdf5_reader_reuse_count`, Step 2 retained pair count, worker RSS, output file
size, `horizontal_seconds / precompute_seconds`,
`normalize_seconds / precompute_seconds`, `group_total_seconds`, and
`local_search_unaccounted_seconds`. See the detailed checklist in
`notes/local-search-memory-speed-handoff.md`.

### Main Runtime Optimization Plan

The next work should optimize bounded dense-window computation rather than
storage layout. Keep the compact 65,536-row HDF5 dataset chunks and
per-precompute reader reuse unless a future remote profile regresses.

1. **Local-search duplicate tracking.**
   The attempted sorted two-pointer merge was reverted. It moved set-union work
   from NumPy's C implementation into a Python loop and the latest
   previous-sweep remote logs showed the expected regression: chr11 dedup rose
   from 134.21 s to 243.58 s, with 239.39 s in the merge path. Keep the
   `np.isin()`/`np.union1d()` path for now and validate that the next profile
   returns dedup near the compact-layout/read-cache baseline.
2. **Dense local-search accumulators plus horizontal grouping.**
   Replace per-breakpoint `sum_vert_by_locus`/`sum_horiz_by_locus`
   dictionaries with dense arrays scoped to the active locus window. Benchmark
   sorting `row_hi` plus `np.add.reduceat()` against the current
   `np.unique(..., return_inverse=True)`/`np.bincount()` path. This targets
   about 77 s horizontal aggregation and some of the 52 s normalization cost on
   chr11.
   Status: implemented locally with per-breakpoint dense accumulators,
   sorted-index `add.reduceat`, and dense lookup/accumulation profiling. The
   latest remote logs do not include this migration; validate it separately.
3. **Metric-pass instrumentation and metadata reuse.**
   The two chr11 streaming metric passes take about 8.8 minutes total. Add
   partition read, normalization, crossing-pair, row-read, and row-crossing
   timings first. Reuse only cheap diagonal/locus metadata or breakpoint lookup
   structures; do not retain normalized covariance rows.
   Status: instrumentation implemented locally; reuse/fusion pending profiling.
4. **Step 3 matrix-to-vector runtime instrumentation and bounded dense
   accumulation.**
   Step 3 is still about 1023 s on chr11. Instrument
   `_process_diag_vector_partition_hdf5()` for HDF5 read, normalization, and
   center-locus accumulation time. Test dense per-partition accumulators within
   the helper scope while preserving helper-return RSS release.
   Status: instrumentation implemented locally; additional accumulator changes
   pending profiling.
5. **Filter-width count cache.**
   Add a `{width: minima_count}` cache inside
   `custom_binary_search_with_trackback()`. Cache counts only; never cache
   smoothed arrays. This is low-risk but lower payoff than the dense-window and
   streaming-pass work.
   Status: implemented locally in `FlexibleBoundedAccessor`.

Validation order for each implemented item:

1. Unit parity for duplicate rows, overlapping partitions, metric sums, and
   vector outputs as applicable.
2. Focused local pytest plus `ruff check` on touched files.
3. Remote chr21/chr22, then chr10/chr11.
4. Acceptance: BED/JSON/HDF5 validation unchanged, max RSS remains near the
   current sub-1 GiB profile, and the target phase seconds improve without
   moving time into another larger bucket.

### Lower memory-risk candidates

#### Cache filter-width search counts

**Affected code:** `custom_binary_search_with_trackback` in
`src/ldetect2/find_minima.py`; `apply_filter_get_minima` in
`src/ldetect2/filters.py`

The filter-width search may evaluate the same width more than once during
binary search and trackback. A small cache of `{width: minima_count}` would
avoid repeated Hanning-window construction, convolution, and minima extraction.

Memory guidance: cache only integer minima counts. Do not cache full smoothed
arrays or filter result dicts unless profiling shows the memory increase is
acceptable.

#### Avoid vector file reread in end-to-end `run`

**Affected code:** `MatrixAnalysis.calc_diag_lean`,
`write_diag_vector_array`, and `find_breakpoints`

The end-to-end CLI writes `vector-{chrom}.txt.gz` and then immediately rereads
it in `find_breakpoints`. An internal fast path could return the vector arrays
directly while still writing the restartable vector file for users.

Memory guidance: the vector is small compared with covariance partitions, so
this should be low to moderate risk. Keep the current file-backed path as the
default fallback until real chromosome memory profiles confirm the peak impact.

### Moderate memory-risk candidates

#### Group local-search windows by overlapping partitions

**Affected code:** `_run_local_search` in `src/ldetect2/pipeline.py`;
`LocalSearch` in `src/ldetect2/local_search.py`

Adjacent breakpoint searches often touch overlapping covariance partitions.
Grouping windows by partition range could reduce repeated partition loads and
precomputation.

Memory guidance: implement groups so partitions are loaded, processed, and
released within a bounded scope. Avoid retaining all grouped windows or all
partition arrays at once.

Status: partially implemented for the single-process normal-float path by
grouping breakpoints with identical partition ranges. Keep future grouping
bounded by partition range; do not promote this into a chromosome-wide cache.

#### Dense local accumulators

**Affected code:** `src/ldetect2/_util/covariance_array.py` and
`src/ldetect2/local_search.py`

Array local search still accumulates `sum_vert_by_locus` and
`sum_horiz_by_locus` in dictionaries before materializing arrays. Replacing
those with dense arrays scoped to the current local-search locus window could
reduce Python object overhead without retaining extra chromosome-wide state.

Memory guidance: only allocate dense arrays for the current breakpoint window
or current bounded partition group. Do not allocate chromosome-wide dense
accumulators.

### Discarded or Default-No Optimizations

These ideas were considered, but should not be implemented as default
optimizations because previous profiling and design review indicate memory or
storage pressure would outweigh the likely runtime win.

#### Pass a chromosome covariance cache through `ldetect2 run` by default

**Affected code:** `src/ldetect2/_cli/cmd_run.py`,
`MatrixAnalysis.calc_diag_lean`, and `find_breakpoints`

Both matrix-to-vector and breakpoint finding can accept a chromosome covariance
cache, but the end-to-end `run` command currently does not build and pass one.
Passing a cache could avoid repeated partition reads across vector generation,
metrics, and local search.

Decision: do not make this a default optimization. `load_chromosome_covariance()`
retains raw partition arrays plus metric arrays. On whole chromosomes this may
hold most or all covariance data in RAM, exactly where memory is already the
limiting risk. Revisit only as an explicit high-memory/debug mode or after a
memory-mapped storage layer exists.

#### Unify full and metric-only covariance caches

**Affected code:** `load_chromosome_covariance`,
`load_metric_covariance`, and `metric_from_arrays` in
`src/ldetect2/_util/covariance_array.py`

`load_metric_covariance()` intentionally builds a slimmer cache for metrics,
while `load_chromosome_covariance()` keeps raw partition arrays needed by
matrix-to-vector and local search. A unified cache could reduce duplicate reads
in fast runs.

Decision: do not unify these caches. Preserving the metric-only path is
important for low-memory runs. A unified cache would retain data that many
pipeline stages do not need and would make worker parallelism more dangerous.

Updated decision: normal metric calculation should now prefer streaming from
files rather than any resident chromosome-wide metric cache. Keep
`load_metric_covariance()` only for tests, explicit caller-supplied caches, and
small/debug workflows.

#### Parallelize cached local search by passing large caches to workers

**Affected code:** `_run_local_search` in `src/ldetect2/pipeline.py`

The current code uses cached in-memory array local search in a single process.
This avoids pickling or copying a large chromosome cache into multiple worker
processes.

Decision: do not pass large covariance caches into a `ProcessPoolExecutor`.
On many platforms this can copy the cache per worker. If parallel local search
is revisited, use whole partition-group workers with bounded inputs, shared
memory, or memory-mapped/chunked storage.

#### Precompute full partition-level normalized rows in memory

**Affected code:** `src/ldetect2/_util/covariance_array.py` and
`src/ldetect2/local_search.py`

Local search repeatedly derives `r²` from `shrink_ld` and diagonal values, so
precomputing normalized rows is tempting. A full in-memory normalized row cache,
however, duplicates information already stored in `shrink_ld` plus diagonals
and can approach another covariance-sized array set.

Decision: do not keep full normalized rows in memory. If normalized rows are
needed, compute them in bounded chunks or store them in a chunked/memory-mapped
format. The current next step is finer precompute instrumentation, not another
large resident cache.

#### Raw `.npy` side caches for local search

**Affected code:** covariance storage and local-search loaders

Raw `.npy` arrays would allow memory mapping, but they would store the compact
hot-path arrays at roughly 16 bytes per covariance row, or about 32 bytes per
row if both original and sorted local-search order are retained. For existing
10-100 MB compressed `.npz` partitions, that likely means a 2-6x disk
multiplier for the minimal cache and potentially 4-12x with a sorted duplicate.

Decision: do not add raw `.npy` side caches. HDF5 now provides the chunked
storage layer without creating parallel raw side-cache files.

#### Historical: delay HDF5 until after non-storage memory validation

**Affected code:** covariance storage, metric readers, local-search loaders

Earlier planning treated HDF5 as a later local-search optimization. The
chr10/chr11 failures changed the priority, and HDF5 has now been implemented.

Decision: HDF5 is no longer delayed. The goal remains bounded chunk reads for
metrics, local search, and matrix-to-vector, not a large resident cache.

#### JIT candidate scoring before precompute instrumentation

**Affected code:** `LocalSearch._search_array()`

EUR chr21/chr22 profiling shows candidate scoring is effectively free:
`search_seconds` was about 0.03 seconds per chromosome while precompute took
70-123 seconds.

Decision: do not optimize or JIT `_search_array()` now. Focus on precompute
instrumentation, row-volume reduction, and bounded storage/layout changes.
