# Performance Optimizations

**Human-oriented reference.** This documents completed, shipped performance work only, for readers who want to understand *why* the pipeline is fast. It is not a task list or a log — in-progress investigation notes and design history live under `notes/logs/`; distilled current-status findings live under `notes/findings/`.

This document summarises the performance improvements applied to `ldetect-lite` since the initial implementation.

---

## 1. Numba JIT for pairwise LD kernel (`shrinkage.py`)

**Affected code:** `_pairwise_ld_impl` in `src/ldetect_lite/shrinkage.py`

The inner pairwise LD kernel was decorated with `@_jit` (`numba.njit(cache=True)` when Numba is available, no-op otherwise). The vectorised inner loop uses `np.sum(a * b)` instead of an explicit Python loop (Numba does not support `np.dot` on `uint8` arrays via BLAS).

**Measured speedup:** ~50× over pure Python on a 200-SNP × 400-haplotype matrix. Numba compilation (~300 ms first run) is disk-cached via `cache=True`, so subsequent calls pay no compile cost.

---

## 2. Parallel covariance calculation (`_cli/cmd_run.py`)

**Affected code:** `src/ldetect_lite/_cli/cmd_run.py`

Covariance partitions are fully independent (each writes to its own `{name}.{start}.{end}.h5` file). The sequential loop was replaced with `concurrent.futures.ProcessPoolExecutor`. The tabix spawn + `calc_covariance` call was extracted into a module-level `_calc_partition(...)` function so it is picklable.

**CLI:** `ldetect run --workers N`

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
ldetect find-minima --high-precision ...
ldetect run --high-precision ...
```

Internally controlled by `use_decimal: bool = False` on `LocalSearch.__init__`, `Metric.__init__`, and `find_breakpoints`.

---

## 5. Parallel local search (`pipeline.py`)

**Affected code:** `_run_local_search`, `_local_search_worker` in `src/ldetect_lite/pipeline.py`

Each breakpoint's local search is independent. The sequential loop over breakpoints was replaced with `ProcessPoolExecutor`. The inner `_run_single` closure was extracted to a module-level `_local_search_worker(...)` function for picklability.

**CLI:** `ldetect find-minima --workers N` and `ldetect run --local-search-workers N`.

**Speedup:** Linear with core count up to the number of breakpoints (~50–100 per chromosome).

---

## 6. Eliminate `math.sqrt` per pair in matrix-to-vector conversion (`matrix_analysis.py`)

**Affected code:** `calc_diag_lean` in `src/ldetect_lite/matrix_analysis.py`

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

## 7. Indexed HDF5 covariance partition files (`shrinkage.py`, `io/covariance_hdf5.py`, `io/partitions.py`)

**Affected code:** `calc_covariance` in `src/ldetect_lite/shrinkage.py`; HDF5 readers/writers in `src/ldetect_lite/io/covariance_hdf5.py`; `read_partition_into_matrix_lean`, `read_partition_into_matrix` in `src/ldetect_lite/io/covariance.py`; `CovarianceStore.partition_path` in `src/ldetect_lite/io/partitions.py`

Partition files are stored as indexed HDF5 files (`.h5`). HDF5 gives chunked reads, persistent row indexes, and a compact schema that supports restartable production runs without materializing full partitions.

**Write side:** `calc_covariance` previously iterated over all pairs in Python with an f-string per row. Current writers emit typed HDF5 datasets for canonical position pairs and shrinkage LD values, plus diagonal and `lo` row-offset indexes:

```python
write_covariance_partition_hdf5(
    output_path,
    i_pos=pos_arr[ii],
    j_pos=pos_arr[jj],
    shrink_ld=ds2_arr,
    i_gpos=gpos_flat[ii],
    j_gpos=gpos_flat[jj],
    naive_ld=d_naive_arr,
    i_id=rs_arr[ii],
    j_id=rs_arr[jj],
)
```

**Compact write side:** `ldetect run` defaults to `--covariance-cache compact`, which writes only `lo`, `hi`, `shrink_ld`, diagonal entries, and row indexes. The compact writer streams bounded chunks into HDF5 so large partitions do not need full `i_pos`/`j_pos`/metadata arrays in memory.

**Read side:** matrix-to-vector, metric, and local-search paths read typed HDF5 arrays and use the `lo_offsets` index for bounded row scans.

**Estimated impact** (based on representative partition profiling):

| Operation | Before | After | Speedup |
|---|---|---|---|
| Write | row-oriented Python formatting | typed HDF5 dataset writes/chunk appends | removes per-row Python formatting |
| Read | row-oriented parsing and broad scans | typed HDF5 reads plus indexed scans | avoids parsing and full scans |
| Working memory | full pair arrays plus metadata | compact chunk streaming in `run` | bounded by chunk size for compact cache |

---

## 8. Local variable caching for hot inner loop (`matrix_analysis.py`)

**Affected code:** `calc_diag_lean` and `calc_diag` in `src/ldetect_lite/matrix_analysis.py`

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

## 9. zstd covariance compression (`io/covariance_hdf5.py`)

**Affected code:** `write_covariance_partition_hdf5`, `write_compact_covariance_partition_hdf5_chunks`/`_append` in `src/ldetect_lite/io/covariance_hdf5.py`

`shrink_ld` dominates covariance partition storage (82% of file size in measurement) and barely compresses under `lzf` (compression ratio 0.972). Switched the default HDF5 compression codec to `zstd` (via the new `hdf5plugin` dependency), which strictly dominates both `lzf` and `gzip` at full float64 precision — smaller, faster to write, and faster to read, with no tradeoff.

**CLI:** `--covariance-compression {lzf,zstd}` on `ldetect run` and `ldetect calc-covariance` (default: `zstd`).

**Measured impact:** validated on the full 1000G dataset (22 chromosomes x 3 populations): 66/66 chromosome x population combinations show byte-identical downstream vectors, exact breakpoints, and exact BED boundaries (compression is lossless), a 12.4% covariance-directory size reduction (1000.65 GB -> 876.43 GB), and a 1.2x aggregate wall-clock speedup.
