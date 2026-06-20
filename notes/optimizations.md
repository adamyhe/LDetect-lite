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

## 8. NumPy binary format for covariance partition files (`shrinkage.py`, `io/covariance.py`, `io/partitions.py`)

**Affected code:** `calc_covariance` in `src/ldetect2/shrinkage.py`; `read_partition_into_matrix_lean`, `read_partition_into_matrix` in `src/ldetect2/io/covariance.py`; `CovarianceStore.partition_path` in `src/ldetect2/io/partitions.py`

Partition files are now stored as compressed NumPy archives (`.npz`) instead of gzipped 8-column text (`.gz`).

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

**Breaking change:** Existing `.gz` partition files must be regenerated by re-running `calc-covariance` or `ldetect2 run`.

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
