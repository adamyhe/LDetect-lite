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

**Affected code:** `_run_local_search`, `_local_search_worker`, `_local_search_group_worker` in `src/ldetect_lite/pipeline.py`

Each breakpoint's local search is independent. The sequential loop over breakpoints was replaced with `ProcessPoolExecutor`. The inner `_run_single` closure was extracted to a module-level `_local_search_worker(...)` function for picklability.

**CLI:** `ldetect find-minima --workers N` and `ldetect run --local-search-workers N`.

**Speedup:** Linear with core count up to the number of breakpoints (~50–100 per chromosome).

**Memory fix:** the multi-worker path originally submitted one task per breakpoint, each independently loading its own covariance partitions from disk with no sharing. Concurrent breakpoints whose windows fell in the same region redundantly reloaded the same large partition, so peak memory scaled with in-flight *breakpoints*, not workers — this caused an OOM `BrokenProcessPool` crash on a large real-world run. Fixed by reusing the same partition-grouping already used by the single-worker path (`_group_local_search_tasks`): the process pool now submits one task per *group* of breakpoints that share partition bounds (`_local_search_group_worker`), loading each partition once per group and running that group's breakpoints sequentially within the worker, so peak memory scales with concurrent workers, not concurrent breakpoints. Verified to produce identical loci/metrics to the single-worker path (`tests/integration/test_pipeline.py::test_find_breakpoints_multiworker_matches_single_worker`).

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

---

## 10. cyvcf2 VCF/BCF I/O (`shrinkage.py`)

**Affected code:** `calc_covariance` in `src/ldetect_lite/shrinkage.py`; call sites in `src/ldetect_lite/_cli/cmd_run.py` and `src/ldetect_lite/_cli/cmd_covariance.py`

`calc_covariance` previously read genotypes via a per-partition `tabix -h <path> <region>` subprocess piped into a naive per-line text parser (`str.split` on each VCF row). Replaced with [cyvcf2](https://github.com/brentp/cyvcf2), a C-extension VCF/BCF reader backed by htslib: `cyvcf2.VCF(path, samples=...)`, region-restricted via `vcf(region)`. This removes the external `tabix` process spawn on every partition and replaces `str.split`-based genotype parsing with htslib's C bindings. `cyvcf2` is a small dependency (~2.2 MB installed; htslib is statically bundled, no system package required).

Because cyvcf2's region-fetch API is format-agnostic given the right index (`.tbi` for `.vcf.gz`, `.csi` for `.bcf`), BCF input support falls out of the rewrite rather than needing separate code — `ldetect calc-covariance --reference-panel panel.bcf` works with no additional implementation.

This also fixed an untested gap: previously, a requested individual missing from the VCF header silently produced empty output (every row dropped, no error). `calc_covariance` now raises `ValueError` naming every missing individual up front.

**Measured impact:** chr21, EUR panel, full `ldetect run` pipeline. VCF.gz and BCF input (identical content) produced byte-identical output (vector sha256, breakpoints, BED boundaries all exact — 23/23 loci match), confirming BCF correctness at real chromosome scale. BCF input was also faster and lower-memory than VCF.gz: 168.9s vs. 189.2s wall-clock (~11% faster) and 4.21 GB vs. 5.99 GB peak RSS (~30% less), consistent with skipping gzip decompression in favor of BCF's binary encoding. For large reference panels, prefer `.bcf`/`.csi` input over `.vcf.gz`/`.tbi` where practical.

**CLI:** `ldetect calc-covariance --reference-panel PATH --region CHROM:START-END` (direct indexed file); `--reference-panel` omitted reads from stdin instead, unchanged from before.

---

## 11. Thread-parallel filter-width search (`find_minima.py`)

**Affected code:** `_find_end`, `_trackback`, `custom_binary_search_with_trackback` in `src/ldetect_lite/find_minima.py`; `find_breakpoints` in `src/ldetect_lite/pipeline.py`

Profiling a real chromosome run (`examples/ldetect_original/plots/EUR-chr21-timeline.pdf`) surfaced a large, previously-uninstrumented single-threaded span between step 3 (matrix-to-vector) and step 4's metric computation: ~34 of ~64 total seconds, flat single-core CPU and RSS. Root cause: `custom_binary_search_with_trackback` (finding the Hanning filter width that yields a target breakpoint count) makes ~40 sequential calls to `apply_filter`, each running `scipy.ndimage.convolve1d` — a direct (non-FFT) convolution costing O(N·width) — at widths in the thousands.

Two of the search's three phases evaluate a boundable, predictable set of candidates per round and were thread-parallelized without changing any numerics: `_find_end` (exponential doubling search) batches up to `search_workers` doubling candidates per round; `_trackback` (coarse/fine refinement sweep) batches each round's candidate window in chunks of `search_workers`. Both apply the exact same first-match-wins decision rule to the concurrently-computed results, in the same order, so the returned width is identical to the sequential result — confirmed empirically that `scipy.ndimage.convolve1d`/`argrelextrema` release the GIL enough for real `ThreadPoolExecutor` speedup (measured ~5x with 8 threads on 8 independent calls).

The core binary search (`find_le_ind`) is **not** parallelized: each step is adaptive on the previous comparison, so it can't be pre-batched the same way without a different (k-ary search) algorithm, and it's a shared utility used elsewhere — out of scope for this pass.

An FFT-based convolution (`scipy.signal.fftconvolve`, O(N log N) instead of O(N·width)) was tried and reverted: it is numerically unsafe for this pipeline. Direct convolution produces bit-identical output across flat/constant stretches of the input vector, so the downstream minima detector's strict `<` comparison never fires there; FFT convolution's rounding error is not shift-invariant and injects distinct noise at every position, breaking exact ties and manufacturing spurious minima (confirmed: 1 real minimum became 23 detected on a synthetic flat-plateau test). Since the search hunts for a width producing an exact minima count, spurious minima anywhere could converge to a materially different breakpoint set. See `notes/logs/multicore-utilization-filter-width-search.md` for the full investigation.

**CLI:** reuses `find_breakpoints`'s existing `workers` parameter (`ldetect find-minima --workers N`, `ldetect run --local-search-workers N`) — safe to share since local search hasn't started yet at the point the filter-width search runs.

---

## 12. numba direct convolution for filter-width search (`filters.py`)

**Affected code:** `_reflect_index`, `_pad_reflect`, `_convolve1d_reflect`, `apply_filter` in `src/ldetect_lite/filters.py`

#11's thread-parallelization sped up two of the filter-width search's three phases, but left the core binary search (`find_le_ind`) sequential and paying the full O(N·width) `scipy.ndimage.convolve1d` cost per call (~13s of a ~20s real chr21 run). Replaced `convolve1d` with a hand-written direct convolution compiled via numba (`@njit(nogil=True, fastmath=True, cache=True)`) — same O(N·width) algorithm and same shift-invariant summation structure as scipy's, so it stays flat-region-safe *by construction* (a constant input run still produces bit-identical output at every position, so the downstream minima detector's strict `<` never fires spuriously there), unlike the FFT approach in #11.

`numpy.pad` is not numba-jittable, so the `mode='reflect'` boundary (== `numpy.pad(..., mode="symmetric")`, edge value repeated) is replicated via a closed-form reflect-index formula that correctly cycles when `width >= len(array)` (real widths here can exceed vector length).

**`fastmath=True` is required for this to be a net win at all**, not just a nicety: without it, LLVM does not auto-vectorize the reduction loop and the compiled kernel measured ~2x *slower* than `scipy.ndimage.convolve1d`; with it, ~2x *faster*. `fastmath` permits floating-point reassociation, but that reassociation is fixed at compile time and applied identically at every output position — verified empirically that a constant input run still produces bit-identical output with `fastmath=True` (flat-region safety is a property of *shift-invariant application*, not of any particular summation order, so this doesn't reopen the FFT-style risk).

**`nogil=True` is required to not regress #11's threading**: numba does not release the GIL by default, and #11's `ThreadPoolExecutor`-based `search_workers` parallelism only overlaps real work if the GIL is actually released during each call.

A genuinely near-tied edge case was found and fixed during validation, not in production code: a pre-existing test (`test_apply_filter_larger_width_fewer_minima`) used a periodic synthetic signal at a width where scipy's *own* result was already at an exact tie (20 minima at both compared widths, not a real margin). Minima count is not monotonic in width for this signal at intermediate widths — confirmed by sweeping widths 2 through 95 under plain scipy: counts go 20, 18, 15, 21, 25, 24, 20, 18, 9, 3, non-monotonic until asymptotically far apart. Any two numerically-non-identical implementations (not specific to numba — even two scipy versions or BLAS builds) can disagree on exact minima count when the smoothed *output* has near-zero local variation from the signal's own structure, which is a distinct, narrower failure mode than the FFT bug's global-noise-on-constant-*input* problem. Fixed by widening the test's compared widths to a decisive, non-fragile margin (2 vs. 95: 20 vs. 3, agreeing exactly under both implementations). Validated this isn't a realistic production risk: 0 mismatches across 80 randomized trials of noisy random-walk vectors (mimicking real covariance-sum vector structure) at production-representative widths up to 9,000.

**Measured impact:** ~2x per-call speedup over `scipy.ndimage.convolve1d` at realistic widths (8,000-18,000), multiplicative with #11's ~5x threading speedup — combined ~10x over the original sequential-scipy baseline on an 8-candidate batch, measured on real chr21-scale synthetic data. Verified exact minima-index equivalence to scipy (not just numerical closeness) on the flat-plateau fixtures, 80 randomized realistic-noise trials, and the full existing test suite unmodified.

**Fallback if numba is unavailable**: `apply_filter` checks `_HAVE_NUMBA` and calls `scipy.ndimage.convolve1d` directly in that case, rather than an un-jitted `_convolve1d_reflect` (a pure-Python O(N·width) triple-nested loop — ~10^8 iterations at production widths, catastrophically slow rather than just non-optimal). Numba is a hard dependency (`pyproject.toml`), so this path should be unreachable in a correctly-installed environment, but the fallback should degrade to "non-optimal," not "unusable," if it's ever hit. Covered by `test_apply_filter_falls_back_to_scipy_when_numba_unavailable` (monkeypatches `_HAVE_NUMBA`, since actually uninstalling numba isn't practical in CI).

**CLI:** no change; transparent to `apply_filter`'s callers.
