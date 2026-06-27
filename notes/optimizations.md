# ldetect2 Performance Optimization Summary

This document is the human-readable optimization summary for `ldetect2`. It is
intended for reports, write-ups, and project context. Detailed agent handoff
notes, active implementation plans, and profiling runbooks live in
`notes/local-search-memory-speed-handoff.md`.

## Executive Summary

The optimization work moved `ldetect2` from a mostly materialized,
whole-chromosome workflow to a bounded, chunked HDF5 workflow. The major change
was not a single micro-optimization; it was replacing repeated full covariance
materialization with streaming readers, compact indexed storage, and
stage-specific bounded accumulators.

The current production path targets:

```text
ldetect2 run --subset fourier_ls --covariance-cache compact
```

On the downloaded EUR chr10/chr11/chr13/chr21/chr22 profiles, whole-run RSS is
now below 1 GiB. Earlier chr10/chr11 runs could exceed tens of GiB because Step
2, Step 3, metrics, and local search each had ways to materialize large
covariance arrays.

The remaining runtime is distributed across covariance generation,
matrix-to-vector conversion, streaming metric passes, and local-search
precompute. Candidate scoring itself is not a meaningful bottleneck.

## Main Improvements

### 1. Faster Pairwise LD Kernel

The Wen/Stephens pairwise LD kernel was moved into a Numba-compiled path. The
hot inner loop now operates on typed NumPy arrays and avoids Python per-pair
overhead. In small benchmarks this was roughly 50x faster than the pure Python
implementation.

Recent local changes also precompute per-SNP allele counts and genetic cutoff
bounds so the compact/full kernels do less repeated inner-loop work. These
latest kernel refinements still need remote real-data validation.

### 2. Parallel Covariance Generation

Covariance partitions are independent, so `ldetect2 run --workers N` computes
them with a process pool. This is still the main coarse-grained parallelism in
the pipeline and scales naturally until limited by available cores, I/O, and
memory.

The current compact path keeps worker memory bounded by writing HDF5 rows in
chunks instead of materializing full retained pair arrays for an entire
partition.

### 3. Compact HDF5 Covariance Storage

The production intermediate format is now compact HDF5 rather than gzipped text
or historical `.npz` files. Compact HDF5 stores canonical `(lo, hi)` positions
and `shrink_ld` values, with indexes for diagonal lookup and row-range reads.

This format gives all downstream stages a shared chunked reader:

- Step 3 matrix-to-vector reads partition chunks;
- streaming metric calculation reads bounded row batches;
- local search reads only relevant segment ranges;
- HDF5 row indexes make repeated partition access cheaper and more predictable.

The storage layout separates write batching from dataset chunking. Production
compact writes use bounded row-generation batches, while HDF5 datasets use
65,536-row storage chunks. That layout recovered much of a previous
local-search read/decompression regression without increasing RSS.

### 4. Bounded Step 2 Covariance Writes

The compact covariance writer removed the old Step 2 memory spike from full
partition pair materialization. Instead of allocating all retained pair indexes
and mapped positions, the compact path writes sorted rows in bounded chunks.

Remote chr11 validation before the latest single-pass local change:

| Metric | Value |
| --- | ---: |
| Whole-run max RSS | 0.837 GiB |
| Compact HDF5 partitions | 378 |
| Retained compact rows | ~8.81B |
| Largest retained partition | ~677.9M rows |
| Compact pair counting, summed across workers | ~1877 s |
| Compact generation/HDF5 writing, summed across workers | ~2593 s |

The latest local code adds a single-pass appendable compact HDF5 writer to
remove the count-then-generate double pass. It is expected to reduce Step 2
runtime while preserving bounded RSS, but remote validation is pending.

### 5. Bounded Step 3 Matrix-To-Vector

Step 3 used to normalize and accumulate large covariance arrays in ways that
could dominate memory. The current HDF5 path streams row chunks, computes `r²`
in bounded batches, and accumulates center-locus sums without retaining
full-partition normalized arrays.

Remote validation showed the memory win clearly:

| Chrom | Step 3 seconds | Step 3 max RSS |
| --- | ---: | ---: |
| chr11 | ~1025 s | ~465 MiB |
| chr21 | ~58 s | ~372 MiB |
| chr22 | ~76 s | ~372 MiB |

Step 3 is no longer the primary RSS risk, but it remains a major wall-time
phase on large chromosomes.

### 6. Streaming Metric Calculation

The normal float metric path now streams from covariance files instead of
loading a chromosome-wide covariance cache. This avoids stacking large metric
arrays with Step 3 or local-search memory.

The tradeoff is runtime: chr11 currently spends roughly 260 seconds on the
Fourier metric and another 264 seconds on the final Fourier-LS metric. Future
work should reduce unnecessary metadata/row reads and consider bounded
partition-level metric workers.

### 7. Selective Breakpoint Subsets

The default `ldetect2 run --subset fourier_ls` computes only the raw Fourier
breakpoints and the Fourier local-search result needed for final BED output.
Uniform local search is skipped unless explicitly requested with
`--all-breakpoint-subsets`.

This avoids doing expensive local-search work for breakpoint sets that the
normal production command will not use.

### 8. Local-Search Streaming and Dense Accumulation

Local search was refactored from repeated full active-row materialization toward
streaming HDF5 segment rows into bounded accumulators. The HDF5 path preserves
partition-order, first-retained-pair semantics at a row-stream boundary.

Validated wins include:

- sorted range slicing instead of full-row masks;
- partition-level diagonal metadata;
- HDF5 reader reuse within each breakpoint precompute;
- grouped horizontal reductions, which cut chr11 horizontal aggregation from
  77.46 s to 16.44 s in the previous remote profile.

The latest local code replaces temporary sum dictionaries with a
per-breakpoint dense accumulator. Remote dense-accumulator profiling is in
progress.

### 9. Failed or Reverted Optimization: Python Duplicate Merge

One attempted local-search duplicate optimization moved set-union work from
NumPy into a Python sorted merge loop. Remote profiling showed this was a clear
regression: chr11 dedup time rose from 134.21 s to 243.58 s, with nearly all
of the new time in `dedup_merge_seconds`.

That path has been reverted. The code is back to NumPy's
`np.isin(..., assume_unique=True)` plus `np.union1d()` behavior for duplicate
tracking.

## Remote Profiling Highlights

Latest downloaded previous-sweep profile, before dense-accumulator validation:

| Chrom | Wall time | Max RSS | Local search |
| --- | ---: | ---: | ---: |
| chr10 | 1872.15 s | 0.587 GiB | 342.27 s |
| chr11 | 3922.00 s | 0.837 GiB | 904.80 s |
| chr13 | 1400.58 s | 0.488 GiB | 304.53 s |
| chr21 | 320.68 s | 0.405 GiB | 45.94 s |
| chr22 | 405.62 s | 0.412 GiB | 82.09 s |

chr11 phase split:

| Phase | Seconds | Notes |
| --- | ---: | --- |
| Step 2 covariance | 1338 | compact HDF5, `workers=4` |
| Step 3 matrix-to-vector | 1025 | bounded RSS, still high wall time |
| Filter/minima before metric | 126 | lower priority |
| Fourier metric | 260 | streaming metric pass |
| Fourier local search | 905 | pre-dense, includes reverted duplicate regression |
| Final Fourier-LS metric | 264 | streaming metric pass |

## Current Remaining Bottlenecks

The main remaining runtime targets are:

1. Step 2 compact covariance generation, especially avoiding duplicate
   count/generate work.
2. Step 3 matrix-to-vector wall time, while keeping parent-owned ordered writes
   and bounded RSS.
3. Streaming metric passes, especially unnecessary row reads and repeated
   partition setup.
4. Local-search HDF5 read/decompression and normalization after dense
   accumulator validation.

Multiprocessing is worth testing for Step 3 and metric partition work, but not
as naive per-breakpoint local-search multiprocessing. Local search should only
use grouped worker units if dense profiling shows enough remaining work to
justify the extra I/O and RSS risk.

## Design Principles That Emerged

- Prefer chunked HDF5 readers over chromosome-wide covariance caches.
- Keep optimized paths bounded by partition, chunk, or breakpoint window.
- Preserve restartable intermediate files even when in-memory shortcuts are
  tempting.
- Add instrumentation before optimizing a phase whose wall time is not yet
  explained.
- Treat duplicate-position compatibility as a stream-boundary/storage concern,
  not something every aggregation kernel should solve independently.
- Keep high-memory optimizations explicit or opt-in.

## Historical Notes

Several earlier optimizations were useful stepping stones but are no longer the
main story:

- `.npz` covariance partitions replaced gzipped text parsing, but have been
  superseded by HDF5.
- Local-search process parallelism exists, but the bounded single-process HDF5
  grouped path is safer for whole-chromosome production runs.
- Full chromosome covariance caches can speed small/debug workflows, but should
  not become default because they recreate the memory pressure this work was
  meant to remove.
