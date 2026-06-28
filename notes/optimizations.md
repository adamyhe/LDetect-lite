# ldetect2 Performance Optimization Summary

This document is the human-readable optimization summary for `ldetect2`. It is
intended for reports, write-ups, and project context. Detailed agent handoff
notes, active implementation plans, and profiling runbooks live in
`notes/optimizations-handoff.md`.

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

The newest profiles include both dense local-search accumulation and the
single-pass compact covariance writer. The largest recent runtime win came from
removing the compact covariance count-then-generate double pass: chr11 Step 2
fell from about 22.3 minutes to about 11.8 minutes, and whole-run wall time fell
from about 65.4 minutes to about 54.0 minutes without increasing RSS.

The remaining runtime is now led by matrix-to-vector conversion, streaming
metric passes, local-search HDF5 reads, and still-nontrivial covariance
generation. Candidate scoring itself is not a meaningful bottleneck.

## Main Improvements

### 1. Faster Pairwise LD Kernel

The Wen/Stephens pairwise LD kernel was moved into a Numba-compiled path. The
hot inner loop now operates on typed NumPy arrays and avoids Python per-pair
overhead. In small benchmarks this was roughly 50x faster than the pure Python
implementation.

The current kernels also precompute per-SNP allele counts and genetic cutoff
bounds so the compact/full kernels do less repeated inner-loop work. These
refinements are part of the remotely profiled single-pass compact covariance
path.

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

Remote chr11 validation before the single-pass writer:

| Metric | Value |
| --- | ---: |
| Whole-run max RSS | 0.837 GiB |
| Compact HDF5 partitions | 378 |
| Retained compact rows | ~8.81B |
| Largest retained partition | ~677.9M rows |
| Compact pair counting, summed across workers | ~1877 s |
| Compact generation/HDF5 writing, summed across workers | ~2593 s |

The newest remote profiles validate the single-pass appendable compact HDF5
writer. Every downloaded compact partition used `single_pass=true`, no fallback
was used, and the compact pair-count profile disappeared:

| Chrom | Compact partitions | Wall time after change | Max RSS after change |
| --- | ---: | ---: | ---: |
| chr10 | 376 | 1505.05 s | 0.598 GiB |
| chr11 | 378 | 3236.81 s | 0.811 GiB |
| chr13 | 274 | 997.88 s | 0.493 GiB |
| chr21 | 103 | 267.31 s | 0.410 GiB |
| chr22 | 98 | 308.48 s | 0.412 GiB |

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
phase on large chromosomes. The latest local implementation starts addressing
that runtime directly: Step 3 now uses compact HDF5 locus indexes for the loci
pass and exposes opt-in bounded partition workers via `--matrix-workers`.
Remote profiling shows `matrix_workers=4` improves runtime without meaningful
RSS inflation. On cached chr19-22 diagnostic runs, Step 3 fell to roughly
14-28 seconds with Step 3 RSS around 222-282 MiB.

### 6. Streaming Metric Calculation

The normal float metric path now streams from covariance files instead of
loading a chromosome-wide covariance cache. This avoids stacking large metric
arrays with Step 3 or local-search memory.

The tradeoff is runtime: chr11 currently spends roughly 260 seconds on the
Fourier metric and another 264 seconds on the final Fourier-LS metric. The
current metric path avoids row streaming during loci discovery by using compact
HDF5 indexes and reports separate loci-index and diagonal-read timings. New
cached chr19-22 profiles show the metadata/index part is now tiny; remaining
metric time is row read, normalization, and crossing classification.

The newest local implementation adds opt-in bounded partition-level metric
workers via `--metric-workers`. This keeps the default single-process behavior
but gives the next profiling run a direct way to test whether the streaming row
passes scale like Step 3 without meaningful RSS inflation.

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

The newest remote profiles include the dense accumulator and the duplicate
merge revert. The dense path is a modest net win, not a dramatic one: chr11
local search moved from 904.80 s in the previous sweep to 844.41 s, while dense
lookup plus dense accumulation now accounts for about 63.6 s. The bigger
cleanup was reverting the Python duplicate merge; `dedup_merge_seconds` is back
to zero and chr11 duplicate tracking fell from 243.58 s to 134.84 s.

With `matrix_workers=4` and `local_search_workers=1`, local search is now the
largest cached-run phase on chr19-22. Its remaining cost is mostly HDF5
read/decompression, duplicate filtering, and dense endpoint accumulation rather
than candidate scoring. The latest local changes target that directly by using
one combined dense endpoint lookup per row chunk for vertical/horizontal sums
and coalescing adjacent local-search HDF5 segments when they share the same
active partition set.

### 9. Failed or Reverted Optimization: Python Duplicate Merge

One attempted local-search duplicate optimization moved set-union work from
NumPy into a Python sorted merge loop. Remote profiling showed this was a clear
regression: chr11 dedup time rose from 134.21 s to 243.58 s, with nearly all
of the new time in `dedup_merge_seconds`.

That path has been reverted and remotely validated. The code is back to
NumPy's `np.isin(..., assume_unique=True)` plus `np.union1d()` behavior for
duplicate tracking.

## Remote Profiling Highlights

Latest downloaded profile, including dense accumulation and single-pass compact
covariance:

| Chrom | Wall time | Max RSS | Local search |
| --- | ---: | ---: | ---: |
| chr10 | 1505.05 s | 0.598 GiB | 327.89 s |
| chr11 | 3236.81 s | 0.811 GiB | 844.41 s |
| chr13 | 997.88 s | 0.493 GiB | 200.44 s |
| chr21 | 267.31 s | 0.410 GiB | 48.63 s |
| chr22 | 308.48 s | 0.412 GiB | 60.06 s |

chr11 phase split:

| Phase | Seconds | Notes |
| --- | ---: | --- |
| Step 2 covariance | 707 | single-pass compact HDF5, `workers=4` |
| Step 3 matrix-to-vector | 1032 | bounded RSS, now larger than Step 2 |
| Filter/minima before metric | 126 | lower priority |
| Fourier metric | 261 | streaming metric pass |
| Fourier local search | 844 | dense accumulation plus duplicate merge revert |
| Final Fourier-LS metric | 262 | streaming metric pass |

Latest cached diagnostic profile with `matrix_workers=4`,
`local_search_workers=1` skipped Step 2 because covariance partitions already
existed, but it isolates the current Step 3/4 behavior:

| Chrom | Wall time | Max RSS | Step 3 | Local search |
| --- | ---: | ---: | ---: | ---: |
| chr19 | 279.32 s | 0.507 GiB | 28.48 s | 121.37 s |
| chr20 | 239.38 s | 0.458 GiB | 24.47 s | 90.96 s |
| chr21 | 132.20 s | 0.400 GiB | 14.27 s | 44.39 s |
| chr22 | 154.43 s | 0.424 GiB | 17.84 s | 58.09 s |

## Current Remaining Bottlenecks

The main remaining runtime targets are:

1. Streaming metric passes, now mostly row read, normalization, and
   crossing-pair classification.
2. Local-search HDF5 read/decompression, duplicate filtering, and dense
   endpoint accumulation.
3. Step 3 validation on larger chromosomes with `matrix_workers=4`.
4. Further compact covariance CPU work, now that the largest double-pass cost is
   gone.

Step 3 multiprocessing is now the preferred scaling knob: `matrix_workers=4`
captures nearly the same runtime as also setting `local_search_workers=4`, but
without the local-search RSS inflation. Metric partition multiprocessing is
now the next most plausible broad runtime win. Naive per-breakpoint local-search
multiprocessing remains a poor fit; local-search optimization should focus on
reducing repeated HDF5 reads and dense accumulation overhead.

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
- Local-search process parallelism exists, but `local_search_workers=4` caused
  significant RSS inflation in remote profiling; keep the bounded
  single-process HDF5 grouped path for whole-chromosome production runs.
- Full chromosome covariance caches can speed small/debug workflows, but should
  not become default because they recreate the memory pressure this work was
  meant to remove.
