# ldetect2 Optimization Handoff

Date: 2026-06-27

## Purpose

This is the agent-facing working note for runtime/RSS optimization. Keep it
practical: what changed, what is validated, what is only local, what to profile
next, and what not to reintroduce. Human-readable write-up material belongs in
`notes/optimizations.md`.

The production target remains:

```text
ldetect2 run --subset fourier_ls --covariance-cache compact
```

The guiding constraint is unchanged: do not trade runtime for chromosome-scale
resident covariance arrays. New speedups should either keep RSS bounded by a
partition/chunk/window or stay opt-in until remote profiles prove otherwise.

## Current State Snapshot

| Area | State | Notes |
| --- | --- | --- |
| Compact HDF5 covariance | Implemented; local single-pass writer added | Bounded compact path is the production cache. Single-pass append writer is local-only until remote validation. |
| Step 2 RSS | Remotely validated | chr10/chr11/chr13/chr21/chr22 all below 1 GiB whole-run RSS before the latest local single-pass writer. |
| Step 3 matrix-to-vector | Bounded and remotely validated | Chunked HDF5 path keeps chr11 Step 3 parent RSS around 0.45 GiB, but wall time remains large. |
| Streaming metrics | Implemented and remotely profiled | Avoids full-chromosome metric arrays; two chr11 metric passes still cost about 8.8 minutes total. |
| HDF5 read layout/reuse | Remotely validated | 65,536-row storage chunks plus per-precompute reader reuse recovered most local-search HDF5 read regression. |
| Local-search dense accumulator | Implemented locally | Remote dense profile is currently running; latest downloaded logs predate it. |
| Horizontal aggregation | Implemented and remotely profiled | Previous sweep cut chr11 horizontal time from 77.46 s to 16.44 s before dense accumulators. |
| Duplicate merge path | Reverted locally | Python sorted merge caused a major dedup regression; code is back to `np.isin()`/`np.union1d()`. |
| Multiprocessing in Step 3/4 | Planned only | Step 3/metric partition multiprocessing looks worth testing; naive per-breakpoint local-search multiprocessing should stay off by default. |

## Working Change Notes

### Bounded Compact Covariance

Validated remote baseline before the latest local single-pass writer:

- chr11 whole-run max RSS: 0.837 GiB.
- Step 2 used `workers=4` and generated 378 compact HDF5 partitions.
- Retained compact rows: about 8.81B total on chr11.
- Largest chr11 compact partition: about 677.9M retained rows.
- Compact pair counting summed to about 1877 s across workers.
- Compact generation/HDF5 writing summed to about 2593 s across workers.

Current local follow-up:

- `calc_covariance()` now precomputes per-SNP allele counts and genetic cutoff
  stop bounds for full and compact kernels.
- Compact production path now attempts a single-pass appendable HDF5 writer.
- The old count-then-generate compact writer remains as a fallback if an
  invariant fails.
- Compact dataset storage chunks remain `HDF5_DATASET_CHUNK_ROWS = 65_536`;
  write batches remain `COVARIANCE_WRITE_CHUNK_ROWS = 1_000_000`.

Remote validation target:

- `compact_pair_counts` should disappear from successful single-pass compact
  runs.
- `compact_hdf5_written ... single_pass=true` should be present.
- Step 2 wall time should improve without raising max RSS or changing output
  validation.

### HDF5 Storage and Reader Layout

Keep this setup unless remote profiles clearly regress:

- compact HDF5 files store `/covariance/lo`, `/covariance/hi`, and
  `/covariance/shrink_ld`;
- `/index/diag_pos`, `/index/diag_val`, `/index/lo_values`, and
  `/index/lo_offsets` are required;
- compact datasets use 65,536-row HDF5 chunks;
- compact write batching is separate from storage chunking;
- local search reuses HDF5 readers within each breakpoint precompute.

Do not add flattened-log layout support. Keep chromosome-prefixed filenames.

### Duplicate Position and Duplicate Pair Semantics

Current production policy:

- duplicate physical VCF positions are collapsed before pairwise LD;
- compact covariance rows are canonical sorted `(lo, hi)`;
- HDF5 local search keeps a duplicate-safe row-stream boundary and preserves
  first-retained-pair precedence across partitions/chunks.

Known divergence from original `ldetect` on duplicate physical positions is
still acknowledged. The current changes do not make it substantially harder to
fix later because duplicate policy is centralized at covariance generation and
the local-search row stream boundary.

Avoid:

- reintroducing Python-level sorted merge loops for local-search duplicate
  tracking;
- scattering duplicate-position compatibility logic into aggregation kernels.

### Local Search

Latest downloaded remote profile does not include dense accumulators. It does
include the previous sweep with horizontal grouped reduction and the now-reverted
Python duplicate merge path.

chr11 previous-sweep local-search split:

| Bucket | Seconds | Interpretation |
| --- | ---: | --- |
| HDF5 read/decompression | 563.47 | Still the largest bucket. |
| Duplicate tracking | 243.58 | Inflated by reverted Python merge path. |
| `dedup_merge_seconds` | 239.39 | Should return to ~0 after revert. |
| Normalization | 51.92 | Secondary target. |
| Horizontal aggregation | 16.44 | Keep the grouped reduction win. |

Dense accumulator expectations for the in-progress remote run:

- `dense_lookup_seconds` and `dense_accumulate_seconds` should be present.
- `horizontal_seconds` should stay near or below the previous-sweep values.
- `dedup_merge_seconds` should be zero or negligible after the revert.
- chr11 dedup should move back toward the compact-layout/read-cache baseline
  of about 134 s, not the previous-sweep 244 s.

Do not prioritize `_search_array()` or JIT candidate scoring. Search time is
effectively zero compared with precompute.

### Step 3 Matrix-To-Vector

Validated remote baseline:

| Chrom | Step 3 seconds | Step 3 max RSS |
| --- | ---: | ---: |
| chr11 | ~1025 s | ~465 MiB |
| chr21 | ~58 s | ~372 MiB |
| chr22 | ~76 s | ~372 MiB |

The current chunked helper solved the memory problem but not the wall-time
problem. Summed profile buckets under-account Step 3 wall time, so add more
instrumentation before assuming where time is going.

Planned optimization:

- partition-level compute workers may be worth restoring;
- parent must own ordered vector writes and `pending_sums`;
- workers should return bounded sparse `(locus, sum)` outputs plus profile
  counters, not write gzip rows directly.

### Streaming Metrics

Current path avoids resident chromosome-wide metric arrays, but chr11 still has
two streaming metric passes of about 260 s each.

Planned optimization:

- first remove row streaming from the metadata/diagonal/loci pass if possible;
- then consider partition-level metric workers that return partial sums and
  counters for parent reduction;
- do not fuse Fourier and Fourier-LS metric passes until a larger pipeline
  restructure makes both breakpoint sets available at once.

## Latest Downloaded Remote Profile

Files:

```text
examples/ldetect_original/results/diagnostics/EUR/logs/
examples/ldetect_original/results/diagnostics/EUR/profiling/
```

Run summary after compact-layout/read-cache baseline plus previous sweep:

| Chrom | Wall time | Max RSS | Local search | HDF5 read | Dedup | Horizontal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr10 | 1872.15 s | 0.587 GiB | 342.27 s | 144.85 s | 127.02 s | 12.69 s |
| chr11 | 3922.00 s | 0.837 GiB | 904.80 s | 563.47 s | 243.58 s | 16.44 s |
| chr13 | 1400.58 s | 0.488 GiB | 304.53 s | 116.04 s | 124.58 s | 13.59 s |
| chr21 | 320.68 s | 0.405 GiB | 45.94 s | 18.99 s | 13.60 s | 2.92 s |
| chr22 | 405.62 s | 0.412 GiB | 82.09 s | 33.14 s | 30.44 s | 4.22 s |

chr11 raw-log wall split:

| Phase | Seconds | Notes |
| --- | ---: | --- |
| Step 1 partitioning | 2 | negligible |
| Step 2 covariance | 1338 | compact HDF5, `workers=4` |
| Step 3 matrix-to-vector | 1025 | bounded RSS, still major wall time |
| Filter/minima before metric | 126 | lower priority |
| Fourier metric | 260 | streaming |
| Fourier local search | 905 | pre-dense, duplicate merge regression included |
| Final Fourier-LS metric | 264 | streaming |

## Next Optimization Plans

### 1. Validate Current Local Changes Remotely

Run chr21/chr22 first, then chr10/chr11:

- dense local-search accumulator;
- duplicate merge revert;
- single-pass compact covariance writer.

Acceptance:

- BED/JSON/HDF5 validation unchanged;
- whole-run max RSS remains near current sub-1 GiB profile;
- compact covariance logs show `single_pass=true`;
- `dedup_merge_seconds` no longer dominates local search;
- dense fields are present and do not move time into a larger bucket.

### 2. Metric Metadata-First Pass

Goal: reduce the two streaming metric passes without resident covariance caches.

Implementation direction:

- derive `loci`, `diag_pos`, and `diag_val` from HDF5 indexes/diagonals rather
  than streaming all rows just to collect loci;
- keep the row streaming pass only for normalized crossing pairs;
- add profile fields for metadata read, row read, normalization, crossing, and
  unaccounted time.

### 3. Bounded Step 3 Multiprocessing

Goal: reduce Step 3 wall time while preserving the bounded helper-scope RSS.

Implementation direction:

- worker processes compute partition partial sums only;
- parent merges in partition order and writes the vector file;
- bound in-flight futures to worker count;
- expose as opt-in `--matrix-workers`, default `1`.

### 4. Metric Partition Multiprocessing

Goal: reduce streaming metric wall time if metadata-first pass is still
expensive.

Implementation direction:

- expose opt-in `--metric-workers`, default `1`;
- each worker streams assigned partitions and returns partial metric sums;
- parent reduces partials deterministically.

### 5. Local-Search Grouped Multiprocessing

Do not restore naive per-breakpoint multiprocessing by default. It discards the
current grouped HDF5 reuse and can multiply decompression and RSS.

Only revisit after dense profiling:

- submit partition groups, not individual breakpoints;
- each worker opens group metadata/readers once and processes breakpoints
  sequentially;
- keep default `--local-search-workers=1`.

## Validation Commands

Use focused tests for touched areas, then full non-integration:

```text
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest tests/test_shrinkage.py tests/test_covariance_io.py -q
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest tests/test_local_search.py tests/test_profile_ldetect2.py -q
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest -m "not integration"
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run ruff check src/ldetect2 tests examples/ldetect_original/scripts
git diff --check
```

Remote validation order:

1. chr21/chr22 for iteration;
2. chr10/chr11 for dense/high-row stress;
3. full all-chromosome run only after output parity and RSS stay stable.
