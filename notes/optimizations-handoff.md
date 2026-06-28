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
| Compact HDF5 covariance | Implemented and remotely validated | Single-pass append writer ran on chr10/chr11/chr13/chr21/chr22 with no fallback and no compact pair-count pass. |
| Step 2 RSS | Remotely validated | chr10/chr11/chr13/chr21/chr22 remain below 1 GiB whole-run RSS after the single-pass writer. |
| Step 3 matrix-to-vector | Bounded and remotely validated | Chunked HDF5 path keeps chr11 Step 3 parent RSS around 0.45 GiB, but wall time remains large. |
| Streaming metrics | Implemented and remotely profiled | Avoids full-chromosome metric arrays; two chr11 metric passes still cost about 8.8 minutes total. |
| HDF5 read layout/reuse | Remotely validated | 65,536-row storage chunks plus per-precompute reader reuse recovered most local-search HDF5 read regression. |
| Local-search dense accumulator | Implemented and remotely profiled | Net local-search result is mixed but mostly positive; dense lookup/accumulate time is now visible and should be watched. |
| Horizontal aggregation | Implemented and remotely profiled | Dense path keeps dictionary growth out of the hot path, but vertical/horizontal buckets include new dense accumulation detail. |
| Duplicate merge path | Reverted and remotely profiled | `dedup_merge_seconds` is back to zero; do not reintroduce the Python sorted merge. |
| Multiprocessing in Step 3/4 | Step 3 opt-in implemented | `--matrix-workers` now parallelizes bounded HDF5 partition compute for Step 3; metric workers remain planned only. |

## Working Change Notes

### Bounded Compact Covariance

Previous remote baseline before the single-pass writer:

- chr11 whole-run max RSS: 0.837 GiB.
- Step 2 used `workers=4` and generated 378 compact HDF5 partitions.
- Retained compact rows: about 8.81B total on chr11.
- Largest chr11 compact partition: about 677.9M retained rows.
- Compact pair counting summed to about 1877 s across workers.
- Compact generation/HDF5 writing summed to about 2593 s across workers.

Current implemented path:

- `calc_covariance()` now precomputes per-SNP allele counts and genetic cutoff
  stop bounds for full and compact kernels.
- Compact production path now uses a single-pass appendable HDF5 writer.
- The old count-then-generate compact writer remains as a fallback if an
  invariant fails.
- Compact dataset storage chunks remain `HDF5_DATASET_CHUNK_ROWS = 65_536`;
  write batches remain `COVARIANCE_WRITE_CHUNK_ROWS = 1_000_000`.

Latest remote validation:

- `compact_hdf5_written ... single_pass=true` appeared for every compact
  partition in the downloaded profiles.
- No fallback was used and `compact_pair_counts` did not appear:

| Chrom | Compact partitions | Single-pass writes | Fallbacks | Pair-count profiles |
| --- | ---: | ---: | ---: | ---: |
| chr10 | 376 | 376 | 0 | 0 |
| chr11 | 378 | 378 | 0 | 0 |
| chr13 | 274 | 274 | 0 | 0 |
| chr21 | 103 | 103 | 0 | 0 |
| chr22 | 98 | 98 | 0 | 0 |

The chr11 Step 2 wall split improved from about 1338 s to about 707 s. Whole
run wall improved from 3922.00 s to 3236.81 s, while max RSS moved from
0.837 GiB to 0.811 GiB.

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

Latest downloaded remote profile includes dense accumulators and the duplicate
merge revert.

| Chrom | Local search | HDF5 read | Dedup | Dense lookup | Dense accumulate | Normalize |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr10 | 327.89 s | 145.42 s | 69.93 s | 34.68 s | 18.28 s | 44.95 s |
| chr11 | 844.41 s | 563.38 s | 134.84 s | 42.22 s | 21.35 s | 52.98 s |
| chr13 | 200.44 s | 83.94 s | 43.20 s | 23.93 s | 12.54 s | 28.22 s |
| chr21 | 48.63 s | 19.16 s | 7.93 s | 6.80 s | 4.09 s | 8.03 s |
| chr22 | 60.06 s | 25.36 s | 10.83 s | 7.44 s | 4.39 s | 8.99 s |

Compared with the previous sweep, the duplicate merge revert did what we
wanted: `dedup_merge_seconds` is zero and chr11 dedup fell from 243.58 s to
134.84 s. Dense accumulation did not create a dramatic standalone speedup;
chr11 dense lookup plus dense accumulation now accounts for about 63.6 s. Net
local-search wall still improved on the large downloaded chromosomes except for
minor small-chromosome noise.

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
problem. Latest local changes add:

- compact-index locus discovery instead of a row-streaming loci pass;
- per-partition debug timing for HDF5 open, locus index read, diagonal read,
  row read, normalization, and center accumulation;
- parent-level `matrix_to_vector_array profile` timing for wall, merge, flush,
  and worker wait;
- opt-in `--matrix-workers` for `ldetect2 run` and `ldetect2 matrix-to-vector`.

The worker path is bounded by at most `matrix_workers` in-flight partition
results. Workers compute partition sums; the parent merges and flushes output in
partition order.

Remote validation target:

- first compare `--matrix-workers 1` with current baseline to measure the
  compact-index loci pass win;
- then test `--matrix-workers 2` and `--matrix-workers 4` on chr21/chr22,
  followed by chr10/chr11 if RSS stays bounded;
- verify vector/BED/JSON validation unchanged.

### Streaming Metrics

Current path avoids resident chromosome-wide metric arrays, but chr11 still has
two streaming metric passes of about 260 s each.

Latest local change:

- metric denominator/loci discovery now uses compact HDF5 `lo_values` indexes
  instead of streaming all owned rows in the metadata pass;
- metric debug profile now splits `loci_index_seconds` and
  `diag_read_seconds` from broader `index_read_seconds`.

Planned optimization:

- next consider partition-level metric workers that return partial sums and
  counters for parent reduction;
- do not fuse Fourier and Fourier-LS metric passes until a larger pipeline
  restructure makes both breakpoint sets available at once.

## Latest Downloaded Remote Profile

Files:

```text
examples/ldetect_original/results/diagnostics/EUR/logs/
examples/ldetect_original/results/diagnostics/EUR/profiling/
```

Run summary after dense accumulation, duplicate-merge revert, and single-pass
compact covariance:

| Chrom | Wall time | Max RSS | Local search | HDF5 read | Dedup | Horizontal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr10 | 1505.05 s | 0.598 GiB | 327.89 s | 145.42 s | 69.93 s | 28.19 s |
| chr11 | 3236.81 s | 0.811 GiB | 844.41 s | 563.38 s | 134.84 s | 33.68 s |
| chr13 | 997.88 s | 0.493 GiB | 200.44 s | 83.94 s | 43.20 s | 19.70 s |
| chr21 | 267.31 s | 0.410 GiB | 48.63 s | 19.16 s | 7.93 s | 6.22 s |
| chr22 | 308.48 s | 0.412 GiB | 60.06 s | 25.36 s | 10.83 s | 6.57 s |

chr11 raw-log wall split:

| Phase | Seconds | Notes |
| --- | ---: | --- |
| Step 1 partitioning | 2 | negligible |
| Step 2 covariance | 707 | single-pass compact HDF5, `workers=4` |
| Step 3 matrix-to-vector | 1032 | bounded RSS, now larger than Step 2 |
| Filter/minima before metric | 126 | lower priority |
| Fourier metric | 261 | streaming |
| Fourier local search | 844 | dense accumulator plus duplicate merge revert |
| Final Fourier-LS metric | 262 | streaming |

## Next Optimization Plans

### 1. Output Parity and Profile Hygiene

The newest profiles validate the main runtime/RSS expectations, but keep the
next remote run focused on correctness and clean counters:

- confirm BED/JSON/HDF5 validation unchanged;
- keep whole-run max RSS near the current sub-1 GiB profile;
- keep `single_pass=true`, zero compact fallbacks, and zero compact pair-count
  profiles;
- keep `dedup_merge_seconds` at zero;
- compare `dense_lookup_seconds` and `dense_accumulate_seconds` across another
  chr10/chr11 run before micro-optimizing the dense path.
- watch new local-search read amplification counters:
  `hdf5_read_calls`, `hdf5_segment_partition_reads`, and `hdf5_segment_loci`.

### 2. Matrix Worker Validation

Goal: decide whether Step 3 partition workers are worth keeping on by default
or exposing in production runbooks.

Run order:

- chr21/chr22 with `--matrix-workers 1`, `2`, and `4`;
- chr10/chr11 with the best small-chromosome setting;
- compare Step 3 wall, whole-run RSS, output validation, and parent
  `worker_wait_seconds`.

### 3. Metric Worker Prototype

Goal: reduce streaming metric wall time after the metadata-first metric pass is
remotely measured.

Implementation direction:

- expose opt-in `--metric-workers`, default `1`;
- each worker streams assigned partitions and returns partial metric sums;
- parent reduces partials deterministically.

### 4. Local-Search Grouped Multiprocessing

Do not restore naive per-breakpoint multiprocessing by default. It discards the
current grouped HDF5 reuse and can multiply decompression and RSS.

Only revisit if later profiles show enough remaining local-search CPU work:

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
