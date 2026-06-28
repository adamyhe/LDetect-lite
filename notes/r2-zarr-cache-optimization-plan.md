# r2-Zarr Cache Optimization Plan

Date: 2026-06-28

## Goal

Make `--pair-cache r2-zarr` the fast, reusable pair-cache path while reducing
cache size and avoiding unnecessary write amplification. The path should remain
exact with respect to HDF5/r2 behavior: duplicate physical pairs keep
first-retained-pair precedence, metric uses the original lower-endpoint
ownership windows, and local search preserves active-partition extent filters.

## Current State

The latest downloaded profile before the v2 layout showed `r2_zarr` as the
fastest mode:

| Chrom | `matrix_hdf5` | `direct_hdf5` | `r2_zarr` |
| --- | ---: | ---: | ---: |
| 13 | 739.87 s | 821.36 s | 568.48 s |
| 21 | 171.32 s | 190.91 s | 134.37 s |
| 22 | 211.96 s | 224.59 s | 160.00 s |

The previous cache sizes were still large: about 14G for `r2_zarr` versus 16G
for the HDF5 modes in the chr21 run directory comparison. Since then, the
working tree has moved new r2-Zarr writes to a compact v2 layout:

- diagonal rows are implicit through `diag_idx`;
- off-diagonal upper endpoints are stored as `hi_delta`;
- `ldetect2 run --pair-cache r2-zarr` builds a chromosome-level `owned_pairs`
  group and deletes partition groups after consolidation;
- explicit compressor choices are available via
  `--r2-zarr-compressor {default,lz4-bitshuffle,zstd-bitshuffle}`.

These v2 changes need fresh remote profiling; the existing downloaded logs are
from the earlier partition-local layout.

## Phase 1: Re-profile v2 Owned Cache

Run the existing exactness/runtime workflow on the same chromosomes:

```text
examples/r2_zarr_exactness
```

Collect:

- final run directory size for `r2_zarr`;
- temporary peak directory size while partition groups exist before
  consolidation;
- Step 2 wall time split between partition generation and owned-cache
  consolidation;
- Step 4 metric/local-search times from the owned cache;
- RSS during consolidation and local search;
- exactness TSVs versus HDF5 modes.

Expected result: final cache size should fall from implicit diagonals,
`hi_delta`, and overlap deduplication. Runtime could move either direction
because v2 currently writes partition groups and then performs a consolidation
pass.

## Phase 2: Reduce Consolidation Overhead

Current v2 run behavior is intentionally simple:

1. parallel workers write partition r2 groups;
2. the parent streams those groups into `owned_pairs`;
3. partition groups are deleted.

Useful follow-up patches:

1. Add consolidation profiling.
   Log rows read, rows written, duplicate rows skipped, diagonal count,
   `hi_delta` dtype, read seconds, write seconds, and delete seconds.
2. Avoid logical-row round trips during consolidation.
   The current reader synthesizes diagonal rows and reconstructs `hi` before
   the owned writer makes diagonals implicit again. Add an internal stored-row
   iterator that streams `positions`, `lo_offsets`, `diag_idx`, `hi_delta`, and
   `r2` directly.
3. Preserve partition groups optionally.
   Add a debug flag only if needed, e.g. `--r2-zarr-keep-partitions`, so
   failures can be inspected without re-running Step 2. Default should delete
   partition groups to save disk.
4. Write temporary partition groups with fast codec settings.
   If final `owned_pairs` uses a stronger compressor, consider writing
   temporary partition groups with default/lz4 and recompressing only the final
   cache. This may reduce Step 2 wall time without changing final size.

## Phase 3: Improve Coordinate Dtypes

`hi_delta` should use the smallest safe unsigned dtype based on maximum delta,
not just total partition/chromosome locus count.

Plan:

1. Track max `hi_delta` while writing a group.
2. If max delta fits in `uint16`, store `uint16`; otherwise use `uint32`.
3. For chromosome-level `owned_pairs`, consider splitting row blocks by
   `hi_delta` dtype only if a single outlier forces `uint32` for the whole
   chromosome. Keep this out of the first pass unless profiling shows a real
   size hit.

Exactness risk is low if validation checks `hi_delta >= 0`, sorted canonical
endpoints, and dtype bounds.

## Phase 4: Compression Sweep

Benchmark the existing compressor choices on v2 `owned_pairs`:

```text
--r2-zarr-compressor default
--r2-zarr-compressor lz4-bitshuffle
--r2-zarr-compressor zstd-bitshuffle
```

Measure:

- final cache size;
- Step 2 partition write time;
- consolidation write time;
- metric read time;
- local-search read time.

Decision rule:

- keep `default` if zstd saves little or slows Step 4 materially;
- prefer `lz4-bitshuffle` if it gives meaningful size reduction with low
  runtime cost;
- keep `zstd-bitshuffle` opt-in unless it is clearly better on both size and
  end-to-end wall time.

## Phase 5: Chunk and Access Layout

The current r2-Zarr reader groups rows by lower endpoint with `lo_values` and
`lo_offsets`. That matches metric and local-search range queries. Remaining
layout knobs:

- dataset chunk rows, currently 65,536 by default;
- local-search `chunk_rows`, currently large enough to create temporary masks;
- whether `owned_pairs` should use larger chunks than temporary partitions.

Plan:

1. Add debug counters for Zarr chunk reads and rows yielded per local-search
   segment.
2. Sweep dataset chunks at 65,536, 262,144, and 1,048,576 rows.
3. Track metric/local-search wall time, RSS, and compression ratio.

Avoid changing the logical row order unless profiling shows row-range reads are
fragmented. The current lower-endpoint grouping is the right shape for metric
and local search.

## Phase 6: Direct Owned-Cache Writer Feasibility

A true direct owned-cache writer would avoid writing temporary partition groups,
but parallel Step 2 workers cannot safely append to one Zarr group without a
central coordinator. Options:

1. Keep current partition-temporary plus consolidation design.
   This is simple, restartable, and parallel, but writes rows twice.
2. Write worker outputs to compact temporary binary chunks, then consolidate.
   This may reduce temporary metadata overhead compared with full partition
   Zarr groups.
3. Single writer process receives worker row streams.
   This reduces disk write amplification but risks IPC bottlenecks and more
   complex failure recovery.

Recommendation: do not start with direct owned writing. First profile v2
consolidation. If consolidation is a minor fraction of Step 2, the current
restartable design is preferable.

## Validation

Every optimization should keep these tests or equivalents green:

- r2-Zarr metric equals HDF5 metric with overlapping partitions;
- r2-Zarr local search equals HDF5 local search;
- owned cache works after partition groups are deleted;
- duplicate physical positions preserve first-pair precedence;
- default HDF5 path remains unchanged.

Fast local validation:

```text
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest tests/test_r2_zarr.py tests/test_shrinkage.py -q
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest -m "not integration"
```
