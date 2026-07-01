# r2-nocache Optimization Findings

Date: 2026-06-28

## Summary

The latest `examples/r2_zarr_exactness` profiling runs strongly suggest that
`--pair-cache r2-nocache` is unlikely to match `r2-zarr` wall time while
remaining a true no-pair-cache mode. It succeeds at the disk goal, but it pays
for repeated LD recomputation during both metric passes and local search.

`r2-nocache` remains useful as a low-disk benchmark and as an escape hatch for
environments where writing multi-GB pair caches is unacceptable. It should not
be treated as the likely fastest path.

## Latest Profile

Source files:

```text
examples/r2_zarr_exactness/results/runtime/EUR.13.runtime.tsv
examples/r2_zarr_exactness/results/runtime/EUR.21.runtime.tsv
examples/r2_zarr_exactness/results/runtime/EUR.22.runtime.tsv
examples/r2_zarr_exactness/results/logs/r2_nocache/EUR/*.ldetect2.log
examples/r2_zarr_exactness/results/logs/r2_zarr/EUR/*.ldetect2.log
```

End-to-end runtime:

| Chrom | `r2_zarr` | `r2_nocache` | Slowdown |
| --- | ---: | ---: | ---: |
| 13 | 568.48 s | 6726.81 s | 11.8x |
| 21 | 134.37 s | 1219.80 s | 9.1x |
| 22 | 160.00 s | 1826.47 s | 11.4x |

No-cache phase timing from the logs:

| Chrom | First metric | Local search | Final metric |
| --- | ---: | ---: | ---: |
| 13 | ~859 s | 4662.56 s | ~864 s |
| 21 | ~179 s | 773.81 s | ~178 s |
| 22 | ~215 s | 1295.49 s | ~216 s |

The cached `r2_zarr` path did the same metric/local-search phases much faster:

| Chrom | First metric | Local search | Final metric |
| --- | ---: | ---: | ---: |
| 13 | ~23 s | 127.55 s | ~23 s |
| 21 | ~5 s | 25.78 s | ~6 s |
| 22 | ~6 s | 34.64 s | ~7 s |

## Local-Search Row Pressure

No-cache local search repeatedly recomputes large active windows. Group-level
load lines show the scale of row work:

| Chrom | Groups | Total group rows | Max group rows | Total load seconds | Max compact payload estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 13 | 61 | 3.21B | 180.9M | 1986.3 s | ~1.7 GiB at 10 B/pair |
| 21 | 23 | 601.9M | 57.9M | 380.2 s | ~552 MiB at 10 B/pair |
| 22 | 23 | 918.5M | 139.6M | 497.3 s | ~1.3 GiB at 10 B/pair |

The compact payload estimate assumes an ideal in-memory layout like
`hi_delta:uint16` plus `r2:float64`. A simple row layout with
`lo:int64`, `hi:int64`, and `r2:float64` would be about 2.4x larger before
temporary masks and dedup arrays.

## Interpretation

The bottleneck is repeated pairwise LD work, not disk output. `r2_nocache`
writes almost nothing compared with cached modes, but the saved I/O is much
smaller than the cost of recomputing the same normalized `r2` rows for:

- the initial Fourier metric;
- each local-search active window;
- the final Fourier-LS metric.

The logs also contain many duplicate-position warnings. Current exactness
policy sends duplicate-position partitions through the canonical row fallback
instead of the fused no-cache fast path. That likely contributes to both wall
time and RSS, especially on chr13 where max RSS reached about 8.4 GiB.

## Optimization Options

These patches could improve no-cache, but they are unlikely to close the full
gap to `r2_zarr` on their own:

1. Use the new no-cache profiling counters.
   The code now logs no-cache VCF query/decode, array prep, row generation,
   duplicate fallback, dosage-cache placeholder, and LD/tile counters. These
   counters should be used to establish whether future wins come from fewer
   VCF decodes, less row recomputation, cheaper duplicate handling, or better
   pair/tile throughput.
2. Parallelize no-cache metric.
   `metric_from_r2_nocache()` currently ignores `workers`. Partition-level
   workers should be exact and could reduce the two metric passes substantially.
3. Add a prepared-partition LRU.
   Cache decoded partition inputs (`hap_mat`, positions, allele sums,
   `diag_shrink`, `j_stop_by_i`) across local-search groups. This avoids
   repeated VCF/BCF decoding without storing pair rows.
4. Improve duplicate-position fast paths.
   Duplicate-position partitions are common in the profiling logs. A
   duplicate-aware fast path could avoid whole-partition canonical row
   materialization when physical duplicate groups are small and exact
   first-pair precedence can be preserved.
5. Add an opt-in bounded row/window cache.
   A compact in-memory cache can target repeated local-search row
   recomputation directly. This is no longer pure no-cache in spirit, but it can
   avoid disk writes while capping memory, e.g. 512 MiB to 2 GiB.
6. Fuse no-cache local-search accumulation.
   A Numba kernel could compute eligible pair `r2` and update dense
   vertical/horizontal accumulators directly. This should wait until duplicate
   fallback behavior is better understood, because duplicates currently limit
   the fast-path coverage.

## Dosage Cache and Tiled r2 Plan

The new dosage-cache and tiled-vectorization notes point in the right
direction: cache an object that scales with SNPs and samples, not with retained
pairs, then compute needed pairs in bounded tiles. Two corrections are
required for exactness in this codebase:

1. The cached vectors must be derived from the same phased haplotype matrix
   used by `shrinkage.py`, not from unphased genotype dosages such as
   `cyvcf2.gt_dosages`. Current LD math operates on `2 * n_ind` haplotypes and
   skips missing or unphased genotypes. A genotype-dosage cache would change
   the statistic unless the whole covariance implementation changed with it.
2. Unit-normalized dosage or haplotype vectors alone give Pearson correlation,
   not the normalized Wen/Stephens shrinkage `r2` that `ldetect2` uses. Exact
   recomputation must still apply:
   - `d_naive = f11 - f1 * f2`;
   - the Wen/Stephens shrinkage factor `(1 - theta)^2 * exp(...)`;
   - the cutoff test on `abs(ds2)`;
   - the diagonal adjustment for `i == j`;
   - normalization by the shrunk diagonal values.

With those corrections, a cache of centered haplotype inputs is still useful.
The exact fast path can precompute, per retained variant:

- physical and genetic position;
- encounter ordinal within the partition/window;
- haplotype sum `n1x`;
- centered float64 haplotype vector `h - mean(h)`;
- optionally its squared norm for diagonal computation;
- a positive-diagonal/monomorphic sentinel;
- diagonal shrink value and `j_stop_by_i` bound.

The dot product of centered vectors recovers `n11 - n1x * nx1 / n_haps`, so a
tiled implementation can reproduce `d_naive` and then apply the existing
shrinkage, cutoff, diagonal, and ownership rules. This is an exact input cache,
not a persisted pair cache.

Monomorphic or cutoff-zero variants must not be silently represented as zero
vectors in the cache. Current code keeps them out of normalized r2 rows by
requiring positive `diag_shrink` before normalization. A cached/tiled path must
preserve that behavior by storing an explicit invalid/zero-diagonal marker and
skipping any pair where either endpoint lacks positive diagonal shrinkage.
This guard should be tested before enabling any normalized-vector cache,
because zero vectors can make the code look numerically stable while quietly
changing the effective locus list.

### Duplicate-position semantics

Do not key hot-path cache entries by physical position alone. Current
duplicate handling retains duplicate-position variants through pairwise LD,
then canonicalizes physical endpoint pairs with first-row precedence. Dropping
all but the first variant per position during VCF iteration would not match
the current HDF5/r2 paths.

Implementation should use an encounter-ordered variant identity internally,
for example a partition-local ordinal plus `(pos, ref, alt)` metadata for
debugging. Physical-position deduplication should remain at the row/canonical
boundary until a duplicate-aware tiled path is proven equivalent. A safe first
implementation can keep the current duplicate-position fallback and enable the
cached/tiled fast path only when physical positions are unique.

### Tiled local-search accumulation

The tiled plan should target local-search precompute first, because local
search is where no-cache repeats the most row work. The algorithm shape:

1. Build or fetch prepared haplotype-vector blocks for the active partitions in
   a local-search group.
2. For each planned local-search segment, iterate `(i_tile, j_tile)` over
   encounter-ordered SNP indexes with `j >= i`.
3. Use matrix multiplication on centered haplotype vectors to compute tile
   dot products.
4. Apply `j_stop_by_i`, physical bounds, ownership, breakpoint-window, cutoff,
   diagonal, and duplicate filters before accumulation.
5. Convert surviving `ds2` values to normalized `r2` using `diag_shrink` and
   update the existing dense vertical/horizontal accumulators directly.

The tile accumulator in the note describes a different block-quality metric
based on left/right/cross means. `ldetect2` local search instead needs the same
per-locus vertical and horizontal sums currently produced by
`DenseLocalSearchAccumulator`. Tiles that straddle a segment or breakpoint
boundary should be split by masks and accumulated into those dense arrays, not
into mean-left/mean-right/mean-cross buckets.

### Suggested implementation order

1. Re-profile current no-cache runs with the new `nocache_*` counters.
2. Add a small prepared-input LRU across local-search groups, still using the
   existing row generator. This isolates VCF decode and array-prep savings.
3. Add a unique-position-only tiled local-search fast path behind an internal
   flag, with current row generation as fallback for duplicate positions.
4. Validate local-search precompute arrays against the existing r2-nocache and
   HDF5 paths on synthetic unique-position and overlapping-partition fixtures.
5. Only after correctness is stable, consider a duplicate-aware tiled path that
   preserves first-retained physical pair precedence at the same boundary as
   `canonical_local_search_rows`.

## Recommendation

Do not prioritize no-cache as the primary fast path. Keep it experimental and
low-disk. If we invest further, use this order:

1. instrumentation-driven profiling;
2. no-cache metric workers;
3. prepared-input LRU;
4. unique-position tiled local-search fast path;
5. duplicate-aware tiled/canonical fast path;
6. optional bounded row/window cache if row recomputation remains dominant;
7. no-cache metric workers or tiled metric reuse, depending on the profiles.

The main performance track should move back to `r2-zarr`, where cached
normalized pairs already deliver the best runtime in the downloaded profiles.
