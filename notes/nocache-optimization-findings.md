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

## Active Implementation Plan

The active no-cache optimization plan is to cache SNP/sample-scale prepared
inputs in memory and compute needed `r2` values in bounded tiles. It should not
cache covariance matrices, normalized pair rows, local-search row chunks, or
whole-window r2 tiles.

Two corrections from the older dosage/tiled sketches are required for exactness
in this codebase:

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

With those corrections, a prepared-input cache is still useful. The exact fast
path can keep, per retained variant:

- physical and genetic position;
- encounter ordinal within the partition/window;
- haplotype sum `n1x`;
- compact phased haplotype row from `hap_mat`;
- a positive-diagonal/monomorphic sentinel via `diag_shrink`;
- diagonal shrink value and `j_stop_by_i` bound.

The dot product of centered vectors recovers `n11 - n1x * nx1 / n_haps`, so a
tiled implementation can reproduce `d_naive` and then apply the existing
shrinkage, cutoff, diagonal, and ownership rules. This is an exact input cache,
not a pair cache. Centered float64 tile matrices should be temporary compute
buffers, not cache entries.

### Stage 1: Bounded prepared-input LRU

Add an in-process LRU for prepared partition inputs only, with a hard byte cap
and eviction based on NumPy array `nbytes`. A default cap such as 512 MiB is
reasonable, with `0` disabling the cache. This gives a clean measurement of how
much repeated VCF decode and array prep costs during one run without allowing
memory to grow with the number of local-search groups.

Thread the cache through metric and local search so initial metric, local
search, and final metric can reuse decoded partition inputs. Instrument cache
hits, misses, evictions, current bytes, VCF decode seconds, and array-prep
seconds.

Monomorphic or cutoff-zero variants must not be silently represented as zero
vectors in the cache. Current code keeps them out of normalized r2 rows by
requiring positive `diag_shrink` before normalization. A cached/tiled path must
preserve that behavior by storing an explicit invalid/zero-diagonal marker and
skipping any pair where either endpoint lacks positive diagonal shrinkage.
This guard should be tested before enabling any normalized-vector cache,
because zero vectors can make the code look numerically stable while quietly
changing the effective locus list.

### Stage 2: Unique-position tiled local search

Add a unique-position-only tiled fast path for local-search precompute. For
each local-search segment, iterate `(i_tile, j_tile)` over encounter-ordered
SNP indexes with `j >= i`, center the two compact haplotype slices on demand,
compute the dot-product tile, and then apply:

- `j_stop_by_i`;
- physical bounds and ownership rules;
- local-search `snp_first`, `snp_last`, and `snp_top` eligibility;
- cutoff on `abs(ds2)`;
- diagonal adjustment;
- positive `diag_shrink` filtering;
- first-seen pair precedence across overlapping partitions.

Convert surviving `ds2` values to normalized `r2` using `diag_shrink` and
update `DenseLocalSearchAccumulator` vertical/horizontal sums directly. Track
tile count, max tile SNPs, pair candidates, pairs after cutoff, and LD compute
seconds.

Peak tiled compute buffers should be approximately:

```text
prepared_cache_mib + 2 * tile_size * n_haps * 8 + tile_size^2 * 8
```

plus the normal local-search arrays.

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

### Stage 3: Metric and broader profiling

After the local-search path is exact, decide from profiles whether to add
partition-level no-cache metric workers, a tiled metric path, or both.
`metric_from_r2_nocache()` currently streams in one process; partition-level
workers should be exact and could reduce the two metric passes substantially.

### Deferred work

Persisted prepared haplotype inputs may help when the same reference panel,
individuals file, map, chromosome, effective population size, cutoff, and
partition/window bounds are reused across separate runs, or across worker
processes that cannot share memory. It does not remove the full repeated
pairwise LD cost by itself, so it should be paired with tiled r2 computation if
local-search row recomputation remains dominant.

If a persisted dosage/prepared cache is added later, it needs its own explicit
opt-in mode and strict invalidation metadata. The stored identity must preserve
encounter-ordered variants, not just physical positions, and it must carry the
monomorphic/zero-diagonal sentinel used to exclude invalid loci. A disk dosage
cache that silently collapses duplicate positions or represents monomorphic
variants as ordinary zero vectors would recreate the exactness problems this
path is trying to avoid.

A bounded row/window cache and a duplicate-aware tiled/canonical fast path are
also deferred. Both target the real row recomputation bottleneck, but both
carry higher exactness and memory-risk than the prepared-input LRU and
unique-position tiled path.

## Recommendation

Do not prioritize no-cache as the primary fast path. Keep it experimental and
low-disk. If we invest further, use this order:

1. instrumentation-driven profiling;
2. bounded prepared-input LRU;
3. unique-position tiled local-search fast path;
4. no-cache metric workers or tiled metric reuse, depending on profiles;
5. duplicate-aware tiled/canonical fast path;
6. optional bounded row/window cache if row recomputation remains dominant.

The main performance track should move back to `r2-zarr`, where cached
normalized pairs already deliver the best runtime in the downloaded profiles.
