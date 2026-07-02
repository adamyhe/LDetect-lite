# Streaming Covariance-Derived Cache Implementation Note

Date: 2026-07-02

## Context

`ldetect2` currently pays for covariance in three different ways:

1. Step 2 calculates shrinkage covariance rows and writes large per-partition
   HDF5 archives.
2. Step 3 rereads those HDF5 files to normalize rows into `r^2` and accumulate
   the diagonal-sum signal vector.
3. Metric and local search reread the same covariance rows, or load large array
   views of them, because they still need pair-level `r^2` values.

The current compact HDF5 schema is already much better than the older full
matrix cache, but it is still storage-heavy and I/O-heavy for whole
chromosome runs. This note proposes a clean branch direction that separates
the cheap win from the harder design problem:

- compute and cache the signal vector directly while covariance rows are being
  produced;
- keep pair-level covariance HDF5 as the default source of truth for metric and
  local search until a recompute cache is proven correct and faster;
- prototype a smaller haplotype/dosage-derived recompute cache as an optional
  mode, not as an immediate replacement for covariance HDF5.

## Current Invariants To Preserve

The vector path is currently implemented in `src/ldetect2/_util/vector_array.py`.
It matches the dictionary-backed legacy scan in `MatrixAnalysis` and has a few
non-obvious rules:

- Each covariance row is canonical `(lo, hi)` and sorted by `(lo, hi)`.
- Duplicates inside one partition are first-row-wins after canonicalization.
- Overlapping partitions are not globally deduplicated. Ownership is determined
  by partition order and output bounds.
- For matrix-to-vector, a non-final partition owns signal centers up to
  `int((partition_end + next_partition_start) / 2)`, while writes flush loci
  before `next_partition_start`.
- The final partition owns centers through `snp_last`, but the effective final
  locus is the last locus present at or before `snp_last`.
- A covariance pair contributes to the center SNP index `floor((lo_idx + hi_idx)
  / 2)`, with the legacy edge-case filter:
  `((hi_idx - lo_idx) % 2 == 0) or (lo_idx > 0)`.
- Rows only contribute when both diagonal shrinkage values are present and
  positive.
- The diagonal row itself contributes `diag^2 / (diag * diag) == 1` when the
  diagonal is positive.

That odd-delta edge-case is easy to miss. In the legacy outward scan, a pair
whose lower endpoint is the first active locus and whose index distance is odd
is never visited, even though `floor((lo_idx + hi_idx) / 2)` exists. A direct
antidiagonal accumulator must keep this behavior unless we intentionally change
the signal definition and update all reference comparisons.

## Proposed Phase 1: Sidecar Signal Cache During Covariance

Add an optional sidecar writer produced by `calc_covariance()` while compact
row chunks are generated. The sidecar should store partition-local signal
contributions, not the final chromosome-wide vector:

```text
{chrom}.{start}.{end}.signal.h5
```

Suggested datasets:

```text
signal/loci       int32 or int64, sorted physical positions
signal/sum_r2     float64, same length as loci
index/diag_pos    optional duplicate of covariance diagonal positions
metadata/start
metadata/end
metadata/snp_count
metadata/cutoff
metadata/ne
metadata/n_ind
```

The accumulator can run inside `_compact_pair_chunks_single_pass()` or a sibling
streaming function. It should avoid materializing all pair rows:

1. Build `pos_arr`, `hap_mat`, `hap_sums`, `gpos_arr`, and `j_stop_by_i` exactly
   as today.
2. Precompute diagonal shrinkage values once for every SNP index. This is cheap
   because the diagonal is one row per SNP and follows the same formula used by
   the pairwise kernel when `i == j`.
3. Allocate `partition_sums = np.zeros(n_snps, dtype=np.float64)`.
4. For each emitted compact chunk `(ii, jj, ds2)` before mapping to physical
   positions:
   - compute `idx_delta = jj - ii`;
   - keep rows with `(idx_delta % 2 == 0) | (ii > 0)`;
   - compute `center_idx = (ii + jj) // 2`;
   - require positive `diag[ii]` and `diag[jj]`;
   - add `ds2 * ds2 / (diag[ii] * diag[jj])` into `partition_sums[center_idx]`.
5. Write nonzero sums with their physical positions.

This accumulates the per-partition signal in the same pass that currently
streams compact covariance rows to HDF5. It removes the need to reread HDF5 for
Step 3 when the sidecar exists.

### Phase 1 Integration

Add a vector assembly path that reads sidecar signals instead of covariance
rows:

- planning should reuse `_plan_diag_vector_partitions()`;
- for each partition, read `signal/loci` and `signal/sum_r2`;
- only emit center loci owned by that partition's center bounds;
- merge and flush in partition order, matching `_finish_diag_vector_partition()`.

Do not aggregate sidecars by physical-position dedup alone. A locus can appear
in overlapping partitions, and whether its contribution is usable depends on
the center ownership bounds for that partition.

Recommended CLI shape:

```text
ldetect2 run --signal-cache {auto,off,only}
ldetect2 calc-covariance --signal-output PATH
ldetect2 matrix-to-vector --prefer-signal-cache
```

`auto` should write the sidecar while still writing covariance HDF5. `only`
should be considered experimental because local search and metric still need
pair-level data.

## Phase 2: Optional Recompute Cache For Metric And Local Search

A signal sidecar is insufficient for metric and local search. Those steps need
pair-level `r^2` crossing arbitrary candidate breakpoints. Dropping covariance
HDF5 without another cache would force expensive pairwise LD recomputation for
every metric/local-search window.

The smaller-cache idea to explore is a per-partition haplotype/dosage cache:

```text
{chrom}.{start}.{end}.geno.h5
```

Possible datasets:

```text
variant/pos       int32 or int64
variant/gpos      float64
variant/hap_sum   float32 or float64
haplotypes/bits   packed uint8 or uint64 bitset, shape roughly n_snps x n_haps
```

With a bit-packed representation, `n11` can be recomputed with bitwise AND plus
popcount. The shrinkage formula then uses cached `hap_sum`, genetic distance,
`theta`, `n_ind`, and `ne`. This avoids VCF parsing and tabix streaming on
recompute, and may be much smaller than pair-level covariance when LD density is
high.

However, it is only attractive if local search can batch candidate windows well.
If each breakpoint independently scans broad pair ranges, recomputing from
haplotypes can still lose to reading compact covariance rows.

## R2 Compute Vectorization Options

There are three plausible ways to speed up pair-level `r^2` computation from
haplotypes. They solve different problems and should be benchmarked separately.

### Row-Vectorized Uint8

This is the lowest-risk speed experiment. Keep `hap_mat` as `uint8`, loop over
the lower SNP index `i`, and compute all eligible upper SNPs `j` for that row
as a vector or small tile:

```python
j = np.arange(i, j_stop_by_i[i])
n11 = hap_mat[j] @ hap_mat[i]
```

Then compute `f11`, `d_naive`, shrinkage `ds2`, diagonal-normalized `r^2`, and
signal-center accumulation with array operations for that row.

Advantages:

- preserves the current sorted `(i, j)` emission order naturally;
- fits the compact HDF5 streaming writer and signal sidecar accumulator;
- avoids full dense `n_snps x n_snps` intermediates;
- is much easier to validate against `_pairwise_ld_impl` than bit-packed
  popcount.

Risks:

- `uint8 @ uint8` accumulation dtype and overflow behavior must be checked
  explicitly. Cast one operand or the block to at least `uint16`/`uint32` or
  `float32` if NumPy would otherwise accumulate too narrowly.
- BLAS acceleration may not apply to `uint8` inputs. This path may be fastest
  as a Numba row kernel rather than pure NumPy matmul.
- If implemented with temporary `hap_mat[j]` slices, chunk sizes must be tuned
  to avoid copying large row windows repeatedly.

### Chunked Dense Matrix Multiplication

For dense local SNP windows, compute `n11` for rectangular tiles:

```python
n11_tile = hap_mat[i0:i1].astype(np.float32) @ hap_mat[j0:j1].astype(np.float32).T
```

This can use highly optimized matrix multiply kernels and may be the fastest
near-term walltime option when many pairs in a tile survive the genetic-distance
and LD cutoff filters.

Advantages:

- simple expression of the core count matrix;
- high arithmetic intensity;
- easy to compare numerically against the current kernel for one tile.

Risks:

- intermediate memory grows as `tile_i * tile_j`. A `1024 x 1024` `float32`
  tile is about 4 MB for `n11` alone, before masks, `ds2`, and output buffers.
- It computes dense tile entries even when `j_stop_by_i` makes the usable upper
  triangle sparse.
- It can disturb output ordering unless tiles are emitted in strict row-major
  partition order.
- Casting `uint8` haplotypes to `float32` can dominate time if done repeatedly;
  pre-casting the whole partition doubles or quadruples haplotype memory.

This path is best treated as a benchmarked fast path for dense partitions, not
as the only implementation strategy.

### Bit-Packed Popcount

Bit-packing stores each haplotype row as machine words and recomputes:

```text
n11 = popcount(bits_i & bits_j)
```

This is the best storage story and a promising long-term recompute backend, but
it is a larger semantic change than row-vectorized `uint8`.

Advantages:

- roughly 8x smaller than `uint8` haplotypes before metadata and compression;
- avoids VCF parsing and tabix streaming when used as a recompute cache;
- can be very fast if popcount is implemented with efficient machine-word loops.

Risks:

- NumPy does not provide a universally optimal SIMD popcount path across all
  supported versions, so this likely needs a Numba kernel, lookup-table kernel,
  or small native extension.
- Padding bits for `n_haps` not divisible by the word size must be masked
  carefully.
- Word endianness, pack order, and reproducible cache schema need explicit
  tests.
- It does not remove the need to reproduce partition ownership, duplicate
  handling, diagonal normalization, or first-locus odd-pair behavior.

Recommended ranking for this clean branch:

1. Implement the signal sidecar with the existing pair stream.
2. Benchmark row-vectorized `uint8` and chunked dense matmul against the current
   Numba pair loop on the same partition.
3. Prototype bit-packed popcount only after the simpler vectorized paths show
   whether compute or storage is the real remaining bottleneck.

## Cost Model

Approximate storage per partition:

- compact covariance HDF5 row: `lo` + `hi` + `shrink_ld`, normally about
  16 bytes raw with int32 positions and float64 shrinkage, before HDF5 overhead
  and compression;
- sidecar signal: one position plus one float64 per SNP, normally about
  12 bytes raw per SNP with int32 positions;
- uncompressed haplotype cache: `n_snps * n_haps` bytes if stored as uint8;
- bit-packed haplotype cache: `n_snps * ceil(n_haps / 8)` bytes, plus variant
  metadata.

For a 5,000-SNP partition and 1,000 diploid samples:

- uint8 haplotypes: about 10 MB raw;
- bit-packed haplotypes: about 1.25 MB raw;
- signal sidecar: about 60 KB raw;
- compact covariance: depends on cutoff and LD density; 10 million retained rows
  is about 160 MB raw before compression.

Walltime expectations:

- Phase 1 should add little CPU cost because every retained covariance row is
  already being visited; the extra work is a few vectorized index operations and
  one grouped accumulation per chunk.
- Phase 1 should reduce Step 3 walltime substantially on large runs by replacing
  HDF5 row reads plus diagonal lookup plus normalization with small sidecar
  reads.
- Phase 2 may reduce storage dramatically, but walltime is uncertain. It trades
  sequential HDF5 row reads for popcount-heavy recomputation. Benchmark before
  changing defaults.

Memory expectations:

- Phase 1 needs one `float64[n_snps]` accumulator and one `float64[n_snps]`
  diagonal array per active covariance worker. That is small relative to
  `hap_mat`.
- Parallel covariance already multiplies `hap_mat` memory by `--workers`; adding
  a signal sidecar does not materially change that.
- A chromosome-level recompute cache should not load all haplotype partitions at
  once. It should stream or memory-map partition windows, especially when
  `--local-search-workers` is greater than one.

## Edge Cases And Traps

### Duplicate Physical Positions

`calc_covariance()` currently skips duplicate physical positions before building
arrays, because covariance rows are keyed by physical position. The sidecar must
be built after this deduplication, using the final `pos_arr`. Never accumulate
against pre-dedup VCF row indexes.

### Unsorted Positions

The compact single-pass writer assumes `pos_arr` is strictly increasing. If it
is not, current code falls back to materialized pair arrays and canonical HDF5
writing. A signal sidecar has the same constraint: index midpoint logic only
matches the legacy locus-list scan when SNP indexes are in physical order. For
unsorted positions, either sort all variant arrays before covariance or disable
the sidecar and use the existing HDF5 vector path.

### Empty Partitions

The current empty-partition path returns before writing output. If sidecar
existence is used as a completion marker, empty partitions need an explicit
empty sidecar or the runner must know that no sidecar is expected. Prefer
writing a valid empty sidecar so restart validation is simple.

### Missing Diagonals

With the current kernel, diagonal rows normally pass the cutoff and are emitted.
Still, the vector definition requires positive diagonal values. The direct
accumulator should use the same diagonal formula as the emitted row, and tests
should include a fixture with zero or absent diagonal values in a manually
written HDF5 partition to keep fallback behavior honest.

### Partition Overlap Ownership

Do not decide final vector rows inside `calc_covariance()`, because a partition
does not know enough about neighboring partition ownership unless the full
partition list is passed in. Store partition-local signal sums, then assemble
with the same partition plan used by matrix-to-vector.

### Boundary Inclusivity

The current vector planner uses inclusive lower bound for the first emitted
partition and exclusive lower bound after the previous partition's `end_locus`.
The HDF5 vector path uses `center_left <= center_idx < center_right`, where
`center_right` is one past the last owned center. Sidecar assembly should copy
that logic, not reimplement it with physical-position comparisons in ad hoc
ways.

### Duplicate Pairs Across Partitions

Overlap means the same `(lo, hi)` pair can appear in multiple partition files.
For the signal, this is fine only because center ownership assigns each center
locus to a partition. For metric/local search, duplicate handling is different:
local search preserves partition and row order and applies first-row-wins for
canonical duplicate pairs in the active stream. A recompute cache must reproduce
that behavior if it replaces HDF5.

### First-Locus Odd Pairs

As noted above, odd-width pairs with `lo_idx == 0` are not reachable by the
legacy antidiagonal scan. This is the most likely off-by-one regression in a
direct signal accumulator. Include an explicit test where loci `[100, 200]`
have a covariance row `(100, 200)` and verify it does not contribute to the
first center.

### Last Partition Final Locus

For the final partition, the legacy path writes through the last locus present
at or before `snp_last`, not necessarily the numeric `snp_last` value. Sidecar
assembly should derive final writable loci from stored sidecar loci, not from a
synthetic range of physical positions.

## Recommended Implementation Order

1. Add a pure helper that accumulates vector sums from index arrays:
   `accumulate_signal_from_pairs(ii, jj, shrink, diag) -> sum_r2`.
2. Unit-test the helper against small legacy `MatrixAnalysis` fixtures,
   especially first-locus odd pairs, diagonal rows, missing/zero diagonals, and
   duplicate positions after VCF deduplication.
3. Add sidecar HDF5 read/write helpers with validation.
4. Teach `calc_covariance()` to optionally write the sidecar while producing
   compact covariance chunks.
5. Add sidecar-based `matrix-to-vector` assembly behind a feature flag.
6. Compare sidecar vector output against the current HDF5 vector path on:
   - toy integration data;
   - synthetic overlapping partitions with cross-partition duplicate pairs;
   - a real small chromosome run, preferably chr22.
7. Add a benchmark harness for four pair kernels on the same partition:
   current Numba pair loop, row-vectorized `uint8`, chunked dense matmul, and
   bit-packed popcount if available.
8. Only after Phase 1 is stable and benchmarks identify the remaining
   bottleneck, prototype the bit-packed haplotype recompute cache for
   metric/local search.

## Validation Checklist

Run the existing focused tests first:

```text
uv run pytest tests/test_shrinkage.py tests/test_covariance_io.py tests/test_covariance_array.py
uv run pytest tests/test_local_search.py tests/test_metric.py tests/test_find_minima.py
uv run pytest tests/integration/test_pipeline.py
```

Add new tests for:

- direct signal accumulator equals the current HDF5 vector path for one
  partition;
- direct signal accumulator equals the current HDF5 vector path for overlapping
  partitions;
- odd first-locus pairs are skipped;
- sidecar assembly respects `snp_first` and `snp_last`;
- empty partitions write and validate as empty sidecars;
- `ldetect2 run` restart logic treats covariance and signal sidecars
  independently.

## Default Recommendation

Make the signal sidecar an additive cache, not a replacement cache. It is small,
cheap, and should remove a full post-covariance scan. Keep compact covariance
HDF5 as the default source for metric and local search until a recompute cache
can match the duplicate, indexing, and partition-boundary semantics under the
existing regression tests.
