# Covariance Cache Redesign: Prototyping Plan

**Agent-oriented working log.** Raw, dated investigation notes — not proofread
for external readability. For current, human-readable status, see
`notes/findings/` and `docs/optimizations.md`.

Date: 2026-07-06

## Context

The covariance HDF5 cache is the dominant storage cost in this pipeline.
Prior work (shipped, `main`): switching the default codec from `lzf` to
`zstd` (`docs/optimizations.md` §9) gave a 12.4% size reduction and a 1.2x
aggregate speedup, losslessly. Prior work (unmerged, unvalidated): two
branches attempt float32 precision for `shrink_ld`
(`covariance-compression-float32`, a since-superseded round-to-fp32-store-as-
fp64 prototype; `shrink-ld-float32-dtype`, a real on-disk float32 dtype,
committed 2026-07-05/06 with a full `Snakefile.compression_diagnostics`
harness that has never actually been run against real data).

Goal for this branch: reduce cache size substantially (walltime/memory held
roughly constant, some precision degradation acceptable) without
compromising the pipeline's demonstrated correctness-sensitive stages. Two
findings from this round of design work change the shape of the problem
versus a pure "pick a better codec" framing:

1. **`_USE_ARRAY_DIAG = True`** (`matrix_analysis.py:21`) means matrix-to-
   vector already runs on the array path, not the dictionary path. The
   "matrix-to-vector dictionary scan" bottleneck in
   `notes/logs/post-covariance-optimization-review.md` (2026-05-03) is
   already resolved on `main`; that note is stale on this point.
2. **The raw Fourier/uniform metric is not a cosmetic diagnostic — it seeds
   every local-search decision, chromosome-wide.** `pipeline.py:481-482`
   passes `fourier_metric["sum"]`/`["N_zero"]` into every `LocalSearch(...)`
   call as `total_sum`/`total_n`; `local_search.py:1607-1623` uses it as
   `curr_sum = self.total_sum` and `min_metric = self.total_sum /
   self.total_n`, compared directly against every candidate position's ratio
   along the search path. Because the denominator (`curr_n`) varies per
   candidate, a biased seed does not cancel out of the ratio — it can flip
   which candidate wins, for every breakpoint on the chromosome, since they
   all share the same seed. This makes the raw metric *more*
   precision-sensitive than the vector itself, not less. It must not be
   served from a lossy bulk cache without independent validation.

## Design: three exact sidecars, not one compression knob

Rather than compressing the persisted bulk covariance cache directly, split
the pipeline's actual consumers and give each the cheapest *exact*
representation it needs:

### 1. Direct-vector sidecar (matrix-to-vector / raw minima)

Already prototyped on old `hdf5-experiments-direct-vector-r2-zarr` /
`hdf5-memory-optimizations` branches (pre-rename, `ldetect2`) as
`--vector-mode direct`: accumulate the correlation-sum vector during
covariance generation (fused with the pairwise LD kernel) instead of
rereading the persisted cache afterward. That prior work found it
"exactness mostly fixed" with residual chr9/chr14 vector-value diffs that
did not change final breakpoints — see
`notes/ldetect_optimization_findings.md` on that branch. Needs porting to
current `main`'s architecture (post-rename, post-`_USE_ARRAY_DIAG`); do not
resurrect the old branch wholesale.

### 2. Metric coverage-array sidecar (raw Fourier/uniform metric) — new

Verified against `_util/covariance_array.py:991` (`metric_from_arrays`):

- `N_zero` is pure combinatorics on block widths
  (`(total² − Σwidth²) / 2`, line 1011) — **zero dependence on covariance
  values**. Needs only locus positions. Already free in principle; today's
  code just happens to load full covariance arrays anyway because `sum`
  needs them.
- `sum` = sum of `r²` over pairs whose two endpoints fall in *different*
  `searchsorted`-assigned blocks (`i_blocks != j_blocks`, line 1018) — a
  boundary-crossing query, not the vector's fixed local-midpoint bucket.

Proposed sidecar: during covariance generation, for each pair `(i, j)`
accumulate a difference array (`diff[i] += r2; diff[j] -= r2`); a single
prefix-sum pass afterward gives, at every locus, the total `r²` mass of
pairs straddling that point. Once breakpoints are chosen, `sum` for the
whole chromosome is a lookup-and-add over the (small) breakpoint set — no
covariance reread.

**Caveat requiring verification before implementation:** this is exact only
if no pair straddles more than one breakpoint. Covariance window default is
5000 SNPs (`shrinkage.py:450`); typical breakpoint spacing
(`n_snps_bw_bpoints`) was ~10000 in the original reproduction — plausible
headroom, not guaranteed for all configs/populations. Check on real
chromosome data (max pair span vs. min breakpoint spacing) before trusting
the flat difference-array form; if violated, the correct generalization is
an offline 2D Fenwick/BIT over sorted pair endpoints, not a redesign of the
overall approach.

### 3. On-demand local-search window recompute — new

Local search currently rereads the *persisted* HDF5 partitions for each
breakpoint's window (`LocalSearchHDF5Partition` /
`local_search_hdf5_partition` in `local_search.py`), inheriting whatever
precision the bulk cache stores. Since local search is the one stage with a
demonstrated history of amplifying small numerical differences into large
output divergence (`notes/logs/local-search-divergence-asn22.md` — root
cause there was an algorithmic bug, not storage precision, but the
amplification mechanism transfers), it should not trust a lossy bulk cache.
Given windows are narrow (bounded by `trackback_delta`, not chromosome-wide)
and partitions already generate in parallel (`ldetect run --workers`),
recomputing each window's covariance from source VCF on demand, at full
float64 precision, once breakpoints are known, is plausibly cheap relative
to a whole-chromosome recompute (the option already ruled out for
walltime).

### Net implication

If sidecars 1 and 2 land, nothing in the *default* breakpoint-finding path
reads the persisted bulk cache except local search — and sidecar 3 removes
that dependency too. The bulk cache then drops off the critical path
entirely for standard runs, becoming a debugging / restart / `--high-precision`
artifact. That reframes its compression as low-stakes: it no longer needs to
protect the pipeline's core correctness, only the fallback/debug paths that
read it directly.

## Independent schema/compression findings (verified against code)

An external review (`notes/chat.md`) proposed several ideas; each was
checked against the current schema in `io/covariance_hdf5.py` and the
dictionary/Decimal fallback paths before accepting.

### Confirmed good, lossless, do first regardless of other directions

**Drop redundant per-row `lo`.** The compact schema
(`write_compact_covariance_partition_hdf5_chunks`,
`io/covariance_hdf5.py:306`) stores `lo` as a full per-row array *and* a
separate `index/lo_values` + `index/lo_offsets` CSR-style index that is
already sufficient to reconstruct which `lo` each row belongs to. Storing
`lo` per-row is genuinely redundant. A v2 compact schema:

```text
positions:    int32[n_snps]          # unique lo values (already lo_values)
row_offsets:  int64[n_snps + 1]      # already lo_offsets
hi_idx/delta: uint16 (or uint32 fallback if a partition exceeds 65535 SNPs)
shrink_ld:    float64[n_rows]
diag_val:     float64[n_snps]
```

Cuts per-row payload from `lo:int32 + hi:int32 + shrink_ld:float64 = 16 B`
to `hi_idx:uint16 + shrink_ld:float64 = 10 B` before compression, and local
deltas/indices should compress better than absolute genomic positions
(lower entropy). Exact, zero precision risk. Extends naturally to storing
local SNP indices instead of genomic positions throughout the row payload
(index-space downstream, convert to genomic coordinates only at
boundaries/output).

### Promising, needs its own validation harness

**Bounded fixed-point quantization of `r²`, not float32 of `shrink_ld`.**
`r²` is bounded in `[0, 1]`; uniform fixed-point quantization
(`r2_q = round(r2 * scale)`, e.g. `uint32`) gives an explicit, provable
worst-case absolute error (`n_crossing_pairs × 0.5 / scale`), unlike
float32's magnitude-dependent relative-precision loss — which gets worse
under squaring/normalization, exactly what produces `r²` from `shrink_ld`.
This is a better-justified lossy lever than either float32 branch, and
cheaper to implement than an error-bounded scientific codec (ZFP/fpzip).
Only usable for whichever tier ends up in the "lossy is fine" category
(candidate: the bulk cache, once sidecars 1-3 above take it off the
critical path) — not a replacement for exact reads elsewhere.

### Rejected as a wholesale replacement (verified against code)

**Store `r²` instead of signed `shrink_ld` + diagonals.** Checked
`read_partition_into_matrix_lean` → `_insert_lean_values`
(`io/covariance.py:245`): the dictionary path inserts raw signed
`shrink_ld`. `Metric._calc_metric_lean` (the `--high-precision` Decimal
path, `metric.py:109`) computes `r2 = cov*cov/(diag_x*diag_y)` itself from
that raw value at read time — it needs `shrink_ld` and diagonals
separately, not a precomputed `r²`. `--high-precision` is this project's
correctness oracle (`notes/logs/local-search-divergence-asn22.md`: "Treat
`--high-precision` as the correctness oracle until array local search is
fixed"). An `r²`-only cache mode would sever that oracle path, not just lose
signed-covariance/heatmap support as originally flagged. Only viable as an
*additional* mode alongside an exact/signed path, never a replacement for
the default cache.

## Priorities

1. **CSR-style schema cleanup** (drop redundant `lo`, local `hi` index/delta).
   Lossless, no precision risk, compounds with everything below. Do this
   first regardless of which other direction is pursued.
2. **Verify the single-breakpoint-crossing assumption** on real chromosome
   data (max pair span vs. min breakpoint spacing, across populations/
   configs actually used). Gates whether the metric sidecar is a simple 1D
   difference array or needs the 2D BIT fallback.
3. **Prototype the metric coverage-array sidecar**; validate its `sum`/
   `N_zero` output against `metric_from_arrays` exactly (bit-for-bit, not
   tolerance-based) on toy integration data and at least one real
   multi-partition chromosome.
4. **Port the direct-vector fused kernel** to current `main`'s architecture
   (the old prototype predates the `ldetect-lite` rename and
   `_USE_ARRAY_DIAG`).
5. **Local-search on-demand recompute** from source VCF for narrow windows,
   replacing reads from the (by then off-critical-path) bulk cache.
6. **Once 1-5 land**, the persisted bulk cache is low-stakes. Apply bounded
   fixed-point `r²` quantization (or the real-float32-dtype design from
   `shrink-ld-float32-dtype`, whichever benchmarks better) to that artifact.
   Run the existing, never-executed `Snakefile.compression_diagnostics`
   smoke test end-to-end to validate — this harness already exists and
   already reports final-BED exactness, not just vector-level diffs, so it
   directly tests for the seed-propagation failure mode identified above.
7. **Local-search task grouping** by overlapping partition/window span
   (already documented as open in
   `notes/logs/post-covariance-optimization-review.md` §2) — pairs well
   with step 5 once local search's "reads" become on-demand recomputes
   instead of cache hits; grouping avoids redundant recompute across
   overlapping windows the same way it would avoid redundant cache reads
   today.

## Explicit non-goals

- Low-rank LD approximations — unlikely to preserve local-search metric
  behavior without validation effort disproportionate to the payoff.
- Blind float32 downcast of `shrink_ld` — superseded by bounded fixed-point
  `r²` quantization (better-justified error model for this specific,
  bounded, squared-and-normalized quantity).
- Vector-only caching as the *sole* persisted artifact — already tried and
  rejected; loses the pairwise access local search and the raw metric need.
- `r²`-only cache mode as a wholesale replacement for signed
  `shrink_ld` + diagonals — breaks the `--high-precision` correctness
  oracle; additive-only.
- Whole-chromosome recompute-on-the-fly (no cache at all) — already tried
  and rejected for walltime; on-demand recompute here is deliberately scoped
  to narrow local-search windows only, not the full matrix.

## Validation methodology to reuse

- `examples/ldetect_original/Snakefile.compression_diagnostics` +
  `compare_compression.py` (from `shrink-ld-float32-dtype`): already
  generalized to an N-candidate-vs-baseline comparison producing exactness
  (vector/BED digest, boundary recall/precision/Jaccard), size, and
  performance rows. Reuse for every new cache-schema/precision mode rather
  than writing new comparison scripts.
- `vector_diffs.tsv`-style top-N divergence localization from the old
  `r2_zarr_exactness` workflow, for any case where exactness isn't bit-exact.
- Acceptance bar: the shipped zstd change hit 66/66 chromosome x population
  exact reproduction. Any lossless schema change (CSR cleanup) should be
  held to the same bar. Lossy modes (fixed-point `r²`, float32 dtype) need
  an explicit, pre-declared tolerance on final BED boundaries — not just
  vector-level max-abs-diff — given that vector-level tolerance and
  local-search-outcome tolerance are now known to be different quantities.
