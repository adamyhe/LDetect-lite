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

## Implementation update (2026-07-06)

Priorities 2, 3, 4, and (a variant of) 6 were prototyped and validated this
round, on this branch, as library-level modules with their own test suites —
**not** wired into `calc_covariance`, the CLI, or `pipeline.py`. Scoping
decision (explicit, asked of the user): schema/structural changes are
prototype-only in this round, deferring the "rewire every consumer +
version-dispatch old files" migration to a later, separately-scoped pass.

### Sidecars 1 and 2 (`src/ldetect_lite/_util/covariance_sidecars.py`)

Built as a single fused accumulator that **tees** the exact
`CovarianceRowChunk` iterator `calc_covariance` already uses to persist the
HDF5 partition (`shrinkage.py`'s `_compact_pair_chunks_single_pass` output),
rather than a second, independently-invoked kernel pass. This was a
deliberate response to the old `ldetect2`-era direct-vector prototype's
unresolved chr9/chr14 vector residual (§1 above): tee-ing guarantees
bit-identical `(lo, hi, shrink_ld)` inputs to both the persisted cache and
the sidecar, structurally ruling out the "two kernel invocations quietly
disagree" failure class rather than just being more careful about it. The
only independently-derived quantity is the per-locus diagonal
(`shrinkage._diag_values_impl`, a closed-form vectorized collapse of the
kernel's `i == j` branch — pure function of `hap_sums`, no second nested-loop
pass), verified bit-exact against the persisted `diag_pos`/`diag_val`.

Results:

- **Direct-vector fragment: bit-exact**, both single-partition and across
  two overlapping partitions, against the existing post-hoc HDF5-based
  vector build (`vector_array.py`). This is the sidecar with the historical
  divergence risk, and it's now the most confidently exact piece of this
  round's work.
- **Metric coverage-array fragment (original flat 1D difference-array
  design): confirmed broken on real data**, not just theoretically fragile.
  Measured directly on the real EUR chr2 fixture
  (`tests/data/cov_matrix/chr2/chr2.39967768.40067768.h5`): max surviving
  pair span is **99,899 bp** against a minimum real breakpoint gap of
  **4,646 bp** (from `fourier_ls` minima) — **91% of all 225,402 surviving
  pairs exceed the minimum gap**. The single-breakpoint-crossing assumption
  §2 flagged as "plausible headroom, not guaranteed" is, on this real
  window, the overwhelming common case of violation, not an edge case. The
  flat difference array double-counts any pair crossing more than one
  breakpoint — see
  `tests/test_covariance_sidecars.py::test_metric_coverage_violates_single_crossing_assumption_when_breakpoints_are_close`
  for a constructive, reproducible counterexample, and
  `test_single_breakpoint_crossing_assumption_on_real_fixture` for the real-
  data measurement above. **This flat-array implementation is superseded by
  the exact decomposition below and should not be built on further** —
  kept in the codebase for now as a documented negative result, not a
  competing design.

### Priority 2's fallback, built and benchmarked (`src/ldetect_lite/_util/banded_metric_coverage.py`)

Per §2's own stated fallback ("if violated, the correct generalization is an
offline 2D Fenwick/BIT over sorted pair endpoints"), replaced the flat array
with an exact decomposition that's correct for *any* number of breakpoints a
pair crosses: `sum_crossing = total_mass - Σ intra_block_mass(block)`, where
blocks are the position ranges between consecutive breakpoints. A pair fully
contained in one block is excluded exactly once; a pair crossing N≥1
breakpoints is never "intra" anywhere and is counted exactly once via
`total_mass` — no double-count, unlike the flat array.

Two variants were built and benchmarked against each other, per an explicit
follow-up question about whether a persisted 2D structure is worth its
storage cost relative to just reading the (smaller) compact cache on demand:

- `sum_crossing_linear_scan`: one O(n_rows) pass over the compact cache's
  own arrays, zero extra storage.
- `MergeSortRangeSumTree`: a persisted O(n log n)-space merge-sort/segment
  tree for O(log² n) queries without touching per-row arrays at query time.

Both are exact against `metric_from_arrays`, including the close-breakpoints
case that broke the flat array. **Benchmark result (real chr2 fixture,
225,402 pairs): the persisted tree is 42.5x larger than the v2 compact cache
(72.7 MB vs. 1.7 MB) for negligible-to-negative query-time benefit** — 0.9x
(i.e. slower) at a realistic 12-breakpoint set, both variants sub-
millisecond regardless. Root cause: a persistent structure only pays for
itself when its O(n log n) build cost is amortized over many repeated
queries, and this metric is evaluated ~4 times per chromosome (once per
fourier/fourier_ls/uniform/uniform_ls subset) via `pipeline.py:357`, not
repeatedly — confirmed by reading `find_minima.custom_binary_search_with_trackback`,
which only evaluates a vector-smoothing function, never `metric_from_arrays`.
**Conclusion: drop the persisted-tree variant; `sum_crossing_linear_scan`
over the compact cache is strictly better here** (exact, no extra storage,
already fast enough). A persisted structure would only be worth revisiting
if some future consumer needed many repeated crossing-sum queries against
the same partition — no such consumer exists in this pipeline today.

### Priority 1, prototyped (`src/ldetect_lite/_util/compact_schema_v2.py`)

v2 schema exactly as sketched above (drop per-row `lo`, `hi` as a rank index
into a per-partition `positions` array). Round-trips bit-exact against v1 on
the real chr2 fixture. **Real-world size reduction: 6.4%** (1,827,417 →
1,710,775 bytes for 226,074 rows) — well short of the ~37.5% pre-compression
estimate (`16 B → 10 B` per row), because zstd already exploits most of the
same redundancy the CSR cleanup targets. Useful calibration for priority 6:
lossless schema tricks are near diminishing returns under zstd; the larger
remaining lever is genuinely lossy (bounded fixed-point `r²` quantization).

### Not done this round

Priority 5 (local-search on-demand recompute) and the task-grouping item
(priority 7) are unstarted — sidecars 1/2 being prototype-only (not wired
into the read paths they're meant to replace) means the "bulk cache drops
off the critical path" net implication (§ above) hasn't actually landed yet
either; it's demonstrated as *feasible*, not *shipped*.

### Open decision: merge and release

Recommended: merge this branch's work into `main` as ordinary internal
development (all of it is additive — new modules, new tests, zero changes
to `calc_covariance`'s default behavior or any existing reader) but **do
not** cut a `v0.1.1` tag for it. Nothing here is wired into `calc_covariance`,
the CLI, or `pipeline.py`, so there is no user-observable change a version
bump would honestly describe yet. Before merging, reconcile the two
metric-coverage implementations living side by side in
`covariance_sidecars.py` (flat, known-wrong) and `banded_metric_coverage.py`
(exact, recommended) so a future reader doesn't mistake the former for a
live option — at minimum, a doc pointer from the flat implementation to its
replacement; ideally, delete the flat implementation once nothing depends on
its tests as a documented negative result.

Next planned step: priority 6, bounded fixed-point `r²` quantization,
scoped down given local search (priority 5, unstarted) still reads the
persisted cache directly — quantization needs its own validation harness
distinguishing vector-level tolerance from final-BED-outcome tolerance, per
the validation methodology above.

## Priority 5 prototype + VCF backend timing (2026-07-06)

Investigated why a coverage sidecar can't reduce total storage (metric
computation and local search both still read the shared bulk cache directly
today) — this reasoning pointed at priority 5 as the actual prerequisite for
priority 6, not just the next item on the list: it decouples local search
from cache precision **without adding storage** (recomputes from source VCF,
which is already on disk for a different reason), unlike any sidecar
artifact (which duplicates O(n_pairs) data). Prototyped and validated,
library-only (not wired into `pipeline.py`/the CLI — no VCF path is threaded
through `find_breakpoints` today; that's a separate, larger change to make
only if this prototype's results justify it).

### On-demand recompute (`src/ldetect_lite/_util/local_search_vcf_recompute.py`)

`recompute_partition_to_hdf5` needed almost no new code: `calc_covariance`
is already exactly the function `cmd_run.py`'s `_calc_partition` uses to
generate one partition on demand (tabix-slice the region, pipe into
`calc_covariance`). Recomputing "on demand" is just calling that same
function again, later, for the same `(start, end)` bounds a local-search
call already needs via `get_final_partitions` — no new pairwise-LD kernel,
and critically, no need to re-derive the ASN22 multi-partition boundary
semantics (dummy loci, `end_locus`-from-next-partition-start), since those
live entirely in `local_search.py`'s existing, already-fixed downstream
reading code, which this prototype never touches — it only changes where
the partition HDF5 file comes from.

Validated on a real slice of the MacDonald2022 chr9 VCF (gitignored,
local-only fixture; 20 EUR individuals, two overlapping windows
chr9:100,000-250,000 / chr9:200,000-400,000, mirroring how real chromosome
partitioning overlaps):

- **Row-level determinism**: calling `recompute_partition_to_hdf5` twice
  independently for the same bounds produces bit-identical HDF5 partitions.
- **End-to-end `LocalSearch` equivalence**: `LocalSearch` reading a "cache"
  store vs. a "recompute" store (both independently generated, same source
  data) returns identical breakpoints and metrics, in both the array
  (`use_decimal=False`) and Decimal (`use_decimal=True`) paths, across a
  window spanning both partitions — the multi-partition path ASN22 broke.

Both pass. This confirms the core hypothesis: recompute is a safe,
zero-extra-storage substitute for reading the persisted partition, at least
for the boundary conditions exercised here (a fuller validation before any
real wiring would want more real regions/populations, matching this
project's existing 66/66 chromosome × population acceptance bar).

### VCF backend timing (`src/ldetect_lite/_util/vcf_backends.py`)

Added `pysam`/`cyvcf2` as a new `vcf-benchmark` optional extra (not a core
dependency — `pyproject.toml`). Three equivalent "extract phased genotypes
for a region" readers (naive per-line text parsing / pysam / cyvcf2),
verified to produce identical output on real data before trusting their
timing. Benchmarked on chr9:100,000-700,000 (~20,500 variants, matching
local search's typical ~10,000-20,000 SNP window scale), swept over
individual count (20 / 100 / 417, the full `EUR_inds.txt`):

| n_individuals | naive  | pysam            | cyvcf2           |
|---------------|--------|------------------|------------------|
| 20            | 0.87s  | 0.85s (1.03x)    | 0.59s (1.49x)    |
| 100           | 1.24s  | 1.77s (0.70x)    | 0.84s (1.47x)    |
| 417           | 2.61s  | 5.23s (0.50x)    | 2.00s (1.31x)    |

**Individual count matters more than region size here, and the two
candidate libraries diverge sharply**: pysam's per-sample dict-like
accessor (`record.samples[ind]["GT"]`) has real per-call Python overhead
that scales *worse* than the naive parser as individual count grows —
2x *slower* than naive at 417 individuals, not a candidate worth adopting.
cyvcf2's flat `variant.genotypes` list stays ahead at every size tested
(1.3-1.5x), a real but modest win, not the order-of-magnitude a C-accelerated
library might suggest — most of the remaining cost is downstream Python-side
haplotype-matrix construction and pairwise LD computation, not VCF parsing
itself. `polars-bio` was not benchmarked (much newer/more niche, not used
anywhere in this ecosystem currently) — noted as un-benchmarked, not
rejected.

**Conclusion: cyvcf2 is worth adopting if/when the VCF-parsing path is
revisited (real but modest 1.3-1.5x win, no scaling cliff); pysam is not.**
Not switching the production parser this round — this was a measurement
pass, and priority 5's actual walltime cost is currently dominated by
`calc_covariance`'s pairwise LD kernel and per-partition subprocess
(`tabix`) startup, not genotype parsing, for the region sizes tested here.

### Next steps

- If priority 5 moves toward real wiring: thread a VCF path through
  `find_breakpoints` → `_run_local_search` → `LocalSearch`, add a CLI flag,
  and validate multiprocessing-safety of per-worker VCF/tabix handles.
- If the VCF backend switch is picked up: swap `calc_covariance`'s parser to
  cyvcf2 for the genotype-extraction step only (keep the existing pairwise
  kernel and array-building logic unchanged), and re-run this benchmark at
  full-chromosome scale to see whether the 1.3-1.5x win holds or shrinks
  further behind the pairwise-kernel cost.

## Priority 5 and direct-vector sidecar wired into `ldetect run` (2026-07-06)

Both wired in as opt-in flags (`--fused-vector`, `--local-search-source
{cache,vcf-recompute}`), off by default. See `_cli/cmd_run.py` and
`pipeline.py`. `--fused-vector` only takes the fast path when every
partition is freshly computed this run (falls back to the post-hoc Step 3
read otherwise, rather than half-solving mixed fresh/cached partitions).
`--local-search-source vcf-recompute` is scoped to the `workers=1`,
non-Decimal, grouped local-search path only — combined with
`--local-search-workers > 1` or `--high-precision`, it raises an error
before Step 2 even runs. Validated bit-exact/behavior-identical against the
existing paths (unit tests, real chr9 data end-to-end through
`find_breakpoints`, and a manual `ldetect run` smoke test producing
byte-identical BED/JSON output on real data). The metric-coverage sidecar
was explicitly **not** wired in — `metric_from_files` was already exact in
production; only the flat-array *replacement attempt* was ever broken.

## v2 lo-less schema: scoped, then explicitly deferred (2026-07-06)

Investigated wiring `compact_schema_v2.py` in as the new default compact
writer. Turned out to be a much larger lift than "swap the writer" once
scoped:

- **No streaming writer exists.** `write_v2_partition` requires fully
  materialized `lo`/`hi`/`shrink_ld` arrays (`np.union1d(lo, hi)` over
  complete data) — incompatible with `calc_covariance`'s default chunked/
  streaming write path (`write_compact_covariance_partition_hdf5_append`),
  which exists specifically to avoid materializing a whole partition in
  memory for large chromosomes.
- **No reader API parity.** Every consumer (`local_search.py`,
  `covariance_array.py`, `vector_array.py`, the `--high-precision`
  dictionary path in `io/covariance.py`) needs `row_count`,
  `read_diagonal`, `read_loci`, `iter_rows`, and `iter_owned_rows`. v2 only
  supports whole-partition reconstruction today.
- **No version-dispatch mechanism exists at all**, even nominally — v1's
  `format`/`version` HDF5 attrs are written but never actually branched on
  by any reader; `HDF5CovariancePartitionReader` reads fixed dataset paths
  unconditionally. Introducing v2 means building dispatch from scratch, not
  extending something that already exists.

Weighed against this cost: the measured real-world gain is **6.4%**, not
the ~37.5% pre-compression estimate, because zstd already exploits most of
the same redundancy the CSR cleanup targets (already noted above, but worth
restating as the decisive factor here). **Decision (confirmed with user):
skip the v2 migration.** Priority 6 (bounded fixed-point `r²` quantization)
is the better-ROI remaining lever — it doesn't require touching the reader
API surface or building version dispatch, and wasn't invalidated by this
finding. `compact_schema_v2.py` and its tests remain in the repo as a
validated prototype/reference, not a migration in progress.

Also explicitly considered and declined: wiring `banded_metric_coverage.py`
into production. Unlike priority 5/direct-vector, it doesn't replace a real
production dependency — `metric_from_files` was already exact the whole
time, so there's nothing broken for it to fix. It's also built directly
against v2's rank-encoded arrays, so using it would mean resurrecting v2 or
re-deriving `metric_from_files`'s own logic against v1 data. No path forward
here that isn't strictly worse than what's already shipped.

## Real-scale profiling harness for priority 5 / --fused-vector (2026-07-06)

Both flags were only validated for *correctness* this session (small real
regions, one single-partition manual smoke test) — never for wall-clock/
memory cost at real chromosome scale. Added a self-contained diagnostic
Snakemake workflow that runs `ldetect run` three ways per chromosome —
baseline, `--fused-vector`, `--local-search-source vcf-recompute` (pinned to
`--workers 1`, the only validated configuration) — against identical real
input, checks BED output is byte-identical across all three (it must be —
these are three exact computations of the same thing), and reports
wall-clock/peak-RSS deltas in `summary.tsv`.

**Location**: first built under `examples/MacDonald2022/` (real chr9 data
was already local there from this session's earlier priority-5 validation
work), using `/usr/bin/time -v` for timing/memory, matching that directory's
main Snakefile's own `run_ldetect` convention. Reconsidered and moved to
`examples/ldetect_original/` instead: that directory already hosts the
closest sibling workflow (`Snakefile.compression_diagnostics`, an
near-identical N-mode real-data comparison pattern) and the family of
diagnostic-only Snakefiles this harness actually belongs to conceptually —
it isn't about reproducing a specific paper's block sets, unlike both
example directories' main pipelines. The move also switched from
`/usr/bin/time -v` (GNU-only, unavailable on macOS, confirmed by trying it
locally) to Snakemake's native `benchmark:` directive — the same mechanism
`Snakefile.compression_diagnostics` already uses, portable, and needs no
custom output parsing (eliminating the class of bug the `/usr/bin/time`
version had — see below). Final files:
`examples/ldetect_original/Snakefile.priority5_profiling`,
`priority5_profiling.yaml`, `scripts/compare_priority5_mode.py`.

**Real chr21/chr22 results** (obtained from the MacDonald2022 version,
`--workers 4` for baseline/`fused_vector`, `--workers 1` for
`vcf_recompute`, before the move — the underlying feature performance
characteristics this measures are about `ldetect run`'s flags, not about
which example directory the harness lives in, so recorded here regardless):

| chrom | mode | bed_exact | baseline | mode | speedup | baseline peak | mode peak | peak ratio |
|---|---|---|---|---|---|---|---|---|
| 21 | fused_vector | True | 105.95s | 101.92s | 1.04x | 1278.1 MiB | 1276.9 MiB | 1.00x |
| 21 | vcf_recompute | True | 105.95s | 538.32s | 0.20x | 1278.1 MiB | 462.2 MiB | 0.36x |
| 22 | fused_vector | True | 107.70s | 103.52s | 1.04x | 1793.7 MiB | 1824.7 MiB | 1.02x |
| 22 | vcf_recompute | True | 107.70s | 544.70s | 0.20x | 1793.7 MiB | 485.8 MiB | 0.27x |

BED output byte-identical in all four rows — correctness holds at real
chromosome scale, not just the small regions tested when these were built.

- **`--fused-vector`**: consistent, modest ~4% wall-clock win, no memory
  cost. Low-risk; a reasonable default-on candidate once more chromosomes
  are checked.
- **`--local-search-source vcf-recompute`**: **~5x slower** but uses only
  **27-36% of baseline's peak memory**. This directly answers the open
  question below: recompute overhead clearly dominates `vcf_recompute`'s
  wall-clock, confirming per-partition recompute (tabix + kernel) is the
  bottleneck, not something else — parallelizing it across breakpoint
  groups would plausibly close most of this gap, and is now justified by
  data rather than speculation. Independent of that follow-up, the
  memory/speed tradeoff is real and already usable today in
  memory-constrained scenarios (e.g. many chromosomes running concurrently
  under a fixed memory budget) without any further work.

**Validation status of the final (`ldetect_original`) location**: DAG
resolves cleanly in dry-run against a real chromosome config
(`--config profiling_chromosomes='[21]'`). Not run live from this location
— doing so requires `bcftools` (for `priority5_filter_vcf`), which isn't
installed on this dev machine; per user instruction, system packages aren't
installed to self-validate a deliverable meant to run elsewhere. Whoever
runs this should confirm `wget`/`bcftools`/`tabix` are on `PATH` first. The
GNU-time parsing bug from the earlier version (the "Elapsed (wall clock)
time" label's own `(h:mm:ss or m:ss)` hint text contains colons, breaking a
naive first-colon split) is moot now that `benchmark:` is used instead, but
is recorded here since it was a real, non-obvious bug worth remembering the
shape of.

Remaining open question, now informed by the results above: whether
multi-worker `vcf_recompute` support is worth building (needs a locking
scheme for concurrent recompute into a shared temp store across worker
processes) is a `speed vs. engineering effort` call now, not a `does this
even matter` call — the 5x gap is large enough that parallelizing across,
say, 4 workers could plausibly bring `vcf_recompute` close to baseline
wall-clock while keeping most of its memory advantage.
