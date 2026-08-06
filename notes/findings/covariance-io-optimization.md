# Covariance I/O optimization — findings

**Findings summary (current as of 2026-08-05).** Distilled for human review.
Full external review text and process notes:
`notes/logs/covariance-io-optimization-review.md`.

## Status: two items verified and shipped, rest rejected or deferred

An external review proposed seven ranked I/O optimizations for the
covariance-calculation stage (`calc_covariance` and its callers), following
up on the long-standing observation in
[`notes/findings/bitpacked-ld-kernel.md`](bitpacked-ld-kernel.md) that
"VCF/BCF ingestion and HDF5 writes are still substantial parts of the
covariance stage." Each ranked item was checked against the current code and,
where the claim was concrete and testable, benchmarked before deciding
whether to implement it.

## Shipped

1. **Per-process genetic-map caching + vectorized parse**
   (`_util/reference_panel.py::read_genetic_map`). Every partition's
   `calc_covariance` call re-read and fully re-parsed the *entire* gzipped
   genetic map from scratch, even though one `ldetect run` invocation always
   passes the same chromosome-wide map to every partition. Now `@cache`d per
   process (a `ProcessPoolExecutor` worker that handles multiple partitions
   parses once, not once per partition) and parsed with `np.loadtxt` instead
   of a manual per-line Python loop. Measured on a synthetic 1M-row map
   (realistic per-chromosome scale): current per-call cost ~0.29s; at a
   realistic ~30 partitions/chromosome that's ~8.8s of pure waste per
   chromosome run, previously invisible in any profile because it's spread
   thinly across many partition tasks rather than showing up as one slow
   step. `functools.cache` is safe here because the returned dict is
   read-only downstream (verified: the only write into it is inside the
   function itself) and every caller passes a stable, unique path per run.

2. **`variant.genotype.array()` instead of `variant.genotypes`**
   (`_util/reference_panel.py::read_reference_panel`). The old path built a
   Python list of `(allele1, allele2, phased)` tuples per variant, then
   looped over every individual in pure Python to unpack and validate them.
   `cyvcf2`'s `.genotype.array()` returns the same data as one vectorized
   `(n_samples, 3)` `int16` array, letting the phased/missing check and the
   haplotype-row build run as vectorized NumPy ops instead of a per-sample
   Python loop. Measured on a synthetic 500-sample/5000-variant panel: 1.71x
   faster, byte-identical output (positions, haplotype rows, and skip counts
   all matched the old path exactly). The now-unused `n_haps` parameter was
   removed from `read_reference_panel`'s signature as part of this change.

Both are covered by `tests/test_reference_panel.py` (new) and the existing
`tests/test_shrinkage.py` `calc_covariance` suite, which continued to pass
unchanged (missing-individual, unphased/missing-genotype, and
duplicate-position edge cases all still exercised through the new code path).

## Rejected or deferred

- **Whole-chromosome packed haplotype cache** (the review's top-ranked item).
  Plausible in principle — avoids repeated decode of genuinely *overlapping*
  partition regions — but unconfirmed: an isolated VCF-open-cost measurement
  found it negligible (~0.22ms), and the two shipped fixes above already
  capture the verified waste. The overlap-driven redundancy this would target
  is real but unmeasured (partitions overlap by a bounded recombination-cutoff
  extension, not a doubling), and the fix is the most invasive item proposed
  (new cache-key/invalidation scheme keyed on panel+index+samples+map+format
  version, node-local storage staging). Needs profiling evidence that
  overlap-driven re-decoding is actually large before it's worth the
  complexity — not undertaken this round.
- **CSR/banded covariance storage schema.** This is the same idea as the "v2
  lo-less schema" already prototyped and shelved on the unmerged
  `covariance-cache-redesign` branch: real measured reduction was 6.4%, far
  short of the ~37.5% pre-compression estimate, because zstd already
  exploits most of the same redundancy. That branch's own measurement — most
  of compressed covariance size is `shrink_ld` (incompressible float64), not
  the position columns a CSR layout would shrink — is consistent with this
  review's independent "shrink_ld is ~82% of storage" observation. No new
  action.
- **Additional HDF5 codec benchmarking** (Blosc2+LZ4+bitshuffle, etc.). Low
  priority: zstd is already the shipped default (`docs/optimizations.md` #9,
  12.4% size reduction, 1.2x speedup), and the CSR finding above means
  `shrink_ld`'s incompressibility bounds how much any codec swap can help.
  Not pursued.
- **PGEN/VCZ persistent panel storage.** Speculative; a real infrastructure
  decision (useful specifically when one panel is reused across many
  populations/runs) that needs its own scoping, not something to adopt
  opportunistically alongside this review.
- **Cython/CuPy for the pairwise kernel.** Already answered independently: a
  chunked-matmul kernel (the natural GPU/BLAS reformulation) benchmarked
  4-45x *slower* than the current bitpacked-popcount kernel at realistic
  partition/haplotype scale (see the matmul benchmark discussion in this
  session's history; not yet its own findings doc). This review's own
  conclusion — "only after I/O and data-layout changes" — agrees.
- **Polars-Bio.** The review's own recommendation is to keep `cyvcf2`/BCF
  rather than adopt Polars-Bio VCF/VCZ for the production hot path. Agreed;
  no action needed.

## Practical guidance

- `read_genetic_map`'s cache assumes callers never mutate the returned dict
  and never rewrite a genetic map file in place at the same path within one
  process. Both hold today (verified across all call sites and the test
  suite); revisit if either assumption changes.
- If the packed-haplotype-cache idea is revisited later, profile the actual
  overlap-driven re-decode fraction first (e.g. count/measure genotype
  fetches for positions shared by 2+ partitions on a real chromosome) rather
  than assuming the review's architectural reasoning translates directly
  into wall-clock savings — the pattern in this review (two confirmed wins,
  one plausible-but-unverified big swing) is worth repeating: benchmark
  before committing to the invasive option.
