# Covariance Optimization: Bitpacked LD Kernel and Chromosome-Grouped Runs

**Agent-oriented working log.** Raw, dated investigation notes — not proofread for external readability. For current, human-readable status, see `notes/findings/`.

Date: 2026-07-11

## Context

`docs/optimizations.md` and `notes/logs/post-covariance-optimization-review.md` both identify Step 2 (`calc_covariance`) as a major remaining cost: per-partition VCF/BCF region reads and a `uint8`-array pairwise LD kernel. Branch `covariance-optimization` (commits `385477f`..`f996c60`) is mid-flight work on two independent speedups for this stage. Neither is documented anywhere yet — this log exists to capture them before they're lost, not because the work is finished.

## What's been added

**Bitpacked LD kernel** (`b0e8f7d`, `157ca3f`): a new pair-count backend that packs haplotypes into `uint64` words and computes pairwise LD via popcount intersection instead of the established `uint8`-array kernel. Opt-in via `--ld-kernel {uint8,bitpacked}` on `ldetect run`/`calc-covariance` (default stays `uint8`); `bitpacked` currently requires `--covariance-cache compact`. Implementation: `_pack_haplotypes_impl`, `_popcount64`, `_compact_pair_chunks_single_pass_bitpacked` in `shrinkage.py`.

**Chromosome-grouped covariance runs** (`c70280c`): a new `--covariance-mode {partition,chromosome}` flag on `ldetect run`. `chromosome` mode loads a chromosome's genotypes once (`load_chromosome_genotypes`) and slices each partition's covariance from the prepared in-memory arrays (`calc_covariance_from_genotypes`), instead of re-reading the VCF/BCF per partition. Currently requires the compact cache. Adds `--profile-covariance PATH` to emit a TSV of one chromosome-load row plus per-partition writer rows.

**Instrumentation** (`157ca3f`): `calc_covariance` and the chromosome-mode path now accept an optional `profile: dict[str, float]` populated with coarse stage timings (`prepare_seconds`, `vcf_seconds`, `array_seconds`, `pack_seconds`, `chunk_seconds`, `write_io_seconds`, `n_pairs`, `n_snps`, `n_haps`, `total_seconds`) — built specifically to attribute where time goes across the `uint8`/`bitpacked` x `partition`/`chromosome` combinations.

**Benchmark infra** (`b0e8f7d`, `529df0f`, `f996c60`): `benchmarks/bench_bitpacked_full_genome.py`, a full-genome (real 1000G download) speed+exactness comparison across the two kernels, with `--include-chromosome-mode` to also exercise chromosome-mode RSS. Edited three times in one day (`529df0f`, `157ca3f`'s instrumentation, `f996c60`) — still being actively shaped, no full-genome run has been committed. `examples/ldetect_example/` also grew comparison scripts (`compare_covariance.py`, `compare_partitions.py`, `compare_bed.py`, `prepare_1000g_region.py`) and Snakefile targets for smaller-scale checks.

CLI docs (`README.md`, `docs/pipeline-steps.md`) were kept in sync with the new flags as part of `c70280c` — that part is not a gap.

## What's validated so far

- Unit-level: `test_calc_covariance_bitpacked_compact_matches_uint8_compact` confirms byte-identical compact HDF5 rows (`lo`, `hi`, `shrink_ld`, diagonal, loci) between `uint8` and `bitpacked` on a small synthetic fixture.
- Toy-scale (`examples/ldetect_example/results/covariance_bitpack_vs_legacy/summary.tsv`, already committed): bitpacked vs. the **original legacy** (not the current `uint8` array kernel) on the chr2 EUR toy example — 108.29s legacy mean vs. 1.89s bitpacked mean (**~57.3x**), min-time speedup ~57.7x, output 0.254x the legacy size (~74.6% smaller, consistent with entry #7's compact-cache reduction), but **higher** peak RSS (824.8 MiB bitpacked vs. 563.5 MiB legacy) — a real tradeoff, not yet reconciled.
- `examples/ldetect_example/results/function_benchmark_bitpacked_check/summary.md`: exactness passes (`calc_covariance` max abs diff 2.2e-16) at toy scale via `benchmark_functions.py --ld-kernel bitpacked`.

## What's still open

- No apples-to-apples bitpacked-vs-**current-`uint8`** (not legacy) speedup number has been recorded anywhere.
- No chromosome-mode-vs-partition-mode speedup number has been recorded anywhere, toy or genome scale.
- `bench_bitpacked_full_genome.py` has not produced a committed genome-scale result yet; the RSS tradeoff seen at toy scale needs checking at real chromosome sizes before considering `bitpacked`/`chromosome` as new defaults.
- Once both are validated at real scale, promote to `docs/optimizations.md` as a new entry (would be #15) and revisit the `uint8`/`partition` defaults.

## Two confirmed gaps in chromosome mode (2026-07-11, user-reported)

1. **No parallelism.** `_calc_chromosome_partitions` (`cmd_run.py`) runs a plain sequential `for start, end in pending` loop — no `ProcessPoolExecutor`, unlike the `partition` path's pool of `--workers` processes. The CLI already surfaces this (`cmd_run.py`: "Note: --covariance-mode chromosome processes this single chromosome serially; --workers still applies to later pipeline stages unless overridden."). On many-core machines this makes `chromosome` mode **slower** than `partition` mode overall, even though it avoids redundant VCF/BCF region reads per partition — the two effects trade off in opposite directions and nothing yet establishes the crossover point (core count / partition count / chromosome size) where `chromosome` mode wins.
2. **Not bit-exact vs. `partition` mode — specifically for `uint8`.** The only genotypes-vs-region equality test, `test_calc_covariance_from_genotypes_matches_region_bitpacked` (`tests/test_shrinkage.py`), only exercises `ld_kernel="bitpacked"`. There is no analogous passing test for `ld_kernel="uint8"`. Per the user, the bitpacked kernel itself is fine in isolation (matches `uint8` when both run in `partition` mode — see `test_calc_covariance_bitpacked_compact_matches_uint8_compact` above) and chromosome-mode-with-bitpacked matches region-mode-with-bitpacked, but chromosome-mode-with-`uint8` does **not** match partition-mode-with-`uint8`. So the divergence is specific to the `uint8` kernel's path through `calc_covariance_from_genotypes`/`load_chromosome_genotypes`, not to bitpacking or to chromosome-mode in general. Root cause not yet identified — worth first diffing how `load_chromosome_genotypes`'s `storage="uint8"` array prep differs from `calc_covariance`'s per-region `uint8` prep (dtype, haplotype ordering, or a slicing-boundary difference are the likely places to look).

**Practical implication:** `--covariance-mode chromosome` is not currently safe to use with the default `--ld-kernel uint8` for anything beyond experimentation — it's silently wrong, not just slow. It's only trustworthy today combined with `--ld-kernel bitpacked`, which itself isn't the default and isn't yet genome-scale-validated (see above). Neither combination should be recommended to end users yet.

## Full-genome exactness diagnostic added, with a division of labor vs. the existing benchmark

Added `examples/ldetect_original/Snakefile.ld_kernel_diagnostics` (+ `ld_kernel_diagnostics.yaml`, `scripts/compare_ld_kernel.py`) as the canonical genome-scale `uint8`-vs-`bitpacked` exactness check: runs the real `ldetect run` CLI end to end across all three 1000G populations by default, `--covariance-mode partition` only (chromosome mode intentionally excluded — see above), and compares final vector/breakpoints/BED, not just covariance rows.

This overlaps substantially with the pre-existing `benchmarks/bench_bitpacked_full_genome.py`, which already did a full-genome `uint8`-vs-`bitpacked` exactness+speed comparison — just scoped to covariance-partition rows (via direct `calc_covariance` calls, one population per run) rather than full pipeline output through the CLI. Rather than duplicate two overlapping full-genome checks, resolved to: the Snakefile is now the canonical full-dataset exactness diagnostic (documented in both `benchmarks/README.md` and the benchmark script's own docstring); `bench_bitpacked_full_genome.py` stays for what the Snakefile doesn't do — per-stage timing/RSS breakdowns and quick smoke checks without Snakemake. Also noted there: `bench_bitpacked_full_genome.py`'s `--include-chromosome-mode` path (`time_calc_covariance_from_genotypes`) hardcodes `ld_kernel="bitpacked"` and has never covered chromosome-mode + `uint8` — consistent with why that inexactness gap wasn't caught by existing benchmark infra.

## Root cause found: this is a partition-mode bug, not a chromosome-mode bug (2026-07-12)

The "not bit-exact" gap above turned out to be real, but **misattributed**. It is not about `uint8` specifically, and chromosome mode is not the buggy side.

**Where the divergence actually lives:** `bench_bitpacked_full_genome.py`'s `--include-chromosome-mode` path only ever runs chromosome mode with `ld_kernel="bitpacked"` (`load_chromosome_genotypes(storage="packed", ...)` and `time_calc_covariance_from_genotypes` both hardcode it — see the division-of-labor section above). So the `chromosome_exact`/`chromosome_max_abs_diff` fields this benchmark reports are a **partition-mode-bitpacked vs. chromosome-mode-bitpacked** comparison, not a `uint8` comparison at all. The script cannot and never did exercise chromosome-mode + `uint8`.

**Reproduction:** ran `bench_bitpacked_full_genome.py --population EUR --chromosomes 22 --include-chromosome-mode --keep-outputs` end to end on real chr22 EUR 1000G data. Result: `chromosome_all_exact: False`, `chromosome_max_abs_diff: inf` (the `inf` specifically means a **row-count/shape** mismatch, not a numeric precision difference — see `compare_outputs`'s `else: max_abs_diff = float("inf")` branch). `all_exact` (partition-mode `uint8` vs. partition-mode `bitpacked`) was `True` with `max_abs_diff: 0.0` — confirming again that the kernel choice itself is exact.

**Pinpointed via a standalone per-partition diff** (recomputing every one of chr22's 98 partitions independently, partition-mode-bitpacked vs. chromosome-mode-bitpacked): exactly 3 partitions mismatch, all row-count (shape) mismatches, partition mode always having *more* rows:

```
idx=14  22595093-23027503   partition=1589082 rows   chromosome=1588874 rows
idx=16  23217586-23713492   partition=2133050 rows   chromosome=2132882 rows
idx=24  25906898-26282765   partition=1259437 rows   chromosome=1259436 rows
```

Diffing the loci sets for partition 14 directly: partition mode includes 2 extra positions chromosome mode doesn't — `22517055` and `22569659` — both **tens of kilobases before** the partition's own start (`22595093`). Looked them up directly in the filtered VCF:

- `22517055` is `esv2670821`, a `<DEL>` structural variant with `POS=22517055` and `INFO/END=22599640` (an 82.5kb deletion). Its span `[POS, END]` overlaps partition 14's range `[22595093, 23027503]` even though `POS` itself is ~78kb before the partition start.
- `22569659` is `esv2670795`, a variant with a very long literal `REF` sequence (a large indel/SV encoded inline rather than symbolically) — same mechanism, span overlap via a long `REF` rather than an `END` field.

**Root cause:** `calc_covariance`'s region-based read (`vcf(region)` where `region = f"{chrom}:{start}-{end}"`, used by the default `partition` covariance mode) has no explicit `start <= pos <= end` check of its own in the Python loop — it only filters by `pos in pos2gpos`, trusting htslib's region-overlap semantics for inclusion. For an ordinary SNP that's a distinction without a difference (span == POS). For a structural variant with an `INFO/END` far from `POS`, or a long-`REF` indel/SV, htslib matches by **span overlap**, so a variant whose true `POS` belongs to an earlier partition can get pulled into a later partition purely because its span reaches into it. `calc_covariance_from_genotypes` (chromosome mode) has no such gap — it slices `genotypes.positions` with `np.searchsorted` on `POS` directly, so it's structurally incapable of including a variant outside `[start, end]` by `POS`.

**This means chromosome mode is the more correct implementation here**, not the buggy one. Partition mode — the long-standing default, used in every existing production/reproduction run (the 1000G source file is literally named `...snps_indels_svs...`, so SVs are expected, not an edge case) — has a latent bug where a large SV or long indel near a partition boundary can be spuriously double-counted into an adjacent partition's covariance calculation. This almost certainly predates the `covariance-optimization` branch entirely: the pre-cyvcf2 `tabix -h region` reference path (see `docs/optimizations.md` #10) uses the same htslib region-overlap semantics and likely has/had the identical behavior.

**Not yet done, by explicit choice:** the fix — an explicit `start <= pos <= end` filter in `calc_covariance`'s region-read loop — has not been implemented. This is default, long-standing partition-mode behavior, not scoped to this branch's new code, so touching it needs its own deliberate pass (plus thinking about whether it has any bearing on the published-block reproduction claims in `notes/findings/ldetect-original-reproduction.md`, which were generated exclusively through partition mode). Flagging this here rather than silently patching it.

**Updated practical implication (supersedes the "Practical implication" paragraph above):** `--covariance-mode chromosome` is not shown to be wrong for either kernel. The exactness question that's actually open is the reverse of what was originally suspected: whether partition mode's SV/long-indel boundary over-inclusion should be fixed, and if so, whether that changes any existing partition-mode-derived output (including the reproduction-status results). `chromosome` mode still isn't the default and still has no parallelism (gap #1 above still stands), so there's no urgency to flip defaults — but it should no longer be treated as "silently wrong."

## Heads up: a separate, unmerged branch explores the same space

`ld-kernel-bitpack-benchmark` (local + `origin`) diverged earlier (before `885144c`) and independently explored "row-vectorized, chunked-matmul, bit-packed popcount" kernel prototypes plus an unrelated "signal cache" HDF5 sidecar feature. It is neither an ancestor nor descendant of `covariance-optimization`. Worth diffing against before doing more kernel-prototype work here, to avoid re-deriving what that branch already tried.
