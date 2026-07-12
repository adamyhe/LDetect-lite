# Bitpacked LD Kernel

**Agent-oriented working log.** Raw, dated investigation notes — not proofread for external readability. For current, human-readable status, see `docs/optimizations.md` #14.

Date: 2026-07-12

## What this is

`--ld-kernel {uint8,bitpacked}` on `ldetect run`/`ldetect calc-covariance` (default: `uint8`). `bitpacked` packs each SNP's haplotype row into `uint64` words (`_pack_haplotypes_impl`) and computes pairwise intersection counts via popcount (`_popcount64`) instead of the established `uint8`-array `np.sum(a * b)` kernel. Both backends compute the same popcount-derived pair counts and feed them through the identical Wen-Stephens shrinkage formula — the two outputs are expected to be exact matches, not merely close. Requires `--covariance-cache compact`.

This log is a distilled port of the original mixed-concerns branch log (`covariance-optimization`'s `notes/logs/covariance-bitpacked-kernel-and-chromosome-mode.md`), stripped of that branch's other, unrelated thread (`--covariance-mode chromosome`, since abandoned — see that log for why, if curious). Everything below is bitpacked-kernel-specific.

## Validated so far

- **Same-codebase exactness (`uint8` vs `bitpacked` within `ldetect-lite`, the claim that matters):** bit-exact, not just close. `tests/test_shrinkage.py::test_calc_covariance_bitpacked_compact_matches_uint8_compact` and `test_bitpacked_compact_chunks_match_uint8_backend` (parametrized across small/large N, word-boundary-crossing haplotype counts) assert `np.testing.assert_array_equal`, not a tolerance. `_pack_haplotypes_impl`/`_popcount64` have their own direct unit tests against a naive Python bit-count reference.
- **Toy-scale, end-to-end:** `examples/ldetect_example/scripts/benchmark_functions.py --ld-kernel bitpacked` on the chr2 EUR toy example (real committed result: `examples/ldetect_example/results/function_benchmark_bitpacked_check/summary.md`) — exactness check passes. Note this specific comparison is bitpacked vs. the *original legacy* ldetect implementation (a different codebase, Decimal/float and summation-order differences expected), reporting `calc_covariance` max abs diff ~2.2e-16 (machine-epsilon-level, not exactly 0.0) — a different, weaker claim than the same-codebase bit-exact tests above. Don't conflate the two: legacy comparison is "matches to floating-point precision," same-codebase comparison is "byte-identical."
- **Toy-scale speed:** bitpacked vs. legacy on the same chr2 EUR toy example (`examples/ldetect_example/results/covariance_bitpack_vs_legacy/summary.tsv`): 108.29s legacy mean vs. 1.89s bitpacked mean (~57.3x), output 0.254x the legacy size, but *higher* peak RSS (824.8 MiB bitpacked vs. 563.5 MiB legacy) — a real tradeoff, not yet reconciled or explained.

## Genome-scale result (2026-07-12)

Ran `Snakefile.ld_kernel_diagnostics` for real: all 22 chromosomes x 3 populations (EUR/AFR/ASN), real 1000G Phase 1 data. `results/ld_kernel_summary.tsv`, 66 rows.

**Exactness: definitive pass, no exceptions.** Every one of the 66 rows: `vector_sha256_equal=True`, `vector_max_abs_diff=0.0`, `loci_exact_match=True`, `bed_jaccard=1.0`. Bitpacked and `uint8` produce byte-identical vectors, breakpoints, and BEDs at full genome scale, not just in unit tests. `covariance_size_ratio=1.0` throughout too — the compact HDF5 schema's on-disk size doesn't depend on which pair-counting kernel produced it, as expected.

**Speed: inconclusive.** Aggregate (sum of all `baseline_seconds` / sum of all `bitpacked_seconds`) = 1.095x, but the per-chromosome `speedup` column ranges from 0.12x (AFR chr22, bitpacked ~8x *slower*) to 11.27x (AFR chr19, ~11x faster), median only 1.03x, and 28/66 rows favor `uint8` over `bitpacked`. The extreme outliers cluster entirely among the shortest-running jobs (`baseline_seconds` ~70-250s); the largest jobs (8,000-13,000s) sit much closer to neutral (0.99x-2.13x). Variance shrinking as job size grows is the signature of a fixed per-run overhead or scheduling contention dominating the measurement, not a real kernel-speed difference — `docs/optimizations.md` #13 already documented exactly this failure mode (Slurm scheduling multiple jobs to the same second) for a different diagnostic, and this Snakefile doesn't apply that fix's thread-count-guard pattern. Not confirmed which mechanism (Numba JIT cold-start cost, scheduling contention, or something else) without digging into the individual `.timing.log`/`.benchmark.tsv` files per job — flagging the pattern, not claiming the cause.

**Memory: no regression.** `max_rss_ratio` (baseline/bitpacked) averages 1.014, range 0.93-1.22 — resolves the toy-scale RSS concern below; that regression was specific to the *legacy* comparison, not `uint8`.

**Verdict:** `bitpacked` is proven correct at genome scale. It is *not* proven faster — the 1.095x aggregate isn't reliable given the underlying noise, and nothing here supports flipping the default away from `uint8`. If a real speed claim is wanted, the diagnostic needs the thread-count-guard treatment (or an isolated/dedicated-node run) before the timing numbers can be trusted.

## Toy-scale-only, unresolved

- **The toy-scale RSS regression (824.8 MiB bitpacked vs. 563.5 MiB legacy) is unexplained** — but now known to be legacy-comparison-specific, not a `uint8`-vs-`bitpacked` issue (see genome-scale memory result above). Packing to `uint64` should reduce memory, not increase it, for the haplotype matrix itself — worth checking whether the toy-scale figure was a transient peak from holding both the `uint8` source array and its packed copy simultaneously during `_pack_haplotypes_impl`, if this is ever revisited.
