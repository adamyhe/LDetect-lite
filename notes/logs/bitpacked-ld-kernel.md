# Bitpacked LD Kernel

**Agent-oriented working log.** Raw, dated investigation notes — not proofread for external readability. For current, human-readable status, see `docs/optimizations.md` #14.

Date: 2026-07-12

## What this is

`--ld-kernel {bitpacked,uint8}` on `ldetect run`/`ldetect calc-covariance` (default: `bitpacked`). `bitpacked` packs each SNP's haplotype row into `uint64` words (`_pack_haplotypes_impl`) and computes pairwise intersection counts via popcount (`_popcount64`) instead of the older `uint8`-array `np.sum(a * b)` kernel. Both backends compute the same popcount-derived pair counts and feed them through the identical Wen-Stephens shrinkage formula — the two outputs are expected to be exact matches, not merely close. Bitpacked requires compact covariance output; `uint8` is retained as a reference/diagnostic backend and for full-schema debug caches.

This log is a distilled port of the original mixed-concerns branch log (`covariance-optimization`'s `notes/logs/covariance-bitpacked-kernel-and-chromosome-mode.md`), stripped of that branch's other, unrelated thread (`--covariance-mode chromosome`, since abandoned — see that log for why, if curious). Everything below is bitpacked-kernel-specific.

## Validated so far

- **Same-codebase exactness (`uint8` vs `bitpacked` within `ldetect-lite`, the claim that matters):** bit-exact, not just close. `tests/test_shrinkage.py::test_calc_covariance_bitpacked_compact_matches_uint8_compact` and `test_bitpacked_compact_chunks_match_uint8_backend` (parametrized across small/large N, word-boundary-crossing haplotype counts) assert `np.testing.assert_array_equal`, not a tolerance. `_pack_haplotypes_impl`/`_popcount64` have their own direct unit tests against a naive Python bit-count reference.
- **Toy-scale, end-to-end:** `examples/ldetect_example/scripts/benchmark_functions.py --ld-kernel bitpacked` on the chr2 EUR toy example (real committed result: `examples/ldetect_example/results/function_benchmark_bitpacked_check/summary.md`) — exactness check passes. Note this specific comparison is bitpacked vs. the *original legacy* ldetect implementation (a different codebase, Decimal/float and summation-order differences expected), reporting `calc_covariance` max abs diff ~2.2e-16 (machine-epsilon-level, not exactly 0.0) — a different, weaker claim than the same-codebase bit-exact tests above. Don't conflate the two: legacy comparison is "matches to floating-point precision," same-codebase comparison is "byte-identical."
- **Toy-scale speed:** bitpacked vs. legacy on the same chr2 EUR toy example (`examples/ldetect_example/results/covariance_bitpack_vs_legacy/summary.tsv`): 108.29s legacy mean vs. 1.89s bitpacked mean (~57.3x), output 0.254x the legacy size, but *higher* peak RSS (824.8 MiB bitpacked vs. 563.5 MiB legacy) — a real tradeoff, not yet reconciled or explained.

## Genome-scale result (2026-07-12/13)

Serial 1000G diagnostics are complete for EUR, ASN, and AFR using `examples/ldetect_original/Snakefile.ld_kernel_diagnostics`, summarized in `examples/ldetect_original/results/{EUR,ASN,AFR}_ld_kernel_summary.tsv`.

**Exactness: definitive pass for the 1000G populations tested.** All 66 chromosome x population runs are exact: `vector_sha256_equal=True`, `vector_max_abs_diff=0.0`, `loci_exact_match=True`, and `bed_jaccard=1.0`. This extends the unit and toy-example checks to full EUR/ASN/AFR population-scale runs.

**Speed: modest end-to-end win.** Combined across EUR/ASN/AFR, aggregate covariance time drops from 32637.68 s (`uint8`) to 30238.12 s (bitpacked), an overall 1.079x speedup. Median chromosome-level speedup is 1.062x. Bitpacked is faster on 63/66 runs. The only slower rows are AFR chr9 (0.908x), AFR chr12 (0.951x), and AFR chr21 (0.960x). ASN is the cleanest population-level result: faster on all 22 chromosomes, with a tight 1.033x-1.113x range. AFR has the strongest aggregate speedup (1.093x) but also the three slow outliers.

**Memory and storage: no unacceptable regression.** Across all 66 runs, the `uint8`/bitpacked peak-RSS ratio has mean 1.016, median 1.002, and range 0.885-1.248. Compact covariance size is identical (`covariance_size_ratio=1.0`) because the backend changes row generation, not the HDF5 cache schema.

**Verdict:** bitpacked is validated across EUR/ASN/AFR and should remain the default backend for compact covariance. Keep `uint8` as an optional reference backend and for full-schema debug caches. Direct vector sidecar work stays abandoned for this merge.

## Toy-scale-only, unresolved

- **The toy-scale RSS regression (824.8 MiB bitpacked vs. 563.5 MiB legacy) is unexplained** — but now known to be legacy-comparison-specific, not a `uint8`-vs-`bitpacked` issue (see genome-scale memory result above). Packing to `uint64` should reduce memory, not increase it, for the haplotype matrix itself — worth checking whether the toy-scale figure was a transient peak from holding both the `uint8` source array and its packed copy simultaneously during `_pack_haplotypes_impl`, if this is ever revisited.
