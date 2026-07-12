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

## Not yet done

- **No genome-scale result has ever been produced.** The infrastructure for this already exists — `examples/ldetect_original/Snakefile.ld_kernel_diagnostics` (+ `ld_kernel_diagnostics.yaml`, `scripts/compare_ld_kernel.py`) runs `ldetect run` twice per chromosome x population on identical real 1000G input (once `--ld-kernel uint8`, once `bitpacked`) and compares vectors, breakpoints, BEDs, covariance-directory size, and Snakemake's own wall-clock/peak-RSS `benchmark:` output. It was built during the original mixed-concerns branch's work but never actually run — defaults to the full 22-chromosome x 3-population dataset, explicitly meant for remote/cluster execution given the compute cost (matches `feedback_no-large-jobs-locally` — don't run this locally).
- **The toy-scale RSS regression (824.8 MiB vs. 563.5 MiB) is unexplained.** Packing to `uint64` should reduce memory, not increase it, at least for the haplotype matrix itself — worth checking whether this is a real steady-state RSS increase or a transient peak from holding both the `uint8` source array and its packed copy simultaneously during `_pack_haplotypes_impl`.
- **No apples-to-apples `uint8`-vs-`bitpacked` genome-scale *speed* number exists either** — only the toy-scale 57.3x figure above, which is bitpacked-vs-*legacy*, not bitpacked-vs-current-uint8.

## Next step

Run `Snakefile.ld_kernel_diagnostics` for real, on a cluster:

```bash
cd examples/ldetect_original/
uv run snakemake -s Snakefile.ld_kernel_diagnostics -n   # dry-run first
uv run snakemake -s Snakefile.ld_kernel_diagnostics --cores 8
```

Outputs land under `results/ld_kernel_diagnostics/{population}/{chrom}/...`, with a genome-wide summary at `results/ld_kernel_diagnostics/summary.tsv`. That result is the actual deliverable this log is waiting on — once it exists, this section should be replaced with the real numbers and an exactness/speed verdict, and `docs/optimizations.md` #14 updated to cite it directly instead of "see `benchmarks/README.md`."
