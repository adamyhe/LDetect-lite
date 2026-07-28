# ldetect_original reproduction — findings

**Findings summary (last updated 2026-07-28).** Distilled for human review — e.g. writing up the paper. Full investigation detail, diagnostic scripts, and dated process notes: `notes/logs/ldetect-original-main-pipeline-audit.md`.

## Status: regression (2026-07-13 to 2026-07-28) found and fixed; pre-existing EUR/AFR residuals still parked

The 2026-07-03 baseline below (ASN exact genome-wide; AFR exact except chr22; EUR exact except chr8-12) held until two later commits introduced a real regression, since fixed:

- `2fa1705` ("Make bitpacked covariance default and add benchmark figures") hoisted `1/n_total` and `(4·ne)/(2·n_ind)` as loop-invariant constants in the shrinkage kernel, replacing per-pair division and the original multiply-multiply-divide exponent order with multiply-by-reciprocal and divide-then-multiply — algebraically equivalent, not bit-identical, causing 1-3 ULP noise on ~40-65% of covariance pairs. Fixed in `a943e28` (reverted to the original per-pair expression order; see `tests/test_shrinkage.py::test_shrink_ld_values_matches_naive_per_pair_formula_bit_exactly`).
- `793928e` ("MacDonald2022 reproduction diagnostics and parity fixes") changed `apply_filter`'s default window from the always-implicit `np.hanning` to `scipy-periodic` (`scipy.signal.get_window(..., fftbins=True)`), needed for MacDonald2022's reproduction. `examples/ldetect_original`'s Snakefile silently inherited this new default instead of pinning its own, which broke `ldetect_original` reproduction specifically — see "The `symmetric`-vs-`scipy-periodic` regression" below for why. Fixed by pinning `filter_window=symmetric` as `ldetect_original`'s own Snakefile default.

Both fixes were needed together; either alone left ASN chr19/21/22 well below 1.0 recall against the published `fourier_ls` reference. With both applied, ASN chr19/21/22 reproduce exactly again (recall/precision/jaccard = 1.0). Full genome-wide ASN/EUR/AFR reverification against the 2026-07-03 baseline is still pending, but there is no reason to expect the pre-existing EUR chr8-12/AFR chr22 residuals (below) to have changed, since those were already present before `scipy-periodic` existed as an option at all.

## The `symmetric`-vs-`scipy-periodic` regression: full root cause

`examples/ldetect_original/scripts/legacy_ldetect/ldetect/baselib/filters.py::apply_filter` calls `scipy.signal.get_window('hanning', 2*width+1)` — always an odd length. This file is an untouched copy of the original `nygcresearch/ldetect` Bitbucket repo (verified byte-identical to that repo's final commit, `a3060d0`, 2015-09-18; `filters.py` itself was never modified after the repo's first commit, `e221cc93`, 2015-03-24 — the whole repo's lifetime is 2015-03-24 to 2015-09-18).

Modern scipy (checked directly: 1.17.1) computes a genuinely asymmetric periodic window for `'hanning'`/`'hann'` at odd lengths — reversing the array changes 8338/8339 elements, max diff 3.8e-4. But **scipy 0.16.0** (July 2015, the latest release available throughout the entire window the original repo was developed) has a confirmed implementation defect in `hann(M, sym)`:

```python
odd = M % 2
if not sym and not odd:      # <-- only adjusts for EVEN M
    M = M + 1
n = np.arange(0, M)
w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1))
if not sym and not odd:
    w = w[:-1]
return w
```

For odd `M`, `not sym and not odd` is always `False`, so the periodic-length adjustment never fires — `sym=True` and `sym=False` compute **bit-identical** output. Since `2*width+1` is always odd, `get_window('hanning', 2*width+1, fftbins=True)` on scipy 0.16.0 silently returned the *symmetric* window despite requesting periodic. This was fixed in scipy 1.1.0 (2018), which rewrote the periodic-window construction to unconditionally extend-then-truncate regardless of parity (`_extend`/`_truncate` in `scipy/signal/windows/_windows.py`).

Consequence: Berisa & Pickrell's actual 2015 analysis, run on contemporary scipy, computed the *symmetric* Hann window despite the code asking for periodic — a real, undetectable-at-the-time library defect, not an intentional choice. Running the same archived code today with modern (post-1.1.0, bug-fixed) scipy produces a genuinely different, correctly-periodic window, which no longer matches what actually generated the published blocks. `ldetect-lite`'s `symmetric` mode (`np.hanning`) is therefore not a deprecated legacy fallback — it's the historically-accurate reproduction of scipy 0.16.0's actual (buggy) output. `scipy-periodic` remains correct for reproductions run against modern-scipy-era published output (MacDonald2022): there is no contradiction, both modes are "correct" for the actual scipy behavior of their respective eras.

Verification chain (all checked directly, not inferred): `sig.get_window('hanning', N)` maps to periodic `hann` since at least scipy 0.12.0 (2013) — ruling out a naming/alias explanation; `scipy.ndimage.convolve1d`'s `mode='reflect'` default and true-kernel-reversal semantics have also been stable since 2013 — ruling out a boundary-handling explanation; `apply_filter`/`baselib.filters` is the only filtering implementation ever called by the legacy pipeline scripts (`P02_minima_pipeline.py`, `E05_find_minima.py`) — ruling out an alternate-implementation explanation.

## Status: parked, not actively being investigated (pre-existing, as of 2026-07-03)

`examples/ldetect_original` reproduces Berisa & Pickrell (2016)'s published LD blocks:

- **ASN**: exact match, all 22 autosomes.
- **AFR**: exact except chr22. (chr11 was previously tracked as divergent too — resolved, see below.)
- **EUR**: exact block *counts* and coverage on every chromosome, but chr8–chr12 (a contiguous run bracketed by exact chr7/chr13 matches) have shifted internal boundary *positions*.

EUR chr8-12 and AFR chr22 are accepted, documented residual divergences. Every concrete, checkable hypothesis for them has been ruled out short of the original authors' own internal processing logs.

## Ruled out

- **VCF release-version provenance** — ruled out twice: once via position-set and phasing-sensitive LD comparisons across releases, and again by a full-pipeline rerun on v1/v2/old2011 and an undocumented `merged_umich` snapshot for every divergent chromosome — none reproduce a divergent chromosome exactly.
- **SNP-only vs. all-variant filtering** — ruled out.
- **Genetic map family** (OMNI vs. HapMap; HapMap Phase 2 Release 22 confirmed as the paper's actual source) — ruled out.
- **`Ne` assignment** — `ldetect-lite` intentionally uses population-specific `Ne` where legacy hardcodes `11418`; a real difference, but ASN (equally affected) still reproduces exactly, so not explanatory.
- **Duplicate-VCF-position / cross-partition-overlap handling** — proven mathematically and empirically equivalent to legacy (regression tests in `tests/test_duplicate_overlap_integration.py` and value-level tests in `test_shrinkage.py`/`test_covariance_io.py`/`test_metric.py`).
- **Sample/panel provenance** — EUR sample list proven byte-identical (379/379) to the panel distributed with the original ldetect toy example.
- **Reference BED file structural integrity** — audited for gaps, overlaps, and duplicate rows across all genome-wide and per-chromosome Bitbucket files; only the AFR chr11 issue below was found.
- **Multiallelic ALT-allele-trimming order** — theoretically plausible but no positive evidence; deprioritized, not disproven.
- **v1/old2011 sample-panel vintage** — ruled out. Neither release ever had its own panel file; 1000G's own documentation for both directs users to the same v3 panel `config.yaml` already uses. v2's distinct panel diffs byte-identical to v3 (1092/1092 samples).

## Resolved: AFR chr11 was never a real divergence

The published reference BED has two corrupted rows at exactly the boundary `ldetect-lite` was flagged for "emitting an extra boundary" at (`chr11 108823642 None` / `chr11 None 111048570`). `ldetect-lite`'s own output (`108823642 -> 109897792 -> 111048570`) is precisely the breakpoint the reference lost to corruption. This is a data bug in the published reference itself, present in the earliest available published source. Treat AFR chr11 as solved.

## The flat-region mechanism

Divergent boundaries consistently fall in flat, low-amplitude, featureless stretches of the smoothed diagonal-sum signal, where many nearby candidate positions are nearly tied — visualized in `examples/ldetect_original/plots/` (EUR chr10, AFR chr22).

This is a real *mechanism*, not proof of unresolvable ambiguity: `--high-precision` (the legacy-equivalent Decimal/dict local-search path) produces byte-identical breakpoints to the default float64/array path on EUR chr10, even in the flat regions causing its divergence — and running the actual legacy downstream scripts on `ldetect-lite`-generated covariance for EUR chr7-13 also fails to reproduce the published chr8-12 reference, while agreeing closely with `ldetect-lite`'s own output. Two numerically distinct implementations agree with each other and disagree with the reference — pointing to the upstream covariance/vector signal itself being subtly different from whatever the original authors computed, not to inherent algorithmic ambiguity. The flat-region correlation explains *why* divergence localizes to specific boundaries (small upstream differences only flip the outcome where the signal is already near-tied), not that it's unfixable in principle.

Caveat: this precision/legacy-downstream check only covers EUR chr8-13, not AFR chr22.

## If this is picked up again

Not currently planned, but in priority order:

0. Re-run the full genome-wide EUR/AFR/ASN comparison now that both the `shrinkage.py` ULP fix (`a943e28`) and the `filter_window=symmetric` Snakefile pin are in place, to confirm the 2026-07-03 baseline below is fully restored (only ASN chr19/21/22 have been directly reverified so far, at 2026-07-28).
1. Close the AFR chr22 gap in the precision/legacy-downstream evidence: run `--high-precision` for AFR chr22, and run `Snakefile.legacy_diagnostics` (already generalized to cover AFR chr21/chr22 alongside EUR) to check whether the same "implementations agree with each other, both disagree with the reference" pattern holds there too.
2. The open question is upstream of covariance/local search: what input or preprocessing step produces a subtly different vector than whatever the original authors used, for exactly these chromosomes? Remaining concrete lead: EUR/AFR/ASN subpopulation-code provenance — EUR is proven byte-identical (379/379) against the original toy example's actual sample list, but no equivalent ground-truth AFR list exists to check the same way; AFR's provenance rests only on population counts matching across VCF releases (246 individuals), not a byte-for-byte proof.
3. Absent new evidence (a new data source, an errata from the original authors), this is close to the practical limit of what's resolvable without the authors' internal processing logs.
