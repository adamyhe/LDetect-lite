# MacDonald2022 reproduction — findings

**Findings summary (current as of 2026-07-26; pyrho numbers below predate the 2026-07-28 non-monotonic-map fix, see "Non-monotonic genetic map bug" below).** Distilled for human review — e.g. writing up the paper. Full investigation detail, diagnostic scripts, and dated process notes: `notes/logs/macdonald2022-pyrho-handoff.md`, `notes/logs/macdonald2022-boundary-diagnostics.md`, and `notes/logs/macdonald2022-interpolation-port.md`.

## Status

`examples/MacDonald2022` now has two distinct reproduction modes:

1. **deCODE/EUR replication.** Running with MacDonald's published deCODE maps and the legacy-compatible `scipy-periodic` filter reproduces the published deCODE/EUR block set essentially exactly. The only remaining structural mismatch is a tiny leading chr7 edge block (`31439-31443`) emitted by the original/legacy BED extraction convention but absent from MacDonald's published BED. All substantive internal boundaries match.
2. **pyrho published-map replication.** Running with MacDonald's published pyrho interpolated maps gave high but not exact reproduction as of 2026-07-26. A real, previously-undiscovered bug affecting covariance windowing on these maps (they're genuinely, severely non-monotonic) was found and fixed 2026-07-28 — see below. The numbers in this table predate that fix and should be treated as a stale baseline pending a rerun, not the current ceiling.

Current downloaded comparison summaries (pre-fix, 2026-07-26):

| block set | ours | ref | delta | mean recall | mean Jaccard | mean bp-Jaccard | exact chroms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EUR` (deCODE) | 1362 | 1361 | +1 | 1.000 | 0.999 | 1.000 | 21/22 |
| `pyrho_AFR` | 1579 | 1580 | -1 | 0.968 | 0.946 | 0.984 | 9/22 |
| `pyrho_EAS` | 1118 | 1121 | -3 | 0.911 | 0.862 | 0.977 | 4/22 |
| `pyrho_EUR` | 1336 | 1336 | 0 | 0.942 | 0.904 | 0.973 | 7/22 |

Worst pyrho chromosomes in the current run:

- `pyrho_AFR`: chr9 dominates the residual mismatch (`recall=0.600`, p90 offset 589.6 kb).
- `pyrho_EAS`: chr4, chr9, and chr17 dominate.
- `pyrho_EUR`: chr19, chr12, and chr9 dominate.

## Focused pyrho boundary diagnostics

A targeted Snakemake diagnostic now runs raw and final boundary diagnostics only for the chromosomes driving the remaining pyrho mismatch:

```bash
cd examples/MacDonald2022
uv run snakemake --cores N pyrho_stage_diagnostics
```

Configured targets:

- `pyrho_AFR`: chr9
- `pyrho_EAS`: chr4, chr9, chr17
- `pyrho_EUR`: chr9, chr12, chr19

Outputs:

```text
results/compare/diagnostics/raw/{block_set}/chr{chrom}.boundary_diagnostics.tsv
results/compare/diagnostics/final/{block_set}/chr{chrom}.boundary_diagnostics.tsv
results/compare/diagnostics/stages/{block_set}/chr{chrom}.breakpoint_stage_diagnostics.tsv
results/compare/diagnostics/local_search/{block_set}/chr{chrom}.local_search_moved_away.tsv
```

These diagnostics annotate out-of-tolerance boundaries with mismatch class, local centromere context, local genetic-map density, and local SNP density. The downloaded `diagnostics/` bundle shows:

- Raw and final are nearly identical: 310 raw mismatch rows vs. 309 final rows across the seven target chromosomes.
- The dominant class is reciprocal shifted boundaries, not extra/missing splits: 262 shifted-boundary rows in both raw and final diagnostics.
- Every reported boundary lies exactly on a filtered SNP and genetic-map position (`nearest_snp_distance_bp=0`, `nearest_map_distance_bp=0` for all rows).
- Centromeres are not the broad explanation: final diagnostics have only 10/309 rows within 2 Mb of a centromere and none inside a centromere.
- Therefore the remaining pyrho divergence is a breakpoint-placement divergence that arises upstream of postprocessing.

The stage diagnostic compared 365 published internal boundaries against both the raw Fourier candidates and the local-search-refined (`fourier_ls`) loci. Only 35 were close at the raw Fourier stage, while 224 were close after local search. The residual 141 misses are mostly cases where local search improved a shifted raw candidate but did not get within 50 kb, or where it worsened an already shifted candidate. A smaller but cleaner subset has raw Fourier close to MacDonald's boundary and local search moving away.

The local-search replay diagnostic checked all 21 of those clean `local_search_moved_away` cases. In every case, MacDonald's boundary was inside the search window and exactly evaluated; replaying the actual `LocalSearch` class reproduced our recorded `fourier_ls` boundary and showed that the selected boundary was the true optimum under our objective. The MacDonald boundary was worse by only 0.003-0.172% in the metric, with median 0.0196%. This rules out a reachability, windowing, or replay mismatch bug and supports the near-tied-objective explanation.

## deCODE map interpolation bug: root cause found and fixed

`src/ldetect_lite/interpolate_maps.py::interpolate()` is a **point**-based linear interpolator (a port of `joepickrell/1000-genomes-genetic-maps`), but deCODE's source map is **interval**-based (`Begin, rate_cM_Mb, cumulative_cM_at_End` per interval, per MacDonald et al.'s `interpolate.R`). Feeding interval data through point-based interpolation is an off-by-one interval shift: every SNP's genetic position used the *next* interval's rate anchored at the *current* interval's endpoint, producing a ~0.001-0.01 cM error per SNP.

**Fix**: added `interpolate_intervals()` (a direct port of the R script's interval-anchoring logic) as an alternative to `interpolate()`, exposed via `ldetect interpolate-maps --mode {point,interval}` (default `point`, preserving prior behavior for true point-sampled maps like HapMap-interpolated 1000G maps). Confirmed on real deCODE data: interval mode is 10-50x closer to MacDonald's own published interpolated map than point mode (MAE 0.00004-0.0002 cM vs. 0.0017-0.002 cM, Pearson r = 1.0 in both).

Historical note: switching EUR from our recomputed interpolation to MacDonald's published already-interpolated deCODE map first resolved the bulk of the divergence. Subsequent legacy-compatibility fixes (`scipy-periodic` filtering and deCODE/EUR postprocessing parity) brought the current deCODE/EUR result to exact substantive boundary correspondence, aside from the tiny chr7 edge-block convention described above.

## Why the pyrho sets aren't exact anywhere

Unlike `examples/ldetect_original` (near-exact genome-wide, since it replays byte-identical archived VCF/map files the original authors used), the MacDonald pyrho block sets are high-concordance but not exact genome-wide. Current final runs have several exact chromosomes, but residual non-exact boundaries remain concentrated in a handful of chromosomes across all three populations. Three known, verified mechanisms explain much of this:

1. **Genetic-map deserts** (Category A — chr9, and 7 other chromosome/population combinations). Some chromosomes have large (tens-of-Mb) genetic-map dead zones (e.g. chr9's 17 Mb desert, consistent with its heterochromatic 9qh block). Legacy LDetect emits one unsplit block across the whole desert; `ldetect-lite` places an extra breakpoint inside it (same total genome-wide breakpoint budget, different placement) — confirmed directly against MacDonald's own git history (raw pre-centromere-removal BEDs).
2. **Non-monotonic genetic map bug in covariance windowing** (found 2026-07-28, fixed — see below). Silently corrupted the covariance pair window on ~10% of affected chromosomes' SNPs, up to 1047-SNP window-size errors on real data (chr9, the single worst `pyrho_AFR` chromosome). Not yet re-run against the fix at the time of writing; likely explains a meaningful share of what was previously attributed entirely to mechanism 3.
3. **Razor-thin local-search margins** (Category B — verified first on EAS chr4 and then across all 21 clean moved-away cases in the focused stage diagnostics). `LocalSearch` correctly finds the true optimum in its search window every time — no bug — but tiny, legitimate numerical differences in the covariance computation (which cannot be byte-compared against legacy — no published intermediates exist) are enough to flip which of two near-tied candidates wins a sub-0.1%-margin race. The replayed moved-away cases had MacDonald's boundary exactly reachable and evaluated, but worse than our chosen boundary by only 0.003-0.172% (median 0.0196%). This investigation (2026-07-04) checked window computation, metric formula, tie-breaking, and array/Decimal path equivalence, but never checked genetic-map monotonicity — a different "window" (the covariance pair window, not the local-search refinement window) that mechanism 2 above was quietly corrupting the whole time.

Mechanisms 1 and 3 converge with the same "flat region" finding documented for `ldetect_original` (see `notes/findings/ldetect-original-reproduction.md`) — this is the same underlying phenomenon showing up pervasively here rather than on 1-2 chromosomes, because MacDonald's pipeline (unlike `ldetect_original`) is a from-scratch rerun against modern VCF releases and pyrho/deCODE maps rather than a byte-identical replay.

Investigated and **ruled out** as explanations: postprocessing, boundary/SNP coordinate artifacts, local-search reachability/windowing, local-search replay mismatch, window-bound computation (matches legacy exactly), metric formula/tie-breaking (identical to legacy; two latent tie-break/denominator-guard bugs were found and fixed along the way, but don't affect any real reproduction output), the historical array-vs-Decimal local-search divergence (verified fixed — see `notes/logs/local-search-divergence-asn22.md`), `N_zero` denominator conditioning (essentially constant across search windows; the sensitivity comes from the numerator), MAF-filter type (`nref` vs. true minor-allele-frequency — tested genome-wide on EAS, makes concordance *worse*, not better), and individual-panel-membership drift (de-risked: selection method is a faithful, deterministic clone of MacDonald's documented recipe).

## Non-monotonic genetic map bug in covariance windowing (found and fixed, 2026-07-28)

MacDonald's own **published** pyrho interpolated maps (used directly by the active `pyrho_AFR`/`pyrho_EAS`/`pyrho_EUR`/`pyrho_SAS` block sets — not just the `macdonald_style` diagnostic below) are genuinely, severely non-monotonic in genetic position. Downloaded and checked directly: GWD chr9 (the single worst `pyrho_AFR` chromosome per the mismatch table above) has **18,130 backward jumps** (3.43% of its 527,983 rows), the worst a **-12.9 cM** jump against a **120.7 cM** total chromosome span. `config.yaml`'s own comments already noted this ("nonmonotone cM coordinates that original ldetect cannot process cleanly") in the context of *legacy* ldetect overflowing on it, but ldetect-lite's own covariance step was never checked for correctness under this condition — it doesn't crash, it silently computes a wrong result.

Root cause: `shrinkage.py::_genetic_stop_bounds_impl` computes each SNP's right-hand covariance pair window with a persistent "two-pointer" scan (`stop` never resets backward across outer iterations) — an O(n) amortized optimization that's only valid when genetic position is non-decreasing. Legacy's own equivalent loop (`P00_01_calc_covariance.py`) resets its scan to `j = i` fresh every row instead, so it has no persistent pointer to invalidate. On real GWD chr9 data, this made the two computations disagree on **51,443 SNPs (9.74%)**, with window-size errors up to **1,047 SNPs** — always over-inclusion relative to the correct per-row scan, never under. This is a real, substantial, previously-undiscovered divergence from legacy behavior, not a tiny numerical artifact, and it has been silently present in every prior pyrho reproduction run (this two-pointer structure predates this investigation).

**Fixed**: `_genetic_stop_bounds_impl` now takes an `assume_monotonic` flag; `calc_covariance` detects non-monotonic genetic maps automatically (`np.diff(gpos_arr) >= 0`) and falls back to the legacy-faithful per-row fresh scan, emitting a stderr warning when it does. No new CLI flag; monotonic maps (the common case) keep the fast path. `partition_chromosome`'s own window-extension loop was checked too and needs no fix — it already does a fresh scan per chunk, matching legacy. No other pipeline stage uses genetic position at all (matrix-to-vector, metric, local search, and filtering all operate on physical positions and already-computed covariance values). Regression test:
`tests/test_shrinkage.py::test_genetic_stop_bounds_non_monotonic_map_needs_assume_monotonic_false`.

**Not yet re-verified against real block output** — the fix has only been checked against the raw genetic map (confirming it changes `j_stop_by_i` substantially on real data) and a synthetic end-to-end `calc_covariance` smoke test, not against a full `pyrho_AFR`/`pyrho_EAS`/`pyrho_EUR`/`pyrho_SAS` rerun compared to MacDonald's published blocks. Re-running those block sets with `--force-covariance` (to regenerate the now-stale cached covariance partitions) is the natural next step.

Also wired up `pyrho_{AFR,EAS,EUR,SAS}_macdonald_style` block sets (`config.yaml`), using `interpolate_macdonald_pyrho()` (already implemented, previously unwired to any block set) to re-interpolate the *raw* pyrho rate maps with MacDonald's own `interpolate_pyhro.R` bugs deliberately reproduced, rather than using their already-interpolated published output directly. This is a diagnostic comparison target: if it agrees closely with the `published`-mode results (now that both go through the same fixed covariance code), that's independent confirmation we understand MacDonald's interpolation bug correctly.

## Bottom line

The razor-thin-margin explanation (mechanism 3) was investigated thoroughly and still stands for whatever divergence remains after the non-monotonic-map fix above, but that fix hasn't been factored into any of the recall/concordance numbers reported earlier in this document — they should be treated as superseded pending a rerun. Do not add a deadband or stability heuristic to coerce near ties toward MacDonald's published boundaries; that would make replication diagnostics look better by changing the algorithm's decision rule. Absent the non-monotonic-map fix changing the picture substantially, current state remains close to the practical ceiling for an independent, from-scratch reproduction against modern data releases without MacDonald's own covariance intermediates, which don't exist publicly.

## Next steps

1. Re-run `pyrho_AFR`/`pyrho_EAS`/`pyrho_EUR`/`pyrho_SAS` with `--force-covariance` (existing cached `.h5` covariance partitions predate the fix and must be regenerated) and regenerate the comparison summaries. Chr9 (the worst `pyrho_AFR` chromosome, and one of the worst for `pyrho_EAS`/`pyrho_EUR` too) is the highest-value target given its 18,130 backward jumps in the real GWD map — expect the largest movement there if this mechanism explains a meaningful share of the residual gap.
2. If the fix substantially closes the gap, re-attribute how much of the previously-documented "razor-thin margins" (mechanism 3) was actually this bug in disguise, and update the recall/Jaccard table above.
3. Optionally run the new `pyrho_{AFR,EAS,EUR,SAS}_macdonald_style` block sets and compare against the corresponding `published`-mode results, now that both paths share the same fixed covariance code — agreement would independently confirm `interpolate_macdonald_pyrho()` faithfully reproduces MacDonald's actual interpolation bug.
