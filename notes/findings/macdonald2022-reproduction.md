# MacDonald2022 reproduction — findings

**Findings summary (current as of 2026-07-05).** Distilled for human review — e.g. writing up the paper. Full investigation detail, diagnostic scripts, and dated process notes: `notes/logs/macdonald2022-pyrho-handoff.md` and `notes/logs/macdonald2022-interpolation-port.md`.

## Status

`examples/MacDonald2022` reproduces MacDonald et al. (2022)'s GRCh38 LD blocks for four block sets: `EUR` (deCODE map), `pyrho_AFR`, `pyrho_EAS`, `pyrho_EUR`. (`pyrho_SAS` is set aside — MacDonald et al. do not document an SAS-specific effective population size.) All four now perform in the same band: mean recall 0.82-0.87, with block counts exact or within one block of the reference.

| block set | ours | ref | delta | mean recall | mean bp-Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `EUR` (deCODE, post-fix) | 1361 | 1361 | 0 | 0.865 | 0.773 |
| `pyrho_AFR` | 1580 | 1580 | 0 | 0.874 | 0.988 |
| `pyrho_EAS` | 1118 | 1121 | -3 | 0.825 | 0.983 |
| `pyrho_EUR` | 1335 | 1336 | -1 | 0.864 | 0.979 |

## deCODE map interpolation bug: root cause found and fixed

`src/ldetect_lite/interpolate_maps.py::interpolate()` is a **point**-based linear interpolator (a port of `joepickrell/1000-genomes-genetic-maps`), but deCODE's source map is **interval**-based (`Begin, rate_cM_Mb, cumulative_cM_at_End` per interval, per MacDonald et al.'s `interpolate.R`). Feeding interval data through point-based interpolation is an off-by-one interval shift: every SNP's genetic position used the *next* interval's rate anchored at the *current* interval's endpoint, producing a ~0.001-0.01 cM error per SNP.

**Fix**: added `interpolate_intervals()` (a direct port of the R script's interval-anchoring logic) as an alternative to `interpolate()`, exposed via `ldetect interpolate-maps --mode {point,interval}` (default `point`, preserving prior behavior for true point-sampled maps like HapMap-interpolated 1000G maps). Confirmed on real deCODE data: interval mode is 10-50x closer to MacDonald's own published interpolated map than point mode (MAE 0.00004-0.0002 cM vs. 0.0017-0.002 cM, Pearson r = 1.0 in both).

Fixing the EUR block set's map source (switching from our own recomputed interpolation to MacDonald's own published already-interpolated deCODE map — the same approach already used for the pyrho maps) resolved the bulk of the divergence: mean recall **0.63 -> 0.865**, block count exact (1361/1361), median boundary offset 0.0 kb on all 22 chromosomes. EUR now sits in the same performance band as the three pyrho block sets instead of being an outlier. Worst remaining chromosomes (chr18 0.65, chr21 0.667, chr15 0.70 recall) show genuine hundreds-of-kb boundary shifts rather than razor-thin margins — likely the same genetic-map-desert mechanism described below, not a new bug.

## Why the pyrho sets aren't exact anywhere

Unlike `examples/ldetect_original` (near-exact genome-wide, since it replays byte-identical archived VCF/map files the original authors used), no MacDonald pyrho chromosome is close to 100% exact: per-chromosome exact-match rates run ~29-96%, non-exact boundary rates ~17-21% genome-wide across all three populations. Two known, verified mechanisms explain this:

1. **Genetic-map deserts** (Category A — chr9, and 7 other chromosome/population combinations). Some chromosomes have large (tens-of-Mb) genetic-map dead zones (e.g. chr9's 17 Mb desert, consistent with its heterochromatic 9qh block). Legacy LDetect emits one unsplit block across the whole desert; `ldetect-lite` places an extra breakpoint inside it (same total genome-wide breakpoint budget, different placement) — confirmed directly against MacDonald's own git history (raw pre-centromere-removal BEDs).
2. **Razor-thin local-search margins** (Category B — verified directly on EAS chr4 by replaying the actual `LocalSearch` class against real covariance data). `LocalSearch` correctly finds the true optimum in its search window every time — no bug — but tiny, legitimate numerical differences in the covariance computation (which cannot be byte-compared against legacy — no published intermediates exist) are enough to flip which of two near-tied candidates wins a sub-0.1%-margin race. The algorithm is a zero-tolerance greedy optimizer by design, so it's maximally sensitive to tiny input differences by construction, not through a defect.

Both mechanisms converge with the same "flat region" finding documented for `ldetect_original` (see `notes/findings/ldetect-original-reproduction.md`) — this is the same underlying phenomenon showing up pervasively here rather than on 1-2 chromosomes, because MacDonald's pipeline (unlike `ldetect_original`) is a from-scratch rerun against modern VCF releases and pyrho/deCODE maps rather than a byte-identical replay.

Investigated and **ruled out** as explanations: window-bound computation (matches legacy exactly), metric formula/tie-breaking (identical to legacy; two latent tie-break/denominator-guard bugs were found and fixed along the way, but don't affect any real reproduction output), the historical array-vs-Decimal local-search divergence (verified fixed — see `notes/logs/local-search-divergence-asn22.md`), `N_zero` denominator conditioning (essentially constant across search windows; the sensitivity comes from the numerator), MAF-filter type (`nref` vs. true minor-allele-frequency — tested genome-wide on EAS, makes concordance *worse*, not better), and individual-panel-membership drift (de-risked: selection method is a faithful, deterministic clone of MacDonald's documented recipe).

## Bottom line

No further code fix is expected to close the remaining gap without MacDonald's own covariance intermediates, which don't exist publicly. Current state is the practical ceiling for an independent, from-scratch reproduction against modern data releases.
