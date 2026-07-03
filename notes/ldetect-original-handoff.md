# ldetect_original reproduction — handoff (2026-07-03)

Current-state summary for picking this investigation back up. For the full
chronological log of every diagnostic run and its reasoning, see
`notes/ldetect-original-main-pipeline-audit.md` (append new dated entries
there as the log of record; this file is a snapshot, not a log).

## Where things stand

Trying to exactly reproduce Berisa & Pickrell (2016)'s published LD blocks
from `examples/ldetect_original`. Status:

- **ASN**: exact match, all 22 autosomes. No open issues.
- **AFR**: exact except chr22. (chr11 was previously tracked as divergent too
  — **now resolved**, see below.)
- **EUR**: exact except chr8, chr9, chr10, chr11, chr12 (contiguous run,
  bracketed by exact chr7/chr13). Block *counts* match exactly everywhere;
  only internal boundary *positions* differ on these chromosomes.

Two genuinely open mysteries remain: **EUR chr8-12** and **AFR chr22**.

## Ruled out this investigation (do not re-litigate without new evidence)

- VCF release-version provenance — ruled out twice: once by position-set +
  phasing-sensitive LD sampling diagnostics (`compare_vcf_positions.py`,
  `compare_vcf_ld.py` via `Snakefile.provenance_diagnostics`), and again by a
  **direct full-pipeline rerun** on v1/v2/old2011 for every divergent
  chromosome (`Snakefile.alternate_source_rerun`) — none of them reproduce
  any divergent chromosome exactly; v2 gives a small (not decisive) recall
  bump on some EUR chromosomes, and makes AFR chr22 worse.
- SNP-only vs. all-variant filtering — ruled out.
- Genetic map family (OMNI vs. HapMap; also confirmed via the paper's
  supplement that HapMap Phase 2 Release 22 is the correct source) — ruled
  out.
- `Ne` assignment, including the discovery that `ldetect2` intentionally uses
  population-specific `Ne` for partition-boundary extension where legacy
  hardcoded `11418` — real difference, but ASN (equally affected) reproduces
  exactly, so not explanatory.
- Duplicate-VCF-position and cross-partition-overlap handling in `ldetect2`
  — proven mathematically and empirically equivalent to legacy (new
  regression tests added: `tests/test_duplicate_overlap_integration.py`,
  `tests/_partition_fixtures.py`, plus value-level tests in
  `test_shrinkage.py`/`test_covariance_io.py`/`test_metric.py`).
- Sample/panel provenance — EUR sample list proven byte-identical (379/379)
  to the actual list distributed with the original `ldetect` toy example
  (`_reference/ldetect_original/ldetect/examples/example_data/eurinds.txt`);
  v1/old2011/v3 panel-vintage question closed (no distinct panel ever
  existed for v1/old2011; v2's own panel is byte-identical to v3's).
- Reference BED file structural integrity — audited for gaps, overlaps,
  degenerate/duplicate rows across all three genome-wide files and all 66
  chromosome-specific files from Bitbucket; only the AFR chr11 issue below
  was found, nothing else.
- Multiallelic ALT-allele-trimming order of operations — theoretically
  plausible, but no positive evidence, and the near-total genome-wide exact
  match rate is hard to reconcile with it being a real, impactful issue.
  Deprioritized, not disproven.
- The `merged_umich` undocumented Phase 1 snapshot — ruled out via the cheap
  position/LD diagnostics (see below); statistically indistinguishable from
  `v2`, so not worth a full-pipeline rerun. This closes out VCF
  release/provenance as a category.

## Resolved: AFR chr11 was never a real divergence

The published reference BED (`AFR_fourier_ls-all.bed`, and independently the
chromosome-specific `AFR/fourier_ls-chr11.bed` from Bitbucket — both
byte-identical) has two corrupted rows at exactly the boundary `ldetect2` was
flagged for "emitting an extra boundary" at:
```
chr11  108823642  None
chr11  None       111048570
```
Our own output has `108823642 -> 109897792 -> 111048570` — precisely the
breakpoint the reference lost to corruption. This is a data bug in the
published reference, present in the earliest available published source (not
introduced by Bitbucket's genome-wide concatenation), so the true intended
value isn't recoverable. Treat AFR chr11 as solved/non-issue going forward.

## Vector/boundary visualization: a mechanism, but not "inherent ambiguity"

Built `examples/ldetect_original/scripts/plot_vector_boundaries.py` — plots
the raw + Hann-smoothed diagonal-sum vector (the actual signal minima are
detected on, at the real `found_width` from `breakpoints-<chrom>.json`) for a
genomic window, overlaid with both our boundaries (blue=matched,
orange=divergent vs. reference within tolerance) and reference boundaries
(red dashed).

Generated 3-window comparisons (concordant-only / divergent-only / mixed) for
both **EUR chr10** and **AFR chr22**, saved at
`examples/ldetect_original/plots/` (untracked; not yet committed):

- EUR chr10 concordant: 61.5-63.1 Mb (boundaries 61891409, 62660140)
- EUR chr10 divergent: 99.5-105.7 Mb
- EUR chr10 mixed: 53.0-57.5 Mb
- AFR chr22 concordant: 31.2-34.5 Mb (boundaries 31439173, 32665304, 34279012)
- AFR chr22 divergent: 47.6-49.15 Mb
- AFR chr22 mixed: 21.2-23.9 Mb

**Consistent pattern across both chromosomes**: concordant boundaries sit on
visually sharp, well-defined troughs in the smoothed vector. Divergent
boundaries consistently fall in comparatively flat, low-amplitude,
featureless stretches of the smoothed signal — no clear minimum, many
nearby points nearly tied.

**Initial framing was too strong and has been corrected** (see main audit
log's "Revisiting the flat-region hypothesis" entry, 2026-07-03). The
original framing was "inherent algorithmic sensitivity — any two reasonable
implementations could tip differently in a flat region, no data/provenance
discrepancy required." But two older diagnostics (found while folding
`ldetect-original-concordance.md`'s history into the main log, originally run
2026-05-06/06-21) already tested exactly this and came back against it:

- `--high-precision` (legacy-equivalent Decimal/dict local-search path) on
  EUR chr10 produced **byte-identical breakpoints** to the default
  float64/array path — two numerically distinct implementations, same
  upstream vector, no disagreement, even in the flat regions causing chr10's
  divergence.
- Running the actual copied **legacy downstream scripts** (not ldetect2) on
  the *same* ldetect2-generated covariance for EUR chr7-13 *also* failed to
  reproduce the published chr8-12 reference, while agreeing closely with
  ldetect2's own output (0.82-0.96 concordance) — much closer to each other
  than either is to the reference.

So the two implementations we have don't actually disagree with each other
in these flat regions — they agree with each other and disagree with the
reference. That points to the vector/covariance signal itself being subtly
different from whatever the original authors computed (an unidentified
upstream input/provenance difference — still not found despite ruling out
VCF release version, sample panel, map family, OMNI-vs-HapMap, Ne, and
duplicate/cross-partition handling), not to unresolvable implementation
ambiguity. The flat-region correlation is still a real and useful
*mechanism* — it explains why divergence is localized to specific boundaries
(small upstream signal differences only flip the outcome where the signal is
already near-tied) — but it should not be read as "unfixable in principle."

**Caveat**: both of the above tests only cover EUR chr8-13, not AFR chr22.
`Snakefile.legacy_diagnostics` was EUR-only (a single hardcoded `population`
config key); generalized on 2026-07-03 to a `chromosomes_by_population` dict
(mirroring `Snakefile.alternate_source_rerun`'s pattern), with all rule
outputs now namespaced under `results/legacy_diagnostics/{population}/{chrom}/`.
`legacy_diagnostics.yaml` now requests `EUR: [8,9,10,11,12,13]` and
`AFR: [21, 22]` (chr21 as the exact-match control, chr22 the divergence) in
one invocation. Not yet run — needs the user's compute environment (same as
the other heavy diagnostics). Snakemake isn't installed in this sandbox, so
the dry-run (`-n`) couldn't be verified here; do a dry-run first when picking
this up:
```bash
cd examples/ldetect_original
uv run snakemake -n -s Snakefile.legacy_diagnostics --configfile config.yaml legacy_diagnostics.yaml
uv run snakemake --cores 8 -s Snakefile.legacy_diagnostics --configfile config.yaml legacy_diagnostics.yaml
```
`--high-precision` for AFR chr22 hasn't been run either — that's the other
half of this check and doesn't need any Snakefile change (`ldetect2 run
--high-precision` already works for any population/chromosome).

## Closed since last handoff: `merged_umich` snapshot

Ran (2026-07-03). Result: **negative, as expected.** `merged_umich` is a
strict position-superset of v3 (0 baseline-only positions, ~5-6%
candidate-only, same shape as `v2`) and, at every shared position, phased
r²/MAF are byte-identical to v3 (`r2_pearson_r`=1, all diffs=0) — for
divergent chromosomes *and* concordant controls alike. Quantitatively
indistinguishable from `v2` on every metric these diagnostics measure, and
`v2`'s own full-pipeline rerun already gave only a non-decisive result. Not
worth a full-pipeline rerun; VCF release/provenance is now exhausted as a
category. Full detail in the main audit log's "`merged_umich` snapshot
check" entry (2026-07-03).

The user obtained the paper's supplementary PDF directly (was previously
unreachable via automated fetch) — already fully mined for findings (see
main audit log). No further paper-side leads outstanding.

## Key scripts/infra built this session (all in `examples/ldetect_original/`)

- `scripts/compare_vcf_ld.py` — phasing-sensitive LD comparison between two
  VCF releases (had two real bugs, both fixed: a `.bed`-extension region-file
  gotcha in bcftools, and a duplicate-position last-write-wins bug).
- `scripts/compare_partition_overlap_duplicates.py` — checks whether
  redundant cross-partition covariance pairs are bit-identical; not yet run
  against real data (needs real materialized `.h5` partitions).
- `Snakefile.alternate_source_rerun` + `alternate_source_rerun.yaml` — full
  end-to-end pipeline rerun on alternate VCF releases for divergent
  chromosomes only. Already run and analyzed (see "ruled out" above).
- `scripts/plot_vector_boundaries.py` — vector/boundary visualization, this
  session's newest tool.
- `tests/_partition_fixtures.py`, `tests/test_duplicate_overlap_integration.py`
  — new regression tests proving duplicate-position/cross-partition handling
  matches legacy.

## Recommended next steps, roughly in priority order

1. Close the AFR chr22 gap in the precision/legacy-downstream evidence: run
   `--high-precision` for AFR chr22, and run the now-generalized
   `Snakefile.legacy_diagnostics` (see above — updated 2026-07-03 to cover
   AFR chr21/chr22 alongside EUR) to check whether the same "implementations
   agree with each other, both disagree with the reference" pattern found
   for EUR chr8-12 also holds there. This would confirm the
   upstream-signal-difference framing applies uniformly rather than being
   EUR-specific.
2. The real open question is now squarely upstream of covariance/local
   search: what input or preprocessing step produces a subtly different
   vector than whatever the original authors used, for exactly these
   chromosomes? The Tier 1 hidden-assumption items are the remaining
   concrete leads here: v1/old2011 sample-panel vintage, EUR/AFR/ASN
   subpopulation-code provenance (see the hidden-assumption-audit plan/log
   entries for detail).
3. If (1) and (2) don't turn up anything, this may be close to the practical
   limit of what's resolvable without the original authors' internal logs —
   worth an explicit conversation with the user about whether to keep
   digging or document this as an accepted, understood residual divergence.
