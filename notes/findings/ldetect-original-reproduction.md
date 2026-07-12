# ldetect_original reproduction — findings

**Findings summary (current as of 2026-07-03).** Distilled for human review — e.g. writing up the paper. Full investigation detail, diagnostic scripts, and dated process notes: `notes/logs/ldetect-original-main-pipeline-audit.md`.

## Status: parked, not actively being investigated

`examples/ldetect_original` reproduces Berisa & Pickrell (2016)'s published LD blocks:

- **ASN**: exact match, all 22 autosomes.
- **AFR**: exact except chr22. (chr11 was previously tracked as divergent too — resolved, see below.)
- **EUR**: exact block *counts* and coverage on every chromosome, but chr8–chr12 (a contiguous run bracketed by exact chr7/chr13 matches) have shifted internal boundary *positions*.

EUR chr8-12 and AFR chr22 are accepted, documented residual divergences. Every concrete, checkable hypothesis for them has been ruled out short of the original authors' own internal processing logs.

## Ruled out

- **VCF release-version provenance** — ruled out twice: once via position-set and phasing-sensitive LD comparisons across releases, and again by a full-pipeline rerun on v1/v2/old2011 and an undocumented `merged_umich` snapshot for every divergent chromosome — none reproduce a divergent chromosome exactly.
- **SNP-only vs. all-variant filtering** — ruled out twice: at the whole-file position-set level, and (2026-07-12) at the actual BED-output level by running the pipeline both ways — filtering to SNPs only *regresses* the match, including on previously-exact control chromosomes and on AFR chr22 itself.
- **SV/indel partition-boundary duplication** — a real, confirmed mechanism (`calc_covariance`'s region-based read spuriously duplicates a boundary-crossing SV/indel into a neighboring partition), but ruled out as an explanation for this divergence: a surgical fix that removes exactly the duplicated rows (`--strict-region-bounds`) produces **byte-identical breakpoints** to the unfixed baseline on every configured chromosome, including AFR chr22 and EUR chr8-12 themselves. The mechanism is real but doesn't perturb the signal enough to flip any local minimum in this pipeline's current runs. Full detail: `notes/logs/sv-boundary-diagnostics-investigation.md`.
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

1. Run `Snakefile.provenance_diagnostics`'s `sv_attribute_comparison_summary.tsv` against the real v1/v2/old2011/merged_umich release archives, to check whether SV `POS`/`END` drift across releases correlates with the divergent chromosomes specifically. (Motivated by the now-ruled-out SV/indel partition-boundary duplication mechanism, but a distinct question: this checks whether SV calling itself differs across releases, independent of the partition-duplication artifact.)
2. Close the AFR chr22 gap in the precision/legacy-downstream evidence: run `--high-precision` for AFR chr22, and run `Snakefile.legacy_diagnostics` (already generalized to cover AFR chr21/chr22 alongside EUR) to check whether the same "implementations agree with each other, both disagree with the reference" pattern holds there too.
3. The open question is upstream of covariance/local search: what input or preprocessing step produces a subtly different vector than whatever the original authors used, for exactly these chromosomes? Remaining concrete lead: EUR/AFR/ASN subpopulation-code provenance — EUR is proven byte-identical (379/379) against the original toy example's actual sample list, but no equivalent ground-truth AFR list exists to check the same way; AFR's provenance rests only on population counts matching across VCF releases (246 individuals), not a byte-for-byte proof.
4. Absent new evidence (a new data source, an errata from the original authors), this is close to the practical limit of what's resolvable without the authors' internal processing logs.
