# SV/Indel Partition-Boundary Duplication Investigation

**Agent-oriented working log.** Raw, dated investigation notes — not proofread for external readability. For current, human-readable status, see `notes/findings/ldetect-original-reproduction.md`.

Date: 2026-07-12

## Context

Found incidentally while investigating an unrelated chromosome-mode covariance exactness gap on the `covariance-optimization` branch (see that branch's `notes/logs/covariance-bitpacked-kernel-and-chromosome-mode.md`). `calc_covariance`'s region-based read (the only covariance strategy this pipeline has today — every existing `ldetect_original`/`MacDonald2022` reproduction run uses it) has no explicit `start <= pos <= end` check of its own; it only requires the position to have a genetic-map entry, and otherwise trusts whatever `vcf(region)` returns. `ldetect-lite` reads via `cyvcf2` (htslib-backed), not a `tabix` subprocess — the main branch moved off tabix streaming a while ago, `docs/optimizations.md` #10 — but `cyvcf2`'s region query goes through the same underlying htslib region-matching code, so the same behavior applies regardless of which one actually invokes it: matching is span-based, not `POS`-based. For an ordinary SNP this is a distinction without a difference (span == `POS`), but for a structural variant with an `INFO/END` far from `POS`, or a long-`REF` indel/SV, the record can be returned for a region query even though its `POS` lies far outside that region — a large SV/indel near a partition boundary can be spuriously double-counted into a neighboring partition, adding extra r² weight to the diagonal-sum vector near that boundary.

This log tracks whether that mechanism explains the parked EUR chr8-12 / AFR chr22 reproduction divergence. The distilled, current-status conclusion lives in `notes/findings/ldetect-original-reproduction.md`; this file is the full blow-by-blow.

## First direct evidence: EUR chr22, not AFR chr22

The original observation that started this thread was a real chr22 EUR run (`bench_bitpacked_full_genome.py --population EUR --chromosomes 22 --include-chromosome-mode`, on the `covariance-optimization` branch): a partition ~78kb away from a variant's `POS` still received that variant's row, because the variant (`esv2670821`) is an 82.5kb `<DEL>` structural variant (`POS=22517055`, `INFO/END=22599640`) whose span reaches into the neighboring partition `[22595093, 23027503]`.

Initially mischaracterized (in an earlier draft of this investigation) as having been found "in exactly the chromosome that diverges" — it wasn't. EUR chr22 is not one of the flagged divergent chromosomes (EUR's divergence is chr8-12 only); EUR chr22 currently reproduces exactly despite having this artifact present. That's not a contradiction of the flat-region mechanism (`notes/findings/ldetect-original-reproduction.md`): a duplicated SV only flips the outcome if it lands somewhere the signal is already near-tied, which apparently isn't the case for EUR chr22. But it does mean AFR chr22 needed to be checked on its own merits, not by analogy.

## AFR chr22 boundary audit

Checked AFR chr22 directly with a standalone, site-only script (no full pipeline run): `bcftools view -G` drops genotypes since span (`POS`, `REF` length, `INFO/END`) is a record-level property, independent of any individual's genotype — this makes the scan seconds, not minutes, even on the full 1092-sample chr22 VCF (first attempt without `-G` took minutes and multiple GB of RSS formatting genotype columns nobody needed; killed and redone with `-G`).

First pass scanned the **raw, all-population** VCF against AFR's actual `partition_chromosome` boundaries (246 individuals, 98 partitions): 251 SV-like/long-`REF` records genome-wide on chr22, **10 boundary-spanning hits** (span crosses a partition boundary the record's own `POS` falls outside of).

That overcounts, per the user's catch: it doesn't account for which SVs are actually polymorphic *in AFR specifically* — an SV monomorphic in AFR would be removed by the real pipeline's population-specific `MAC[0]>=1` filter before `calc_covariance` ever sees it, so it can't actually trigger this mechanism for an AFR run even if it's present in the raw multi-population file. Rerun through the same filter the real pipeline applies (`bcftools view -S <AFR individuals> | bcftools view -i 'MAC[0]>=1' -m2 -M2`, i.e. `examples/ldetect_original/Snakefile`'s `filter_vcf` rule) before scanning: 341,490 variants survive the AFR-specific filter, only 166 of the original 251 SV-like records are still polymorphic in AFR, and **7 of those have a span crossing a partition boundary** — including the same `esv2670821`/`esv2670795` pair found in the EUR check. Three of the original 10 raw-scan hits (`esv2661366`, `esv2673637`, `esv2677311`) dropped out once restricted to AFR-polymorphic sites — direct, concrete evidence that population-specific allele frequency is a real modulator of which chromosomes/populations this mechanism can affect, independent of anything else.

This is audit-only (no covariance/pipeline run): it shows the mechanism *can* fire on AFR chr22's actual boundaries, on the exact variant set AFR's real pipeline run would see. It does not by itself show that it *does* change AFR chr22's output relative to the reference.

Scripts: `examples/ldetect_original/scripts/audit_boundary_spanning_variants.py` (the cleaned-up, reusable version); wired into `Snakefile.sv_boundary_diagnostics` as `svb_boundary_audit`/`svb_boundary_audit_summary`, using the `all` mode's already-filtered input VCF and the partition file `ldetect run` itself computed (not a recomputation) — so the cluster run reuses this same audit for EUR chr8-13 too, which wasn't checked locally (would have required fresh full-chromosome downloads; held off per instruction not to run large jobs locally).

## Reference-implementation precedent, both ways

- Original ldetect (`_reference/ldetect_original/ldetect/examples/P00_01_calc_covariance.py`): fed via `tabix -h <full-panel-VCF> <region>`, no additional `POS` filtering in the Python parser. Its own documented example command uses the same `...snps_indels_svs...` file. Exposed to the identical mechanism.
- MacDonald et al. (2022) (`_reference/LDblocks_GRCh38/scripts/subsetVcf.sh`): filters to SNPs only *before* any partitioning/covariance step (`bcftools view --min-af 0.01 --types snps`), structurally eliminating this mechanism. `examples/MacDonald2022/Snakefile` already mirrors that filter, so that reproduction isn't exposed to this issue. `examples/ldetect_original/Snakefile` deliberately does *not* filter by type (its own comment: "Do not add ... --types snps"), for fidelity to the original's stated methodology — a choice made before this mechanism was known.

## Full-pipeline test: `Snakefile.sv_boundary_diagnostics`, negative result

Ran the diagnostic for real (on the user's cluster, not locally) across EUR chr8-13 / AFR chr11,13,22: `ldetect run` twice per chromosome, current `all`-variant-types filtering vs. an added `snps_only` filter, each compared against the published reference.

Reading `summary.tsv`: rows with `our_n=0` are chromosomes this diagnostic's configured subset didn't run (`compare_blocks.py` reports every chromosome in either BED, and the reference BED has all 22) — not regressions, just absent data. The real signal is in the nonzero rows:

| Chromosome | Role | `all` recall/precision/jaccard (tol0) | `snps_only` recall/precision/jaccard (tol0) |
|---|---|---|---|
| AFR chr13 | control, previously exact | 1.0 / 1.0 / 1.0 | 0.52 / 0.49 / 0.34 |
| AFR chr11 | control, previously ~exact | 0.99 / 1.0 / 0.98 | 0.35 / 0.33 / 0.21 |
| AFR chr22 | the actual divergence target | 0.69 / 0.71 / 0.54 | 0.44 / 0.43 / 0.28 |
| EUR chr8-12 | divergent target | ~0.06-0.19 across the board | mixed, mostly flat or marginally worse |

`snps_only` regresses everything checkable — including previously-exact controls and AFR chr22 itself. Not a surprise on reflection: the original's published blocks were generated *using* the full `snps_indels_svs` file, so removing that data from our input doesn't undo a bug relative to what they computed, it just makes our input diverge further from theirs. This is the same conclusion the earlier position-Jaccard "SNP-only vs. all-variant" test already reached, now confirmed at the actual BED-output level, more decisively.

This doesn't kill the underlying mechanism — it kills blanket SNP-only filtering as a way to test it. That approach conflates removing the spuriously-duplicated rows with removing legitimate SV signal that correctly belongs to its own partition; it can't distinguish "duplication is the problem" from "SVs matter and shouldn't be touched at all." The surgical fix (explicit `start <= pos <= end` filter in `calc_covariance`'s region-read loop) would isolate just the duplication artifact without discarding real data, and remains unimplemented and untested — see `notes/findings/ldetect-original-reproduction.md`'s "If this is picked up again" for current priority.
