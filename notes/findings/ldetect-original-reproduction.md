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
- **SNP-only vs. all-variant filtering, at the whole-file level** — ruled out: bulk-removing indels/SVs from the input doesn't move the overall called-position set (Jaccard) differently for divergent vs. control chromosomes. This does **not** rule out a narrower, per-boundary mechanism — see "New candidate mechanism" below, found 2026-07-12 and not yet tested against this reproduction.
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

## New candidate mechanism: SV/indel partition-boundary duplication (found 2026-07-12, not yet tested here)

Found while investigating an unrelated chromosome-mode covariance discrepancy on a separate performance-optimization branch (not yet merged as of this writing; see that branch's own notes if present). `calc_covariance`'s region-based read (the only covariance strategy this pipeline has today, i.e. every existing reproduction run) has no explicit `start <= pos <= end` check of its own — it only requires the position to have a genetic-map entry, and otherwise trusts whatever `vcf(region)` returns. htslib/tabix region matching is **span**-based, not `POS`-based: for an ordinary SNP this is a distinction without a difference (span == `POS`), but for a structural variant with an `INFO/END` far from `POS`, or a long-`REF` indel/SV, the record can be returned for a region query even though its `POS` lies far outside that region (confirmed directly on real chr22 EUR data: a partition ~78kb away from a variant's `POS` still received that variant's row, because the variant is an 82.5kb `<DEL>` structural variant whose span reaches into the partition).

This means a large SV/indel near a partition boundary can be **spuriously double-counted** — once by its own true partition, once by a neighboring partition whose region query its span reaches into — adding extra r² weight to the diagonal-sum vector at specific positions near that boundary. That is exactly the kind of small, localized upstream-signal perturbation the flat-region mechanism above says would flip a near-tied outcome, and it is mechanistically distinct from the whole-file "SNP-only vs. all-variant" test already ruled out (that test only checks aggregate position-set membership, not per-boundary double-counting).

**Why this specifically points at EUR chr8-12 / AFR chr22:** the 1000G Phase 1 source file is literally `...snps_indels_svs...` — SVs and indels are present in every chromosome's real input, not a contrived edge case. `Ne` (already known to differ from the original's hardcoded `11418`, previously dismissed as non-explanatory on its own) directly controls `partition_chromosome`'s overlap-extension logic, i.e. exactly which partition boundaries a given SV's span would cross — so a different `Ne` between this reproduction and the original authors' (undocumented) run would cause the *same* underlying mechanism to duplicate *different* SVs across *different* boundaries, consistent with "same bug class, different specific chromosomes affected."

**Reference-implementation precedent, both ways:**
- The original ldetect (`_reference/ldetect_original/ldetect/examples/P00_01_calc_covariance.py`) is fed via `tabix -h <full-panel-VCF> <region>` with no additional `POS` filtering in the Python parser either — its own documented example command uses the same `...snps_indels_svs...` file. It is exposed to the identical mechanism; the published reference blocks were plausibly generated with this same artifact present, just affecting different specific boundaries.
- MacDonald et al. (2022) (`_reference/LDblocks_GRCh38/scripts/subsetVcf.sh`) instead filters to SNPs only *before* any partitioning or covariance step (`bcftools view --min-af 0.01 --types snps`), which structurally eliminates this mechanism (no indel/SV left to have a mismatched span). `examples/MacDonald2022/Snakefile` in this repo already mirrors that filter, so the MacDonald2022 reproduction is not exposed to this issue. `examples/ldetect_original/Snakefile` deliberately does *not* filter by type (see its "Do not add ... --types snps" comment), for fidelity to the original's stated methodology — a choice made before this mechanism was known.

**Not yet tested**: whether removing indels/SVs (mirroring MacDonald2022's `--types snps`) changes the *actual BED output* for EUR chr8-12 / AFR chr22 specifically, closer to the published reference. The old "ruled out" position-jaccard test never checked this. `Snakefile.sv_boundary_diagnostics` (added 2026-07-12, see `examples/ldetect_original/README.md`) runs the full pipeline both ways (current `all`-variant-types filtering vs. a `snps_only` mode) on the same EUR chr8-13 / AFR chr11,13,22 chromosome set used by the provenance diagnostics, and compares each mode's BED against the published reference at both the loose 100kb tolerance (should stay matching) and an exact 0bp tolerance (the metric that actually tests this hypothesis), plus `snps_only` directly against `all` to quantify how much the filter itself shifts boundaries.

## If this is picked up again

Not currently planned, but in priority order:

1. Run `Snakefile.sv_boundary_diagnostics` (see above) — the most concrete, previously-untested lead, and the cheapest to check since it requires no code changes, only an upstream VCF filter.
2. Close the AFR chr22 gap in the precision/legacy-downstream evidence: run `--high-precision` for AFR chr22, and run `Snakefile.legacy_diagnostics` (already generalized to cover AFR chr21/chr22 alongside EUR) to check whether the same "implementations agree with each other, both disagree with the reference" pattern holds there too.
3. The open question is upstream of covariance/local search: what input or preprocessing step produces a subtly different vector than whatever the original authors used, for exactly these chromosomes? Remaining concrete lead: EUR/AFR/ASN subpopulation-code provenance — EUR is proven byte-identical (379/379) against the original toy example's actual sample list, but no equivalent ground-truth AFR list exists to check the same way; AFR's provenance rests only on population counts matching across VCF releases (246 individuals), not a byte-for-byte proof.
4. Absent new evidence (a new data source, an errata from the original authors), this is close to the practical limit of what's resolvable without the authors' internal processing logs.
