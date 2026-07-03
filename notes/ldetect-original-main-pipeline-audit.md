# ldetect_original Main Pipeline Audit

Date: 2026-07-02

Inputs audited:

- `examples/ldetect_original/results/{AFR,ASN,EUR}/`
- `examples/ldetect_original/results/{AFR,ASN,EUR}_LD_blocks.bed`
- `examples/ldetect_original/results/compare/*_block_comparison.tsv`
- `examples/ldetect_original/results/logs/{AFR,ASN,EUR}/`
- `examples/ldetect_original/resources/ldetect_ref/*_fourier_ls-all.bed`

The main Snakemake pipeline completed all 22 autosomes for AFR, ASN, and EUR.
Each chromosome has a BED, breakpoint JSON, vector, partition list, log,
timing log, and benchmark file. No tracebacks, killed jobs, explicit errors, or
nonzero `/usr/bin/time` exit statuses were found in `results/logs`.

## Summary

| Population | Our blocks | Ref blocks | Delta | Mean recall at 100 kb | Mean bp-Jaccard | Remaining divergences |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AFR | 2584 | 2581 | +3 | 0.9894 | 0.9992 | chr11, chr22 |
| ASN | 1445 | 1445 | 0 | 1.0000 | 1.0000 | none |
| EUR | 1703 | 1703 | 0 | 0.8576 | 1.0000 | chr8-12 |

ASN is an exact match across all chromosomes.

EUR reproduces block counts and base-pair coverage for every chromosome, but
chr8-12 have shifted internal boundaries. The pattern is contiguous and is
bracketed by exact matches on chr7 and chr13, matching the earlier diagnostic
batch.

AFR is nearly exact. The only residual differences are chr11 and chr22. AFR
chr11 has two extra blocks but still near-perfect 100 kb boundary recall. AFR
chr22 has one extra block and a localized boundary disagreement.

**Update 2026-07-03: AFR chr11 is resolved — see "Supplementary PDF obtained"
below.** The "126 ref blocks" figure below is a parsing artifact of two
corrupted rows in the published reference BED file (`"None"` literal instead
of a real coordinate at exactly the boundary ldetect2 was flagged for
emitting an "extra" boundary at). Our output is very likely correct there;
treat AFR chr11 as no longer an open divergence. AFR chr22 remains open.

## AFR divergences

| Chrom | Our blocks | Ref blocks | Recall | Precision | Boundary Jaccard | Median offset kb | p90 offset kb | bp-Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chr11 | 128 | 126 | 0.9922 | 1.0000 | 0.9922 | 0.0 | 0.0 | 0.9835 |
| chr22 | 35 | 34 | 0.8056 | 0.8286 | 0.6905 | 0.0 | 376.2 | 1.0000 |

Interpretation:

- AFR chr11 looks like a small count-level discrepancy rather than a shifted
  chromosome-wide run. Reference boundaries are all matched at 100 kb, but
  ldetect2 emits one additional internal boundary at `109897792`, about
  1.07 Mb from the nearest reference boundary (`108823642`). Coverage endpoints
  are identical: `70855-134946452`.
- AFR chr22 has one additional block and several boundaries displaced beyond
  100 kb, but base-pair coverage is still identical
  (`16050408-51243298`). The unmatched boundaries are localized rather than
  chromosome-wide, clustering around roughly 22.6-25.2 Mb, 39.8-40.8 Mb,
  43.6 Mb, and 47.4-50.1 Mb. All AFR chr22 boundaries match by 500 kb
  tolerance.

AFR recall by tolerance:

| Chrom | 0 kb | 10 kb | 25 kb | 50 kb | 100 kb | 250 kb | 500 kb | 1 Mb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chr11 | 0.9922 | 0.9922 | 0.9922 | 0.9922 | 0.9922 | 0.9922 | 0.9922 | 0.9922 |
| chr22 | 0.6944 | 0.7500 | 0.7778 | 0.7778 | 0.8056 | 0.8333 | 1.0000 | 1.0000 |

## EUR divergences

| Chrom | Blocks | Recall | Boundary Jaccard | Median offset kb | p90 offset kb | bp-Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chr8 | 94 / 94 | 0.4105 | 0.2583 | 161.0 | 682.8 | 1.0000 |
| chr9 | 74 / 74 | 0.3467 | 0.2097 | 237.1 | 671.2 | 1.0000 |
| chr10 | 85 / 85 | 0.2907 | 0.1701 | 264.2 | 723.8 | 1.0000 |
| chr11 | 84 / 84 | 0.4588 | 0.2977 | 117.4 | 647.7 | 1.0000 |
| chr12 | 82 / 82 | 0.3614 | 0.2206 | 194.4 | 706.8 | 1.0000 |

Interpretation:

- The EUR issue remains a boundary-placement divergence, not a block-count,
  chromosome-coverage, or BED-combination issue.
- Exact matches on EUR chr1-7 and chr13-22 argue against a global current-code
  bug in filtering, partitioning, vector generation, minima selection, local
  search, or genome-wide BED concatenation.
- This continues to support the earlier hypothesis that published EUR chr8-12
  were generated from different upstream inputs/provenance, or from a narrow
  upstream implementation detail affecting only those chromosomes.

EUR boundary recall by tolerance:

| Chrom | 0 kb | 10 kb | 25 kb | 50 kb | 100 kb | 250 kb | 500 kb | 1 Mb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chr8 | 0.1895 | 0.3474 | 0.3579 | 0.3684 | 0.4105 | 0.5789 | 0.7474 | 0.9579 |
| chr9 | 0.0800 | 0.1867 | 0.2000 | 0.2267 | 0.3467 | 0.5467 | 0.7867 | 0.9733 |
| chr10 | 0.0581 | 0.2093 | 0.2093 | 0.2326 | 0.2907 | 0.4884 | 0.8140 | 0.9884 |
| chr11 | 0.1529 | 0.3294 | 0.3294 | 0.3882 | 0.4588 | 0.6588 | 0.8588 | 0.9882 |
| chr12 | 0.0843 | 0.2771 | 0.3133 | 0.3494 | 0.3614 | 0.5422 | 0.6747 | 0.9639 |

The 1 Mb recall recovery is high, but exact/near-exact recall is low. This is
consistent with broad local displacement of many internal boundaries, not with
a small number of missing/extra terminal intervals. Coverage endpoints are
identical for each divergent EUR chromosome.

## Runtime notes

Benchmark files show successful runs with expected scale by chromosome and
population. The largest wall-clock outliers are large SNP-count chromosomes and
chr11, especially EUR chr11 (`0:49:46`, max RSS about 3.0 GB by Snakemake
benchmark) and AFR chr11 (`0:45:18`, max RSS about 2.8 GB). These runtime
outliers do not correspond to process failures in the logs.

## Upstream provenance checks

The 1000 Genomes technical archive does not show an obvious chr11- or
chr22-specific Phase 1 patch directory analogous to the AFR residual
divergences. The visible chromosome-specific Phase 1 working directories are
mostly chr20 test/release material or unrelated chrX/chrY updates.

The same global Phase 1 release-version churn remains relevant for AFR:

- v3 current pipeline inputs:
  `ALL.chr11.phase1_release_v3.20101123.snps_indels_svs.genotypes.vcf.gz`
  (`6.7G`) and
  `ALL.chr22.phase1_release_v3.20101123.snps_indels_svs.genotypes.vcf.gz`
  (`1.8G`), posted in the final `release/20110521` directory.
- v2 archive:
  `ALL.chr11.phase1_release_v2.20101123.snps_indels_svs.vcf.gz`
  (`7.0G`) and
  `ALL.chr22.phase1_release_v2.20101123.snps_indels_svs.vcf.gz`
  (`1.9G`).
- v1 archive:
  `ALL.chr11.phase1_integrated_calls.20101123.snps_indels_svs.genotypes.vcf.gz`
  (`7.0G`) and
  `ALL.chr22.phase1_integrated_calls.20101123.snps_indels_svs.genotypes.vcf.gz`
  (`1.8G`).
- old 2011 archive:
  `ALL.chr11.merged_beagle_mach.20101123.snps_indels_svs.genotypes.vcf.gz`
  (`8.2G`) and
  `ALL.chr22.merged_beagle_mach.20101123.snps_indels_svs.genotypes.vcf.gz`
  (`2.2G`).

The 2013 Phase 1 haplotype-error position list overlaps the AFR residual
windows. Counts are not by themselves causal because the list is genome-wide
and dense, but chr22 is worth keeping as a local-quality candidate:

| Window | Haplotype-error positions | Rate |
| --- | ---: | ---: |
| AFR chr11 extra-boundary gap, 108823642-109897792 | 45 | 41.9/Mb |
| AFR chr22 cluster 22.6-25.2 Mb | 204 | 78.5/Mb |
| AFR chr22 cluster 39.8-40.8 Mb | 48 | 48.0/Mb |
| AFR chr22 cluster 43.0-44.2 Mb | 90 | 75.0/Mb |
| AFR chr22 cluster 47.4-50.1 Mb | 258 | 95.6/Mb |
| AFR chr22 full LDetect span, 16050408-51243298 | 2469 | 70.2/Mb |
| AFR chr11 full LDetect span, 70855-134946452 | 8155 | 60.5/Mb |

The Phase 1 mask README describes accessibility masks and notes that most
project variant calling did not use hard masks, instead relying on VQSR.
Therefore masks are a lower-priority explanation for LDetect BED divergence
unless the published run applied an additional SNP/accessibility filter that
our pipeline does not.

Best AFR follow-up comparisons:

1. Compare filtered AFR variant positions for chr11, chr22, and exact-match
   controls across v3, v2, v1, and the old 2011 Phase 1 archive.
2. Run a focused SNP-only v3 diagnostic for AFR chr22 first, then chr11. This
   tests whether indel/SV inclusion explains the residual extra/split blocks.
3. If SNP-only does not move chr22 toward the reference, run v2 or old-archive
   chr22 through covariance/vector/breakpoint generation before spending time
   on all AFR chromosomes.
4. Overlay unmatched AFR chr22 boundaries with haplotype-error positions and
   accessibility-mask intervals. This is mainly a triage step for local-quality
   artifacts, not yet a primary explanation.

## Provenance position/LD diagnostics follow-up

Date: 2026-07-02

Ran item 1 above (`Snakefile.provenance_diagnostics`) across EUR chr8-13 and
AFR chr11/13/22, using chr13 as an exact-match control in each population, for
v3/snps, v2/all, v2/snps, v1/all, and old2011/all against the v3/all baseline.
Results in `results/provenance_diagnostics/position_comparison_summary.tsv`.

### Finding: VCF release-version and SNP/indel filtering are ruled out

Position-jaccard between v3/all and every candidate is essentially identical
whether the chromosome is divergent or an exact-match control:

```text
                v3/snps   v2/all   v1/all
EUR chr8  (div) 0.943     0.944    0.806
EUR chr9  (div) 0.940     0.940    0.808
EUR chr10 (div) 0.938     0.943    0.813
EUR chr11 (div) 0.939     0.943    0.809
EUR chr12 (div) 0.936     0.940    0.802
EUR chr13 (ctl) 0.934     0.942    0.802
AFR chr11 (div) 0.951     0.948    0.877
AFR chr13 (ctl) 0.947     0.945    0.870
AFR chr22 (div) 0.949     0.945    0.881
```

The divergent chromosomes track their population's control chromosome almost
exactly. This rules out "SNP-only vs. all-variant" filtering and "which Phase
1 release version" as differentiators between divergent and matched
chromosomes: whatever's different about EUR chr8-12 and AFR chr11/chr22 is not
visible in which physical positions are called.

Also checked commit history on the two upstream data repositories this
pipeline depends on, since a "some chromosomes were silently patched later"
explanation would produce exactly the observed contiguous-range symptom:

- `joepickrell/1000-genomes-genetic-maps`: every
  `interpolated_from_hapmap/chr*.interpolated_genetic_map.gz` file was added
  in a single 2014-06-19 commit (`73cbe924`, message "hapmap") and has never
  been modified since (checked via `gh api .../commits?path=...` per
  chromosome for chr7-13).
- `nygcresearch/ldetect-data`: all EUR `fourier_ls-chr*.bed` files were added
  together in one commit (`777d32666e`, "added EUR dataset", 2015-04-15); AFR
  and ASN were likewise added together in one commit (`f651dc409a`, "Added
  AFR and ASN", 2015-04-30). There are only 14 commits in the repo's entire
  history and none of them touch a subset of per-chromosome BED files after
  the initial per-population add.

This rules out both "the published reference blocks were patched for some
chromosomes after initial publication" and "the genetic map was regenerated
for some chromosomes" as explanations — both artifacts have been static,
whole-population, single-commit additions since 2014-2015.

### New diagnostic: `scripts/compare_vcf_ld.py`

The position-set comparison only sees which SNP positions are *called*, not
genotype or phasing content at *shared* positions. That is the main gap,
since 1000G Phase 1 v1->v2->v3 differ mainly in re-phasing/re-imputation, not
just which sites are called, and phasing feeds directly into the Wen &
Stephens shrinkage covariance in `ldetect2.shrinkage`.

Added `scripts/compare_vcf_ld.py`: for a deterministic, evenly-spaced sample
of nearby SNP pairs at positions shared between a baseline and candidate VCF,
it computes minor allele frequency and phased-haplotype r^2 independently
within each VCF, then compares the resulting values (Pearson r and mean/
median/max absolute difference) between releases. r^2 computed within one VCF
is invariant to a whole-chromosome per-individual haplotype relabeling (swapping
which copy is called "haplotype 1" vs "haplotype 2" for an individual doesn't
change any pairwise haplotype r^2), so no cross-VCF haplotype-index alignment
is required — only local switch errors or re-calling differences between
releases would show up as a difference in the compared r^2 values.

Wired into `Snakefile.provenance_diagnostics` as `compare_ld_sets` /
`combine_ld_comparisons`, writing
`results/provenance_diagnostics/ld_comparison_summary.tsv` using the same
`comparison_candidates` list and the already-filtered VCFs from the position-set
diagnostic (no new downloads required; confirmed via `snakemake -n`). New
`ld_window_bp`/`ld_max_anchors`/`ld_pairs_per_anchor` knobs added to
`provenance_diagnostics.yaml`. Not run against real data in the session that
wrote it — bcftools is not available in that environment. The script's pure
logic (pair selection, MAF, r^2, output aggregation) and full `compare()` flow
were validated against a synthetic mocked-`bcftools` fixture: identical
phasing between two "releases" produces `abs_diff` ≈ 0 for shared pairs, a
scrambled/switch-error-like site produces `abs_diff` = 0.75.

#### Bug found on first real run: `.bed`-extension region file (fixed)

Date: 2026-07-02 (later same day)

The user ran `compare_ld_sets` for real (EUR chr8-13, AFR chr11/13/22, all 5
candidates each) and every single row of `ld_comparison_summary.tsv` came back
`n_pairs=0` with every downstream stat `nan` — including the MAF stats, not
just r^2. That total, uniform failure (46/46 rows, both populations, all
chromosomes and candidates) is not a plausible biological result; it means the
haplotype extraction step returned nothing for every position in every run.

Root cause: `compare()` wrote its temporary region file as
`Path(tmpdir) / "regions.bed"`, then read it back with `bcftools query -R`.
`bcftools` autodetects the region-file coordinate convention from the file
**extension**: a `.bed`-named file is parsed as 0-based, half-open BED
intervals; anything else uses the plain 1-based-inclusive `CHROM\tBEG\tEND`
convention. `write_region_file()` writes `pos\tpos` (start == end) intending
"select exactly this 1-based position" — correct for the plain convention,
but a zero-width, half-open `[pos, pos)` interval under BED semantics, which
matches **no** records. Because the file was named `regions.bed`, every
`bcftools -R` genotype query returned empty output, for every position, in
every comparison — exactly matching the observed symptom.

This is a real gap in the earlier validation: the mocked-`bcftools` unit test
stubbed out the `_run()` subprocess call entirely, so it could never exercise
real bcftools' region-file-parsing semantics — only the pure Python logic
downstream of that call.

**Fix**: renamed the temp file to `regions.txt` (`compare_vcf_ld.py`, in
`compare()`), so bcftools uses the plain 1-based-inclusive convention that
`write_region_file()` was already written for. No other code changed.

#### Real run after the fix: release-version phasing differences do not distinguish divergent chromosomes either

Date: 2026-07-02 (later same day)

Reran `compare_ld_sets` for real after the `.bed`-extension fix. Data is now
non-degenerate (`n_pairs` ~2500/row, matching `max_anchors=500 *
pairs_per_anchor=5`).

**v1/old2011 vs v3/all (different 1000G release): decisive negative result.**
The exact-match control chromosomes show r^2 discordance across releases
that's statistically indistinguishable from — or worse than — the divergent
chromosomes:

```text
                    r2_pearson_r  r2_mean_abs_diff  r2_max_abs_diff
EUR chr8  (div)     0.9820        0.00767           0.9998
EUR chr9  (div)     0.9829        0.00622           0.99998
EUR chr10 (div)     0.9786        0.00810           0.999993
EUR chr11 (div)     0.9940        0.00436           0.9525
EUR chr12 (div)     0.9782        0.00716           0.9917
EUR chr13 (ctl)     0.9738        0.00703           0.999997
AFR chr11 (div)     0.9732        0.00623           0.999996
AFR chr13 (ctl)     0.9729        0.00719           0.999967
AFR chr22 (div)     0.9855        0.00539           0.999996
```

EUR chr13 (control) has the *lowest* Pearson r of the whole EUR set; AFR
chr13 (control) is essentially tied with AFR chr11. This extends the earlier
position-set-only negative result to actual phasing-sensitive LD: VCF release
version is now ruled out both for which positions are called and for the LD
estimated at shared positions.

**v2/all vs v3/all (closer release): one inconclusive signal, not pursued
further.** AFR chr11 (divergent) shows real discordance (`max=0.938`) against
a perfectly-concordant AFR chr13 control (`max=0`) and a near-perfect AFR
chr22 (divergent, `max=0.0143`) — but EUR chr11 is the EUR outlier
(`max=0.755` vs control's `0.165`) while EUR chr9 (also divergent) is *more*
concordant than the control (`max=0.00127`). Inconsistent across the two
divergent EUR chromosomes checked; not treated as evidence either way.

**Second bug found and fixed: duplicate-position last-write-wins in
`read_phased_haplotypes()`.** The `v3/snps` vs `v3/all` rows (same release,
only the SNP-only filter differs) showed implausibly large discordance (e.g.
AFR chr22 `max=0.566`, EUR chr9 `max=0.467`) that should be ~0, since a SNP's
genotype calls don't change when unrelated indel records elsewhere in the
file are filtered out. Cause: `read_phased_haplotypes()` built
`haps[pos] = arr` as a plain, unconditional dict assignment, so when a
physical position had more than one VCF record (e.g. a SNP and a co-located
indel), whichever record `bcftools query` emitted *last* silently won —
arbitrary with respect to variant type, and can differ between VCF releases
or between the "all"/"snps" views of the same release.

Fix: keep only the *first* record per position (`if pos in haps: continue`
before parsing), matching how `ldetect2.shrinkage.calc_covariance` resolves
the same situation when it streams a VCF (see the "Duplicate-position /
cross-partition equivalence" section below) — `bcftools query -R` traverses
the same on-disk sorted/indexed file order as `calc_covariance`'s own
VCF-stream parsing, so "first" here should correspond to the same record the
real production pipeline would actually use at that locus. This is not
"pick the true SNP" in the abstract — it's "match what the real pipeline
would see," which is what this diagnostic is trying to measure. One
assumption not independently verified (bcftools unavailable in this
environment): that `bcftools query -R`'s same-POS tie order matches the
tabix-streamed order `calc_covariance` sees; this is standard, documented
htslib behavior (neither tool re-sorts), so treated as reliable. Verified
against a mocked-`bcftools` fixture with two records at one position and
different GT values — first now wins.

#### Final confirmed results after both fixes

Date: 2026-07-02 (later same day)

Reran `compare_ld_sets` again after the `read_phased_haplotypes()` fix. This
resolves the ambiguity above and gives a clean, final picture:

- **`v3/snps` vs `v3/all` (same release, filter-only): perfect concordance
  everywhere** — `r2_pearson_r=1`, `mean=0`, `max=0` for every single row.
  Confirms the fix: the previous implausible discordance was entirely the
  last-write-wins bug, not a real effect.
- **`v2/all` vs `v3/all`: also perfect concordance everywhere** — `1`/`0`/`0`
  for every chromosome, both populations, divergent and control alike. The
  previously-reported "AFR chr11 v2 mismatch (`max=0.938`)" flagged as a
  "mildly interesting, inconclusive signal" was *also* entirely the
  last-write-wins bug. v2 and v3 are genuinely byte-identical at every
  sampled shared position on every chromosome checked. No ambiguity remains
  here at all.
- **`v1/all`/`old2011/all` vs `v3/all`: still substantial, still uniform
  across divergent and control chromosomes.** Updated numbers (small shifts
  from the pre-second-fix table, same conclusion):

```text
                    r2_pearson_r  r2_mean_abs_diff  r2_max_abs_diff
EUR chr8  (div)     0.9871        0.00655           0.9913
EUR chr9  (div)     0.9829        0.00622           0.9999983
EUR chr10 (div)     0.9786        0.00808           0.999993
EUR chr11 (div)     0.9940        0.00425           0.9525
EUR chr12 (div)     0.9808        0.00669           0.9917
EUR chr13 (ctl)     0.9768        0.00649           0.999997
AFR chr11 (div)     0.9732        0.00623           0.999996
AFR chr13 (ctl)     0.9729        0.00717           0.999967
AFR chr22 (div)     0.9855        0.00538           0.999996
```

EUR chr13 (control) again has the lowest Pearson r of the EUR set; AFR chr13
(control) is again essentially tied with AFR chr11 (divergent) and slightly
worse than AFR chr22 (divergent) on mean_abs_diff.

**Conclusion: VCF-release-version/phasing provenance is now decisively and
comprehensively ruled out**, at every level checked: which positions are
called (earlier position-set diagnostic), and now actual phasing-sensitive
LD at shared positions, for both a near release (v2, perfectly identical to
v3 everywhere) and a distant one (v1/old2011, substantial but
uniformly-distributed discordance uncorrelated with reproduction success).
Combined with the earlier duplicate-position/cross-partition-handling
elimination (also empirically closed this session), essentially all
"cheap" upstream-input hypotheses for the EUR chr8-12 / AFR chr11/chr22
divergence are now exhausted. The leading remaining explanation is still the
one from the "Upstream provenance checks" section above: the published
reference blocks for those specific chromosomes were generated from a
different upstream snapshot/pipeline state than what's reproducible from
current public inputs, or a narrow implementation detail not yet identified.
Both remaining avenues (archaeology into the original authors' exact
run/errata, or a full alternate-VCF-release end-to-end rerun) are
substantially more expensive than anything tried so far.

### Duplicate/multiallelic-position handling: prior work exists on other branches, correcting an initial overclaim

`ldetect2.shrinkage.calc_covariance` (`src/ldetect2/shrinkage.py:634-654` on
`ldetect-original-fix`) deduplicates VCF rows by physical position while
parsing: it keeps the *first* row seen at a given `POS` and drops the rest,
logging the dropped count. This was initially flagged here as a "real,
previously-undocumented difference" from the original `P00_01_calc_covariance.py`,
which appends every row to `allpos` with no dedup at all. **That framing was
too strong; both parts need correction:**

1. **This exact question was already investigated, on other branches — the
   user's recollection was correct.** `further-optimizations` -> `nocache-mode`
   (and downstream `hdf5-experiments-direct-vector-r2-zarr` /
   `hdf5-memory-optimizations`) contain commits `c9a18f0` ("Dedup"), `145f07d`
   ("Fixed another duplicate handling issue"), and `47a060d` ("Patched matrix
   and direct method divergence"). None of these branches are merged into
   `ldetect-original-fix` or `main` (they diverge from the common ancestor
   `2e7579d`). That work was pursued as part of a large memory/performance
   optimization effort (compact HDF5 caching, an `r2-zarr` pair cache, tiled
   local search) and specifically fixed *internal* inconsistency between three
   computational backends (`matrix_hdf5`, `direct_hdf5`, `r2_zarr`) that had
   accidentally implemented different duplicate-position precedence rules from
   each other. `notes/optimizations-handoff.md` on `nocache-mode` states
   explicitly: "Known divergence from original `ldetect` on duplicate physical
   positions is still acknowledged" — i.e. that work did not claim to close,
   and did not verify closing, the gap against the *published Berisa/Pickrell
   reference blocks*. There is no evidence on those branches of an
   EUR-chr8-12/AFR-chr11/chr22 reproduction rerun tied to this change.

2. **Reading the legacy matrix-insertion code directly weakens the "different
   precedence" theory.** `_reference/ldetect_original/ldetect/baselib/flat_file.py:104`
   (`insert_into_matrix`) only inserts `matrix[l]['data'][r]` `if r not in
   matrix[l]['data']` — i.e. legacy is **first-insert-wins**, not last-wins as
   originally assumed here. Because covariance rows are written in `allpos`
   iteration order (VCF encounter order) and two duplicate-position variants at
   the same physical position share an identical genetic-distance window (the
   window depends only on physical position via `pos2gpos`), the
   first-VCF-encountered duplicate variant's row for any given neighbor `r`
   is always written — and therefore always kept — before the second
   duplicate's row for that same `r`. That means legacy's dict-overwrite
   dedup and ldetect2's parse-time "keep first, drop rest" dedup select the
   *same* surviving variant and should produce the *same* LD values for
   cross-position pairs. `notes/covariance-streaming-cache-implementation-note.md`
   (a design-only branch, also forked from `2e7579d`) independently documents
   this same "current" skip-duplicates-before-arrays behavior as an accepted
   baseline, not a bug.

**What remains a genuinely open, less-scrutinized candidate** is *cross-partition*
duplicate handling, not same-VCF duplicate physical positions. Partitions
overlap (`io/partitions.py` / `relevant_subpartitions`), so the same canonical
`(lo, hi)` pair can appear in more than one partition file. The
`covariance-streaming` design note flags this as a distinct mechanism ("Duplicate
Pairs Across Partitions") where local search "preserves partition and row order
and applies first-row-wins for canonical duplicate pairs in the active stream" —
this ownership/ordering logic is materially different from the simple
same-VCF-row case just ruled out above, has its own edge cases (see also
"Partition Overlap Ownership", "Boundary Inclusivity", "First-Locus Odd Pairs"
in that note), and has not been checked against the reference reproduction
either.

### Recommended next steps, in cost order

1. Run the new `compare_ld_sets` diagnostic (cheap — reuses already-filtered
   VCFs, no new downloads) for EUR chr8-13 and AFR chr11/13/22. If the
   divergent chromosomes show elevated `r2_mean_abs_diff` / degraded
   `r2_pearson_r` relative to the chr13 control, that is strong evidence of a
   phasing/re-calling difference between VCF releases, and a full old-VCF
   pipeline rerun becomes worth the cost.
2. Settle the duplicate-position question empirically instead of re-deriving it
   from first principles, but do **not** check out, diff against, or port code
   from `nocache-mode`/`further-optimizations`/etc. as part of this. Those
   branches were a large, never-merged optimization effort, and the same
   commits that touched duplicate handling there were fixing bugs the branch
   itself introduced (`47a060d` "Patched matrix and direct method divergence"),
   so their code is not a trustworthy oracle and pulling from it risks
   importing a different, undiscovered problem instead of resolving this one.
   Instead, write a small standalone script (in `examples/ldetect_original/scripts/`,
   independent of `src/ldetect2`) that parses one divergent chromosome's
   partition VCF twice — once reproducing `calc_covariance`'s current
   keep-first-drop-rest dedup, once with a from-scratch "retain every row,
   canonicalize with first-row-wins per `(lo, hi)` pair" implementation
   written directly against the legacy semantics described above — and diff
   the resulting covariance rows/vectors directly. This tests the hypothesis
   with fresh, purpose-built code instead of a legacy oracle port or
   another branch's code.
3. Only after (1) and (2) come back inconclusive is it worth running the full
   `ldetect2 run` pipeline end-to-end on an old VCF release (v1/old2011) for
   one divergent chromosome. The position-set diagnostic already shows the
   divergent chromosomes are indistinguishable from the exact-match control by
   which release is used, so a blind full-pipeline VCF swap is unlikely to fix
   anything on its own; it is informative only once paired with a real
   phasing-level anomaly detected in (1), or to test a specific "the original
   authors mixed up VCF sets for a batch of chromosomes" hypothesis after (1)
   and (2) leave that as the remaining explanation.

## Duplicate-position / cross-partition equivalence: closed empirically

Date: 2026-07-02

Item 2 above ("settle the duplicate-position question empirically") is now
closed. Direct comparison of legacy `_reference/ldetect_original/ldetect/` code
against current `src/ldetect2` (branch `ldetect-original-fix`), followed by new
regression tests, both point the same way: **duplicate-VCF-position and
cross-partition-overlap handling in the current codebase are legacy-faithful,
not the source of the EUR chr8-12/AFR chr11/chr22 divergence.**

### Why they're equivalent (the reasoning, not just the conclusion)

**Within-partition duplicate VCF positions** (two records sharing one `POS`):

- Legacy `baselib/flat_file.py::insert_into_matrix` (`if r not in
  matrix[l]['data']: ...`, line ~104-105) and `insert_into_matrix_lean` (`if r
  not in matrix[l]: ...`, line ~163) are **first-write-wins** dict inserts — a
  second write to an already-populated key is silently skipped, not applied.
  This was previously misread as last-write-wins in this file's earlier
  "Duplicate/multiallelic-position handling" section above; that was wrong.
- `P00_01_calc_covariance.py` never deduplicates `allpos` — every VCF row,
  including duplicates, gets its own row(s) in the flat covariance output,
  driven by array *index* (not unique position).
- The key insight: two duplicate-position VCF rows sit at the *same physical
  position*, so they get an *identical* genetic-distance window (window only
  depends on `pos2gpos[pos]`). Because legacy's output-row order follows
  outer-loop index order (= VCF encounter order), the first-VCF-encountered
  duplicate's row for any neighbor `r` is *always* written, and therefore
  always kept, before the second duplicate's row for that same `r`. So
  first-write-wins always resolves to "use the first-VCF-encountered
  duplicate's genotype data for every neighbor" — which is exactly what
  `ldetect2.shrinkage.calc_covariance` (`shrinkage.py:634-657`) already does
  by dropping every row after the first at a given `POS`, before computing
  anything. Same outcome via a different mechanism (upfront drop vs.
  downstream insert-once), not a divergence.

**Cross-partition overlap** (the same canonical `(lo, hi)` pair, redundantly
computed by two adjacent, overlapping partitions):

- Legacy `E07_metric.py`/`E08_local_search.py` and current `metric.py`/
  `local_search.py`/`_util/covariance_array.py::_owned_bounds` all use the
  same "next-partition-start" locus-ownership boundary
  (`end_locus = partitions[p+1][0]`).
- Legacy `E03_matrix_to_vector.py::calc_diag` and current `matrix_analysis.py`/
  `_util/vector_array.py` both use the same "midpoint" boundary
  (`floor((partitions[p][1] + partitions[p+1][0]) / 2)`).
  These are two intentionally different legacy-faithful conventions (one per
  algorithm), not an internal inconsistency in `ldetect2`.
- Which *value* wins for a redundant pair is decided by a shared primitive,
  `io/covariance.py::_insert_lean_values` (`if hi not in matrix[lo]:
  matrix[lo][hi] = shrink`) — plain partition-read-order first-write-wins,
  used identically by `MatrixAnalysis._calc_diag_lean_legacy`,
  `Metric._calc_metric_lean`, and `LocalSearch`'s dict/`use_decimal=True`
  path. This is a direct, already-tested port of legacy's
  `insert_into_matrix` precedence.
- The array-backed paths resolve the same redundancy differently, but also
  correctly: `metric.py`/`covariance_array.py::_owned_in_range_mask` use a
  *structural* guarantee — ownership is decided purely by a row's lower
  position (`i_pos`) against fixed, non-overlapping numeric windows, so a
  redundant pair's two copies (which always share the same `i_pos`) can never
  both be "owned" — no identity-based dedup is needed or possible to get
  wrong. `local_search.py` is the one array path that genuinely needs
  identity-based resolution (its windows load full overlapping partition row
  sets), and it has explicit, tested first-partition-wins logic
  (`canonical_local_search_rows`, `_first_seen_pair_mask`).
- One consequence of this reasoning: `_util/covariance_array.py::
  _deduplicate_metric_pairs` was dead code (defined, zero call sites) — its
  job is already structurally guaranteed by `_owned_in_range_mask`. Deleted.

### What was empirically verified (new tests, all passing on `ldetect-original-fix`)

- `tests/test_shrinkage.py::test_calc_covariance_duplicate_position_matches_first_encountered_variant`
  — value-level check (not just sortedness/uniqueness like the pre-existing
  `test_calc_covariance_canonicalizes_duplicate_physical_positions`): two
  duplicate-position variants engineered to give provably opposite-sign LD
  with a shared neighbor; asserts the deduped output matches an independently
  hand-computed "first-encountered-only" value, not the second variant's.
- `tests/test_covariance_io.py::test_matrix_to_vector_array_matches_legacy_with_divergent_overlap_pair`
  and `tests/test_metric.py::test_array_metric_matches_legacy_with_divergent_overlap_pair`
  — the pre-existing overlapping-partition fixtures for these two modules
  happened to give both overlapping partitions *identical* values for their
  shared boundary rows, so they never exercised actual overlap-resolution
  precedence. New tests use a fixture (`tests/_partition_fixtures.py::
  divergent_overlap_partitions`) where the redundant pair has genuinely
  different values (`0.7` vs `0.2`) in each partition, cross-checked against
  an independent from-scratch oracle (`first_write_wins_pair_value`, plain
  Python, no imports from `ldetect2`). Both pass.
- `tests/test_duplicate_overlap_integration.py` — the one combination flagged
  as untested anywhere: a duplicate VCF position that also sits inside a
  cross-partition overlap zone, exercised through real (not hand-typed)
  `calc_covariance()` runs on two overlapping VCF slices
  (`tests/_partition_fixtures.py::build_two_overlapping_partitions_with_duplicate_position`).
  Confirms (a) the two partitions' independently-computed redundant pairs are
  **bit-identical**, not just close, and (b) `matrix_analysis`, `metric`, and
  `local_search` all agree between their array/fast paths and legacy dict
  paths on this combined scenario. All four tests pass.

### Conclusion

Duplicate-position and cross-partition handling are not implicated. Diagnostic
effort should focus on (1) the `compare_ld_sets` phasing/re-calling diagnostic
and, if that's inconclusive, (3) a targeted old-VCF-release rerun, per the
"Recommended next steps" above — not further duplicate/overlap-handling code
changes.

## Hidden-assumption audit: data preprocessing & provenance

Date: 2026-07-03

While the alternate-source-rerun pipeline runs, audited implicit
preprocessing/provenance assumptions that were never surfaced as an explicit
hypothesis to test (as opposed to another empirical diagnostic). Method: read
the actual legacy code/config, compare against what `ldetect2`/the example
pipelines do today, and use the existing success/failure pattern (ASN exact
genome-wide; EUR chr1-7/13-22 and AFR chr1-10/12-21 exact; EUR chr8-12 and
AFR chr11/chr22 the only failures) as a control on each candidate.

### Partition-boundary `Ne` hardcoding — real divergence, doesn't explain the failures

`_reference/ldetect_original/ldetect/examples/P00_00_partition_chromosome.py`
line 53 hardcodes `exp(-4.0*11418.0*df/(2.0*nind))` — literally `11418.0`,
not a parameter — for the partition-boundary-extension step, for *every*
population including AFR and ASN. Current `ldetect2`
(`src/ldetect2/_cli/cmd_run.py::_run()`,
`partition_chromosome(..., ne=args.ne)`) instead passes the population-
specific `--ne` into that same step (17469 for AFR, 14269 for ASN) — a real,
previously-unverified divergence from legacy for every non-EUR population.

This doesn't explain the observed failures: ASN is subject to the identical
substitution (Ne=14269 instead of legacy's implicit 11418 for partition
boundaries) and still reproduces exactly, genome-wide, on every chromosome.
If this substitution meaningfully shifted boundary placement, ASN should show
some symptom too — it shows none. No further action; not a candidate root
cause, but worth documenting since nobody had actually checked `ldetect2`'s
own behavior here before (only speculated about it, in
`notes/local-search-divergence-asn22.md`).

### EUR sample list — definitively confirmed correct, byte-for-byte

`_reference/ldetect_original/ldetect/examples/example_data/eurinds.txt` is
the *actual* 379-individual EUR sample list distributed with the original
`ldetect` toy example — a real artifact from the original authors, not an
inference. Diffed directly against `examples/ldetect_original/resources/EUR_inds.txt`
(our pipeline's own subpopulation-filter output, derived from
`config.yaml`'s `EUR: [CEU, TSI, FIN, GBR, IBS]`):

```text
diff <(sort eurinds.txt) <(sort EUR_inds.txt)   # empty diff, exit 0
```

**Exact match, 379/379 individuals, zero differences.** This is a definitive,
not inferred, confirmation that our EUR subpopulation-code choice and
sample-selection logic exactly reproduce the original authors' actual EUR
sample list. Since EUR chr8-12 still diverge despite the EUR sample list
being *proven* identical, this closes off "wrong EUR sample composition" as
an explanation entirely — not just unlikely, but directly disproven.

No equivalent original AFR/ASN individual list is distributed anywhere in
`_reference/` (the toy example is EUR/chr2 only), so the same direct check
isn't available for AFR — AFR sample-list correctness remains only as
well-supported as the existing count-matching checks (246 individuals,
consistent across all four VCF releases), not a byte-for-byte proof like EUR.

### Multiallelic ALT-allele-trimming order of operations — no evidence of impact, deprioritized

Confirmed via repo-wide grep that no `bcftools view -a`/`--trim-alt-alleles`
or `bcftools norm` step exists anywhere in any example pipeline
(`Snakefile`, `Snakefile.provenance_diagnostics`,
`Snakefile.alternate_source_rerun`, MacDonald2022). The filter chain
(`bcftools view -S individuals -Ou vcf | bcftools view -i 'MAC[0]>=1' -m2 -M2`)
subsets samples before the biallelic filter without trimming unobserved ALT
alleles first, so a site triallelic in the *full* panel is excluded even if
only 2 alleles are observed within the analyzed population. This exact
command sequence is what reproduces the toy chr2 example's 672-SNP count, but
that window is only 100 kb (chr2:39967768-40067768) — too small to fully
validate genome-wide-scale behavior on its own.

Not pursued further for now: if this caused a meaningful, systematic SNP-
inclusion difference from whatever the original pipeline did, it would most
likely show up as some boundary or count divergence on *most* chromosomes
(density-dependent, not chr8-12/chr11/chr22-specific), which we don't
observe — ASN matches exactly on all 22 chromosomes and EUR/AFR match exactly
on all but 7 chromosomes total. Not disproven the way the two items above
are, but deprioritized given no positive evidence points at it and the
existing near-total genome-wide exact-match rate is hard to reconcile with a
real, impactful trimming discrepancy.

### v1/old2011 panel vintage — definitively closed, no discrepancy ever existed

Date: 2026-07-03 (external/web research, approved)

Directly browsed the 1000G FTP archive (network access confirmed working via
plain `curl ftp://...`):

- `technical/working/20120213_phase1_integrated_release_version1/` (v1) and
  `technical/working/20111111_old_phase1_release_files/` (old2011) each
  contain **only** VCF/TBI files plus a README — no panel file of their own.
  v1's own README explicitly directs users to the standard
  `phase1_integrated_calls.20101123.ALL.panel` (the same file `config.yaml`
  already uses for v3) — so reusing the v3 panel for v1/old2011 isn't a gap
  in our config, it's what the original 1000G documentation itself always
  specified. There was never a period-specific panel for these releases to
  have missed.
- `technical/working/20120316_phase1_integrated_release_version2/` (v2) does
  have its own distinct panel file,
  `phase1_integrated_calls_v2.20101123.ALL.panel` (already the one
  `provenance_diagnostics.yaml` uses for v2). Downloaded it and diffed
  directly against the final v3 panel
  (`release/20110521/phase1_integrated_calls.20101123.ALL.panel`):
  **zero differences, 1092/1092 samples identical.**

Combined with the EUR sample-list exact-match finding above, sample/panel
provenance across every tested release is now about as thoroughly closed as
it can be without the original authors' own logs.

### Original paper / Bitbucket archaeology — thin, but one new lead surfaced

Date: 2026-07-03 (external/web research, approved)

- Berisa & Pickrell (2016), Bioinformatics 32(2):283 (PMC free full text:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4731402/). Main text methods are
  thin: *"We applied this method to sequencing data from European, African
  and East Asian populations in the 1000 Genomes Phase 1 dataset"* and a mean
  block size of 10,000 SNPs (matches our `n_snps_bw_bpoints` default) — no
  release date/version, no population-code list, no Ne values, no filtering
  parameters given in the main text.
- The Supplementary Data PDF (`supp_btv546_document-sup.pdf`, linked from the
  article) could not be retrieved — the old Oxford Journals DOI-based
  supplementary link (`bioinformatics.oxfordjournals.org/lookup/suppl/...`)
  now redirects to the generic current article page, not the file; the
  bioRxiv preprint (`biorxiv.org/content/10.1101/020255v2.full`) returned
  HTTP 403. Not resolved — would need a subscriber login or a different
  mirror to actually read it.
- `nygcresearch/ldetect-data`'s Bitbucket README says only *"Thanks to: Yue Li
  (comments pointing to several ambiguities)"* with no detail on what those
  ambiguities were. Both `ldetect` and `ldetect-data` repos have their issue
  trackers disabled (API returns `Gone`, `has_issues: False`) and no wiki;
  the Wayback Machine has zero archived snapshots of anything under
  `bitbucket.org/nygcresearch/ldetect*` (checked via CDX API). The original
  content of Yue Li's ambiguity reports is not recoverable through any
  channel tried.
- **New lead: a previously-untested, undocumented Phase 1 snapshot exists.**
  Browsing `technical/working/` directly (not referenced anywhere in our
  config or in any note before now) turned up
  `20120117_new_phase1_intgrated_genotypes/`, dated Jan 19 2012 — chronologically
  between `old2011` (Nov 2011) and `v1` (Feb 13 2012). Filename template
  `ALL.chr{chrom}.merged_umich.20101123.snps_indels_svs.vcf.gz`, genome-wide,
  all chromosomes present. No README, no panel file, no documentation of any
  kind in that directory — it reads as an internal University-of-Michigan
  integration-center snapshot that happened to be left on the public FTP
  server, not one of the citable, documented public releases. Assessed as
  lower-priority than the four already-tested releases (v1/v2/v3/old2011 are
  the only ones with any public documentation/citation trail, making them
  far more likely candidates for what a careful published analysis would
  have actually used), but it is a genuine, previously-unknown-to-us
  candidate that has not been ruled out. Not added to
  `Snakefile.alternate_source_rerun` without checking with the user first,
  since that pipeline is already running.

### Recommended posture going forward

Sample/panel provenance (EUR composition, panel-file vintage across all four
tested releases) is now closed about as tightly as it can be from first
principles — two independent, direct, byte-level proofs (EUR sample list;
v2-vs-v3 panel diff) rather than inference. The paper/Bitbucket trail is
exhausted short of a subscriber login for the supplementary PDF. The
remaining live threads are: (a) the `merged_umich` snapshot, if worth the
compute cost given it's undocumented and the four documented releases already
showed no chromosome-specific signal; (b) reading the actual supplementary
PDF if a copy can be obtained some other way; (c) waiting on the
`alternate_source_rerun` results themselves, which are a more direct test
than any further provenance archaeology can offer.

## Supplementary PDF obtained — and it resolves AFR chr11 completely

Date: 2026-07-03

The user obtained `document-sup.pdf` (the paper's supplementary material,
previously unreachable via automated web fetch) and provided it directly.

### Confirms existing assumptions, no new leads there

- Genetic map source: *"Genome-wide recombination rates were obtained from
  Phase 2 HapMap Release 22 (Frazer et al., 2007) and interpolated to all
  positions in the 1000 Genomes dataset."* Matches what
  `joepickrell/1000-genomes-genetic-maps` already provides and what we've
  been using — confirmed correct, not a new lead.
- Mean segment size: *"For the published blocks, we used 10^4 SNPs for the
  mean segment size"* — matches `n_snps_bw_bpoints=10000`, our existing
  default — confirmed correct.
- Shrinkage/covariance formula (Section 1) matches `ldetect2.shrinkage`'s
  implementation exactly (population-scaled recombination rate in the
  exponential decay term, sample-size normalization).
- No new information on exact 1000G release version or population
  definitions beyond "1000 Genomes Project Phase 1 dataset" — doesn't add
  anything beyond the empirical byte-level proofs already established
  (EUR sample list, panel diffs).

### Fig. 2's per-chromosome block-count table — real signal, and a real bug found

Fig. 2 gives an exact block count per chromosome per population. Cross-checked
directly against our local `resources/ldetect_ref/{POP}_fourier_ls-all.bed`
files and our own `results/{POP}_LD_blocks.bed`, for all 22 chromosomes x 3
populations (66 rows):

**Every single row satisfies `supplement_count == raw_bed_row_count + 1`,
with zero exceptions** — a clean, uniform off-by-one, almost certainly a
"number of breakpoints vs. number of blocks" counting-convention difference
between the paper's figure-generation script and the distributed `.bed`
files, not a real discrepancy. Not investigated further; the `.bed` files
remain the correct, authoritative comparison target (as already used
everywhere in this investigation), and this cross-check just confirms that
target is internally consistent with the paper's own reported statistics
once the off-by-one convention is accounted for.

**The one place this cross-check initially looked inconsistent — AFR chr11 —
turned out to be a real, previously-unknown bug in the published reference
BED file itself, not in our pipeline or in the off-by-one theory.**

Investigation: `AFR_fourier_ls-all.bed` has 128 raw rows for chr11 (matches
our own `results/AFR_LD_blocks.bed`'s 128 rows for chr11 exactly), but
`ldetect2.io.bed.read_genome_bed()` (used by `compare_blocks.py`) parses only
126 valid blocks for chr11 — it silently drops 2 malformed rows that fail the
"is this a valid integer coordinate" check
(`src/ldetect2/io/bed.py::_iter_bed_records`, correct, defensive behavior).
Found the actual malformed rows:

```text
chr11 	 107843326 	 108823642
chr11 	 108823642 	 None       <- malformed: end coordinate is the literal string "None"
chr11 	 None 	 111048570      <- malformed: start coordinate is the literal string "None"
chr11 	 111048570 	 112221476
```

This is exactly the pair of coordinates (`108823642`, `111048570`) previously
flagged in this file as "AFR chr11 emits one additional internal boundary at
`109897792`, about 1.07 Mb from the nearest reference boundary
(`108823642`)" (see "AFR divergences" above) — and our own current output
confirms it precisely:

```text
results/AFR_LD_blocks.bed, chr11:
107843326  108823642
108823642  109897792   <- exactly the block whose end got corrupted to "None" in the reference
109897792  111048570   <- exactly the block whose start got corrupted to "None" in the reference
111048570  112221476
```

**Conclusion: AFR chr11 is not a real reproduction divergence at all.** The
published reference file has a data-corruption bug — two block-boundary
values were serialized as the literal string `"None"` instead of the real
breakpoint (`109897792`) — isolated to this one exact location in the AFR
file (confirmed via `grep -c "None"`: zero occurrences anywhere in the EUR or
ASN reference files, and no other occurrence anywhere else in the AFR file).
Our pipeline's output is very likely *correct* here; the "extra boundary"
was never extra, it's the reference artifact that's missing it. This fully
resolves what was previously the second-most-tracked AFR residual alongside
chr22. AFR chr22 remains unexplained — no similar corruption found there,
and its divergence pattern (1 extra block, boundaries displaced beyond
simple tolerance in places) doesn't match this specific failure mode.

### Updated residual list

- ~~AFR chr11~~ — resolved: reference-file corruption, not a real divergence.
- AFR chr22 — still open, no explanation found.
- EUR chr8-12 — still open, no explanation found.

### Broader structural audit of the reference BED files — nothing else found

Date: 2026-07-03

Audited all three reference files (and our own genome-wide outputs, for
comparison) for gaps, overlaps, zero/negative-length blocks, out-of-order
boundaries, and duplicate rows (parsing with the same permissive-but-safe
logic as `ldetect2.io.bed`, skipping the harmless header row).

```text
REF AFR: 1 issue  — chr11: GAP of 2,224,928 bp (108823642 -> 111048570)
REF EUR: 0 issues
REF ASN: 0 issues
OURS AFR/EUR/ASN: 0 issues each
```

The one flagged issue is the same chr11 corruption already found — filtering
out the two malformed `"None"` rows leaves a large gap exactly where they
were, which is confirming triangulation from a different angle (contiguity)
rather than a new finding. No overlaps, no degenerate blocks, no ordering
problems, no duplicates anywhere else in any of the three reference files;
our own outputs are perfectly contiguous everywhere as expected. This rules
out "silent structural data-quality issues elsewhere in the reference files"
as a contributor to AFR chr22 or EUR chr8-12 — there is nothing else to find
in this direction.

## Alternate-source full-pipeline rerun results — decisive, VCF release is not the answer

Date: 2026-07-03

`Snakefile.alternate_source_rerun` completed (21 full `ldetect2 run` executions:
v1/v2/old2011 x EUR chr8/9/10/11/12 x AFR chr11/22). This is the direct
empirical test the earlier `compare_ld_sets` diagnostic could only infer from
sampled positions/LD pairs. Extracted the 21 real per-chromosome result rows
from `results/alternate_source_rerun/comparison_summary.tsv` (the file also
contains ~440 padding rows from `compare_blocks.py` iterating the full
22-chromosome reference against each single-chromosome run — filtered to
`our_n != 0`).

Recall at 100kb tolerance, vs. the v3 baseline already on record:

```text
pop  chrom  v3(baseline)   v1      v2      old2011
EUR  chr8   0.4105         0.4330  0.4526  0.4330
EUR  chr9   0.3467         0.3158  0.3467  0.3158
EUR  chr10  0.2907         0.3068  0.3023  0.3068
EUR  chr11  0.4588         0.4138  0.4824  0.4138
EUR  chr12  0.3614         0.3810  0.3902  0.3810
AFR  chr22  0.8056         0.7222  0.7778  0.7222
AFR  chr11  0.9922*        0.7984  0.9070  0.7984
```
\* AFR chr11's v3 recall is measured against the corrupted reference file
(see above) and isn't a clean number; already resolved independently as a
reference-file bug.

Findings:

- **v1 and old2011 produce byte-identical results on every chromosome** —
  consistent with the earlier `compare_ld_sets` finding that these two
  "releases" are functionally the same processed dataset for our filtered,
  individual-subsetted universe.
- **v2 gives a modest recall improvement on 4 of 5 EUR chromosomes**
  (chr8, chr11, chr12 up; chr9 flat) but nowhere near the ~1.0 recall seen on
  exact-match control chromosomes (chr7, chr13). Not remotely sufficient to
  call this "the fix."
- **For AFR chr22, every alternate release is worse than v3** — v3 is the
  best-performing source of the four for that chromosome.

**Conclusion: swapping the 1000G VCF release does not fix EUR chr8-12 or AFR
chr22.** This directly confirms (rather than just infers, as the phasing/LD
diagnostic did) that VCF-release provenance is not the explanation. Combined
with everything else ruled out this investigation (SNP-only filtering,
genetic map family, Ne assignment, duplicate-position/cross-partition
handling, sample/panel provenance, reference-file structural integrity), the
input-and-implementation search space is now essentially exhausted for these
two remaining chromosomes. Candidates not yet tried: the undocumented
`merged_umich` snapshot (low expected value given the four documented,
better-candidate releases already failed to move the needle), or accepting
this as an unresolved provenance mismatch with the original authors' exact
(unrecoverable) process.

## Chromosome-specific reference BED audit — reconfirms consistency, AFR chr11 corruption traced to the earliest available source

Date: 2026-07-03

Downloaded all 66 chromosome-specific reference files
(`https://bitbucket.org/nygcresearch/ldetect-data/raw/master/{POP}/fourier_ls-chr{N}.bed`,
3 populations x 22 chromosomes) and compared each against the corresponding
chromosome's slice of the genome-wide `{POP}_fourier_ls-all.bed`, using
raw string-preserving parsing (not normalized/whitespace-stripped, so
malformed values like `"None"` are caught rather than silently coerced).

**All 66 pairs are byte-identical.** This reconfirms and extends the earlier
`reference_bed_consistency.tsv` check (which found the same "all 66 match")
with a stricter comparison method, and it still holds perfectly — the
all/chromosome-specific published files are fully consistent everywhere.

**The AFR chr11 corruption exists identically in the chromosome-specific file
too**, not just the genome-wide one:

```text
AFR/chr11 (chromosome-specific file) line 103: (108823642, None)
AFR/chr11 (chromosome-specific file) line 104: (None, 111048570)
```

This was checked hoping the more "primitive" per-chromosome file might carry
the real, uncorrupted boundary value (if the corruption were introduced
during Bitbucket's genome-wide concatenation) — it doesn't. The corruption is
present in the earliest, most direct data source the original authors
published, so it must originate in whatever script generated/exported the
AFR breakpoints in the first place. There is no more-authoritative published
source left to check; the true intended value at that boundary is not
recoverable from anything Bitbucket has published. Does not change the
conclusion (AFR chr11 remains a reference-data bug, not a real pipeline
divergence) — just confirms the trail ends here.
