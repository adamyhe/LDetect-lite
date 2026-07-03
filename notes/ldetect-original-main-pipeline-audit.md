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
