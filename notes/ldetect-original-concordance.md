# ldetect_original fp64/high-precision concordance notes

Date: 2026-05-06

## Current observation

Downloaded fp64 BED files and comparison TSVs under:

```text
examples/ldetect_original/results/
```

The current genome-wide comparison state is:

- ASN reproduces completely genome-wide.
- AFR is nearly exact, with residual issues on chr11, chr16, and chr22.
- EUR reproduces most chromosomes exactly, but chr8-12 have poor boundary
  concordance despite matching block counts.

For EUR chr10, a rerun with high precision produced the same poor concordance
against `EUR_fourier_ls`:

```text
chr10  our_n=85  ref_n=85  recall=0.2907  precision=0.2907
       median_offset=264.2 kb  p90_offset=723.8 kb  bp_jaccard=1.0
```

The Decimal and float64 breakpoint JSONs are identical.

## Interpretation update

This rules out ordinary float64-vs-Decimal numerical drift for the observed EUR
chr10 failure.

It also makes the rebuilt array local-search path unlikely to be the primary
cause for EUR chr10, because `--high-precision` routes local search through the
Decimal/dictionary path yet produces identical breakpoints.

The remaining likely divergence point is upstream of local search:

- covariance values,
- matrix-to-vector output,
- Hanning/minima stage,
- filter-width targeting,
- partition/covariance inputs,
- or a population/chromosome-specific input/config mismatch.

`bp_jaccard=1.0` is not evidence of boundary correctness here. Since both BEDs
tile the same chromosome span contiguously, base-pair interval Jaccard can be
1.0 even when many internal boundaries differ.

## Per-population pattern from downloaded comparison TSVs

Problem chromosomes by population:

```text
EUR: chr8, chr9, chr10, chr11, chr12
AFR: chr11, chr16, chr22
ASN: none
```

EUR failure pattern:

```text
chr8   recall=0.4105  median_offset=161.0 kb  p90=682.8 kb
chr9   recall=0.3467  median_offset=237.1 kb  p90=671.2 kb
chr10  recall=0.2907  median_offset=264.2 kb  p90=723.8 kb
chr11  recall=0.4588  median_offset=117.4 kb  p90=647.7 kb
chr12  recall=0.3614  median_offset=194.4 kb  p90=706.8 kb
```

All five failing EUR chromosomes have the correct block count. That means
targeting reaches the same number of blocks, but the selected boundary
locations differ.

## Bitbucket reference BED consistency check

Date: 2026-06-20

After the `nygcresearch/ldetect-data` Bitbucket repository became available
again, I checked whether the genome-wide BED files differ from the
chromosome-specific BED files. This had been a plausible explanation for the
EUR chr8-12 pattern, because the repository stores both:

```text
{POP}/fourier_ls-all.bed
{POP}/fourier_ls-chr{N}.bed
```

The check downloaded `fourier_ls-all.bed` and `fourier_ls-chr1.bed` through
`fourier_ls-chr22.bed` for `AFR`, `ASN`, and `EUR` from:

```text
https://bitbucket.org/nygcresearch/ldetect-data/raw/master/{POP}/...
```

Each per-chromosome file was compared with the matching chromosome slice of
the corresponding `fourier_ls-all.bed`, after normalizing whitespace and
skipping the header.

Result: all 66 comparisons matched exactly.

Important EUR counts also matched:

```text
EUR chr8   all_n=94   chr_n=94
EUR chr9   all_n=74   chr_n=74
EUR chr10  all_n=85   chr_n=85
EUR chr11  all_n=84   chr_n=84
EUR chr12  all_n=82   chr_n=82
EUR chr13  all_n=62   chr_n=62
```

This rules out an internally inconsistent `fourier_ls-all.bed` as the cause
of poor EUR chr8-12 concordance. The genome-wide and chromosome-specific
published reference BEDs are equivalent for all three populations.

## Recommended next diagnostic

For one failing EUR chromosome, compare the raw `fourier` breakpoints before
local search between our JSON and the best available reference/previous
known-good output if available.

If no reference vector or minima is available, regenerate chr10 while saving:

- vector checksum/statistics,
- `found_width`,
- raw `fourier` loci,
- `fourier_ls` loci,
- covariance partition summary.

Then compare those against a chromosome that reproduces exactly, such as EUR
chr7 or chr13, to identify whether the EUR chr8-12 issue begins at vector
construction, filter-width/minima selection, or covariance generation.

## Updated next checks

The next most informative checks should focus on inputs and early pipeline
state, not on reference BED packaging:

1. Compare public VCF metadata and variant counts for EUR chr8-12 against
   matched chromosomes that reproduce, such as chr7 and chr13. Record raw
   VCF record counts, post-EUR-subset biallelic `MAC[0] >= 1` counts, and
   vector row counts. This tests whether the failing chromosomes have a
   distinctive SNP-density or filtering profile.
2. Test recombination-map provenance on one failing chromosome, ideally EUR
   chr10. The current pipeline uses Pickrell `interpolated_from_hapmap`
   maps. Run the same diagnostic with Pickrell `interpolated_OMNI` for chr10
   if available, because OMNI is CEU/EUR-specific and was published in the
   same map repository. A large improvement would implicate map provenance.
3. Compare partition boundaries for failing and matched chromosomes. If
   chr8-12 have partition boundaries or counts that differ unexpectedly from
   the original workflow, downstream covariance/vector differences can occur
   even when final block counts match.
4. Preserve one full diagnostic bundle for a failing chromosome and one
   matched chromosome: filtered VCF index stats, map summary, partition file,
   covariance summary, vector hash/statistics, `found_width`, raw `fourier`
   loci, and final `fourier_ls` loci. This is the smallest bundle needed to
   isolate whether divergence starts in covariance, vector construction, or
   minima selection.

These checks are now wired into
`examples/ldetect_original/Snakefile.diagnostics`. The key outputs are
`results/diagnostics/{POP}/input_summary.tsv`,
`results/diagnostics/{POP}/diagnostic_summary.tsv`, and
`results/diagnostics/reference_bed_consistency.tsv`.

## Diagnostic summary review

Date: 2026-06-21

Downloaded diagnostic summaries for EUR chr10 as the failing case and EUR
chr13 as the matched control were reviewed under:

```text
examples/ldetect_original/results/diagnostics/EUR/
```

Key observations:

- chr10 still fails against the published EUR reference despite matching the
  expected block count:

```text
chr10: our_n=85, ref_n=85, recall=0.2907, precision=0.2907,
       median_offset=264.2 kb, p90_offset=723.8 kb
```

- chr13 remains an exact control:

```text
chr13: our_n=62, ref_n=62, recall=1.0, precision=1.0
```

- The CEU OMNI-map rerun for chr10 does not rescue the mismatch:

```text
chr10 OMNI: our_n=85, ref_n=85, recall=0.314,
            median_offset=266.8 kb, p90_offset=728.0 kb
```

  The OMNI vector differs substantially from the HapMap-interpolated vector
  and increases covariance rows, but final reference concordance remains poor.
  This makes a simple "wrong Pickrell map family" explanation unlikely.

- `reference_bed_consistency.tsv` contains no mismatches. The Bitbucket
  `fourier_ls-all.bed` files and chromosome-specific `fourier_ls-chrN.bed`
  files are internally consistent for the checked populations/chromosomes.

- Input scale differs as expected between chr10 and chr13, but the current
  summaries do not show an obvious pathological map or partition issue:

```text
chr10 filtered_vcf_records=845717, vector_rows=845019,
      partitions=376, cov_rows=2289855822, found_width=4305
chr13 filtered_vcf_records=611065, vector_rows=610559,
      partitions=274, cov_rows=1362130544, found_width=4861
```

One caveat: `raw_vcf_records` is reported as `0` for both chromosomes. This is
probably a diagnostic-script limitation from using `bcftools index -n` on these
raw `.tbi` files, not evidence that the raw VCFs are empty. The filtered VCF
counts are informative.

Updated interpretation: the leading hypotheses are now (1) original published
EUR chr8-12 were generated from a different 1000G/public-input snapshot or
filtering state than the current release files, or (2) there is still an
unidentified upstream implementation/provenance detail before vector/minima
selection. The simple reference-BED packaging and OMNI-vs-HapMap-map-family
explanations are both disfavored.

## EUR chr7-13 diagnostic batch

Date: 2026-06-21

The diagnostic workflow was rerun for EUR chr7-13 with covariance workers set
to 4 and local search capped at 1 worker. Only high-level outputs and chr7-13
diagnostic files were inspected, because other chromosome directories may
contain stale files from earlier runs.

Concordance pattern:

```text
chr7:  our_n=99, ref_n=99, recall=1.0000, median_offset=0.0 kb
chr8:  our_n=94, ref_n=94, recall=0.4105, median_offset=161.0 kb
chr9:  our_n=74, ref_n=74, recall=0.3467, median_offset=237.1 kb
chr10: our_n=85, ref_n=85, recall=0.2907, median_offset=264.2 kb
chr11: our_n=84, ref_n=84, recall=0.4588, median_offset=117.4 kb
chr12: our_n=82, ref_n=82, recall=0.3614, median_offset=194.4 kb
chr13: our_n=62, ref_n=62, recall=1.0000, median_offset=0.0 kb
```

All chr7-13 runs produce the expected block count, and all reported
`bp_jaccard=1.0`, so the bad chromosomes appear to have the correct genome
coverage and number of intervals but shifted internal boundaries. The failures
are localized to a contiguous interval, chr8-12, bracketed by exact controls
chr7 and chr13.

Input/map summaries do not show a simple chr8-12-only pathology:

- filtered EUR sample count is consistently 379.
- genetic maps have zero inversions for all chr7-13.
- chr8-12 are not the only chromosomes with large map gaps; chr13 has the
  largest max cM gap in this batch but is exact.
- partition counts and vector row counts scale with chromosome size/SNP count
  as expected.

Runtime/memory observations:

- local search dominates wall time even when capped at one worker.
- `--local-search-workers 4` previously caused chr13 to be OOM-killed
  (`exit 137`), so diagnostic local search is now capped at one worker.
- chr11 is unusually heavy in this batch: about 108 GB maximum RSS and 6h20m
  wall time despite skipping already-completed covariance partition generation.
  Its compact covariance footprint is also much larger than neighboring
  chromosomes (`cov_rows=8,816,825,678`, `cov_bytes=79,739,645,889`).

Interpretation after chr7-13 batch:

- The contiguous chr8-12 failure pattern is real and reproducible.
- The agreement on chr7 and chr13 argues against a global bug in current
  filtering, vector construction, minima counting, local search, Ne, or
  reference BED comparison.
- The failures are still most consistent with either a chromosome-specific
  provenance/input discrepancy in the published EUR reference blocks or a very
  specific upstream implementation/detail that affects chr8-12 but not adjacent
  chromosomes.
- Since all failing chromosomes have the correct number of blocks and
  full-base-pair coverage, future diagnostics should compare breakpoint
  positions, vector summaries, and covariance/vector provenance rather than
  block counts.

## Legacy ldetect downstream diagnostic

Date: 2026-06-21

Legacy diagnostic comparison files were reviewed under:

```text
examples/ldetect_original/results/legacy_diagnostics/EUR/{8,9,10,11,12,13}/compare/
```

These diagnostics run copied legacy downstream scripts on covariance data
derived from the same ldetect2-generated full covariance partitions. The goal
is to test whether the EUR chr8-12 mismatch is caused by ldetect2 downstream
steps after covariance generation.

Summary:

```text
chrom  ldetect2-vs-ref  legacy-vs-ref  ldetect2-vs-legacy
chr8   0.4105           0.4211         0.8737
chr9   0.3467           0.3467         0.8800
chr10  0.2907           0.3372         0.8256
chr11  0.4588           0.4471         0.9647
chr12  0.3614           0.3855         0.8795
chr13  1.0000           0.9365         0.9365
```

The legacy run is not bit-identical to ldetect2: vector row counts match, but
vector SHA256 hashes differ on every checked chromosome, and final loci are not
all identical. However, legacy and ldetect2 are much closer to each other than
either is to the published reference on chr8-12.

Interpretation:

- Running legacy downstream code on the same covariance does not reproduce the
  published EUR chr8-12 reference blocks.
- This argues against the chr8-12 mismatch being primarily an ldetect2-only
  bug in matrix-to-vector, minima selection, or local search.
- The chr13 result is a useful caveat: ldetect2 exactly matches the published
  reference, while the copied legacy downstream diagnostic is only 0.9365
  concordant. Therefore this diagnostic should not be treated as a perfect
  reference implementation oracle. It is still informative because it fails in
  the same direction as ldetect2 on chr8-12 when supplied the same covariance.
- The best remaining explanation remains upstream provenance/input/covariance
  differences between the current public inputs and whatever generated the
  published EUR chr8-12 reference BEDs.
