# Reproducing the Original ldetect LD Blocks

This example pipeline attempts to reproduce the EUR, AFR, and ASN LD blocks
published with Berisa and Pickrell (2016):

- Original implementation: <https://bitbucket.org/nygcresearch/ldetect>
- Published block files: <https://bitbucket.org/nygcresearch/ldetect-data>

The workflow starts from public 1000 Genomes Phase 1 VCFs and runs the modern
`ldetect2` implementation end to end.

## Quick Start

```bash
cd examples/ldetect_original

# Dry-run all configured chromosomes and populations.
snakemake -n

# Run a small smoke test.
snakemake --cores 8 --config chromosomes='[22]'

# Run the configured analysis.
snakemake --cores 32
```

The main outputs are:

- `results/{POP}/{chrom}/{chrom}-ld-blocks.bed`
- `results/{POP}_LD_blocks.bed`
- `results/compare/{POP}_block_comparison.tsv`

## Important Reproduction Detail: SNP Filtering

The published paper and original ldetect command examples use
`n_snps_bw_bpoints = 10000`. Early attempts with public Phase 1 VCFs produced
far too many SNPs and too many blocks. Tuning this parameter to larger values
can make the block count look closer for one population, but it does not
reproduce boundary locations and does not transfer across EUR, AFR, and ASN.

The missing detail is the SNP universe. The reference covariance file in the
original ldetect toy example contains 672 unique SNPs in:

```text
chr2:39967768-40067768
```

That count is reproduced from the public Phase 1 VCF when the VCF is filtered
after sample subsetting:

```bash
bcftools view \
  -S eurinds.txt \
  -Ou data/raw/ALL.chr2.phase1_release_v3.20101123.snps_indels_svs.genotypes.vcf.gz \
  2:39967768-40067768 |
bcftools view -H -i 'MAC[0]>=1' -m2 -M2 |
wc -l
```

Expected result:

```text
672
```

This means the original effective filter is:

- subset to the population's individuals first,
- keep variants polymorphic in that population (`MAC[0] >= 1`),
- keep biallelic records (`-m2 -M2`),
- do not add `FILTER=PASS` or SNP-type-only restrictions for exact
  reproduction.

The Snakefile implements this with:

```bash
bcftools view -S {population_individuals} -Ou {raw_vcf} |
bcftools view -i 'MAC[0]>=1' -m2 -M2 -Oz -o {filtered_vcf}
tabix -f -p vcf {filtered_vcf}
```

## Individual Lists

The pipeline builds population sample lists from the 1000 Genomes panel and
the VCF sample header:

```text
resources/EUR_inds.txt
resources/AFR_inds.txt
resources/ASN_inds.txt
```

For the original EUR chr2 toy window, exact reproduction requires 379 EUR
individuals. If an old cached `resources/EUR_inds.txt` has 378 individuals,
remove it and let Snakemake regenerate it, or compare it to the reference
`eurinds.txt` from the toy example. In particular, missing `HG00096` changes
the chr2 toy-window filtered count from 672 to roughly 641.

Useful checks:

```bash
wc -l resources/EUR_inds.txt
grep -x HG00096 resources/EUR_inds.txt
bcftools query -l data/raw/ALL.chr2.phase1_release_v3.20101123.snps_indels_svs.genotypes.vcf.gz |
  grep -x HG00096
```

## Boundary Comparison

After running the pipeline, compare against the published BED files:

```bash
python scripts/compare_blocks.py \
  --ours results/EUR_LD_blocks.bed \
  --ref resources/ldetect_ref/EUR_fourier_ls-all.bed \
  --output results/compare/EUR_block_comparison.tsv \
  --tolerance 100000
```

The comparison reports per-chromosome block counts, boundary recall/precision,
boundary Jaccard, nearest-boundary offsets, and base-pair interval Jaccard.

Published reference block counts are:

| Population | Genome-wide blocks | chr2 blocks |
| ---------- | ------------------ | ----------- |
| EUR        | 1703               | 144         |
| AFR        | 2583               | 220         |
| ASN        | 1445               | 122         |

Matching block counts alone is not enough. Boundary offsets and recall should
be used to decide whether the analysis is reproducing the original result.

## Diagnostic Workflow

`Snakefile.diagnostics` runs a focused case/control investigation for boundary
divergence. The default `diagnostics.yaml` compares EUR chr10 against EUR
chr13, records upstream input summaries, checks published reference BED
consistency, and optionally reruns chr10 with the Pickrell CEU OMNI map.

Dry-run the default diagnostic:

```bash
snakemake -s Snakefile.diagnostics -n
```

Run the default diagnostic:

```bash
snakemake -s Snakefile.diagnostics --cores 4
```

Run the five problematic EUR chromosomes plus chr13 as a control:

```bash
snakemake -s Snakefile.diagnostics --cores 4 \
  --config chromosomes='[8,9,10,11,12,13]' \
           case_chromosome=10 \
           control_chromosome=13 \
           population=EUR
```

By default, each diagnostic `ldetect2 run` job claims
`ldetect2_job_threads: 4` Snakemake cores and passes up to
`cov_workers: 4` to `ldetect2 --workers`. This gives within-chromosome
parallel covariance generation. With `--cores 4`, chromosomes run one at a
time. To run multiple chromosomes concurrently instead, use one internal
worker per chromosome and give Snakemake more cores:

```bash
snakemake -s Snakefile.diagnostics --cores 4 \
  --config chromosomes='[8,9,10,11,12,13]' \
           population=EUR \
           cov_workers=1 \
           local_search_workers=1 \
           ldetect2_job_threads=1
```

Useful outputs:

- `results/diagnostics/{POP}/input_summary.tsv`: raw/filtered VCF counts,
  sample counts, map stats, and partition stats for each diagnostic
  chromosome.
- `results/diagnostics/{POP}/diagnostic_summary.tsv`: vector, covariance,
  breakpoint, and final BED comparison summaries.
- `results/diagnostics/{POP}/case_vs_control.tsv`: compact side-by-side
  comparison of the configured case and control chromosomes.
- `results/diagnostics/reference_bed_consistency.tsv`: comparison of
  `fourier_ls-all.bed` slices against chromosome-specific `fourier_ls-chrN.bed`
  files for the configured reference populations.
- `results/diagnostics/{POP}/{chrom}/omni_summary.tsv` and
  `omni_block_comparison.tsv`: optional OMNI-map rerun summaries.

## Effective Population Size

The covariance shrinkage step uses an effective population size (`Ne`). The
original ldetect README example uses `11418` and notes that it is appropriate
for European populations. The SHAPEIT documentation for HapMap II maps lists
population-specific values that are the likely source of these defaults:

| Population | HapMap II source | Default `Ne` |
| ---------- | ---------------- | ------------ |
| EUR        | CEU              | 11418        |
| AFR        | YRI              | 17469        |
| ASN        | CHB+JPT          | 14269        |

These are configured in `config.yaml` and passed to `ldetect2 run --ne`.

## Pipeline Steps

1. Download public Phase 1 VCFs, genetic maps, panel metadata, and published
   BED references.
2. Build population-specific individual lists.
3. Create population-specific VCFs filtered to biallelic records with
   population-specific `MAC[0] >= 1`.
4. Run `ldetect2 run` per chromosome and population using
   `n_snps_bw_bpoints = 10000`.
5. Combine chromosome BEDs into genome-wide BEDs.
6. Compare the generated BEDs against the published ldetect blocks.

By default, this pipeline sets `covariance_cache: compact` and passes
`ldetect2 run --covariance-cache compact`. The resulting compact `.npz`
partitions contain `i_pos`, `j_pos`, and `shrink_ld`, which are the fields used
by the array-backed matrix-to-vector path. Set `covariance_cache: full` only if
you need full covariance metadata for debugging or heatmap generation.

## Notes for Developers

- `ldetect2.shrinkage` intentionally applies the covariance cutoff before
  adding the diagonal shrinkage term. This matches the original ldetect script
  and drops population-monomorphic variants from the covariance output.
- This ordering fixed an important compatibility bug. An earlier ldetect2
  version added the diagonal shrinkage term before checking the covariance
  cutoff. For diagonal entries, that made population-monomorphic variants look
  nonzero and kept them in the covariance matrix. The original ldetect script
  computes `Ds2`, applies `abs(Ds2) < CUTOFF`, and only then adds the diagonal
  shrinkage term for retained variants. Matching that order is necessary for
  the vector SNP count to reflect the population-polymorphic SNP set.
- The toy `examples/ldetect_example` pipeline is still the strictest
  implementation test because it starts from the original reference covariance
  matrix and should reproduce the reference BED exactly.
- If changing filtering or sample-list behavior, first rerun the 100 kb chr2
  count check above before launching a full-genome run.
