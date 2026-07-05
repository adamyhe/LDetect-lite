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
uv run snakemake -n

# Run a small smoke test.
uv run snakemake --cores 8 --config chromosomes='[22]'

# Run the configured analysis.
uv run snakemake --cores 32
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
uv run python scripts/compare_blocks.py \
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
uv run snakemake -s Snakefile.diagnostics -n
```

Run the default diagnostic:

```bash
uv run snakemake -s Snakefile.diagnostics --cores 4
```

Run the five problematic EUR chromosomes plus chr13 as a control:

```bash
uv run snakemake -s Snakefile.diagnostics --cores 4 \
  --config chromosomes='[8,9,10,11,12,13]' \
           case_chromosome=10 \
           control_chromosome=13 \
           population=EUR
```

By default, each diagnostic `ldetect2 run` job claims
`ldetect2_job_threads: 4` Snakemake cores and passes up to
`cov_workers: 4` to `ldetect2 --workers`. Local search is defensively capped
at one worker in this diagnostic workflow to avoid multiplying memory use.
With `--cores 4`, chromosomes run one at a time. To run multiple chromosomes
concurrently instead, use one internal covariance worker per chromosome and
give Snakemake more cores:

```bash
uv run snakemake -s Snakefile.diagnostics --cores 4 \
  --config chromosomes='[8,9,10,11,12,13]' \
           population=EUR \
           cov_workers=1 \
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

## Provenance Diagnostic Workflow

`Snakefile.provenance_diagnostics` runs the targeted upstream checks suggested
by the remaining EUR and AFR divergences. It compares the current v3/all-record
population-polymorphic VCF universe against SNP-only and archived Phase 1
release candidates for only the divergent chromosomes:

- EUR chr8-12 plus chr13 as an exact-match control.
- AFR chr11 and chr22 plus chr13 as an exact-match control.

Dry-run the provenance diagnostic:

```bash
uv run snakemake -s Snakefile.provenance_diagnostics -n
```

Run it:

```bash
uv run snakemake -s Snakefile.provenance_diagnostics --cores 1
```

The default comparison matrix is configured in
`provenance_diagnostics.yaml`. It uses `v3/all` as the baseline and compares:

- `v3/snps`, testing whether published blocks effectively used SNP-only input.
- `v2/all` and `v2/snps`, testing the March 2012 Phase 1 v2 archive.
- `v1/all`, testing the February 2012 Phase 1 v1 archive.
- `old2011/all`, testing the November 2011 old Phase 1 archive.

Outputs:

- `results/provenance_diagnostics/position_comparison_summary.tsv`: one row per
  population/chromosome/candidate comparison with record counts, unique
  position counts, shared/only counts, duplicate-position counts, and position
  Jaccard.
- `results/provenance_diagnostics/{POP}/chr{N}/position_sets/*.tsv`: the same
  metrics split by comparison.
- `results/provenance_diagnostics/filtered_vcf/{source}/{filter}/{POP}/`: the
  staged population-specific filtered VCFs used for comparison.

This workflow is intentionally input-focused. If one candidate produces a
position-set change concentrated in the divergent chromosomes, rerun that
candidate through `Snakefile.diagnostics` or the main workflow before launching
larger release-matrix runs.

## Legacy Diagnostic Workflow

`Snakefile.legacy_diagnostics` compares the modern pipeline against a minimal
vendored copy of the original ldetect downstream scripts. It stages compact
ldetect2 HDF5 covariance partitions into the original text format, runs legacy
matrix-to-vector, minima, local-search, and BED extraction steps, and compares
both outputs against the published reference BED.

Dry-run the default legacy diagnostic:

```bash
uv run snakemake -s Snakefile.legacy_diagnostics -n
```

Run a memory-bounded diagnostic for one chromosome:

```bash
uv run snakemake -s Snakefile.legacy_diagnostics --cores 1 \
  --config chromosomes='[11]'
```

`legacy_diagnostics.yaml` keeps `local_search_workers` conservative. Raising
`cov_workers` or `local_search_workers` can spawn multiple large
covariance/local-search processes inside a single chromosome job; raising
`--cores` can also allow Snakemake to run multiple chromosome jobs at once
unless `ldetect2_job_threads` claims the cores for each heavy job. Staging
HDF5 covariance partitions into legacy `.gz` files is independent per
partition and can be parallelized with `staging_workers`.

The vendored legacy checkout intentionally excludes the original toy
`example_data` tree and old exploratory scripts because they are not used by
the diagnostic Snakemake workflow. To inspect or manually run those legacy toy
examples in a local checkout, download the data on demand:

```bash
uv run python scripts/download_legacy_example_data.py
```

The downloaded files are written under
`scripts/legacy_ldetect/ldetect/examples/example_data/` and are ignored by Git.

## Compression Diagnostic Workflow

`Snakefile.compression_diagnostics` measures the performance, exactness, and
**covariance cache size** of the `--covariance-compression` codec choice and
the `--shrink-ld-precision` value against this pipeline's real 1000G data.
For each configured chromosome x population, it runs `ldetect2 run` three
times on identical filtered input:

- `baseline`: `--covariance-compression lzf` (the prior default);
- `zstd`: `--covariance-compression zstd` (the new default — lossless, no
  accuracy risk);
- `zstd_f32`: `--covariance-compression zstd --shrink-ld-precision float32`
  (an additional, larger storage lever — `shrink_ld` values are rounded to
  float32 precision before writing, still stored as float64 on disk. This is
  **lossy** and not yet validated against real breakpoints — that is exactly
  what this diagnostic is for).

Each of `zstd`/`zstd_f32` is then compared against `baseline`, producing one
comparison row per (population, chromosome, candidate) triple:

- **vector/breakpoints/BED**: row count, sha256 digest, exact loci match, and
  BED boundary recall/precision/Jaccard at `tolerance` bp (default `0`). For
  `zstd` (lossless) these are expected to match exactly — any divergence
  there indicates a real bug, not floating-point noise. For `zstd_f32`
  (lossy) divergence is possible; the question this diagnostic answers is
  whether it ever shifts a real breakpoint;
- **covariance directory size**: total bytes of every `.h5` partition under
  each mode's covariance directory, plus the size ratio and percent
  reduction — this is the actual point of this diagnostic;
- **performance**: wall-clock seconds and peak RSS from each mode's
  Snakemake `benchmark:` record, plus the speedup/reduction ratios.

This defaults to the **full 22-chromosome, 3-population dataset**, since a
comprehensive storage-size comparison is only meaningful at full-genome
scale — run it on a remote server/cluster given the compute cost.

Dry-run the default (full-genome) diagnostic:

```bash
uv run snakemake -s Snakefile.compression_diagnostics -n
```

Run it for real:

```bash
uv run snakemake -s Snakefile.compression_diagnostics --cores 8
```

Smoke-test a smaller subset instead of the full genome:

```bash
uv run snakemake -s Snakefile.compression_diagnostics --cores 4 \
  --config chromosomes_by_population='{EUR: [11, 22]}'
```

Outputs:

- `results/compression_diagnostics/{population}/{chrom}/{baseline,zstd,zstd_f32}/` —
  each mode's full `ldetect2 run` output directory, including its covariance
  partitions under `{mode}/{chrom}/`.
- `results/compression_diagnostics/{population}/{chrom}/logs/{mode}.benchmark.tsv` —
  Snakemake's wall-clock/peak-memory record for that mode.
- `results/compression_diagnostics/{population}/{chrom}/compare/compression_vs_baseline.{candidate}.tsv` —
  one row per candidate (`zstd`, `zstd_f32`) combining the exactness, size,
  and performance comparison against baseline for that chromosome x
  population.
- `results/compression_diagnostics/summary.tsv` — all comparison rows
  concatenated (both candidates, all chromosomes, all populations).

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
`ldetect2 run --covariance-cache compact`. The resulting compact HDF5
partitions contain canonical position pairs, `shrink_ld`, diagonal entries, and
lookup indexes, which are the fields used by the array-backed matrix-to-vector
path. Set `covariance_cache: full` only if you need full covariance metadata for
debugging or heatmap generation.

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
