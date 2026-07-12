# Reproducing the Original ldetect LD Blocks

This example pipeline attempts to reproduce the EUR, AFR, and ASN LD blocks
published with Berisa and Pickrell (2016):

- Original implementation: <https://bitbucket.org/nygcresearch/ldetect>
- Published block files: <https://bitbucket.org/nygcresearch/ldetect-data>

The workflow starts from public 1000 Genomes Phase 1 VCFs and runs the modern
`ldetect-lite` implementation end to end.

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

`--cores N` lets Snakemake schedule multiple chromosome/population jobs concurrently (each claiming `workers`-many cores). `run_ldetect` exports `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS`/`NUMBA_NUM_THREADS` to match `workers` so BLAS/numba don't oversubscribe the shared node when several jobs land on it at once (see `docs/optimizations.md` #13) — no action needed unless you're invoking `ldetect run` directly outside this Snakefile.

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

By default, each diagnostic `ldetect run` job claims
`ldetect_job_threads: 4` Snakemake cores and passes up to
`workers: 4` to `ldetect --workers`. Local search is defensively capped
at one worker in this diagnostic workflow to avoid multiplying memory use.
With `--cores 4`, chromosomes run one at a time. To run multiple chromosomes
concurrently instead, use one internal worker per chromosome and
give Snakemake more cores:

```bash
uv run snakemake -s Snakefile.diagnostics --cores 4 \
  --config chromosomes='[8,9,10,11,12,13]' \
           population=EUR \
           workers=1 \
           ldetect_job_threads=1
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
ldetect-lite HDF5 covariance partitions into the original text format, runs legacy
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
`workers` or `local_search_workers` can spawn multiple large
covariance/local-search processes inside a single chromosome job; raising
`--cores` can also allow Snakemake to run multiple chromosome jobs at once
unless `ldetect_job_threads` claims the cores for each heavy job. Staging
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
**covariance cache size** of the `--covariance-compression` codec choice
against this pipeline's real 1000G data. For each configured chromosome x
population, it runs `ldetect run` twice on identical filtered input — once
with `--covariance-compression lzf` (the prior default) and once with
`--covariance-compression zstd` (the new default) — then compares the two
runs':

- **vector/breakpoints/BED**: row count, sha256 digest, exact loci match, and
  BED boundary recall/precision/Jaccard at `tolerance` bp (default `0`).
  Compression is lossless, so these are expected to match exactly — any
  divergence here indicates a real bug, not floating-point noise from a
  different computation order;
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

Smoke-test a smaller subset instead of the full genome. `--config` merges
into `chromosomes_by_population` key by key, so a population left out of the
override keeps its full 1-22 default — override every population you want
restricted, or the "smoke test" will still run the others genome-wide:

```bash
uv run snakemake -s Snakefile.compression_diagnostics --cores 4 \
  --config chromosomes_by_population='{EUR: [22], AFR: [22], ASN: [22]}'
```

Outputs:

- `results/compression_diagnostics/{population}/{chrom}/{baseline,zstd}/` —
  each mode's full `ldetect run` output directory, including its covariance
  partitions under `{mode}/{chrom}/`.
- `results/compression_diagnostics/{population}/{chrom}/logs/{mode}.benchmark.tsv` —
  Snakemake's wall-clock/peak-memory record for that mode.
- `results/compression_diagnostics/{population}/{chrom}/compare/compression_vs_baseline.tsv` —
  one row combining the exactness, size, and performance comparison for that
  chromosome x population.
- `results/compression_diagnostics/summary.tsv` — all comparison rows
  concatenated.

## SV-Boundary Diagnostic Workflow

`Snakefile.sv_boundary_diagnostics` tests a candidate explanation for the
parked EUR chr8-12 / AFR chr22 reproduction divergence (see
`notes/findings/ldetect-original-reproduction.md`, "New candidate mechanism:
SV/indel partition-boundary duplication"): `calc_covariance`'s region-based
read has no explicit `start <= pos <= end` check of its own, so a structural
variant or long indel whose span crosses a partition boundary can be
spuriously double-counted into a neighboring partition's covariance
calculation. The original ldetect is exposed to this; MacDonald et al. (2022)
sidesteps it entirely by filtering to SNPs only upstream.

For each chromosome in the same divergent + control set used by
`Snakefile.provenance_diagnostics` (EUR chr8-13, AFR chr11/13/22), this
workflow runs the full pipeline twice on identical population-filtered
input — once with the current `all`-variant-types filtering (matches the
main `Snakefile`/original ldetect methodology) and once with an added
`snps_only` filter (`bcftools view --types snps`, matching MacDonald2022) —
then compares each mode's BED against the published Berisa & Pickrell
reference at a loose 100kb tolerance (sanity check — both modes should stay
matching) and an exact 0bp tolerance (the metric that actually tests the
hypothesis), plus `snps_only` directly against `all` to quantify how much
the filter itself shifts boundaries.

Dry-run the default (divergent + control chromosome set) diagnostic:

```bash
uv run snakemake -s Snakefile.sv_boundary_diagnostics -n
```

Run it for real:

```bash
uv run snakemake -s Snakefile.sv_boundary_diagnostics --cores 8
```

Smoke-test a smaller subset (same `--config` merge caveat as above — override
every population you want restricted):

```bash
uv run snakemake -s Snakefile.sv_boundary_diagnostics --cores 4 \
  --config chromosomes_by_population='{EUR: [10], AFR: [22]}'
```

Outputs:

- `results/sv_boundary_diagnostics/{population}/{chrom}/{all,snps_only}/` —
  each mode's full `ldetect run` output directory for that chromosome.
- `results/sv_boundary_diagnostics/{population}/{all,snps_only}_LD_blocks.bed` —
  each mode's combined BED across this diagnostic's chromosome subset.
- `results/sv_boundary_diagnostics/{population}/compare/{mode}_vs_ref_tol{0,100000}.tsv` —
  each mode vs. the published reference, per chromosome, at both tolerances.
- `results/sv_boundary_diagnostics/{population}/compare/snps_only_vs_all_tol0.tsv` —
  `snps_only` directly against `all`, at exact tolerance.
- `results/sv_boundary_diagnostics/summary.tsv` — all comparison rows
  from every population and comparison, concatenated and tagged by which
  comparison they came from.

**Reading the result:** if the hypothesis holds, `snps_only_vs_ref_tol0`
should show improved (higher) `recall`/`precision`/`jaccard` specifically on
EUR chr8-12 and AFR chr22 relative to `all_vs_ref_tol0`, while both stay
similarly high on the control chromosomes (chr13 for both populations, chr11
for AFR) and at the loose 100kb tolerance. If `snps_only` doesn't move the
divergent chromosomes closer to the reference, that's a real negative result
worth recording back in `notes/findings/ldetect-original-reproduction.md`,
not a reason to rerun with different parameters first.

## Resource Profiling

Two scripts under `scripts/` turn this replication's resource usage into
plots. Part A needs no new run -- it aggregates data the main `Snakefile`
already produces (Snakemake's `benchmark:` record per `run_ldetect` job).
Part B is a genuine time-series trace of one run, since `benchmark:` only
records a single summary row per job.

**Part A -- scaling across the whole replicated dataset:**

```bash
uv run python scripts/plot_resource_scaling.py \
  --results-root results \
  --output-dir results/profiling \
  --workers-cap 4
```

Aggregates every `results/logs/{population}/{chrom}.benchmark.tsv` (peak
RSS, wall-clock time, mean CPU load) against each chromosome's partition
count, and plots peak memory / runtime / core utilization vs. dataset size,
one series per population, into `results/profiling/`.

**Part B -- detailed memory/CPU trace for one run:**

```bash
uv run python scripts/profile_run.py \
  --interval 1.0 \
  --output results/profiling/EUR-chr21.csv \
  --log-output results/profiling/EUR-chr21.log \
  -- uv run ldetect run --genetic-map data/maps/chr21.interpolated_genetic_map.gz \
  --reference-panel results/filtered_vcf/EUR/ALL.chr21.phase1_release_v3.20101123.snps_indels_svs.genotypes.population-polymorphic.vcf.gz \
  --individuals resources/EUR_inds.txt \
  --chromosome 21 --output-dir results/profiling/EUR/21 --workers 4

uv run python scripts/plot_profile_timeline.py \
  --csv results/profiling/EUR-chr21.csv \
  --log results/profiling/EUR-chr21.log \
  --title "EUR chr21" \
  --output results/profiling/EUR-chr21-timeline
```

`profile_run.py` requires the `profiling` extra (`uv sync --extra profiling`,
for `psutil`) and polls the wrapped command's whole process tree -- covering
every `ProcessPoolExecutor` worker and `tabix` subprocess it spawns, not just
the top-level process -- so the trace reflects real peak usage across
parallel steps. `plot_profile_timeline.py` correlates the trace against the
run's own `Memory checkpoint` log lines to shade each pipeline step. Point
`--output-dir` at a fresh directory rather than an existing populated one --
`ldetect run` skips any covariance partition that already has a valid HDF5
cache, which would produce an unrepresentative fast/idle trace.

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

These are configured in `config.yaml` and passed to `ldetect run --ne`.

## Pipeline Steps

1. Download public Phase 1 VCFs, genetic maps, panel metadata, and published
   BED references.
2. Build population-specific individual lists.
3. Create population-specific VCFs filtered to biallelic records with
   population-specific `MAC[0] >= 1`.
4. Run `ldetect run` per chromosome and population using
   `n_snps_bw_bpoints = 10000`.
5. Combine chromosome BEDs into genome-wide BEDs.
6. Compare the generated BEDs against the published ldetect blocks.

By default, this pipeline sets `covariance_cache: compact` and passes
`ldetect run --covariance-cache compact`. The resulting compact HDF5
partitions contain canonical position pairs, `shrink_ld`, diagonal entries, and
lookup indexes, which are the fields used by the array-backed matrix-to-vector
path. Set `covariance_cache: full` only if you need full covariance metadata for
debugging or heatmap generation.

## Notes for Developers

- `ldetect_lite.shrinkage` intentionally applies the covariance cutoff before
  adding the diagonal shrinkage term. This matches the original ldetect script
  and drops population-monomorphic variants from the covariance output.
- This ordering fixed an important compatibility bug. An earlier ldetect-lite
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
