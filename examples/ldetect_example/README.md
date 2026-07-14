# ldetect Toy Example

This workflow reproduces the original EUR chr2 toy example distributed with
`ldetect`.

The workflow downloads the original BitBucket reference intermediates and also
fetches the matching 1000 Genomes Phase 1 chr2 interval. The main ldetect-lite
path starts from the VCF:

```text
work/vcf/1000G.phase1.EUR.2.39967768-40067768.vcf.gz
```

It filters to the toy example's EUR individuals, computes covariance from the
VCF, and compares each downstream artifact to the original LDetect reference
files:

```text
work/{chrom}/{chrom}.{start}.{end}.h5
```

The original gz covariance fixture remains a reference only; it is no longer
used as the starting point for the main example pipeline. The Snakemake
covariance rule intentionally uses `--ld-kernel uint8` because the covariance
exactness fixture checks full-schema legacy fields (`naive_ld`, genetic
positions, and SNP IDs). Production runs default to the compact bitpacked
backend.

## Run

```bash
uv run snakemake -n
uv run snakemake --cores 1
```

Expected comparison outputs:

```text
results/compare_covariance.tsv
results/compare_staged_partitions.tsv
results/compare_vector.tsv
results/compare_bpoints.tsv
results/compare_bed.tsv
results/compare_generated_partitions.tsv
results/plots/*.svg
```

The exactness checks fail the Snakemake job if the VCF-start covariance,
matrix-to-vector, breakpoint, or BED outputs diverge from the original
intermediates. The independently regenerated partition file is emitted as a
diagnostic comparison because the toy reference contains a one-window partition
fixture; its plot is zoomed to that toy fixture interval rather than treating
the whole-chromosome generated partition set as an exactness target.

To benchmark individual functions after the workflow has prepared `ref/` and
`work/vcf/`, run:

```bash
uv run --extra heatmap python scripts/benchmark_functions.py --warmups 1 --repeats 5
```

The benchmark calls the Python APIs directly to avoid command-launch overhead
and writes `results/function_benchmark/timings.tsv`,
`results/function_benchmark/exactness.tsv`, `summary.md`, and `timings.svg`.
It defaults to the compact bitpacked covariance backend; pass
`--ld-kernel uint8` to benchmark the reference backend.

To benchmark average downstream-stage CLI runtimes against the vendored
original LDetect scripts, run:

```bash
uv run --extra heatmap python scripts/benchmark_legacy_pipeline.py --warmups 1 --repeats 5
```

This writes `results/legacy_pipeline_benchmark/timings.tsv`, `summary.tsv`,
and independent per-step timing plots comparing mean original LDetect and
ldetect-lite CLI runtime.

To benchmark covariance generation as a command-level comparison against the
original LDetect script, run:

```bash
uv run --extra heatmap python scripts/benchmark_legacy_covariance.py --warmups 1 --repeats 2
```

This streams the prepared VCF into the vendored original
`P00_01_calc_covariance.py` script and compares it with
`ldetect calc-covariance` on the same interval. It writes
`results/legacy_covariance_benchmark/timings.tsv`, `summary.tsv`,
`timings.svg`, and `timings-calc-covariance.svg`.

This benchmark requires the prepared 1000 Genomes VCF interval and the original
covariance fixture under `ref/cov_matrix/`. The output-size comparison is
between the original gzipped text covariance file and the current full HDF5
`calc-covariance` CLI output; compact-cache storage comparisons should be made
with the function-level or full-genome bitpacking benchmarks instead.

## VCF/BCF Ingestion Profiling

The example workflow prepares both VCF.gz and BCF versions of the same EUR chr2
interval:

```text
work/vcf/1000G.phase1.EUR.2.39967768-40067768.vcf.gz
work/vcf/1000G.phase1.EUR.2.39967768-40067768.bcf
```

To time VCF.gz to BCF conversion on this interval:

```bash
mkdir -p results/bcf_conversion_profile

/usr/bin/time -v \
  -o results/bcf_conversion_profile/vcf_to_bcf.timing.log \
  bcftools view -Ob \
    -o results/bcf_conversion_profile/1000G.phase1.EUR.2.39967768-40067768.bcf \
    work/vcf/1000G.phase1.EUR.2.39967768-40067768.vcf.gz

/usr/bin/time -v \
  -o results/bcf_conversion_profile/bcf_index.timing.log \
  bcftools index -f \
    results/bcf_conversion_profile/1000G.phase1.EUR.2.39967768-40067768.bcf
```

To split bitpacked covariance input preparation into raw cyvcf2 iteration,
genotype decoding, current list-backed panel construction, current
array-plus-bitpack construction, direct one-pass packing, and packed HDF5
sidecar writing, run the profiler on both prepared inputs:

```bash
uv run python scripts/profile_bitpack_ingestion.py \
  --reference-panel work/vcf/1000G.phase1.EUR.2.39967768-40067768.bcf \
  --region 2:39967768-40067768 \
  --genetic-map ref/chr2.interpolated_genetic_map.gz \
  --individuals ref/eurinds.txt \
  --output results/bitpack_ingestion_profile_bcf.tsv \
  --repeats 7

uv run python scripts/profile_bitpack_ingestion.py \
  --reference-panel work/vcf/1000G.phase1.EUR.2.39967768-40067768.vcf.gz \
  --region 2:39967768-40067768 \
  --genetic-map ref/chr2.interpolated_genetic_map.gz \
  --individuals ref/eurinds.txt \
  --output results/bitpack_ingestion_profile_vcf.tsv \
  --repeats 7
```

These diagnostics are intentionally small-scale. They are useful for deciding
whether optimization effort should target input format, cyvcf2 iteration,
Python haplotype construction, direct bitpacking, or sidecar writes before
testing larger chromosome-scale changes.
