# ldetect Toy Example

This workflow reproduces the original EUR chr2 toy example distributed with
`ldetect`.

The workflow downloads the original BitBucket reference intermediates and also
fetches the matching 1000 Genomes Phase 1 chr2 interval. The main ldetect-lite
path starts from the VCF:

```text
work/vcf/1000G.phase1.EUR.2.39967768-40067768.vcf.gz
```

It filters to the toy example's EUR individuals, computes covariance using the
established JIT uint8 backend, and compares each downstream artifact to the
original LDetect reference files:

```text
work/{chrom}/{chrom}.{start}.{end}.h5
```

The original gz covariance fixture remains a reference only; it is no longer
used as the starting point for the main example pipeline.

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
Pass `--ld-kernel bitpacked` to benchmark the compact bitpacked covariance
backend.
