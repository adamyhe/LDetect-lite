# Benchmarks

## Original LDetect toy chr2 example

The VCF-start toy example benchmark now lives with the reproducibility workflow
under `examples/ldetect_example/`. First prepare the reference fixtures and the
matching filtered 1000 Genomes interval:

```bash
cd examples/ldetect_example
uv run snakemake --cores 1
```

Then run the function-level benchmark:

```bash
uv run --extra heatmap python scripts/benchmark_functions.py \
  --warmups 1 \
  --repeats 10
```

The script calls ldetect-lite APIs directly to avoid command-launch overhead.
It times covariance calculation from the prepared VCF, matrix-to-vector,
breakpoint search, and BED extraction, and writes:

- `results/function_benchmark/timings.tsv`
- `results/function_benchmark/exactness.tsv`
- `results/function_benchmark/summary.md`
- `results/function_benchmark/timings.svg`

Use `--ld-kernel bitpacked` to benchmark the compact bitpacked covariance
backend.

## Bitpacked LD kernel, full-genome scale

`bench_bitpacked_full_genome.py` downloads real 1000G data and compares the
`uint8` and `bitpacked` LD kernels at the covariance-partition layer only
(exactness of `lo`/`hi`/`shrink_ld`/diagonal rows, plus per-stage timing and
RSS profiling). For full-dataset *pipeline-output* exactness (vector,
breakpoints, BED, across all three populations, via the actual `ldetect run`
CLI), use `examples/ldetect_original/Snakefile.ld_kernel_diagnostics`
instead — see that directory's README. The two are complementary: the
Snakefile is the canonical exactness check; this script is for fine-grained
per-stage timing.
