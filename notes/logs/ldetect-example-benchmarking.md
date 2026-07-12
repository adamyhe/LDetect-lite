# LDetect Example Benchmarking

**Agent-oriented working log.** Raw investigation and update notes. For the
current distilled summary, see `notes/findings/ldetect-example-benchmarks.md`.

Date: 2026-07-12

## Motivation

The application note needed benchmark claims tied to the original LDetect toy
example, starting from VCF input rather than relying on a local `_reference/`
checkout or pre-existing generated intermediates. Earlier benchmark drafts mixed
legacy function calls with LDetect-lite CLI calls and also overemphasized
parallelization, which is not the main source of the implementation speedup.

The benchmark was reworked to compare user-visible commands:

- original covariance: restored vendored `P00_01_calc_covariance.py`, VCF
  streamed to stdin;
- LDetect-lite covariance: `ldetect calc-covariance`;
- original downstream stages: `run_legacy_ldetect.py --stage ...`;
- LDetect-lite downstream stages: `ldetect matrix-to-vector`,
  `ldetect find-minima`, and `ldetect extract-bpoints`.

## Exactness workflow

`examples/ldetect_example/Snakefile` now prepares the matching 1000 Genomes
Phase 1 VCF interval, regenerates LDetect-lite covariance from that VCF, and
compares all downstream artifacts to downloaded copies of the original LDetect
fixtures. It emits SVG figures under `examples/ldetect_example/plots/`.

Current exactness tables:

```text
results/compare_covariance.tsv
results/compare_vector.tsv
results/compare_bpoints.tsv
results/compare_bed.tsv
results/compare_staged_partitions.tsv
results/compare_generated_partitions.tsv
```

Observed results:

- covariance: 226,074 rows, exact keys, max shrinkage difference
  `5.551115e-17`;
- vector: 671 shared loci, all equivalent, max absolute difference
  `7.460699e-14`;
- breakpoints: exact for all four subsets;
- BED: exact 13/13 blocks and 14/14 boundaries;
- staged toy partition: exact;
- generated whole-chromosome partition: diagnostic only and not expected to
  match the one-window toy fixture.

## Timing commands

Downstream command-level benchmark:

```bash
cd examples/ldetect_example
UV_CACHE_DIR=/Users/adamhe/github/ldetect-lite/.uv-cache \
  uv run --extra heatmap python scripts/benchmark_legacy_pipeline.py \
  --warmups 1 --repeats 5 --plot-format svg
```

Result:

```text
stage              legacy_mean_seconds  lite_mean_seconds  speedup
matrix_to_vector   1.044236             0.203742           5.125282
find_minima        10.785921            1.200439           8.984978
extract_bpoints    0.607418             0.137519           4.416980
```

Covariance command-level benchmark:

```bash
cd examples/ldetect_example
UV_CACHE_DIR=/Users/adamhe/github/ldetect-lite/.uv-cache \
  uv run --extra heatmap python scripts/benchmark_legacy_covariance.py \
  --warmups 0 --repeats 1 --plot-format svg
```

Result:

```text
legacy_mean_seconds              99.935503
lite_mean_seconds                 3.222010
mean_speedup                     31.016509
legacy_peak_rss_mib_max        398.750000
lite_peak_rss_mib_max          524.984375
legacy_output_bytes         6209257
lite_output_bytes          18206607
output_size_ratio_lite_vs_legacy  2.932172
```

The covariance output-size comparison here is full HDF5 versus legacy gz text,
not compact-cache HDF5 versus legacy gz text. Use full-genome or function-level
bitpacking/compact-cache benchmarks for production storage claims.

## Plots

Tracked SVGs:

```text
examples/ldetect_example/plots/timings-calc-covariance.svg
examples/ldetect_example/plots/timings-matrix_to_vector.svg
examples/ldetect_example/plots/timings-find_minima.svg
examples/ldetect_example/plots/timings-extract_bpoints.svg
examples/ldetect_example/plots/covariance.svg
examples/ldetect_example/plots/vector.svg
examples/ldetect_example/plots/breakpoints.svg
examples/ldetect_example/plots/bed.svg
examples/ldetect_example/plots/generated_partitions.svg
```

`docs/optimizations.md` links the timing plots and `docs/exactness.md` links
the exactness plots.

## Interpretation

The largest measured speedup on the toy example is covariance generation
(~31x), reflecting compiled pairwise calculations plus more efficient
genotype/covariance I/O. Downstream stages also improve substantially through
typed/indexed HDF5 reads, array-backed vector/metric calculations, and avoiding
legacy Decimal arithmetic on the default path.

Parallel worker orchestration remains useful operationally, but should not be
presented as the main optimization relative to original LDetect because the
original independent scripts can also be scheduled externally across
chromosomes or partitions.
