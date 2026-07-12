# LDetect Example Benchmarks

**Findings summary (current as of 2026-07-12).** Distilled for human review
and manuscript drafting. Full process notes: `notes/logs/ldetect-example-benchmarking.md`.

## Scope

The benchmark target is the original LDetect EUR chromosome 2 toy interval
(`chr2:39,967,768-40,067,768`, hg19). The updated example workflow starts from
the matching 1000 Genomes Phase 1 VCF interval, subsets to the original EUR
individual list, regenerates LDetect-lite artifacts, and compares them to
downloaded copies of the original LDetect fixtures.

These timings are deliberately command-level comparisons where possible:

- original covariance uses the vendored `P00_01_calc_covariance.py` script with
  the prepared VCF streamed through stdin;
- LDetect-lite covariance uses `ldetect calc-covariance` on the same indexed
  VCF interval;
- downstream legacy stages use the compatibility wrapper
  `examples/ldetect_original/scripts/run_legacy_ldetect.py --stage ...`;
- downstream LDetect-lite stages use the installed `ldetect` CLI.

The result is a fairer comparison of user-visible command execution than the
earlier function-level-only benchmark. Function-level benchmarks are still
useful for detailed profiling and backend experiments, especially bitpacking.

## Current command-level timings

Measured on the toy interval after one Snakemake preparation run. Downstream
stages used one warmup and five measured repeats; covariance used one measured
repeat because the original script is slow.

| Stage | Original LDetect mean seconds | LDetect-lite mean seconds | Speedup |
|---|---:|---:|---:|
| `calc-covariance` | 99.936 | 3.222 | 31.02x |
| `matrix-to-vector` | 1.044 | 0.204 | 5.13x |
| `find-minima` | 10.786 | 1.200 | 8.98x |
| `extract-bpoints` | 0.607 | 0.138 | 4.42x |

Covariance peak RSS in the command-level run was 398.75 MiB for original
LDetect and 524.98 MiB for LDetect-lite. Output sizes in this specific
benchmark were 6.21 MB for original gzipped text and 18.21 MB for LDetect-lite
full HDF5. This is **not** the compact-cache storage comparison: the
`calc-covariance` CLI currently writes the full HDF5 schema for debugging and
heatmap support, whereas production `ldetect run` defaults to compact caches.

## Exactness status

The VCF-start example reproduces the original fixtures to exact or
roundoff-equivalent precision:

- covariance has exact row keys and 226,074 rows; shrinkage values differ by at
  most `5.55e-17`;
- matrix-to-vector output has all 671 loci equivalent; max absolute difference
  is `7.46e-14`;
- breakpoint JSON matches exactly for `fourier`, `fourier_ls`, `uniform`, and
  `uniform_ls`;
- BED output matches exactly: 13/13 blocks and 14/14 boundaries.

The independently generated whole-chromosome partition comparison is diagnostic
only. The toy reference fixture contains a single staged window; that staged
partition file is the exactness target for the downstream example and matches
exactly.

## Artifacts

Timing and exactness figures are tracked under:

```text
examples/ldetect_example/plots/
```

Current human-facing documentation:

```text
docs/optimizations.md
docs/exactness.md
```

## Caveats for manuscript use

The chr2 toy interval is a small example, not a whole-genome throughput
benchmark. It is valuable because it is directly tied to the original LDetect
fixture and gives a reproducible apples-to-apples command comparison. For
manuscript claims, report the exact command, repeat count, hardware/software
environment, and whether thread-count environment variables were pinned.
