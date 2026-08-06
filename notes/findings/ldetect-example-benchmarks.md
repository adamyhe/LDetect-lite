# LDetect Example Benchmarks

**Findings summary (current as of 2026-08-05).** Distilled for human review
and manuscript drafting. Full process notes: `notes/logs/ldetect-example-benchmarking.md`
(2026-07-12 methodology notes; timings below were re-measured on 2026-08-05 on
a different host — see the hardware note below).

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

Measured on the toy interval after one Snakemake preparation run, on
`cbsugpu01` (Intel Xeon E5-2620 v4 @ 2.1GHz). Downstream stages used one
warmup and five measured repeats; covariance used one warmup and two measured
repeats because the original script is slow.

| Stage | Original LDetect mean seconds | LDetect-lite mean seconds | Speedup |
|---|---:|---:|---:|
| `calc-covariance` | 259.278 | 4.651 | 55.75x |
| `matrix-to-vector` | 2.112 | 0.365 | 5.79x |
| `find-minima` | 23.844 | 1.892 | 12.60x |
| `extract-bpoints` | 1.082 | 0.296 | 3.66x |

This supersedes an earlier measurement (99.936s / 3.222s / 31.02x for
`calc-covariance`, etc.) taken on a different, faster host (an AMD EPYC
9554) — absolute times are not comparable across hosts, but legacy and
LDetect-lite were timed together on the same host in both cases, so each
speedup ratio is a valid same-environment comparison.

Covariance peak RSS in this command-level run was 445.87 MiB for original
LDetect and 716.95 MiB for LDetect-lite. Output sizes were 6.21 MB for
original gzipped text and 1.58 MB for LDetect-lite. This **is** now the
compact-cache storage comparison: `calc-covariance` ties `compact_output` to
`--ld-kernel` (bitpacked, the default, always writes compact output; `uint8`
writes the full schema), so the default CLI invocation and production
`ldetect run` write the same compact schema. This corrects an earlier version
of this finding, which reported an 18.21 MB full-schema LDetect-lite output
from before that coupling existed.

## Exactness status

The VCF-start example reproduces the original fixtures to exact or
roundoff-equivalent precision:

- covariance has exact row keys and 226,074 rows; shrinkage values differ by at
  most `2.78e-17`;
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
