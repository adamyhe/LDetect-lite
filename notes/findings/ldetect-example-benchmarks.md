# LDetect Example Benchmarks

**Findings summary (current as of 2026-11-22).** Distilled for human review
and manuscript drafting. Full process notes: `notes/logs/ldetect-example-benchmarking.md`
(2026-07-12 methodology notes). The timings below were re-measured for v0.5.3 on
2026-11-12 on an exclusive AMD EPYC 7702 node; raw log:
`notes/logs/benchmark-figures-0.5.3-2026-11.md`. GT-only BCF timings were added
on 2026-11-21/22 on the same node; raw log:
`notes/logs/benchmark-figures-bcf-2026-11.md`.

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

Measured with `ldetect-lite` v0.5.3 after one Snakemake preparation run, on
**cbsubscb17, an AMD EPYC 7702 64-Core Processor** (1 socket, 64 cores, 2 threads
per core, max 3.35 GHz). The node was allocated exclusively, load average ~1, and
all commands were pinned to 4 physical cores plus their hyperthread siblings
(CPUs 0-3,64-67). Downstream stages used one warmup and five measured repeats.
Covariance used one warmup and two measured repeats, because the original script
is slow.

| Stage | Original LDetect mean seconds | LDetect-lite mean seconds | Speedup |
|---|---:|---:|---:|
| `calc-covariance` | 213.458 | 3.850 | 55.45x |
| `matrix-to-vector` | 1.832 | 0.335 | 5.46x |
| `find-minima` | 21.036 | 1.646 | 12.78x |
| `extract-bpoints` | 0.916 | 0.272 | 3.38x |

These ratios are within ~10% of the previous measurement, taken on `cbsugpu01`
(Intel Xeon E5-2620 v4 @ 2.1GHz) with earlier code: 55.75x / 5.79x / 12.60x /
3.66x. The v0.5.x Step 2 and Step 4 optimizations mostly pay off at chromosome
scale, not on this one-window interval. Absolute times are not comparable across
hosts. Each speedup is a same-host, same-job comparison.

A run of the same job on a heavily shared node (cbsubscb19, load average ~200)
gave noisier and inconsistent ratios: `find-minima` 8.43x, `calc-covariance`
63.5x. Use exclusive allocations for figures.

Covariance peak RSS in this command-level run was 446.1 MiB for original
LDetect and 717.3 MiB for LDetect-lite. Output sizes were 6.21 MB for
original gzipped text and 1.58 MB for LDetect-lite. This **is** now the
compact-cache storage comparison: `calc-covariance` ties `compact_output` to
`--ld-kernel` (bitpacked, the default, always writes compact output; `uint8`
writes the full schema), so the default CLI invocation and production
`ldetect run` write the same compact schema. This corrects an earlier version
of this finding, which reported an 18.21 MB full-schema LDetect-lite output
from before that coupling existed.

## GT-only BCF input

Same host and pinning (Slurm jobs 4128926, 4141431, 4141689).

| Measurement | `.vcf.gz` | GT-only BCF |
|---|---:|---:|
| Toy chr2 `calc-covariance`, mean s (original LDetect 203.128 s) | 3.876 (52.40x) | 3.642 (55.78x) |
| EUR chr21 `ldetect run`, Step 2 s (`workers=4`, warm) | 29 | 19 |
| EUR chr21 `ldetect run`, wall s (2 repeats each) | 56.7 | 46.4 |
| EUR chr21 summed PSS peak, whole process tree | ~1.37 GB | ~1.37-1.53 GB |
| EUR chr21 largest single process | 535 MiB | 549-574 MiB |
| Filtered EUR chr21 panel size | 334 MB | 12 MB |

- The one-time toy conversion took 0.142 s.
- Outputs are identical: toy covariance to 2.78e-17 with exact positions;
  chr21 vectors (sha256), breakpoints and BEDs byte-identical.
- BCF speeds up only Step 2 (genotype decoding). On the toy interval startup
  dominates, so the gain is 1.06x; on chr21 it is 1.5x for Step 2 and 1.22x
  end to end.
- Peak memory does not improve. The older "~30% less memory" BCF result
  predates the streaming Step 2.
- With covariance cached, the chr21 main workflow took 99.07 s, matching
  v0.5.3's 99.12 s, as expected since only Step 2 reads the panel.
- For manuscript use, quote the chr21 Step 2 and end-to-end figures, not the
  main-workflow clean-run total, which varied from 113 s to 151 s with how warm
  the checkout was (first-import startup).

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
