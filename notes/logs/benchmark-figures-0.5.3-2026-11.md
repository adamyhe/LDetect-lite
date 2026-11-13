# Benchmark figures for v0.5.3 — 2026-11-12

Raw log behind the timing tables in `docs/optimizations.md` ("Example benchmark
figures") and `notes/findings/ldetect-example-benchmarks.md`. It regenerated the
original-LDetect vs `ldetect-lite` figures after the v0.5.3 release, which
contains optimizations #19-#21.

- **Job script:** `/home2/ayh8/ldetect-bench/figures_0.5.3_excl.sh`, out of tree.
  The shared-node variant is `figures_0.5.3.sh`.
- **Raw results:** `/home2/ayh8/ldetect-bench/results-figures-0.5.3{,-excl}/`.
  `env.txt` and `load.txt` record the CPU, the allocation and the load at each
  phase.

## Protocol

Each run was one Slurm job that did everything on one node:

1. `git worktree` of tag `v0.5.3` into node-local `/workdir`.
   `uv sync --frozen --group dev` (Python 3.13.5).
2. **Toy chr2** (`examples/ldetect_example`):
   - `snakemake --cores 1` (prep and exactness checks).
   - `benchmark_legacy_pipeline.py --warmups 1 --repeats 5`.
   - `benchmark_legacy_covariance.py --warmups 1 --repeats 2`.
3. **Full EUR chr21** (`examples/ldetect_original`):
   - `snakemake --cores 8 results/EUR/21/21-ld-blocks.bed`. This is a clean
     run: it computes covariance and includes JIT compilation.
   - A second run with `--forcerun run_ldetect`, which reused the cached
     covariance.
   - `Snakefile.legacy_diagnostics` staging.
   - The `/usr/bin/time -v` legacy downstream run.
   - `runtime_benchmark.py`.

Node scratch is wiped between jobs, so every input was rebuilt inside the job.

## Runs

| | Shared (job 4116901) | **Exclusive (job 4116908), used for the figures** |
|---|---|---|
| Node | cbsubscb19 | cbsubscb17 |
| CPU | 2x AMD EPYC 9554 64-Core, max 3.76 GHz | AMD EPYC 7702 64-Core (1 socket), max 3.35 GHz |
| Allocation | `-c 8`: CPUs 1,3,4,10,134-137 (not sibling pairs) | `--exclusive`, `taskset` to CPUs 0-3,64-67 (4 cores + HT siblings) |
| Load average | ~190-200 | ~1 (up to ~3 during legacy staging) |
| Wall time | 3:36:01 | 3:46:38 |

### Toy chr2 (mean seconds; original LDetect → ldetect-lite)

| Stage | Shared | Exclusive |
|---|---|---|
| `calc-covariance` | 285.08 → 4.488 (63.52x) | 213.46 → 3.850 (55.45x) |
| `matrix-to-vector` | 1.540 → 0.338 (4.55x) | 1.832 → 0.335 (5.46x) |
| `find-minima` | 16.68 → 1.978 (8.43x) | 21.04 → 1.646 (12.78x) |
| `extract-bpoints` | 0.807 → 0.265 (3.05x) | 0.916 → 0.272 (3.38x) |

- **Exclusive run spread:** `ldetect-lite` covariance mean 3.850 s vs min
  3.842 s; legacy 213.5 s vs 199.9 s.
- **Shared run spread:** `ldetect-lite` 4.488 s vs 4.170 s; legacy 285.1 s vs
  268.9 s.
- **Exactness, both runs:** 226,074 covariance rows, exact positions, max
  |diff| 2.78e-17. Vector, breakpoints and BED all pass. Peak RSS was about
  446 MiB for legacy and 717 MiB for lite; output sizes were 6.21 MB and
  1.58 MB.

### Full EUR chr21

| | Shared | Exclusive |
|---|---:|---:|
| `ldetect-lite` clean run (covariance + JIT + LD-neighborhood plot) | 138.88 s | **136.06 s** |
| `ldetect-lite` with covariance cached | 101.03 s | 99.12 s |
| Original LDetect downstream (single process, 99% CPU) | 7582 s | **9018 s** |
| Speedup (clean run) | 54.6x | **66.28x** |
| Speedup (cached covariance) | 75.0x | 91.0x |

- **Which `ldetect-lite` time is the headline:** the figure uses the clean
  run, as the README protocol describes (one run from a clean checkout). The
  previous 168.86 s figure was made the same way. The `--forcerun` run skipped
  all 103 partitions (`Skipping 103 already-completed partition(s)`) because
  the covariance cache from the first run was still on disk.
- **Peak RSS:** legacy ~1.32 GB; `ldetect-lite` ~494 MB with covariance cached
  and ~583 MB for the clean run.
- **Cached-covariance step times** (exclusive run):

  | Step | Time |
  |---|---:|
  | Step 3 | ~11 s |
  | Step 4 | ~24 s |
  | Step 5 (BED, then LD-neighborhood plot) | ~62 s |

  Almost all of Step 5 is the plot, which the Snakemake rule always
  requests.
- **Legacy was faster on the shared EPYC 9554 node, despite the load.** The
  legacy run is a single process, so it was less affected by contention, and
  the 9554 has higher per-core speed. That is why the shared-node ratios
  differ. Ratios are only comparable within a host.

## Exactness on chr21

- **`ldetect-lite`:** the 0.5.3 `fourier_ls` BED (24 blocks) is
  byte-identical to the published Bitbucket `ldetect-data` EUR chr21 blocks in
  both runs.
- **Original LDetect:** its downstream scripts, run on covariance converted
  from `ldetect-lite`'s, produced the same 24-block BED in both runs. That BED
  differs from `ldetect-lite` and from the published blocks at 2 of the 23
  internal boundaries:

  | `ldetect-lite` = published | Legacy downstream |
  |---:|---:|
  | 26,469,122 | 25,901,176 |
  | 46,177,105 | 46,043,396 |

This is the same pattern recorded for EUR chr7-13 in
`notes/findings/ldetect-original-reproduction.md`: the two implementations
agree closely but not exactly, which is expected near nearly-tied minima. Here,
`ldetect-lite` is the one matching the published reference. The legacy BED also
uses `chr21` labels and CRLF line endings, unlike `ldetect-lite`'s `21`. This
does not affect the timings.
