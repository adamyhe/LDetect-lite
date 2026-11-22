# BCF benchmark figures (2026-11-21/22)

Raw log for the `.vcf.gz` vs GT-only BCF measurements on branch
`bcf-replications` (code at `b440585`, ldetect-lite 0.5.3 + BCF workflow
changes; no `src/` changes vs v0.5.3). Distilled results are in
`docs/optimizations.md` ("Example benchmark figures") and
`notes/findings/ldetect-example-benchmarks.md`.

All jobs: Slurm on **cbsubscb17** (AMD EPYC 7702, 1 socket, 64 cores, 2
threads/core), `--exclusive --mem=64G`, every command pinned with
`taskset -c 0-3,64-67` (4 physical cores + HT siblings), data on node-local
`/workdir`, Python 3.13.5, bcftools 1.20. Same host and pinning as the v0.5.3
figures (job 4116908, `benchmark-figures-0.5.3-2026-11.md`). Load average ~1-2
throughout. Job scripts live outside the repo (`~/ldetect-bench/`).

GT-only BCF = `bcftools annotate -x INFO,^FORMAT/GT -Ob` + `bcftools index`.

## Jobs

| Job | What | Outcome |
|---|---|---|
| 4124619 | full figure script | OOM after 56 min: `--exclusive` keeps the 1 GB default `--mem`. Toy numbers from it discarded (1 GB cgroup also capped page cache; lite `.vcf.gz` 5.31 s vs 3.85 s normally). |
| 4128926 | full figure script, `--mem=64G` | Toy chr2 and chr21 main clean run usable. chr21 `.vcf.gz` vs BCF **not usable** (ran concurrently, see below). Cached-covariance step hit the 4 h limit (see below). |
| 4141431 | chr21 main pipeline: clean, then cached covariance | Complete. |
| 4141689 | chr21 `.vcf.gz` vs BCF, serialized, with process-tree memory sampling | Complete. Source of the format comparison. |

### Two harness bugs found in 4128926

1. **Concurrent vcf/bcf runs.** `Snakefile.cyvcf2_migration_profiling` was
   run with `--cores 8 --config workers=4`. The run rule has
   `threads: workers`, so Snakemake started the vcf and bcf jobs 6 s apart on
   the same 8 pinned CPUs; both timings (92.45 s / 62.13 s) were contended.
   Fix: `--cores` equal to `workers`.
2. **`--forcerun` swallowed the target.** `snakemake --cores 8 --forcerun
   run_ldetect results/EUR/21/21-ld-blocks.bed` parses the BED path as a second
   rule name, so the default `all` target (66 chromosome x population runs)
   was scheduled and the job timed out downloading. Fix: put the target before
   `--forcerun`.

Also seen during the accidental genome-wide run: `bgzf_uncompress` inflate
error reading the freshly downloaded raw chr12 VCF (likely a truncated
download). Unrelated to chr21; not investigated.

## Toy chr2 `calc-covariance` (job 4128926)

`benchmark_legacy_covariance.py --warmups 1 --repeats 2`.

| Backend | Mean s | Min s | Peak RSS MiB |
|---|---:|---:|---:|
| Original LDetect | 203.128 | 198.609 | 402.7 |
| ldetect-lite, `.vcf.gz` | 3.876 | 3.849 | 717.1 |
| ldetect-lite, GT-only BCF | 3.642 | 3.597 | 717.3 |
| One-time BCF conversion | 0.142 | 0.142 | - |

- Speedup vs original: 52.40x (`.vcf.gz`), 55.78x (BCF). BCF vs `.vcf.gz`:
  1.06x. The interval is ~100 kb, so fixed startup dominates.
- All six runs: 226,074 rows, exact positions, max |diff| 2.78e-17.
- `.vcf.gz` mean matches v0.5.3 job 4116908 (3.850 s).

## EUR chr21 `.vcf.gz` vs BCF (job 4141689)

Filtered panel: 334 MB `.vcf.gz` vs 12 MB GT-only BCF.

**A. Snakemake, serialized** (`--cores 4 --config workers=4`): vcf 87.80 s,
bcf 46.15 s. Not used as a headline: the vcf run was the first `ldetect` call
in the fresh worktree and paid ~30 s of first-import startup (io_in 243 MB vs
0.5 MB for bcf). `compare.tsv`: vector rows 233,282 equal, vector sha256
equal, 23/23 loci exact, BED exact.

**B. Direct `ldetect run`, warm**, under `sample_tree_mem.py` (psutil, whole
process tree every 0.2 s) and `/usr/bin/time -v`; one warm-up per format, then
vcf, bcf, vcf, bcf. `.venv/bin/ldetect` called directly (no `uv run` wrapper).

| Run | Wall s | Step 2 s | Step 4 s | Peak sum PSS MiB | Peak sum USS MiB | Peak sum RSS MiB | Largest process MiB | `time -v` max RSS KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vcf-0 | 56.83 | 29 | 24 | 1369.1 | 1243.3 | 1959.8 | 535.2 | 590,596 |
| vcf-1 | 56.54 | 29 | 24 | 1361.3 | 1244.2 | 1897.5 | 535.6 | 591,116 |
| bcf-0 | 46.50 | 19 | 25 | 1371.6 | 1249.9 | 1967.7 | 548.7 | 592,356 |
| bcf-1 | 46.20 | 19 | 24 | 1530.2 | 1402.9 | 2105.1 | 574.3 | 592,232 |
| warmup-vcf | 56.70 | | | 1461.7 | 1332.8 | 2001.1 | 540.0 | 583,284 |
| warmup-bcf | 47.27 | | | 1378.9 | 1250.1 | 2009.9 | 566.6 | 586,148 |

- **Speed:** Step 2 1.5x faster on BCF (29 s -> 19 s); Steps 3-5 unchanged;
  end to end 56.7 s -> 46.4 s (1.22x).
- **Memory:** max 5 processes (main + 4 workers). Summed PSS peaks at about
  1.37 GB for both formats; one BCF repeat reached 1.53 GB (peak during Step
  2). Largest single process 535 MiB (vcf) vs 549-574 MiB (bcf). No consistent
  difference; BCF is not a memory optimization at this scale.
- All six BEDs md5 `eaaf55d419c864838e9d3de4797bf520`.
- Why job 4128926's Snakemake `max_rss` showed 2057 vs 1562 MB: Snakemake's
  benchmark sums RSS over the process tree (shared pages counted per
  process), sampling every 0.5 s for ~15 s and then every 30 s, so a 60-90 s
  run gets only a couple of samples after Step 2 starts. Plus the runs were
  concurrent.

## EUR chr21 main reproduction pipeline on BCF (jobs 4128926, 4141431)

`Snakefile` (now GT-only BCF by default), `run_ldetect` rule, `workers=4`,
includes the LD-neighborhood plot.

| Run | Job | Wall s | Notes |
|---|---|---:|---|
| clean | 4128926 | 113.28 | earlier commands had already run in the worktree; ~1 s startup |
| clean | 4141431 | 150.78 | first command in a fresh worktree: ~32 s before Step 1, io_in 243 MB |
| cached covariance | 4141431 | 99.07 | Step 2 skipped (103 partitions cached); Step 3 non-fused ~11 s |

- v0.5.3 (`.vcf.gz`, job 4116908): clean 136.06 s, cached 99.12 s. Cached
  times agree to 0.05 s, as expected (Step 2 is the only format-dependent
  step). The clean-run totals depend on how warm the checkout is, so the
  format effect is taken from job 4141689 B instead.
- Step times, clean BCF run (4141431): Step 2 ~23 s, Step 4 ~24 s, Step 5
  (BED + plot) ~71 s.
- Peak RSS (`time -v`): 602 MB clean, 498 MB cached (v0.5.3: ~583 / ~494 MB).
- BED md5 `eaaf55d419c864838e9d3de4797bf520`, identical to the 4141689 BEDs.
