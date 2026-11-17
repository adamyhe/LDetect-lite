# Roadmap

Planned and in-progress work. Released changes are in
[`CHANGELOG.md`](CHANGELOG.md).

## In progress

### DP-optimal LD block splitter (`dp-ld-splitter` branch)

A new `dp` breakpoint subset that finds the exact-optimal partition for the
same sum(r²) objective that `fourier_ls`/`uniform_ls` only optimize locally by
hill-climbing. It solves every block count up to a maximum in one pass,
following Privé (2022)'s `snp_ldsplit()`. Candidate cut points are restricted
so memory and compute stay bounded on whole chromosomes.

- Done on the branch: `src/ldetect_lite/dp_split.py`; SNP-count and
  physical-distance block-size constraints; `thr_r2`/`max_r2` pruning; CLI
  wiring in `run`, `find-minima` and `extract-bpoints`. Tests cover brute
  force, `metric.py`'s cost definition, and the cost compared with
  `fourier_ls` on the toy fixture.
- Design note: `notes/shrinkage_ld_optimized_dynamic_programming.md` on the
  branch. It describes running the DP directly on the sparse
  shrinkage-correlation pairs in the compact covariance cache, with interval
  scores generated during the sweep instead of stored.
- To merge: the branch forked from v0.5.1, so it needs rebasing onto the
  v0.5.3 Step 2/Step 4 changes (`pipeline.py` and `cmd_run.py` both changed),
  then a genome-scale run comparing DP blocks with `fourier_ls`.

### Whole-genome runtime benchmark (`dp-ld-splitter` branch)

`examples/ldetect_original/Snakefile.runtime_benchmark` compares original
LDetect and ldetect-lite end to end, including original LDetect's own
covariance step (`scripts/run_legacy_covariance.py`). The current chr21
comparison starts original LDetect from ldetect-lite's covariance, so it
understates the speedup. Original LDetect runs on chr19-22 and ldetect-lite
on all 22 autosomes, with CPU-seconds as the main metric.

- Done on the branch: the workflow, a collector script, and local validation
  on the chr2 toy interval.
- Also on the branch: `run_legacy_ldetect.py --filter-window`. This confirmed
  on the original code itself that only the symmetric Hann window reproduces
  the published toy blocks.
- To do: the full cluster run, then merging alongside or ahead of the DP
  splitter.

### BCF benchmark figures

- Re-measure the toy chr2 covariance and EUR chr21 figures with GT-only BCF
  input, plus the one-time conversion cost (Slurm job 4124619, pending).
- Rerun the replications genome-wide on BCF to confirm byte-identical blocks.

## Planned

- **Array-backed multi-partition local search.** Multi-partition windows and
  `--high-precision` still use the dictionary path, because the array path
  doesn't yet reproduce the legacy multi-partition locus list
  (`notes/logs/local-search-divergence-asn22.md`).

## Shelved

- **Whole-chromosome decode cache for Step 2.** It would save about 1-2 s of
  13 s at 4 workers on BCF input. Worth revisiting for much larger panels
  (`docs/optimizations.md` §20).
- **EUR chr8-12 and AFR chr22 reproduction divergence.** Accepted and
  documented; next steps are in
  `notes/findings/ldetect-original-reproduction.md`.
