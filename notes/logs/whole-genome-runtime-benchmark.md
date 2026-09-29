# Whole-Genome Runtime Benchmark (LDetect vs ldetect-lite)

**Agent-oriented working log.** Raw investigation and build notes for
`examples/ldetect_original/Snakefile.runtime_benchmark`.

Date: 2026-09-10

## Motivation

The existing chr21 head-to-head (README, "Full EUR chr21 Runtime Benchmark",
`scripts/runtime_benchmark.py`) needed generalizing to all 22 autosomes for
EUR, with results-concordance stats alongside runtime.

## The existing chr21 comparison is not like-for-like

Found while reading the prior setup, before building anything:

- lite arm: full `ldetect run` — partition, covariance, matrix-to-vector,
  find-minima, extract-bpoints.
- legacy arm: `scripts/run_legacy_ldetect.py`, which is `P01`/`P02`/`P03`
  **only**, started from lite-generated covariance converted to legacy text
  format by `stage_legacy_dataset`.

So legacy never runs `P00_01_calc_covariance.py` — measured at **55.75x**
slower than lite on the chr2 toy interval
(`notes/findings/ldetect-example-benchmarks.md`) — and gets its dominant stage
for free. Both rows are still labelled `full_chromosome` in `timings.tsv`. The
published chr21 speedup therefore *understates* lite.

The README section is now marked downstream-only and superseded, rather than
deleted: it documents the staged-dataset route `Snakefile.legacy_diagnostics`
also uses.

## Scope decision

Full end-to-end legacy on all 22 autosomes is impractical. Extrapolating the
chr2 toy interval (226,074 pairs / 259 s on `cbsugpu01`) to EUR genome-wide —
~17-20M biallelic MAC>=1 SNPs, a ±0.267 cM window at Ne=11418/n=379, so ~1300
forward partners per SNP — gives order 10^3-10^4 CPU-hours. Order of magnitude
only; the extrapolation ignores partition-edge truncation.

Per the user: legacy runs end-to-end on **a few chromosomes only**
(`legacy_full_chromosomes`, default chr19-22), lite runs all 22. No
downstream-only arm.

## What parallelizes in legacy

Checked against the vendored source, since it determines whether wall clock
means anything:

| Stage | Parallel? | Evidence |
|---|---|---|
| `P00_01` covariance | Yes, trivially | Independent process per partition, VCF region on stdin. How the original authors ran it. |
| `P01` matrix→vector | No, as written | `E03:297 calc_diag_lean` streams partitions in order into a running diagonal sum with dynamic locus deletion. |
| `P02` filter-width search | No, fundamentally | `E05:98 custom_binary_search_with_trackback`: exponential → binary → trackback, each probe depends on the previous. |
| `P02` local search | Independent in principle, serial as written | `P02:159` — each `run_local_search_single` reads only the original breakpoint list and fixed `total_sum`/`total_N`. Exploiting it means editing vendored code. |
| Across chromosomes | Yes, 22-way | Both arms. |

Consequence: legacy's per-chromosome wall-clock floor is its downstream time,
whatever the core count. Hence **CPU-seconds (user+sys) is the primary
metric**; wall clock is recorded but tagged with its worker count.

## Two hazards found while building

**Unmapped positions.** `ldetect-lite` retains only variants present in
`pos2gpos` (`src/ldetect_lite/_util/reference_panel.py:75`). Legacy does a bare
`pos2gpos[pos1]` lookup and would `KeyError`. The chr2 toy window never exposed
this because every position in it is in the map. `run_legacy_covariance.py`
pre-filters the VCF stream with a `bcftools -T` targets file built from the
map, which is exactly lite's rule, so both arms see an identical variant set.

**The Hann window.** `run_legacy_ldetect.py` maps `'hanning'`→`'hann'` but
leaves `fftbins=True`, so modern scipy returns the *periodic* window — while
the 2015 run actually got the *symmetric* one via the scipy 0.16.0 defect
(`notes/findings/ldetect-original-reproduction.md`). Left alone, the legacy arm
would diverge from lite (whose default is `symmetric`) for reasons unrelated to
either implementation. Added an opt-in `--filter-window` flag, defaulting to
`scipy-periodic` so `Snakefile.legacy_diagnostics` is unaffected; the benchmark
passes `symmetric`. See verification below.

## Verification (chr2 toy interval, local)

`run_legacy_covariance.py` on the single toy partition
(`2:39967768-40067768`, EUR, Ne=11418, cutoff=1e-7), against the original
LDetect fixture `ref/cov_matrix/chr2/chr2.39967768.40067768.gz`:

```text
rows compared     : 226074      (matches the fixture row count exactly)
key mismatches    : 0
differing values  : 320 / 452148 (0.0708%)
max abs difference: 2.776e-17
max rel difference: 3.434e-16
```

`2.776e-17` is the same roundoff bound `notes/findings/ldetect-example-benchmarks.md`
records for lite against this fixture. Wall 93.7 s, peak RSS 484 MiB, output
6,209,272 bytes (fixture: 6,209,257 — the delta is float repr digits).

Full legacy chain on that driver-produced dataset, `--n-snps-bw-bpoints 50`:

- vector: 671 rows, matches the reference vector (roundoff only, e.g.
  `13.603287339218475` vs `...477`).
- `--filter-window scipy-periodic`: `fourier_ls_n=11` → **12 blocks**, does not
  match the reference.
- `--filter-window symmetric`: `fourier_ls_n=12` → **13 blocks, exact match**
  to `ref/bed/EUR-chr2-50-39967768-40067768.bed`.

That is a clean, direct empirical confirmation of the scipy-window root cause
in `notes/findings/ldetect-original-reproduction.md`, on the actual vendored
code path rather than by reasoning about scipy versions.

## Build notes

- `run_legacy_covariance.py` uses `os.posix_spawn` + `os.wait4` rather than
  `subprocess`, so rusage is attributed to the exact child without racing the
  subprocess module's reaping. The command is `bash -c 'bcftools ... | python
  P00_01 ...'`; the shell reaps both stages, so its rusage covers bcftools too.
- Per-partition timing rows are flushed incrementally — a crash mid-chromosome
  does not lose completed measurements. `--skip-existing` resumes, and the
  collector refuses to summarize a TSV containing skipped rows, since their
  runtime was never measured.
- The `bcftools -T` targets file is written to `--dataset-dir`'s *parent*, so
  the dataset directory stays a faithful replica of the layout legacy's
  flat-file loader expects.
- `collect_runtime_benchmark.py`'s `/usr/bin/time -v` parser anchors the
  wall-clock regex to end-of-line: the label itself contains colons
  (`"(h:mm:ss or m:ss)"`), which a naive `[^:]*` pattern fails on.

## Not done

Nothing has been run at scale — this is local validation on the chr2 toy
interval only. The real 22-autosome run is a remote-cluster job.
