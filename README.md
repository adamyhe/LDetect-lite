# LDetect2

[![PyPI](https://img.shields.io/pypi/v/ldetect2)](https://pypi.org/project/ldetect2/)
[![Tests](https://github.com/adamyhe/ldetect2/actions/workflows/tests.yml/badge.svg)](https://github.com/adamyhe/ldetect2/actions/workflows/tests.yml)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/ldetect2?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/ldetect2)

A modern, fast re-implementation of [LDetect](https://bitbucket.org/nygcresearch/ldetect), a method for calculating approximately independent linkage disequilibrium (LD) blocks in the human genome. The algorithm is described in [Berisa & Pickrell, 2016](https://academic.oup.com/bioinformatics/article/32/2/283/1743626).

## Installation

Install from PyPI via:

```bash
pip install ldetect2
```

Or, with [uv](https://docs.astral.sh/uv/):

```bash
uv add ldetect2
```

The main `ldetect2 run` pipeline also requires [htslib](https://www.htslib.org/). Specifically, `tabix` is used to stream VCF files to `ldetect2 calc-covariance`, and so must be on PATH.

**Optional** (`--generate-heatmap`): install matplotlib with `pip install "ldetect2[heatmap]"`, or use `uv sync --extra heatmap` from a source checkout. Generating covariance heatmaps requires a matplotlib install.

### Development

Install from source

```bash
git clone https://github.com/adamyhe/ldetect2.git
cd ldetect2
uv sync --extra dev
```

From a development checkout, run CLI commands through `uv run` so they use the managed environment.

## Usage

### End-to-end pipeline

```bash
uv run ldetect2 run \
  --genetic-map chr2.interpolated_genetic_map.gz \
  --reference-panel 1000G.chr2.vcf.gz \
  --individuals eurinds.txt \
  --chromosome chr2 \
  --output-dir results/chr2/
```

This writes `results/chr2/chr2-ld-blocks.bed` — a BED file of approximately independent LD blocks.

Global options (before the subcommand):

- `-v / --verbosity {debug,info,warning,error}` — logging verbosity (default: `info`; use `warning` to silence progress messages, `debug` for full detail)

Options:

- `--ne FLOAT` — effective population size Ne used by the Wen & Stephens shrinkage estimator (default: 11418.0, the CEU/HapMap II value; reproduction configs may override this for non-European populations)
- `--cov-cutoff FLOAT` — LD pairs with absolute shrinkage correlation below this threshold are not written to disk, reducing storage (default: 1e-7)
- `--covariance-cache {compact,full}` — partition cache schema for `ldetect2 run` (default: `compact`). Compact caches write only canonical position pairs, `shrink_ld`, diagonals, and lookup indexes, which is enough for restartable matrix-to-vector, metric, and local-search steps. Use `full` when debugging or when later running full-matrix/heatmap readers.
- `--covariance-compression {lzf,zstd}` — HDF5 compression codec for covariance partitions (default: `zstd`). `zstd` is smaller and faster to read/write than `lzf` at equal precision — see `docs/optimizations.md`.
- `--n-snps-bw-bpoints N` — target mean number of SNPs between consecutive breakpoints; controls block granularity (default: 10000, following Berisa & Pickrell 2016). The target breakpoint count is `ceil(n_snps / N - 1)`. Mutually exclusive with `--n-bpoints`.
- `--n-bpoints N` — directly specify the number of breakpoints, bypassing the `--n-snps-bw-bpoints` formula; useful when replicating a published analysis with a known block count
- `--subset {fourier,fourier_ls,uniform,uniform_ls}` — which of the four breakpoint sets to write to the BED file (default: `fourier_ls`; see `docs/pipeline-steps.md` step 4)
- `--all-breakpoint-subsets` — compute all four breakpoint sets in the JSON output. By default, `run` computes only the requested `--subset` and its dependencies to avoid unused local-search work.
- `--workers N` — parallel workers for the pipeline (default: 1); set to the number of available cores to speed up covariance calculation (step 2) and, unless overridden below, matrix-to-vector, local search, and metric scoring as well
- `--matrix-workers N` — override parallel workers for matrix-to-vector partition processing (default: inherit `--workers`)
- `--local-search-workers N` — override parallel workers for local search (default: inherit `--workers`). Higher values can multiply RAM use because each worker loads its own covariance window.
- `--metric-workers N` — override parallel workers for streaming metric row passes during breakpoint scoring (default: inherit `--workers`)
- `--high-precision` — use 50-digit Decimal arithmetic for local search instead of the default float path (slower; mainly useful for exact reference comparisons)

Each of the five stages (partition, covariance, matrix-to-vector, find-minima, extract-bpoints) can also be run individually, along with a `covariance-summary` inspection utility — see `docs/pipeline-steps.md`.

### Interpolate genetic maps

Convert a recombination rate map (e.g. the [deCODE map](https://www.science.org/doi/10.1126/science.aau1043) or [HapMap-interpolated 1000G maps](https://github.com/joepickrell/1000-genomes-genetic-maps)) to per-SNP genetic positions required by steps 1 and 2:

```bash
uv run ldetect2 interpolate-maps \
  --snp-file snps.bed.gz \
  --genetic-map recombination_map.gz \
  --output chr2.interpolated_genetic_map.gz
```

Arguments:
- `--snp-file PATH` — bgzipped BED file of SNP positions (columns: `chrom start end rs_id`); typically extracted from a filtered VCF with `bcftools query -f '%CHROM\t%POS0\t%POS\t%ID\n'`
- `--genetic-map PATH` — gzipped recombination map; interpolation is used to assign a cM value to each SNP position
- `--output PATH` — gzipped output map in the 3-column format expected by steps 1 and 2 (`rs_id  position  cM`)
- `--mode {point,interval}` (default: `point`) — interpolation algorithm:
  - `point` — treats `--genetic-map` as discrete `(position, cM)` points and linearly interpolates between the two points bracketing each SNP. Correct for point-sampled maps (e.g. HapMap-interpolated 1000G maps).
  - `interval` — treats each map row as the start of a genomic interval with its own recombination rate (`Begin, rate_cM_Mb, cumulative_cM_at_End`), matching MacDonald et al.'s R interpolation scripts ([`interpolate.R`](https://github.com/jmacdon/LDblocks_GRCh38/blob/master/scripts/interpolate.R)/[`interpolate_pyhro.R`](https://github.com/jmacdon/LDblocks_GRCh38/blob/master/scripts/interpolate_pyhro.R)). Required for interval-rate maps such as the deCODE map — feeding those into `point` mode silently uses the *next* interval's rate for SNPs in the *current* interval, an off-by-one bug that produced a ~0.001–0.003 cM error per SNP in earlier testing (see `notes/findings/macdonald2022-reproduction.md`).

## Algorithm

The pipeline detects LD block boundaries by finding local minima in a smoothed diagonal-sum signal derived from the shrinkage LD covariance matrix:

1. **Partition** — chromosome split into ~5000-SNP overlapping windows at low-recombination boundaries
2. **Covariance** — Wen & Stephens shrinkage estimator applied to phased haplotypes; shrinks sample correlations toward the expected LD decay to reduce finite-sample noise
3. **Matrix → vector** — each covariance matrix reduced to a `[position, diagonal_sum]` signal; troughs correspond to LD block boundaries
4. **Find minima** — binary search for optimal Hanning-window filter width; `scipy.signal.argrelextrema` finds local minima; local search refines each breakpoint using sum of squared inter-block correlations as the quality metric
5. **Extract** — chosen breakpoint set written as BED

The available breakpoint sets are `fourier` and `uniform` (raw minima from Fourier-filtered and uniformly-spaced candidates), plus `fourier_ls` and `uniform_ls` (after local search refinement). `fourier_ls` is the recommended output.

## Known limitations

`ldetect2` reproduces the published Berisa & Pickrell (2016) 1000 Genomes LD blocks exactly for ASN (all 22 autosomes) and AFR (all chromosomes except chr22), and matches EUR block counts and coverage exactly but with shifted internal boundaries on chr8–chr12. These two residual divergences (EUR chr8-12, AFR chr22) are understood to stem from an unidentified upstream input/provenance difference from the original authors' pipeline, not a bug in this implementation — an extensive diagnostic effort ruled out VCF release-version provenance, SNP filtering, genetic map family, `Ne` assignment, duplicate/cross-partition handling, and reference-BED integrity as causes. See `notes/findings/ldetect-original-reproduction.md` for the full writeup, and `notes/findings/macdonald2022-reproduction.md` for the equivalent status reproducing MacDonald et al. (2022)'s GRCh38 blocks.

## Pre-computed LD blocks

Pre-computed BED files for 1000 Genomes reference populations are available from in hg19 coordinates from the [original LDetect data repository](https://bitbucket.org/nygcresearch/ldetect-data) and in hg38 coordinates from a more recent effort by 
