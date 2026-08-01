# LDetect-lite

[![PyPI](https://img.shields.io/pypi/v/ldetect-lite)](https://pypi.org/project/ldetect-lite/)
[![Tests](https://github.com/adamyhe/ldetect-lite/actions/workflows/tests.yml/badge.svg)](https://github.com/adamyhe/ldetect-lite/actions/workflows/tests.yml)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/ldetect-lite?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/ldetect-lite)

A modern, fast re-implementation of [LDetect](https://bitbucket.org/nygcresearch/ldetect), a method for calculating approximately independent linkage disequilibrium (LD) blocks in the human genome. The algorithm is described in [Berisa & Pickrell, 2016](https://academic.oup.com/bioinformatics/article/32/2/283/1743626).

## Installation

LDetect-lite is available through PyPI:

```bash
pip install ldetect-lite
```

Or, with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install ldetect-lite
```

This installs three equivalent CLI entry points — `ldetect-lite`, `ldetect`, and `ldl` — so pick whichever is most convenient; examples below use `ldetect`.

The main `ldetect run` pipeline reads the VCF/BCF reference panel via [cyvcf2](https://github.com/brentp/cyvcf2), a core dependency installed automatically — no separate `tabix` binary or htslib system package is required to *run* the pipeline. However, the VCF/BCFs must be indexed before running `ldetect run` — `tabix -p vcf` (for `.vcf.gz`) or `bcftools index` (for `.bcf`), from [htslib](https://www.htslib.org/)/[bcftools](https://samtools.github.io/bcftools/) — since region-based partition reads require one.

### Development

Install from source

```bash
git clone https://github.com/adamyhe/ldetect-lite.git
cd ldetect-lite
uv sync --group dev
```

From a development checkout, run CLI commands through `uv run` so they use the managed environment.

## Usage

### End-to-end pipeline

```bash
ldetect run \
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
- `--covariance-cache {compact,full}` — partition cache schema for `ldetect run` (default: `compact`). Compact caches write only canonical position pairs, `shrink_ld`, diagonals, and lookup indexes, which is enough for restartable matrix-to-vector, metric, and local-search steps. Use `full` when debugging or when later running full-matrix/heatmap readers.
- `--covariance-compression {lzf,zstd}` — HDF5 compression codec for covariance partitions (default: `zstd`). `zstd` is smaller and faster to read/write than `lzf` at equal precision — see `docs/optimizations.md`.
- `--ld-kernel {bitpacked,uint8}` — compact covariance pair-count backend (default: `bitpacked`). `bitpacked` uses packed haplotypes and popcounts. `uint8` keeps the older array-sum backend available for reference and diagnostics; use it when requesting `--covariance-cache full`.
- `--n-snps-bw-bpoints N` — target mean number of SNPs between consecutive breakpoints; controls block granularity (default: 10000, following Berisa & Pickrell 2016). The target breakpoint count is `ceil(n_snps / N - 1)`. Mutually exclusive with `--n-bpoints`.
- `--n-bpoints N` — directly specify the number of breakpoints, bypassing the `--n-snps-bw-bpoints` formula; useful when replicating a published analysis with a known block count
- `--subset {fourier,fourier_ls,uniform,uniform_ls}` — which of the four breakpoint sets to write to the BED file (default: `fourier_ls`; see `docs/pipeline-steps.md` step 4)
- `--all-breakpoint-subsets` — compute all four breakpoint sets in the JSON output. By default, `run` computes only the requested `--subset` and its dependencies to avoid unused local-search work.
- `--workers N` — parallel workers for the pipeline (default: 1); set to the number of available cores to speed up covariance calculation (step 2) and, unless overridden below, matrix-to-vector, local search, and metric scoring as well
- `--matrix-workers N` — override parallel workers for matrix-to-vector partition processing (default: inherit `--workers`)
- `--local-search-workers N` — override parallel workers for local search (default: inherit `--workers`). Higher values can multiply RAM use because each worker loads its own covariance window.
- `--metric-workers N` — override parallel workers for streaming metric row passes during breakpoint scoring (default: inherit `--workers`)
- `--high-precision` — use 50-digit Decimal arithmetic for local search instead of the default float path (slower; mainly useful for exact reference comparisons)
- `--filter-window {scipy-periodic,symmetric}` — Hanning window mode for breakpoint filtering (default: `scipy-periodic`). `scipy-periodic` matches modern SciPy's periodic Hann window (`scipy.signal.get_window(..., fftbins=True)`) and is correct for new analyses. `symmetric` uses `np.hanning` — not a legacy fallback, but the setting needed to reproduce published reference blocks generated under scipy <1.1 (e.g. Berisa & Pickrell's original 2015 analysis), where a scipy defect made periodic odd-length windows bit-identical to symmetric ones until it was fixed in scipy 1.1.0 (2018). See `notes/findings/ldetect-original-reproduction.md` for the full root-cause writeup.
- `--filter-workers N` — minimum Numba threads for adaptive single-filter convolutions during the Step 4 filter-width search (default: 1)
- `--generate-ld-neighborhood-plot` — write a left/across/right LD-separation box plot alongside the BED output: r² distributions sampled from the covariance cache around every computed breakpoint, comparing LD within each neighboring window against pairs that cross it. A useful set of breakpoints should show lower "across" LD than "left"/"right" neighborhood LD. Self-contained sanity check — needs no reference BED. Written to `{output-dir}/{chromosome}-ld-neighborhood.svg`.
- `--ld-neighborhood-window-bp BP` — window size on each side of a breakpoint sampled for `--generate-ld-neighborhood-plot` (default: 500000)

`ldetect` manages native BLAS/OpenMP/NumExpr/Numba thread caps at CLI startup, before importing numpy/scipy/numba-backed pipeline modules, so `--workers` controls process-level parallelism without multiplying into hidden native thread pools. The default process-parallel stages pin BLAS/OpenMP-style pools to one thread. Step 4's adaptive filter-width search gets its own within-call threading via `--filter-workers`, implemented as a row-chunked `ThreadPoolExecutor` rather than Numba `prange` (see `docs/optimizations.md` #15 for why).

Each of the five stages (partition, covariance, matrix-to-vector, find-minima, extract-bpoints) can also be run individually, along with a `covariance-summary` inspection utility — see `docs/pipeline-steps.md`.

The overview command writes `pipeline-overview.svg`; the per-step schematic
command writes SVG/PDF figures under `schematics/plots/`.

### Interpolate genetic maps

Convert a recombination rate map (e.g. the [deCODE map](https://www.science.org/doi/10.1126/science.aau1043) or [HapMap-interpolated 1000G maps](https://github.com/joepickrell/1000-genomes-genetic-maps)) to per-SNP genetic positions required by steps 1 and 2:

```bash
ldetect interpolate-maps \
  --snp-file snps.bed.gz \
  --genetic-map recombination_map.gz \
  --output chr2.interpolated_genetic_map.gz
```

Arguments:
- `--snp-file PATH` — bgzipped BED file of SNP positions (columns: `chrom start end rs_id`); typically extracted from a filtered VCF with `bcftools query -f '%CHROM\t%POS0\t%POS\t%ID\n'`
- `--genetic-map PATH` — gzipped recombination map; interpolation is used to assign a cM value to each SNP position
- `--output PATH` — gzipped output map in the 3-column format expected by steps 1 and 2 (`rs_id  position  cM`)
- `--mode {point,interval,hapmap,macdonald-decode,macdonald-pyrho}` (default: `point`) — interpolation algorithm:
  - `point` — treats `--genetic-map` as discrete `(position, cM)` points and linearly interpolates between the two points bracketing each SNP. Correct for point-sampled maps (e.g. HapMap-interpolated 1000G maps).
  - `interval` — treats each map row as the start of a genomic interval with its own recombination rate (`Begin, rate_cM_Mb, cumulative_cM_at_End`). Correct for interval-rate maps such as deCODE; feeding those into `point` mode silently uses the *next* interval's rate for SNPs in the *current* interval.
  - `hapmap` — treats each map row's cM as the cumulative genetic position at that row's physical position, with the row's rate applying to the following interval. Correct for pyrho/HapMap-format maps.
  - `macdonald-decode` / `macdonald-pyrho` — compatibility modes for reproducing MacDonald et al.'s R interpolation scripts, including their dataframe/indexing conventions. Use these only for replication diagnostics; use `interval`/`hapmap` for corrected coordinates.

## Algorithm

The pipeline detects LD block boundaries by finding local minima in a smoothed diagonal-sum signal derived from the shrinkage LD covariance matrix:

1. **Partition** — chromosome split into ~5000-SNP overlapping windows at low-recombination boundaries
2. **Covariance** — Wen & Stephens shrinkage estimator applied to phased haplotypes; shrinks sample correlations toward the expected LD decay to reduce finite-sample noise
3. **Matrix → vector** — each covariance matrix reduced to a `[position, diagonal_sum]` signal; troughs correspond to LD block boundaries
4. **Find minima** — binary search for optimal Hanning-window filter width; `scipy.signal.argrelextrema` finds local minima; local search refines each breakpoint using sum of squared inter-block correlations as the quality metric
5. **Extract** — chosen breakpoint set written as BED

The available breakpoint sets are `fourier` and `uniform` (raw minima from Fourier-filtered and uniformly-spaced candidates), plus `fourier_ls` and `uniform_ls` (after local search refinement). `fourier_ls` is the recommended output.

## Known limitations

`ldetect-lite` reproduces the published Berisa & Pickrell (2016) 1000 Genomes LD blocks exactly for ASN (all 22 autosomes) and AFR (all chromosomes except chr22), and matches EUR block counts and coverage exactly but with shifted internal boundaries on chr8–chr12. These two residual divergences (EUR chr8-12, AFR chr22) likely stem from an unidentified upstream input/provenance difference from the original authors' pipeline, not a bug in this implementation — an extensive diagnostic effort ruled out VCF release-version provenance, SNP filtering, genetic map family, `Ne` assignment, duplicate/cross-partition handling, and reference-BED integrity as causes. Reproducing this result requires `--filter-window symmetric` (see above); the `scipy-periodic` default reflects the window shape the original ldetect code actually intended, and reproduces exactly what a from-scratch analysis on modern scipy would produce, but not the original 2015-era output, due to a since-fixed scipy defect. MacDonald et al. (2022) reproduction is exact for substantive deCODE/EUR boundaries and high-concordance but not exact for pyrho maps; the remaining pyrho gap is documented as local-search near-tie sensitivity plus map-desert effects, not a planned code change. See `notes/findings/ldetect-original-reproduction.md` and `notes/findings/macdonald2022-reproduction.md` for the full writeups.

## Pre-computed LD blocks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21733349.svg)](https://doi.org/10.5281/zenodo.21733349)
