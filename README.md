# LDetect2

A modern reimplementation of [LDetect](https://bitbucket.org/nygcresearch/ldetect), a method for calculating approximately independent linkage disequilibrium (LD) blocks in the human genome. The algorithm is described in [Berisa & Pickrell, 2016](https://academic.oup.com/bioinformatics/article/32/2/283/1743626).

## Installation

```bash
pip install ldetect2
```

Or from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone ...
uv sync
```

**Optional**: install [numba](https://numba.pydata.org/) for a ~50x speedup on the covariance calculation step (included as a dependency by default).

**Required for `ldetect2 run`**: [htslib](https://www.htslib.org/) (`tabix` must be on PATH).

## Usage

### End-to-end pipeline

```bash
ldetect2 run \
  --genetic-map chr2.interpolated_genetic_map.gz \
  --reference-panel 1000G.chr2.vcf.gz \
  --individuals eurinds.txt \
  --chromosome chr2 \
  --output-dir results/chr2/
```

This writes `results/chr2/chr2-ld-blocks.bed` — a BED file of approximately independent LD blocks.

Options:
- `--ne FLOAT` — effective population size (default: 11418.0, tuned for Europeans)
- `--cov-cutoff FLOAT` — LD pairs below this are excluded (default: 1e-7)
- `--n-snps-bw-bpoints N` — target mean SNPs between breakpoints (default: 50)
- `--subset {fourier,fourier_ls,uniform,uniform_ls}` — breakpoint set for output (default: `fourier_ls`)

### Step-by-step

The pipeline has five stages that can be run individually:

**Step 1 — Partition chromosome** into overlapping windows:

```bash
ldetect2 partition-chromosome \
  --genetic-map chr2.interpolated_genetic_map.gz \
  --n-individuals 379 \
  --output scripts/chr2_partitions
```

**Step 2 — Calculate covariance** from a phased VCF (reads stdin):

```bash
tabix -h 1000G.chr2.vcf.gz chr2:39967768-40067768 | \
  ldetect2 calc-covariance \
    --genetic-map chr2.interpolated_genetic_map.gz \
    --individuals eurinds.txt \
    --output cov_matrix/chr2/chr2.39967768.40067768.gz
```

**Step 3 — Matrix to vector**:

```bash
ldetect2 matrix-to-vector \
  --dataset-path cov_matrix/ \
  --name chr2 \
  --output vector-chr2.txt.gz
```

**Step 4 — Find breakpoints**:

```bash
ldetect2 find-minima \
  --input vector-chr2.txt.gz \
  --chr-name chr2 \
  --dataset-path cov_matrix/ \
  --n-snps-bw-bpoints 50 \
  --output breakpoints-chr2.json
```

**Step 5 — Extract to BED**:

```bash
ldetect2 extract-bpoints \
  --name chr2 \
  --dataset-path cov_matrix/ \
  --breakpoints breakpoints-chr2.json \
  --subset fourier_ls \
  --output chr2-ld-blocks.bed
```

### Interpolate genetic maps

Convert a recombination rate map (e.g. from [1000 Genomes genetic maps](https://github.com/joepickrell/1000-genomes-genetic-maps)) to per-SNP genetic positions:

```bash
ldetect2 interpolate-maps \
  --snp-file snps.bed \
  --genetic-map genetic_map_chr2_combined_b37.txt.gz \
  --output chr2.interpolated_genetic_map.gz
```

## Algorithm

The pipeline detects LD block boundaries by finding local minima in a smoothed diagonal-sum signal derived from the shrinkage LD covariance matrix:

1. **Partition** — chromosome split into ~5000-SNP windows with low-recombination boundaries
2. **Covariance** — Wen & Stephens shrinkage estimator applied to phased haplotypes from a VCF reference panel
3. **Matrix → vector** — each covariance matrix reduced to a `[position, diagonal_sum]` signal
4. **Find minima** — binary search for optimal Hanning-window width; `scipy.signal.argrelextrema` finds local minima; local search refines each breakpoint; quality metric is sum of squared correlations across blocks (50-digit decimal precision)
5. **Extract** — chosen breakpoint set written as BED

Four breakpoint sets are produced: `fourier` and `uniform` (raw minima), `fourier_ls` and `uniform_ls` (after local search refinement). `fourier_ls` is the recommended output.

## Pre-computed LD blocks

Pre-computed BED files for 1000 Genomes reference populations are available from the [original ldetect repository](https://bitbucket.org/nygcresearch/ldetect-data).
