# LDetect2

A modern reimplementation of [LDetect](https://bitbucket.org/nygcresearch/ldetect), a method for calculating approximately independent linkage disequilibrium (LD) blocks in the human genome. The algorithm is described in [Berisa & Pickrell, 2016](https://academic.oup.com/bioinformatics/article/32/2/283/1743626).

## Installation

```bash
pip install ldetect2
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
- `--cov-cutoff FLOAT` — LD pairs with shrinkage correlation below this threshold are not written to disk, reducing storage (default: 1e-7)
- `--covariance-cache {compact,full}` — partition cache schema for `ldetect2 run` (default: `compact`). Compact caches write only `i_pos`, `j_pos`, and `shrink_ld`, which is enough for restartable matrix-to-vector, metric, and local-search steps. Use `full` when debugging or when later running full-matrix/heatmap readers.
- `--n-snps-bw-bpoints N` — target mean number of SNPs between consecutive breakpoints; controls block granularity (default: 10000, following Berisa & Pickrell 2016). The target breakpoint count is `ceil(n_snps / N - 1)`. Mutually exclusive with `--n-bpoints`.
- `--n-bpoints N` — directly specify the number of breakpoints, bypassing the `--n-snps-bw-bpoints` formula; useful when replicating a published analysis with a known block count
- `--subset {fourier,fourier_ls,uniform,uniform_ls}` — which of the four breakpoint sets to write to the BED file (default: `fourier_ls`; see Step 4 below)
- `--workers N` — parallel workers for covariance calculation (default: 1); set to the number of available cores to speed up step 2 significantly
- `--local-search-workers N` — parallel workers for local search (default: 1). Higher values can multiply RAM use because each worker loads its own covariance window.

### Step-by-step

The pipeline has five stages that can be run individually:

---

**Step 1 — Partition chromosome** into overlapping windows:

```bash
uv run ldetect2 partition-chromosome \
  --genetic-map chr2.interpolated_genetic_map.gz \
  --n-individuals 379 \
  --output chr2_partitions.txt
```

The chromosome is split into overlapping windows of approximately `--window-size` SNPs (default: 5000) each. Window boundaries are placed at positions where the recombination fraction between adjacent SNPs falls below `--cutoff` (default: 1.5e-8), so that windows break at low-LD regions. Adjacent windows overlap so that SNPs near a boundary appear in two windows, preventing edge artifacts when building the covariance matrices. The output is a text file with one `start end` pair per line (base-pair positions), which drives step 2.

Arguments:
- `--genetic-map PATH` — gzipped 3-column map: `chr  position  cM`
- `--n-individuals N` — number of diploid individuals in the reference panel; used to set the recombination fraction threshold relative to the sample size
- `--window-size N` — target SNPs per window (default: 5000)
- `--ne FLOAT` — effective population size (default: 11418.0)
- `--cutoff FLOAT` — recombination fraction threshold for placing window boundaries (default: 1.5e-8)

---

**Step 2 — Calculate covariance** from a phased VCF (reads stdin):

```bash
tabix -h 1000G.chr2.vcf.gz chr2:39967768-40067768 | \
  uv run ldetect2 calc-covariance \
    --genetic-map chr2.interpolated_genetic_map.gz \
    --individuals eurinds.txt \
    --output cov_matrix/chr2/chr2.39967768.40067768.npz
```

This step must be run once per partition. `ldetect2 run --workers N` runs partitions in parallel automatically.

Reads phased haplotypes from a VCF stream and applies the [Wen & Stephens (2010)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2950123/) shrinkage estimator to compute pairwise LD. The estimator shrinks the sample correlation toward an expected decay curve based on the genetic distance between SNPs and Ne, reducing noise from finite sample sizes. Only pairs whose shrinkage correlation exceeds `--cutoff` are written, keeping file sizes manageable. Output is a compressed NumPy file (`.npz`) with arrays for SNP positions, genetic positions, naive LD, and shrinkage LD.

Arguments:
- `--genetic-map PATH` — gzipped 3-column map used to convert physical positions to genetic distances (cM) for the shrinkage estimator
- `--individuals PATH` — plain-text file with one individual ID per line; only these samples are extracted from the VCF
- `--ne FLOAT` — effective population size for the shrinkage estimator (default: 11418.0)
- `--cutoff FLOAT` — pairs with shrinkage LD below this are excluded from the output (default: 1e-7)

---

**Step 3 — Matrix to vector**:

```bash
uv run ldetect2 matrix-to-vector \
  --dataset-path cov_matrix/ \
  --name chr2 \
  --output vector-chr2.txt.gz
```

Assembles all partition matrices for a chromosome and reduces them to a 1-D signal: for each SNP position, the sum of squared shrinkage correlations with all other SNPs in its window (the diagonal of the assembled correlation matrix). This produces a `[position, diagonal_sum]` vector over the full chromosome. Positions with many strong LD partners have a high diagonal sum; positions near LD block boundaries where correlations decay have a low diagonal sum. These troughs are the candidate breakpoints detected in step 4.

Arguments:
- `--dataset-path PATH` — root directory containing the partition `.npz` files and the partition list
- `--name TEXT` — chromosome name, used to locate files under `dataset-path`
- `--snp-first / --snp-last INT` — restrict the vector to a sub-range of positions (auto-detected from partition boundaries if omitted)
- `--generate-heatmap` — also write a PNG heatmap of the assembled covariance matrix alongside the output (requires `ldetect2[heatmap]`)

`--generate-heatmap` requires full-schema covariance partitions. If your cache was created by the default `ldetect2 run` mode, rerun with `ldetect2 run --covariance-cache full` or create full partitions with standalone `ldetect2 calc-covariance`.

---

**Step 4 — Find breakpoints**:

```bash
uv run ldetect2 find-minima \
  --input vector-chr2.txt.gz \
  --chr-name chr2 \
  --dataset-path cov_matrix/ \
  --n-snps-bw-bpoints 10000 \
  --output breakpoints-chr2.json
```

This is the core block-detection step. It applies a Hanning (raised cosine) smoothing filter to the diagonal-sum vector and finds local minima. The filter width is chosen by binary search: the width is increased until the number of minima matches the target breakpoint count derived from `--n-snps-bw-bpoints` (or `--n-bpoints` directly). Two initial candidate sets are produced — `fourier` (minima from the Fourier-filtered signal) and `uniform` (minima spaced uniformly across the chromosome).

Each candidate breakpoint is then refined by a local search (`fourier_ls`, `uniform_ls`): nearby positions are evaluated using the sum of squared inter-block correlations as the quality metric, computed to 50-digit decimal precision to avoid floating-point ties. The position that minimises this metric is chosen as the final breakpoint.

The output JSON contains four breakpoint sets: `fourier`, `fourier_ls`, `uniform`, `uniform_ls`. `fourier_ls` is the recommended output.

Arguments:
- `--input PATH` — gzipped vector file from step 3
- `--chr-name TEXT` — chromosome name
- `--dataset-path PATH` — covariance matrix root directory (used by local search to load partition data)
- `--n-snps-bw-bpoints N` — target mean SNPs per block; drives the binary search for filter width (default: 10000)
- `--n-bpoints N` — directly set the target breakpoint count, bypassing the formula (overrides `--n-snps-bw-bpoints`)
- `--trackback-delta / --trackback-step` — search range and step size for the coarse local search phase (defaults: 200 / 20)
- `--init-search-loc` — initial filter width for the binary search (default: 1000)
- `--workers N` — parallel workers for the local search phase
- `--high-precision` — use 50-digit Decimal arithmetic throughout (default: only in the final metric comparison; slower)

---

**Step 5 — Extract to BED**:

```bash
uv run ldetect2 extract-bpoints \
  --name chr2 \
  --dataset-path cov_matrix/ \
  --breakpoints breakpoints-chr2.json \
  --subset fourier_ls \
  --output chr2-ld-blocks.bed
```

Reads the chosen breakpoint set from the step 4 JSON and writes a 3-column BED file (`#chr start stop`). The first block starts at the first SNP position in the partition and the last block ends at the last SNP position. Each breakpoint position becomes both the end of one block and the start of the next.

Arguments:
- `--breakpoints PATH` — JSON file from step 4
- `--subset {fourier,fourier_ls,uniform,uniform_ls}` — which breakpoint set to extract (`fourier_ls` recommended)
- `--output PATH` — output BED file; writes to stdout if omitted

---

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
- `--genetic-map PATH` — gzipped recombination map; linear interpolation is used to assign a cM value to each SNP position
- `--output PATH` — gzipped output map in the 3-column format expected by steps 1 and 2 (`rs_id  position  cM`)

## Algorithm

The pipeline detects LD block boundaries by finding local minima in a smoothed diagonal-sum signal derived from the shrinkage LD covariance matrix:

1. **Partition** — chromosome split into ~5000-SNP overlapping windows at low-recombination boundaries
2. **Covariance** — Wen & Stephens shrinkage estimator applied to phased haplotypes; shrinks sample correlations toward the expected LD decay to reduce finite-sample noise
3. **Matrix → vector** — each covariance matrix reduced to a `[position, diagonal_sum]` signal; troughs correspond to LD block boundaries
4. **Find minima** — binary search for optimal Hanning-window filter width; `scipy.signal.argrelextrema` finds local minima; local search refines each breakpoint using sum of squared inter-block correlations (50-digit decimal precision) as the quality metric
5. **Extract** — chosen breakpoint set written as BED

Four breakpoint sets are produced: `fourier` and `uniform` (raw minima from Fourier-filtered and uniformly-spaced candidates), `fourier_ls` and `uniform_ls` (after local search refinement). `fourier_ls` is the recommended output.

## Pre-computed LD blocks

Pre-computed BED files for 1000 Genomes reference populations are available from the [original ldetect repository](https://bitbucket.org/nygcresearch/ldetect-data).
