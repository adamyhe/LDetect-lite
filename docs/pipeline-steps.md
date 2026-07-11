# Pipeline: individual steps

`ldetect run` (see `README.md`) chains all five stages below end-to-end and is the recommended way to run the pipeline. This doc covers running each stage individually — useful for debugging, restarting a partial run, or inspecting intermediate outputs.

Several of these stages accept their own `--workers`/`--metric-workers`. The same BLAS/OMP oversubscription risk described in `README.md` applies here too — but the automatic startup warning is only wired up in `ldetect run`, not these standalone commands, so export `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS`/`NUMBA_NUM_THREADS` yourself to match your worker count if you're driving these directly (e.g. from your own Snakefile rules) on a shared node.

The pipeline has five stages that can be run individually:

---

**Step 1 — Partition chromosome** into overlapping windows:

```bash
ldetect partition-chromosome \
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

**Step 2 — Calculate covariance** from a phased VCF/BCF partition:

```bash
ldetect calc-covariance \
  --reference-panel 1000G.chr2.vcf.gz \
  --region chr2:39967768-40067768 \
  --genetic-map chr2.interpolated_genetic_map.gz \
  --individuals eurinds.txt \
  --output cov_matrix/chr2/chr2.39967768.40067768.h5
```

This standalone step must be run once per partition. `ldetect run --workers N` runs partitions in parallel automatically in the default `--covariance-mode partition` path. For full pipeline runs, `ldetect run --covariance-mode chromosome` instead loads the chromosome genotypes once and slices each covariance partition from prepared arrays; this avoids repeated VCF/BCF region reads, currently requires the compact cache schema, and emits optional timing rows with `--profile-covariance`.

`ldetect calc-covariance` reads directly from an indexed `--reference-panel` (`.vcf.gz` with a `.tbi`, or `.bcf` with a `.csi`) via [cyvcf2](https://github.com/brentp/cyvcf2), restricting to `--region` if given. If `--reference-panel` is omitted, it instead reads VCF text from stdin with no region restriction of its own — e.g. `tabix -h 1000G.chr2.vcf.gz chr2:... | ldetect calc-covariance ...` (omitting `--reference-panel`/`--region`), useful for piping in output from arbitrary preprocessing (`bcftools view -i ...`, `zcat`, etc.) rather than a plain indexed file.

For large reference panels, prefer `.bcf`/`.csi` over `.vcf.gz`/`.tbi` — same output, but faster and lower-memory to read (see `docs/optimizations.md`).

Reads phased haplotypes and applies the [Wen & Stephens (2010)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2950123/) shrinkage estimator to compute pairwise LD. The estimator shrinks the sample correlation toward an expected decay curve based on the genetic distance between SNPs and Ne, reducing noise from finite sample sizes. Only pairs whose absolute shrinkage correlation exceeds `--cutoff` are written, keeping file sizes manageable. Output is an indexed HDF5 covariance partition (`.h5`) containing canonical SNP-position pairs, shrinkage LD values, diagonal entries, and lookup indexes. The standalone command writes the full schema, including naive LD, genetic positions, and SNP IDs; `ldetect run` defaults to the compact schema described above.

Arguments:
- `--reference-panel PATH` — VCF/BCF reference panel path, indexed with `tabix -p vcf` or `bcftools index`. If omitted, reads from stdin instead.
- `--region CHROM:START-END` — restrict to this region via an indexed fetch. Requires `--reference-panel`; omit to read the whole file/stream.
- `--genetic-map PATH` — gzipped 3-column map used to convert physical positions to genetic distances (cM) for the shrinkage estimator
- `--individuals PATH` — plain-text file with one individual ID per line; only these samples are extracted from the VCF
- `--ne FLOAT` — effective population size for the shrinkage estimator (default: 11418.0)
- `--cutoff FLOAT` — pairs with absolute shrinkage LD below this are excluded from the output (default: 1e-7)
- `--covariance-compression {lzf,zstd}` — HDF5 compression codec for the output partition (default: `zstd`). `zstd` is smaller and faster to read/write than `lzf` at equal precision — see `docs/optimizations.md`.

---

**Step 3 — Matrix to vector**:

```bash
ldetect matrix-to-vector \
  --dataset-path cov_matrix/ \
  --name chr2 \
  --output vector-chr2.txt.gz
```

Assembles all partition matrices for a chromosome and reduces them to a 1-D signal: for each SNP position, the sum of squared shrinkage correlations with all other SNPs in its window (the diagonal of the assembled correlation matrix). This produces a `[position, diagonal_sum]` vector over the full chromosome. Positions with many strong LD partners have a high diagonal sum; positions near LD block boundaries where correlations decay have a low diagonal sum. These troughs are the candidate breakpoints detected in step 4.

Arguments:
- `--dataset-path PATH` — root directory containing the partition `.h5` files and the partition list
- `--name TEXT` — chromosome name, used to locate files under `dataset-path`
- `--snp-first / --snp-last INT` — restrict the vector to a sub-range of positions (auto-detected from partition boundaries if omitted)
- `--generate-heatmap` — also write a PNG heatmap of the assembled covariance matrix alongside the output (requires `ldetect-lite[heatmap]`)
- `--workers N` — parallel workers for partition-level vector computation (default: 1)

`--generate-heatmap` requires full-schema covariance partitions. If your cache was created by the default `ldetect run` mode, rerun with `ldetect run --covariance-cache full` or create full partitions with standalone `ldetect calc-covariance`.

---

**Step 4 — Find breakpoints**:

```bash
ldetect find-minima \
  --input vector-chr2.txt.gz \
  --chr-name chr2 \
  --dataset-path cov_matrix/ \
  --n-snps-bw-bpoints 10000 \
  --output breakpoints-chr2.json
```

This is the core block-detection step. It applies a Hanning (raised cosine) smoothing filter to the diagonal-sum vector and finds local minima. The filter width is chosen by binary search: the width is increased until the number of minima matches the target breakpoint count derived from `--n-snps-bw-bpoints` (or `--n-bpoints` directly). Two initial candidate sets can be produced — `fourier` (minima from the Fourier-filtered signal) and `uniform` (minima spaced uniformly across the chromosome).

Each candidate breakpoint is then refined by a local search (`fourier_ls`, `uniform_ls`): nearby positions are evaluated using the sum of squared inter-block correlations as the quality metric. The default path uses native floats for speed; add `--high-precision` to use 50-digit Decimal arithmetic for exact reference-style comparisons. The position that minimises this metric is chosen as the final breakpoint.

By default, the standalone command computes all four breakpoint sets for backward compatibility: `fourier`, `fourier_ls`, `uniform`, `uniform_ls`. Use repeated `--subset` flags to compute only selected sets. `fourier_ls` is the recommended output.

Arguments:
- `--input PATH` — gzipped vector file from step 3
- `--chr-name TEXT` — chromosome name
- `--dataset-path PATH` — covariance matrix root directory (used by local search to load partition data)
- `--n-snps-bw-bpoints N` — target mean SNPs per block; drives the binary search for filter width (required for standalone `find-minima`; `run` defaults to 10000)
- `--n-bpoints N` — directly set the target breakpoint count, bypassing the formula (overrides `--n-snps-bw-bpoints`)
- `--trackback-delta / --trackback-step` — search range and step size for the coarse local search phase (defaults: 200 / 20)
- `--init-search-loc` — initial filter width for the binary search (default: 1000)
- `--workers N` — parallel workers for the local search phase
- `--metric-workers N` — parallel workers for streaming metric row passes (default: inherit `--workers`)
- `--high-precision` — use 50-digit Decimal arithmetic for local search and metric comparisons instead of the default float path (slower)
- `--subset {fourier,fourier_ls,uniform,uniform_ls}` — breakpoint subset to compute; repeat to compute multiple subsets. If omitted, all subsets are computed.

---

### Inspect covariance cache size

```bash
ldetect covariance-summary \
  --dataset-path cov_matrix/ \
  --name chr2 \
  --format tsv
```

Reads the partition list and covariance row counts, then reports per-partition and total row counts plus estimated memory needed by the covariance-array readers. This is useful before running local search on large chromosomes or when choosing worker counts.

Arguments:
- `--dataset-path PATH` — root directory containing the partition `.h5` files and the partition list
- `--name TEXT` — chromosome name, used to locate files under `dataset-path`
- `--snp-first / --snp-last INT` — restrict the summary to a sub-range of positions (auto-detected from partition boundaries if omitted)
- `--format {tsv,json}` — output format (default: `tsv`)
- `--output PATH` — write the summary to a file instead of stdout

---

**Step 5 — Extract to BED**:

```bash
ldetect extract-bpoints \
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
