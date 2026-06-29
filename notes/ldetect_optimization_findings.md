# LDetect Optimization Findings

This document summarizes a technical audit of the [LDetect codebase](https://bitbucket.org/nygcresearch/ldetect/src/master/) (Berisa & Pickrell 2016), covering performance bottlenecks, numerical issues, and architectural improvements. It is intended as a specification for a coding agent tasked with rewriting or refactoring the pipeline.

---

## Pipeline overview

LDetect detects approximately independent linkage disequilibrium (LD) blocks in the human genome. The pipeline has five sequential steps:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `P00_00_partition_chromosome.py` | Split chromosomes into partitions by large genetic distances |
| 2 | `P00_01_calc_covariance.py` | Compute Wen–Stephens shrinkage covariance matrix from VCF stdin |
| 3 | `P01_matrix_to_vector_pipeline.py` | Convert covariance matrix to antidiagonal-sum vector |
| 4 | `P02_minima_pipeline.py` | Find minima in filtered vector (Hann low-pass + local search) |
| 5 | `P03_extract_breakpoints.py` | Write breakpoints to BED file |

Performance problems concentrate in steps 2–4. Usability problems are spread across all five.

---

## Implementation status in `ldetect2`

Several findings in this audit are now implemented experimentally in this code base:

- Direct vector mode is available through `ldetect2 run --vector-mode direct`. It computes partition-local correlation-sum vector fragments during covariance generation and merges them with the same ownership boundaries used by the full-chromosome covariance path.
- `ldetect2 run --pair-cache r2-zarr` is an opt-in experimental path that writes direct vector fragments plus a normalized float64 `r²` Zarr cache for metric and local-search. The default remains the HDF5 covariance cache.
- Full covariance/HDF5 caches are still required for compatibility, Decimal legacy paths, and matrix-to-vector comparisons unless the experimental `r²` cache is selected.

---

## Step 2: Covariance computation (critical bottleneck)

### Problem: Row-by-row VCF parsing in Python
The script reads VCF records one line at a time from stdin via `tabix` pipe. Every line triggers Python string splitting, genotype parsing, and dosage extraction with no C-level acceleration.

**Fix:** Replace with `cyvcf2` or `pysam` (C backends). Alternatively, pre-convert the reference panel to PLINK `.bed` binary format and use `bed-reader` or numpy-backed loading. Expected speedup: **5–20×**.

### Problem: O(n²) nested Python loop for shrinkage estimation
For each SNP pair (i, j), the Wen–Stephens shrinkage penalty is computed in pure Python. Even for ~2,000 SNP partitions this is very slow.

**Fix:** Express as matrix math. Compute dosage matrix `G` (shape: n_snps × n_samples), then:
```python
cov_empirical = (G @ G.T) / n_samples          # BLAS-backed
D = np.abs(positions[:, None] - positions[None, :])  # pairwise distances
S = np.exp(-rho * D / (2 * Ne))                # shrinkage weights
cov_shrunk = cov_empirical * S
```
Expected speedup: **10–100×** over the pure Python loop.

### Problem: No intra-partition parallelism
Parallelism requires manual shell scripting across SGE/SLURM. Single-machine multi-core use is undocumented.

**Fix:** Add `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor` inside the covariance script. Expected speedup: **linear in available cores**.

### Problem: Hard FTP dependency on 1000 Genomes Phase 1
The example pipeline streams directly from EBI FTP (`tabix -h ftp://...`), which is slow, unreliable, and uses outdated data.

**Fix:** Document local download + tabix index workflow. Support modern reference panels (1KG Phase 3, HGDP, UK Biobank).

### Note: Numba for the shrinkage loop
`@numba.njit(parallel=True)` with `prange` is viable for the shrinkage penalty application (tight numerical loop over a 2D array). Expected gain over vectorized NumPy: **2–4×**, and avoids the large intermediate `D` matrix allocation. Requires `cache=True` to avoid per-run JIT compilation overhead. Do **not** use Numba for `G @ G.T` — NumPy already calls optimized BLAS there.

---

## Step 3: Matrix-to-vector conversion

### Problem: `decimal` module for antidiagonal accumulation
The code uses Python's `decimal` module for high-precision arithmetic when summing antidiagonals. `decimal` is 50–200× slower than native float64 and bypasses all NumPy vectorization. The precision is unnecessary: values are shrinkage-regularized correlations in [-1, 1] with no catastrophic cancellation risk, and the downstream Hann filter washes out any float64 rounding error.

**Fix:** Replace with NumPy trace-based computation:
```python
n = matrix.shape[0]
vector = np.array([np.trace(matrix, k) for k in range(-(n-1), n)])
```
Or use `math.fsum` per antidiagonal if extra caution is desired. Expected speedup: **50–100×**.

### Problem: float32 is not a useful optimization here
The matrix-to-vector step is memory-bandwidth-bound, not compute-bound. float32 does not improve memory bandwidth on CPU. float32 also carries genuine precision risk when summing thousands of small covariance values. **Do not use float32** for this step.

### Problem: Full matrix materialized in memory
Peak memory up to ~100GB for large chromosomes, preventing multi-chromosome parallelization.

**Preferred fix:** Eliminate the persisted matrix entirely — see "Recommended architectural change" below.

**Secondary fix:** If the matrix must be stored, use Zarr with chunked storage (see Step 2/3 storage section).

---

## Step 4: Minima computation

### Problem: Known bug in uniform-minima path
Lines 103–105 of `P02_minima_pipeline.py` crash when computing the metric for uniform breakpoints with local search. The README vaguely hints that the file "can be tweaked" but does not explain what to change or why. This bug has been independently discovered by multiple downstream users (e.g., `LDblocks_GRCh38` project).

**Fix:** Remove the uniform-minima path entirely, or fix the metric computation. The `fourier_ls` (Fourier low-pass with local search) path is what virtually all users need.

### Problem: Four algorithm variants computed unconditionally
The pipeline computes uniform, uniform+local-search, Fourier, and Fourier+local-search breakpoints on every run. The last variant is what users actually use.

**Fix:** Make `fourier_ls` the default and only required output. This cuts Step 4 runtime by ~75%.

### Problem: Intermediate `.pickle` files
Minima are serialized as Python pickles, which are version-sensitive, opaque, and not restartable across Python versions.

**Fix:** Replace with CSV or HDF5/Zarr checkpoints.

---

## Storage: gzipped TSVs, HDF5, and Zarr

### Problems with current gzipped TSV format
- Full sequential decompression required before any data is usable
- String-to-float parsing for every value
- Filename encodes metadata (start/stop coordinates) — fragile, causes silent failures if misnamed
- No support for partial/streaming reads
- gzip compression cannot exploit matrix structure

### Why Zarr may be preferable for new caches
- Chunks stored as individual files — multiple processes can read different partitions simultaneously without file handle contention or locking
- Lazy loading without materialization (arrays behave like NumPy arrays but don't load until indexed)
- Native Blosc support (`blosc:lz4` for speed, `blosc:zstd` for ratio) — 3–4× faster decompression than gzip with better compression ratios on smooth correlation matrices
- Simpler API for this use case than parallel HDF5

In `ldetect2`, Zarr is not treated as a blanket HDF5 replacement. Existing HDF5 covariance partitions remain the default compatibility format because they store shrinkage covariance rows and support the legacy Decimal paths. Zarr is currently used for the new normalized pair cache described below.

### Experimental minimal `r²` Zarr cache

The experimental cache stores normalized pair rows rather than raw covariance/shrinkage rows:

```
<dataset>/<chrom>.r2.zarr/
  partitions/
    <start>_<end>/
      positions      # physical SNP positions for the partition
      lo_values      # lower endpoints represented in row order
      lo_offsets     # offsets into hi_idx/r2 for each lo value
      hi_idx         # upper endpoint as index into positions
      r2             # float64 normalized r² values
```

Partition attrs include `format=ldetect2-r2-zarr`, `version=1`, `chrom`, `start`, `end`, `ne`, `cutoff`, `n_pairs`, and `position_dtype`.

Exactness notes:
- Rows are canonical and sorted as `(lo, hi)`.
- Rows whose endpoints lack positive diagonal shrinkage are skipped.
- Every positive-diagonal SNP is stored explicitly as `(lo_idx == hi_idx, r2 = 1.0)`. This preserves the local-search locus bookkeeping used by the HDF5 path.
- Metric ignores diagonal rows with `lo == hi`; local search consumes them because diagonal rows define loci and contribute to the same vertical/horizontal bookkeeping as the original normalized covariance path.
- Overlapping partitions are merged with the same ownership/deduplication boundaries as the full-chromosome behavior.

Tradeoffs:
- The cache is much smaller than storing full covariance metadata because it stores only `hi_idx` plus normalized `r²`.
- It cannot support the legacy Decimal covariance path because raw shrinkage values are intentionally not retained.
- It is experimental and opt-in via `--pair-cache r2-zarr`; HDF5 remains the default.

### Recommended Zarr store layout
```
chr2.zarr/
  .zattrs              # chromosome, n_samples, Ne, cov_cutoff, genome_build
  partitions/
    0/
      .zattrs          # start, stop, n_snps, snp_ids
      vector/          # antidiagonal-sum vector (preferred — see below)
      matrix/          # full covariance matrix (optional / avoid if possible)
    1/
      ...
```

### Chunk shape guidance
- For square matrix storage: 256×256 float64 chunks as a starting point
- Antidiagonal access is worst-case for any chunk shape — each antidiagonal intersects ~2n/c chunks for chunk size c, with utilization ~c/(2n) per chunk
- Diagonal-oriented chunks (e.g., 16×4096) reduce intersections for antidiagonal access but hurt other patterns
- **Best solution: don't store the matrix at all** (see below)

---

## Recommended architectural change: eliminate matrix materialization

### Core insight
The Wen–Stephens covariance matrix entry `C[i,j]` contributes to exactly one antidiagonal of the vector, at offset `k = j - i`. The matrix never needs to be fully materialized — the vector can be accumulated directly during covariance computation.

In the current `ldetect2` implementation, direct vector mode uses the existing Numba pairwise kernel style rather than the `np.einsum` sketch below. That keeps the Wen–Stephens cutoff, diagonal handling, and ownership-boundary behavior aligned with the covariance implementation while avoiding full matrix materialization.

### Implementation

The antidiagonal sum at offset k is:
```
trace(C, k) = sum_i C[i, i+k]
            = sum_i (G[i,:] · G[i+k,:]) * shrinkage[i, i+k]
```

This is a batched dot product between rows of G offset by k:

```python
# Precompute
G_norm = dosages - dosages.mean(axis=1, keepdims=True)
G_norm /= G_norm.std(axis=1, keepdims=True)
n = n_snps
vector = np.zeros(2 * n - 1)

# Precompute genetic distances by diagonal offset (1D, length n)
# distances_by_offset[k] = array of genetic distances for SNP pairs (i, i+k)

for k in range(n):
    # Batched dot products for all pairs at offset k
    dots = np.einsum('ij,ij->i', G_norm[:n-k], G_norm[k:]) / n_samples
    weights = np.exp(-rho * distances_by_offset[k] / (2 * Ne))
    antidiag_sum = (dots * weights).sum()
    vector[k + n - 1] = antidiag_sum
    if k > 0:
        vector[-k + n - 1] = antidiag_sum  # symmetric matrix
```

### Memory profile of new approach

| Quantity | Size | Notes |
|----------|------|-------|
| Dosage matrix G | O(n_snps × n_samples) | Irreducible — needed for all computations |
| Genetic distance offsets | O(n_snps) | 1D precomputed array |
| Working buffer (einsum) | O(n_snps) | One antidiagonal at a time |
| Output vector | O(n_snps) | Final output |
| **Covariance matrix** | **0** | **Never materialized** |

Peak memory for typical partition (2000 SNPs, 2500 samples): ~40MB. No quadratic memory term.

### Trade-offs
- Loses the `G @ G.T` BLAS matrix-multiply path (replaced by `einsum` per diagonal)
- `einsum` is well-optimized but not as aggressively tuned as BLAS `dgemm`
- Difference is modest in practice — shrinkage weight computation dominates runtime

### Impact on parallelization
With no quadratic memory term, multi-chromosome parallelization becomes straightforward:
- Each worker holds one chromosome's dosage matrices
- Peak memory per worker is bounded and predictable
- Workers write vectors directly to Zarr — no large intermediate files

---

## Packaging and dependency issues

| Issue | Fix |
|-------|-----|
| `commanderline` dependency (obscure, not on conda-forge) | Replace with `argparse` from stdlib |
| `setup.py` using deprecated `distutils` | Migrate to `pyproject.toml` with `setuptools` or `flit` |
| Not available on conda-forge or bioconda | Add conda recipe; document `tabix`/htslib as system dependency |
| No test suite | Add pytest suite running the full pipeline on the included example data (chr2 small region) |
| No progress reporting in Step 2 (can run for days) | Add periodic logging with SNP count, partition, and elapsed time |

---

## Output correctness issues

| Issue | Location | Fix |
|-------|----------|-----|
| `None` string values in BED output | Step 5 / ldetect-data repo | Catch and handle during BED writing; replace with midpoint coordinate |
| Metadata inferred from filename | All steps | Store in Zarr `.zattrs` or a manifest TSV |
| hg19 / Phase 1 hardcoded throughout | Steps 1–2 | Parameterize genome build and effective population size; document GRCh38 workflow |

---

## Prioritized implementation order

1. **Eliminate matrix materialization** — accumulate vector directly during covariance computation (resolves 100GB memory pressure and unblocks parallelization)
2. **Add normalized `r²` cache** — implemented experimentally with the aggressive Zarr partition-row cache for fast metric/local-search comparisons without HDF5 covariance reads
3. **Vectorize covariance inner loop** with NumPy broadcasting; optionally add Numba `@njit(parallel=True)` for shrinkage application (10–100× speedup)
4. **Replace VCF parsing** with `cyvcf2` (5–20× speedup)
5. **Remove `decimal`** from matrix-to-vector step; use `np.trace` (50–100× speedup — moot if matrix is eliminated, but relevant if kept as fallback)
6. **Fix or remove uniform-minima bug** in Step 4; make `fourier_ls` the only required output (75% reduction in Step 4 runtime)
7. **Adopt Zarr** where it fits new normalized/cache-oriented storage with metadata in `.zattrs`
8. **Add `multiprocessing.Pool`** for per-partition parallelism within a chromosome
9. **Replace `commanderline`** with `argparse`; migrate to `pyproject.toml`
10. **Add Snakemake or Nextflow workflow** to encode step dependencies and naming conventions explicitly
11. **Add progress logging**, pytest suite, and GRCh38 documentation
