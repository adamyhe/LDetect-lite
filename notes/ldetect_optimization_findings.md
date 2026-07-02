# Historical LDetect Optimization Audit

This file preserves the original audit themes that motivated the optimization
work. It is no longer the active implementation plan. For current status, use:

- `notes/optimizations.md` for the human summary;
- `notes/optimizations-handoff.md` for the active engineering runbook.

## Original Bottlenecks

The legacy LDetect workflow had five major steps:

| Step | Legacy script | Main issue |
| --- | --- | --- |
| 1 | `P00_00_partition_chromosome.py` | global path/config conventions |
| 2 | `P00_01_calc_covariance.py` | Python VCF parsing and pairwise LD loops |
| 3 | `P01_matrix_to_vector_pipeline.py` | full covariance matrix materialization |
| 4 | `P02_minima_pipeline.py` | computes unused variants and expensive local search |
| 5 | `P03_extract_breakpoints.py` | fragile intermediate naming/output handling |

The largest problems were memory blowups from materialized covariance matrices,
slow text I/O, repeated row parsing, and unconditional computation of outputs
that most users did not need.

## What Was Implemented In ldetect2

Implemented or partially implemented:

- argparse-based CLI and `ldetect2 run`;
- compact indexed HDF5 covariance partitions;
- Numba-backed Wen/Stephens pairwise LD kernels;
- process-pool partition generation;
- bounded matrix-to-vector, metric, and local-search readers;
- direct vector fragments via `--vector-mode direct`;
- opt-in normalized r2 Zarr cache via `--pair-cache r2-zarr`;
- cyvcf2-backed decoding where file-backed VCF/BCF access is available;
- JSON breakpoint output instead of pickle;
- focused real-data exactness/runtime Snakemake workflow in
  `examples/r2_zarr_exactness`.

## Current Architecture Decisions

### HDF5 Is Still The Compatibility Baseline

Compact HDF5 remains the default because it stores raw shrinkage covariance
rows and supports compatibility/debugging paths. It is bounded and indexed, so
it no longer implies full matrix materialization.

### Zarr Is Only For The r2 Pair Cache

Zarr is not a blanket HDF5 replacement. The useful current Zarr role is the
normalized r2 pair cache:

```text
<dataset>/<chrom>.r2.zarr/
  partitions/<start>_<end>/
    positions
    lo_values
    lo_offsets
    hi_delta
    r2
    diag_idx
```

This cache is float64, opt-in, and experimental. It stores normalized r2 rows
for metric/local-search, with implicit diagonals and a chromosome-level owned
pair cache to avoid overlap duplication.

### Direct Vector Is A Fused Kernel, Not A Row-Stream Replacement

Direct vector mode should preserve its main optimization: accumulate vector
values directly during pairwise LD work and avoid materializing row caches just
to build the vector. Row-stream accumulation can be a diagnostic or fallback,
but it should not become the main direct-vector path without profiling.

## Updated Lessons From The Audit

- `np.einsum` was not adopted as the production direct-vector kernel. The
  current direct path uses the existing Wen/Stephens pairwise kernel style to
  keep cutoff, diagonal, and active-locus behavior aligned with covariance
  generation.
- float32 is still not recommended for r2/vector/cache exactness work.
- full covariance matrix materialization remains the thing to avoid.
- r2-nocache proved that cache size can be minimized, but runtime was not
  competitive with r2-zarr.
- duplicate physical positions and overlapping partition ownership are the
  recurring exactness hazards.

## Historical Recommendations That Changed

Some original audit recommendations have been superseded:

- "Use Zarr for matrices" became "use Zarr for normalized r2 rows only."
- "Eliminate matrix materialization entirely" remains directionally correct,
  but HDF5 compatibility is still needed as the baseline.
- "Replace matrix-to-vector with trace/einsum" was replaced by bounded HDF5
  matrix-to-vector plus direct vector mode.
- "Make r2 nocache the minimal storage path" was rejected for runtime.

## Remaining Open Question

The current exactness workflow has resolved vector key-support mismatches, but
chr9 and chr14 still show vector-value residuals in direct/r2-zarr versus
matrix-HDF5. Final `fourier_ls` breakpoints and BED files were exact in the
latest downloaded summaries.

Use the current diagnostic artifact:

```text
examples/r2_zarr_exactness/results/compare/{POP}.{chrom}.vector_diffs.tsv
```

to locate the top differing loci before changing any kernels.
