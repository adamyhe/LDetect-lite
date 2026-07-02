# ldetect2 Optimization Summary

This is the current human-readable optimization summary. Agent-facing runbook
details and next-step profiling instructions live in
`notes/optimizations-handoff.md`. The original audit is preserved as historical
context in `notes/ldetect_optimization_findings.md`.

## Current Recommendation

For production-style runs, the best validated path is:

```text
ldetect2 run --subset fourier_ls --covariance-cache compact --pair-cache r2-zarr
```

`r2_zarr` is still experimental and opt-in, but it has been the fastest path in
the recent real-data comparisons while keeping final breakpoint/BED outputs
equivalent to the compact-HDF5 baseline. The default HDF5 path remains the
compatibility baseline because it stores raw shrinkage covariance rows and
supports all legacy/debug readers.

## What Changed

### Compact HDF5 Baseline

The main baseline moved from materialized text/matrix intermediates to compact,
indexed HDF5 partitions. Compact HDF5 stores canonical `(lo, hi)` rows plus
`shrink_ld`, with indexes for diagonals and lower-endpoint row ranges. This
lets matrix-to-vector, metrics, and local search stream bounded row chunks
instead of loading chromosome-scale covariance arrays.

Key properties:

- partition/chunk-bounded RSS;
- restartable intermediate files;
- shared reader semantics for matrix-to-vector, metric, and local search;
- compatibility with raw covariance/shrinkage debugging.

### Direct Vector Mode

`--vector-mode direct` computes the matrix-to-vector signal during covariance
generation without materializing a covariance matrix. It keeps the fast fused
pairwise accumulation path and writes per-partition vector fragments.

Important exactness notes:

- center positions are based on active matrix loci, not raw VCF SNP indexes;
- final partition support matches matrix flushing when trailing SNPs are
  inactive;
- duplicate-position behavior is still an exactness-sensitive area and should
  be tested with real vector-difference diagnostics.

Recent comparisons show matching vector key sets after the boundary fixes.
Residual chr9/chr14 vector-value differences remain under investigation; final
`fourier_ls` breakpoints and BED outputs were exact in the latest downloaded
compare files. The benchmark workflow now emits `vector_diffs.tsv` to identify
the exact loci driving those residual vector differences.

### r2-Zarr Pair Cache

`--pair-cache r2-zarr` writes normalized float64 `r²` rows to a Zarr v2 cache
and uses that cache for metric/local-search. This avoids retaining raw
covariance metadata when downstream stages only need normalized pair values.

Current layout:

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

The current space-saving format stores diagonal rows implicitly through the
positive-diagonal index and stores upper endpoints as `hi_delta = hi_idx -
lo_idx`. A chromosome-level owned-pair cache removes overlap duplication while
preserving first-pair precedence. The earlier fully explicit row layout was
larger and is no longer the target schema.

Why Zarr here:

- it fits the normalized pair-cache use case;
- it avoids treating HDF5 as a blanket replacement target;
- it has been faster than matrix/direct HDF5 in recent exactness benchmark
  runs;
- it is opt-in while the HDF5 baseline remains the compatibility path.

### Nocache Mode

The r2-nocache path proved useful as a size/RSS experiment, but it is not
competitive with r2-zarr for runtime. It avoids large pair-cache output, but it
recomputes enough pair information that whole-run time can be much higher.

Current posture:

- keep nocache as experimental/research code if needed;
- do not treat it as the primary performance path;
- prefer r2-zarr when runtime matters and HDF5 compatibility is not required.

### VCF/BCF I/O

cyvcf2 support was added to accelerate VCF/BCF decoding in paths that can use
file-backed input. `tabix` streaming remains available for compatibility with
the existing end-to-end partition workflow.

## Recent Real-Data Status

The latest downloaded exactness summaries show:

- chr20/21/22 vector key support now matches across matrix, direct HDF5, and
  r2-zarr;
- chr15 key support and final outputs match;
- chr9 and chr14 still have vector-value max differences in direct/r2_zarr vs
  matrix, but downstream `fourier_ls` breakpoints and BED outputs were exact in
  the downloaded summaries;
- r2-zarr was fastest in the updated chr20-22 runtime files.

Example updated runtimes:

| Chrom | matrix_hdf5 | direct_hdf5 | r2_zarr |
| --- | ---: | ---: | ---: |
| chr20 | 465.45 s | 493.18 s | 384.73 s |
| chr21 | 281.09 s | 300.71 s | 222.81 s |
| chr22 | 331.06 s | 333.12 s | 199.65 s |

## Optimization Lessons

- Preserve the compact HDF5 path as the correctness and compatibility baseline.
- Keep r2-zarr experimental but prioritized for runtime comparisons.
- Do not replace the fused direct-vector kernel with row-stream accumulation
  unless profiling shows the optimization no longer matters.
- Avoid r2-nocache as the main runtime path; its cache-size win is outweighed
  by recomputation cost.
- Add diagnostics before changing exactness-sensitive kernels. Summary max
  differences are not enough; use `vector_diffs.tsv` to locate the actual loci.
- Treat duplicate-position handling and partition-boundary ownership as
  correctness-critical.

## Historical or Rejected Paths

- Full covariance matrix materialization is no longer acceptable for large
  chromosomes.
- Full HDF5 covariance caches remain useful for compatibility/debugging but are
  not the preferred speed path.
- Earlier r2-zarr schemas with explicit diagonal rows and duplicated overlap
  rows were replaced by the smaller implicit-diagonal/owned-pair design.
- r2-nocache is not expected to catch r2-zarr runtime without a much deeper
  in-memory/window-cache redesign.
- Speculative local-search micro-optimizations should not be reintroduced
  without fresh profiling evidence.
