# ldetect2 Optimization Handoff

Date: 2026-07-01

This is the active engineering handoff for optimization work. Keep this file
practical: current state, open questions, profiling targets, and commands.
Narrative summary belongs in `notes/optimizations.md`.

## Current Branch State

Primary branch: `further-optimizations`

Current benchmark modes:

| Mode | Purpose | Status |
| --- | --- | --- |
| `matrix_hdf5` | compact HDF5 covariance plus matrix-to-vector baseline | compatibility baseline |
| `direct_hdf5` | compact HDF5 covariance plus fused direct vector fragments | exactness mostly fixed; vector-value residuals on chr9/14 |
| `r2_zarr` | direct vector plus normalized r2 Zarr pair cache | fastest recent path; experimental |
| `r2_nocache` | recompute r2 without a pair cache | size experiment; not runtime-competitive |

The recommended comparison workflow is `examples/r2_zarr_exactness`.

## Current Recommendation

Use r2-zarr for runtime-oriented experimental runs:

```text
ldetect2 run --subset fourier_ls --covariance-cache compact --pair-cache r2-zarr
```

Keep compact HDF5 as the baseline for correctness, raw shrinkage-row
compatibility, and matrix-to-vector comparisons.

## Current Exactness Status

Resolved:

- direct/r2 vector centers use active matrix loci instead of raw VCF SNP
  indexes;
- final direct vector support matches matrix behavior when the chromosome tail
  contains inactive SNPs;
- chr20/21/22 vector key support now matches baseline;
- chr15 final breakpoints and BED match baseline;
- r2-zarr final `fourier_ls` BED outputs match baseline in the latest
  downloaded summaries.

Open:

- chr9 and chr14 still show direct/r2 vector-value differences against
  matrix-HDF5 despite matching vector keys;
- chr9 max vector absolute difference remains about `0.114`;
- chr14 max vector absolute difference remains about `4.5e-06`;
- latest summaries show no downstream `fourier_ls` BED divergence from these
  vector differences, but the source should still be understood.

Important: do not replace the fused direct-vector kernel with canonical
row-stream accumulation as a "fix." That removes the main direct-vector
optimization advantage. Use diagnostics first.

## New Diagnostic Artifact

The exactness workflow now writes:

```text
results/compare/{POP}.{chrom}.vector_diffs.tsv
```

It reports the largest vector differences for each mode vs baseline:

- comparison;
- rank;
- position;
- baseline/mode presence;
- baseline/mode value;
- absolute and relative difference;
- nearest baseline partition boundary and distance.

Use this file to localize chr9/chr14 residual vector differences. The summary
`exactness.tsv` is no longer enough.

## r2-Zarr Cache State

Current design:

- Zarr v2;
- float64 normalized r2 values;
- implicit diagonal rows via `diag_idx`;
- upper endpoints stored as `hi_delta = hi_idx - lo_idx`;
- chromosome-level owned-pair cache removes overlap duplication while
  preserving first-pair precedence;
- HDF5 covariance cache remains the default compatibility path.

Do not restore the older explicit-diagonal/duplicated-overlap schema unless a
profile proves the current compact schema is incorrect.

## Nocache State

r2-nocache has a strong cache-size story but weak runtime:

- it avoids large pair-cache writes;
- it recomputes pair information repeatedly enough to be much slower in whole
  runs;
- bounded in-memory window/pair caching would be required to make it
  competitive, and that is a rewrite-scale project.

Keep nocache separate from the main r2-zarr merge path.

## Runtime Status

Recent chr20-22 exactness workflow runtimes:

| Chrom | matrix_hdf5 | direct_hdf5 | r2_zarr |
| --- | ---: | ---: | ---: |
| chr20 | 465.45 s | 493.18 s | 384.73 s |
| chr21 | 281.09 s | 300.71 s | 222.81 s |
| chr22 | 331.06 s | 333.12 s | 199.65 s |

r2-zarr is the fastest of these modes in the current comparison set.

Older compact-HDF5 optimization results remain useful context:

- single-pass compact HDF5 writer removed the compact count-then-generate pass;
- whole-run RSS stayed below 1 GiB on downloaded chr10/11/13/21/22 profiles;
- `matrix_workers=4` and `metric_workers=4` are useful bounded worker knobs;
- `local_search_workers=4` inflates RSS and should not be the default.

## Next Investigation

For chr9/chr14 vector-value residuals:

1. Rerun/download `vector_diffs.tsv`.
2. Inspect the top loci for both `direct_hdf5_vs_matrix_hdf5` and
   `r2_zarr_vs_matrix_hdf5`.
3. Check whether the top loci cluster:
   - near partition starts/ends;
   - near duplicate physical positions;
   - near inactive/zero-diagonal SNP stretches;
   - in one or a few partitions only.
4. If clustered by partition, add a focused test using synthetic overlapping
   partitions before touching the direct kernel.
5. If clustered by duplicate positions, compare direct active-index mapping
   against compact HDF5 canonical row ordering.
6. If not clustered, add row-level instrumentation around the top locus only.

Do not infer another global kernel change from max-diff summaries alone.

## Validation Commands

Fast local checks:

```text
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest -m "not integration"
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run ruff check
git diff --check
```

Focused checks for the exactness workflow diagnostics:

```text
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run pytest tests/test_r2_zarr_exactness_compare.py -q
cd examples/r2_zarr_exactness
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache uv run snakemake -n results/compare/EUR.22.vector_diffs.tsv
```

Remote validation order:

1. chr14, because the vector residual is small and likely faster to inspect;
2. chr9, because it has the large residual;
3. chr20-22 sanity rerun only if the diagnostic/script changes need workflow
   validation.

## Do Not Reintroduce

- chromosome-scale resident covariance arrays;
- row-stream vector accumulation as the default direct-vector path;
- r2-nocache as the main runtime path;
- local-search per-breakpoint multiprocessing as default;
- speculative duplicate merge rewrites without remote profiling;
- broad Zarr replacement of HDF5 covariance compatibility storage.
