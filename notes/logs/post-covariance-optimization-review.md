# Post-Covariance Optimization Review (THIS NOTE IS STALE)

**Agent-oriented working log.** Raw, dated investigation notes — not proofread for external readability. For current, human-readable status, see `notes/findings/`.

Date: 2026-05-03

## Context

The reproduction pipelines can now accurately reproduce the original LDetect
paper LD blocks, but the stages after covariance calculation remain slow and
I/O-heavy:

1. `MatrixAnalysis.calc_diag_lean()` converts covariance partitions to the
   correlation-sum vector and writes `vector-{chrom}.txt.gz`.
2. `find_breakpoints()` reads that vector, finds raw minima, computes metrics,
   runs local search, recomputes metrics, and writes breakpoint JSON/BED.
3. Normal float metrics use array-backed covariance loading; high precision
   still uses the legacy dictionary path.
4. Multi-partition local search currently falls back to the dictionary path
   because the array-backed implementation was shown to diverge from legacy
   behavior.

## Low-Risk Change Implemented

`find_breakpoints()` used to load all covariance partitions and build the
array-backed `r²` representation four times:

- raw Fourier metric;
- raw uniform metric;
- Fourier after local search;
- uniform after local search.

This was redundant. The pipeline now loads the chromosome-wide covariance
arrays once for the two pre-local-search metrics, releases them during local
search, then reloads once for the two post-local-search metrics.

This reduces metric partition reads and `r²` construction from four passes to
two passes while avoiding peak memory stacking during local search.

Files changed:

```text
src/ldetect2/pipeline.py
```

## Major Remaining Bottlenecks

### 1. Matrix-to-vector dictionary scan

`MatrixAnalysis.calc_diag_lean()` reads covariance partitions into nested Python
dictionaries, then for each locus scans outward through neighboring loci. This
is slow and object-heavy.

The vector contribution can likely be computed directly from arrays:

- For each covariance row `(i_pos, j_pos)`, map endpoints to partition-local SNP
  indices.
- The current legacy loop contributes that pair to the center locus with index
  `floor((i_index + j_index) / 2)`.
- Sum `r²` into that center locus with `np.bincount` or `np.add.at`.
- Apply the same partition center ownership as legacy
  `calc_diag_lean()`:
  - non-final partition output through
    `int((partition_end + next_partition_start) / 2)`;
  - final partition output through `snp_last`.

This would avoid building nested dictionaries and could also make it possible
to keep the vector in memory rather than writing and rereading
`vector-{chrom}.txt.gz`.

Risk: overlap ownership must match legacy exactly. The existing integration
test against the toy pipeline should be the first correctness gate.

### 2. Local search repeatedly reloads partitions

`_run_local_search()` launches one `LocalSearch` per breakpoint. Each local
search worker rereads the covariance partitions overlapping its search window.
With multiprocessing, many workers can load overlapping partition data
simultaneously. This is both slow and memory-intensive.

Short-term operational mitigation:

- Use low `--workers`/`cov_workers` for whole-chromosome runs.
- Avoid high Snakemake `--cores` values unless memory has been profiled.

Longer-term implementation options:

- Build a chromosome-level covariance cache once and use it for local search.
- Or process local-search windows grouped by overlapping partition ranges.
- Or implement exact array local search and reduce reliance on Python
  dictionaries.

### 3. Array local-search bug

The previous array local-search path diverged from the legacy dictionary path
on real multi-partition chromosomes. EUR reproduced exactly only after
multi-partition windows were routed back through the dictionary path.

The source-level issue is the effective locus list:

- legacy local search builds `precomputed["locus_list"]` through partition
  reads and dynamic deletion side effects;
- array local search used a simple interval slice of covariance loci;
- `N_zero` updates depend on the exact indices in that locus list.

A correct array rewrite must first reproduce the legacy effective locus list
for multi-partition windows, then compute `sum_vert`, `sum_horiz`, and
candidate metric deltas from arrays.

The regression tests currently enforce:

- array local search may be used for single-partition windows;
- multi-partition normal-float local search uses the legacy-compatible path;
- multi-partition normal-float output matches Decimal/dictionary output on a
  synthetic overlapping-partition fixture.

## Suggested Implementation Order

1. Keep the current correctness-preserving local-search fallback.
2. Add an array-backed matrix-to-vector implementation behind an internal flag.
3. Compare array vector vs legacy vector on:
   - toy integration data;
   - a synthetic overlapping-partition fixture;
   - one real chr21/chr22 run.
4. If vector equivalence is exact or within expected float tolerance, make it
   the default and keep legacy as fallback.
5. Revisit array local search after the vector path is stable.

## Validation To Run After Changes

```text
uv run pytest tests/test_metric.py tests/test_local_search.py tests/test_shrinkage.py
uv run pytest tests/integration/test_pipeline.py
uv run ruff check src/ldetect2 tests
cd examples/ldetect_original && uv run snakemake -n --config chromosomes='[22]'
```
