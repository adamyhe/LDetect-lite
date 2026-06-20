# ASN chr22 Local-Search Divergence Notes

Date: 2026-04-30

## Context

The `examples/ldetect_original` pipeline now gets the ASN chr22 block count
right, but the default `fourier_ls` boundaries are still offset from the
published ldetect reference. The available published reference is only the
final `fourier_ls` BED, not the intermediate vector or raw Fourier minima.

Reference BED fetched from Bitbucket:

```text
examples/ldetect_original/resources/ldetect_ref/ASN_fourier_ls-chr22.bed
```

Our breakpoint files:

```text
examples/ldetect_original/results/ASN/22/breakpoints-22.json
examples/ldetect_original/results/ASN/22/breakpoints-22-hp.json
```

`breakpoints-22-hp.json` was generated with high precision, which routes local
search through the legacy Decimal/dictionary implementation.

## Key Findings

Default and high-precision agree exactly before local search:

```text
n_bpoints: 19
found_width: 4169
fourier: 19/19 exact default vs hp
uniform: 19/19 exact default vs hp
```

Therefore filter-width targeting and raw minima extraction are not responsible
for the default-vs-high-precision difference.

After local search, default and high precision diverge strongly:

```text
fourier_ls default vs hp:
  exact: 3/19
  median abs difference: 194.5 kb
  mean abs difference:   277.8 kb
  max abs difference:    843.9 kb

uniform_ls default vs hp:
  exact: 2/19
  median abs difference: 252.8 kb
  mean abs difference:   315.0 kb
  max abs difference:    1475.2 kb
```

Against the ASN chr22 published `fourier_ls` reference:

```text
default fourier_ls:
  exact: 2/19
  median offset: 277.0 kb
  <=100kb: 5/19

hp fourier_ls:
  exact: 8/19
  median offset: 41.3 kb
  <=100kb: 11/19
```

The old Decimal/dictionary path is much closer to the published reference.

Metric sums also differ by orders of magnitude after local search:

```text
fourier_ls default sum: 530.426
fourier_ls hp sum:       1.209

uniform_ls default sum: 138.200
uniform_ls hp sum:       1.879
```

This is too large to be normal floating-point precision drift. The array-backed
local-search implementation is not behavior-equivalent to legacy local search
on full-chromosome, multi-partition data.

## Current Hypothesis

The bug is in array local-search precomputation/search, not in:

- BED extraction.
- filter-width search.
- raw Fourier minima.
- raw uniform breakpoints.
- global metric for the initial Fourier/uniform breakpoints.

Most likely candidates:

- array precompute misses `sum_horiz` contributions where `j_pos` lies above
  the candidate search window but within `snp_top`;
- array precompute/search uses inconsistent candidate-locus bounds vs legacy;
- array loader's overlap ownership filtering is appropriate for global metric
  but too aggressive for local-search windows;
- array local search counts `N_zero` with a different effective locus list than
  the legacy precomputed dictionary path.

## Recommended Next Steps

1. Treat `--high-precision` as the correctness oracle until array local search
   is fixed.
2. Compare `LocalSearch._init_search_array()` and `_search_array()` directly
   against legacy `LocalSearch.init_search_lean()` and `search()` for ASN chr22.
3. For one divergent breakpoint, dump candidate metrics from both paths across
   the same search window and find the first candidate where `sum` or `N_zero`
   differs.
4. Once fixed, add a regression that compares default vs high-precision local
   search for a multi-partition fixture.

## Detailed Local-Search Source Check

The reference implementation lives at:

```text
_reference/ldetect_original/ldetect/pipeline_elements/E08_local_search.py
```

The current implementation lives at:

```text
src/ldetect2/local_search.py
```

The dictionary/Decimal path in `src/ldetect2/local_search.py` is intentionally
very close to the legacy `init_search_lean()` and `search()` implementation.
The main search loops match the legacy method:

- find `snp_bottom_ind` and `snp_top_ind` from the precomputed locus list;
- locate the current breakpoint with `find_le_ind`;
- walk right, updating `sum` by `-sum_horiz + sum_vert` and `N_zero` by
  `-horiz_N + vert_N`;
- reset to the initial metric;
- walk left, updating `sum` by `+sum_horiz - sum_vert` and `N_zero` by
  `+horiz_N - vert_N`;
- use the same left-vs-right tie-break behavior.

The divergence is not in those loops. It is in the newer array-backed
precomputation path, which tried to build `loci`, `sum_vert`, and `sum_horiz`
directly from covariance arrays. The legacy precomputed locus list is not a
simple interval slice. In multi-partition windows it is created as a side
effect of:

- pre-reading overlapping partitions below `snp_bottom`;
- reading each selected partition in order;
- choosing non-final partition `end_locus` values from the next partition start;
- clipping only the final partition to `snp_last`;
- deleting matrix loci below the current `end_locus`.

That side-effect-derived list controls both `snp_top_ind` and the candidate
indices used in the denominator updates. A straightforward array slice such as
`snp_bottom <= locus <= snp_last` misses some dummy loci required by the
legacy denominator in some multi-partition windows; widening it to `snp_top`
fixes that synthetic case but breaks the established toy-pipeline reference.

## Current Code Fix

As a correctness-preserving workaround, default local search now uses the
array-backed path only for single-partition windows. Multi-partition windows
fall back to the legacy-compatible dictionary path even in normal float mode.
High precision still uses the Decimal dictionary path.

This preserves the validated toy pipeline output and avoids the unsafe
multi-partition array behavior seen on ASN chr22. It does not yet implement the
fully vectorized multi-partition local search. That rewrite should explicitly
reproduce the legacy effective locus list before re-enabling the array path for
multi-partition windows.

Regression coverage added:

- default local search matches Decimal local search across an overlapping
  multi-partition synthetic fixture;
- default local search explicitly leaves `_array_loci` unset for
  multi-partition windows;
- the existing toy integration BED comparison still matches the reference.

Validation run:

```text
uv run pytest tests/test_metric.py tests/test_local_search.py
uv run pytest tests/test_local_search.py tests/integration/test_pipeline.py
uv run ruff check src/ldetect2/local_search.py tests/test_local_search.py
```

## EUR Retest and Population-Specific Hypothesis

After disabling array local search for multi-partition windows, EUR chr21 and
chr22 reproduce the published Berisa/Pickrell `fourier_ls` BED exactly:

```text
chr21 recall=1.0 precision=1.0 jaccard=1.0 median_offset=0.0 kb
chr22 recall=1.0 precision=1.0 jaccard=1.0 median_offset=0.0 kb
```

The same output did not reproduce correctly with the array method. This
confirms at least one real bug in the array-backed local-search method.

Residual ASN/AFR differences should now be treated as a separate issue. The
most likely candidate is a population-specific covariance/shrinkage
hyperparameter, especially `N_e`, rather than the final BED extraction or the
local-search loop itself.

## Reference-Code Notes on `N_e`

The original ldetect README documents `N_e` as a covariance-script argument:

```text
P00_01_calc_covariance.py <input_genetic_map> <input_individuals_file>
    <effective_population_size> <cov_cutoff> <output_cov_matrix_partition>
```

Its worked example uses `11418`, and the README explicitly says this is
appropriate for European populations.

In code:

- `_reference/ldetect_original/ldetect/examples/P00_01_calc_covariance.py`
  reads `NE = float(sys.argv[3])`.
- `_reference/ldetect_original/ldetect/examples/P00_00_partition_chromosome.py`
  hardcodes `11418.0` in the partition-extension calculation.

For the MacDonald2022 reference:

- `_reference/LDblocks_GRCh38/scripts/runAllCov.sh` accepts `popsize=$3`.
- `_reference/LDblocks_GRCh38/scripts/runCov.sh` passes that value to
  `P00_01_calc_covariance.py`.
- `_reference/LDblocks_GRCh38/README.md` shows `runAllCov.sh eurinds.txt EUR
  11418` and describes the final argument as the effective population size for
  Europeans.

I did not find a population-specific `N_e` table in the checked-in reference
code. The available scripts support passing different values, but the examples
and prose only document `11418` for EUR. The original partitioning step is also
implicitly European because of the hardcoded `11418`.

The likely source of the intended population-specific values is the SHAPEIT
documentation for HapMap II maps, which lists:

```text
CEU = 11418
YRI = 17469
CHB+JPT = 14269
```

The example configs now use these as defaults where they map cleanly:

- `examples/ldetect_original`: EUR=11418, AFR=17469, ASN=14269.
- `examples/MacDonald2022`: EUR=11418, AFR=17469, EAS=14269.
- MacDonald SAS remains 11418 because the cited SHAPEIT/HapMap II values do
  not include a SAS-specific default.
