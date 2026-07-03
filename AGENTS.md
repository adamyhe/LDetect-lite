# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**ldetect2** is a modern reimplementation of [ldetect](https://bitbucket.org/nygcresearch/ldetect), a bioinformatics tool that calculates approximately independent linkage disequilibrium (LD) blocks in the human genome. The algorithm is described in [Berisa & Pickrell, 2016](https://academic.oup.com/bioinformatics/article/32/2/283/1743626).

The complete reference implementation lives in `_reference/ldetect_original/ldetect/` (also `_reference/LDblocks_GRCh38/` for the MacDonald et al. 2022 GRCh38 reproduction). The new implementation is in `src/ldetect2/`.

## Commands

```bash
# Install in editable mode (uv recommended)
uv sync

# Install with heatmap support (matplotlib, required for --generate-heatmap)
uv sync --extra heatmap

# Install dev extras (pytest, ruff, mypy, snakemake — needed for examples/)
uv sync --extra dev

# Run unit tests only (fast)
uv run pytest -m "not integration"

# Run a single test
uv run pytest tests/test_filters.py::test_unimodal_single_minimum -v

# Run integration tests (downloads ~5 files from BitBucket on first run, cached to tests/data/)
uv run pytest -m integration

# Run all tests
uv run pytest

# Lint / type-check
uv run ruff check src tests
uv run mypy src

# CLI entry point
uv run ldetect2 --help
```

## Architecture

The package is under `src/ldetect2/`. Key modules:

| Module | Role |
|--------|------|
| `shrinkage.py` | Steps 1–2: `partition_chromosome` and `calc_covariance` (Wen/Stephens shrinkage LD estimator). The pairwise LD kernel (`_pairwise_ld_impl`) is JIT-compiled with Numba when available (~50x speedup), falling back to pure Python. Writes indexed HDF5 covariance partitions. |
| `matrix_analysis.py` | Step 3: `MatrixAnalysis` class — reduces covariance partitions to a `[position, diagonal_sum]` vector. `calc_diag_lean` (dictionary path) and `calc_diag_array` (array-backed path, via `_util/vector_array.py`) are the two production paths; `calc_diag`/`generate_img` support the `--generate-heatmap` PNG output and require full-schema partitions. |
| `filters.py` | Hanning-window convolution (`np.hanning`, symmetric) and `scipy.signal.argrelextrema` minima extraction |
| `find_minima.py` | Step 4 core: `FlexibleBoundedAccessor`, binary search for optimal filter width, `custom_binary_search_with_trackback` |
| `metric.py` | `Metric` class — sum of squared correlations across blocks. Float by default; `use_decimal=True` opts into 50-digit `decimal.Decimal` precision (dictionary path) instead of the array-backed path. |
| `local_search.py` | `LocalSearch` — refines each breakpoint by local search. Array-backed path (`_init_search_array`/`_search_array`) is used for single-partition windows only; multi-partition windows and `--high-precision` fall back to the legacy dictionary/Decimal path (`init_search`/`search`) because the array path does not yet reproduce the legacy multi-partition effective locus list (see `notes/local-search-divergence-asn22.md`). |
| `pipeline.py` | `find_breakpoints` — orchestrates steps 4–5: binary search for filter width, minima extraction, metric computation, local search, JSON output with four subsets (`fourier`, `fourier_ls`, `uniform`, `uniform_ls`). Supports computing only requested subsets plus their local-search dependencies, and reuses an in-memory `ChromosomeCovariance` cache across metric calls where possible. |
| `interpolate_maps.py` | `interpolate()` — maps SNP positions to genetic positions via linear interpolation |
| `io/partitions.py` | `CovarianceStore` dataclass (replaces reference's global `flat_file_consts.py` dict), `read_partitions`, `relevant_subpartitions`, `first_last`, `get_final_partitions` |
| `io/covariance.py` | Legacy dictionary-based covariance matrix I/O: `insert_into_matrix_lean`, `read_partition_into_matrix_lean`, `delete_loci_*`, `write_corr_vector` |
| `io/covariance_hdf5.py` | Indexed HDF5 partition I/O: `write_covariance_partition_hdf5` (full schema), `write_compact_covariance_partition_hdf5_chunks`/`_append` (compact, streamed), `HDF5CovariancePartitionReader`, `open_covariance_reader`, `validate_covariance_hdf5` |
| `io/bed.py` | `write_bed` |
| `io/vcf.py` | `read_vcf_samples` — sample IDs via `bcftools query -l` |
| `_util/binary_search.py` | `find_le/ge/lt/gt` and `*_ind` variants wrapping `bisect` |
| `_util/covariance_array.py` | Array-backed covariance loading for metrics/local search: `ChromosomeCovariance`, `CovariancePartition`, `load_covariance_arrays`, `load_chromosome_covariance`, `metric_from_arrays`, `metric_from_files` (parallel row-pass streaming metric via `--metric-workers`) |
| `_util/vector_array.py` | Array-backed matrix-to-vector implementation used by `calc_diag_array`; streams and chunks HDF5 partitions to bound memory |
| `_util/covariance_summary.py` | `summarize_covariance` — per-partition row counts and estimated covariance-array memory, backing `ldetect2 covariance-summary` |
| `_util/memory.py` | `current_rss_mib`/`max_rss_mib`/`log_memory_checkpoint` — process RSS diagnostics logged at pipeline checkpoints (debug logging) |
| `_util/intervals.py` | Interval helpers |
| `_util/logging.py` | `log_msg`/`log_debug`/`configure_logging` wrapping stdlib logging (replaces `print_log_msg`) |
| `_cli/` | argparse subcommands: `partition-chromosome`, `calc-covariance`, `covariance-summary`, `matrix-to-vector`, `find-minima`, `extract-bpoints`, `interpolate-maps`, `run`; global `-v/--verbosity {debug,info,warning,error}` flag handled in `main.py` |

## Key Design Decisions

- **`CovarianceStore`** (frozen dataclass) replaces the reference's `flat_file_consts.py` global path-config dict. Every function that previously took `input_config: dict` now takes `store: CovarianceStore`.
- **Numba JIT**: `_pairwise_ld_impl` in `shrinkage.py` is decorated with `@_jit` — a decorator defined as `njit(cache=True)` when Numba is available, or a no-op otherwise. The inner k loop was replaced with `np.sum(a * b)` / `np.sum(a)` / `np.sum(b)` to let LLVM vectorize (~50x speedup over pure Python).
- **Parallel covariance**: `ldetect2 run --workers N` uses `concurrent.futures.ProcessPoolExecutor` to calculate covariance matrices for multiple partitions concurrently. Each partition is independent (unique output file, no shared state). The tabix spawn + `calc_covariance` call is extracted into a module-level `_calc_partition` function in `_cli/cmd_run.py` so it is picklable by the process pool.
- **Pickle → JSON**: Intermediate breakpoint output uses `.json` (not `.pickle`) for portability.
- **Flat file → indexed HDF5**: Covariance partitions are stored as indexed HDF5 (`.h5`), not the reference's gzipped space-delimited flat files. `full` schema (naive LD, genetic positions, SNP IDs) supports `--generate-heatmap` and debugging; `compact` schema (canonical `lo`/`hi` pairs, `shrink_ld`, diagonal entries, lookup indexes) is the `ldetect2 run` default and supports restartable production runs with bounded memory.
- **Array-backed vs. dictionary paths**: `matrix_analysis.py`, `metric.py`, and `local_search.py` each have a legacy dictionary-based path (ported near-verbatim from the reference, precise but slow) and a newer NumPy array-backed path (faster, vectorized). Float-mode single-partition local search and normal-float metrics use the array path by default; `--high-precision` (Decimal) and multi-partition local-search windows use the dictionary path because the array-backed local search does not yet reproduce the legacy multi-partition effective locus list exactly. Treat `--high-precision` as the correctness oracle when investigating local-search discrepancies.
- **Float arithmetic by default**: The reference used 50-digit `decimal.Decimal` for all breakpoint metric/local-search arithmetic (~10–30x slower than float, with no practical difference for typical LD data). `float` is now the default; `--high-precision` opts back into Decimal.
- **Symmetric Hann window**: `filters.py` uses `np.hanning(2*width+1)` — the symmetric variant. `scipy.signal.get_window('hann', N)` (default `fftbins=True`) is a periodic DFT window and gives different results.
- **Test data**: Downloaded on first run from BitBucket raw URLs into `tests/data/` (gitignored). Session-scoped fixtures in `tests/conftest.py` handle this.

## Input Data Formats

- **Genetic map**: gzipped TSV, columns: `chr  position  genetic_position_cM`
- **Covariance matrix**: indexed HDF5 partition file (`.h5`); `full` schema also readable/writable as the reference's gzipped space-delimited 8-column flat format (`i_id  j_id  i_pos  j_pos  i_gpos  j_gpos  naive_ld  shrink_ld`) via the legacy `io/covariance.py` path
- **Partition file**: space-delimited `start end` pairs, one per line
- **Reference panel**: phased VCF piped via `tabix`
- **Breakpoint JSON** (from `find-minima`): `{"n_bpoints", "found_width", "computed_subsets", "skipped_subsets", "fourier"/"fourier_ls"/"uniform"/"uniform_ls": {"loci": [...], "metric": {"sum", "N_nonzero", "N_zero"}}}`. `metric.sum`/`N_zero` are stored as strings to preserve precision (relevant when `--high-precision` produces Decimal values).

## Repository Layout Beyond `src/`

- `notes/` — agent-facing design notes, optimization history, and reproduction-divergence investigations. `notes/optimizations.md` is the up-to-date human-readable summary of implemented performance work; `notes/implementation-plan.md` is a historical record of the original port design; `notes/ldetect-original-*.md` and `notes/local-search-divergence-asn22.md` track ongoing work to reproduce the published Berisa & Pickrell BED files exactly (see Reproduction Status below).
- `examples/` — Snakemake reproduction workflows, each with its own `README.md`/`config.yaml`:
  - `ldetect_example/` — toy EUR chr2 example bundled with the original ldetect, used as a correctness fixture.
  - `ldetect_original/` — full genome-wide reproduction of the published Berisa & Pickrell EUR/AFR/ASN LD blocks from public 1000 Genomes Phase 1 VCFs, plus diagnostic Snakefiles (`Snakefile.diagnostics`, `Snakefile.legacy_diagnostics`, `Snakefile.provenance_diagnostics`) used to isolate divergences from the reference.
  - `MacDonald2022/` — reproduction of MacDonald et al. (2022) GRCh38 LD blocks (deCODE map + pyrho superpopulation maps).
  - `r2_zarr_exactness/` — Zarr-based exactness checks.
- `benchmarks/` — standalone perf benchmarks (e.g. `bench_ld_kernel.py` for the Numba LD kernel, referenced in `notes/optimizations.md`).
- `_reference/` — vendored reference implementations used for parity testing and porting: `ldetect_original/ldetect/` (original Berisa & Pickrell code) and `LDblocks_GRCh38/` (MacDonald et al. 2022 scripts).

## Reproduction Status

Tracked in detail in `notes/ldetect-original-main-pipeline-audit.md` and `notes/ldetect-original-concordance.md`. As of the latest full genome-wide run against the published Bitbucket `ldetect-data` reference BEDs:

- **ASN**: exact match, all 22 autosomes.
- **AFR**: near-exact; residual boundary divergence on chr11 and chr22 only.
- **EUR**: exact block counts and coverage on every chromosome, but chr8–12 have shifted internal boundaries relative to the published reference (contiguous range bracketed by exact chr7/chr13 matches). Leading hypothesis is upstream input/provenance divergence for those chromosomes rather than a current ldetect2 bug — see the audit notes for the evidence ruling out covariance/vector/local-search/BED-packaging causes.

When investigating reproduction mismatches, read the relevant `notes/ldetect-original-*.md` file first; it likely already rules out several hypotheses.

## CLI Subcommands

`partition-chromosome`, `calc-covariance`, `covariance-summary`, `matrix-to-vector`, `find-minima`, `extract-bpoints`, `interpolate-maps`, `run`. Full argument documentation and worked examples for each are in `README.md` — that is the source of truth for CLI usage; keep it in sync with `_cli/cmd_*.py` when changing flags.
