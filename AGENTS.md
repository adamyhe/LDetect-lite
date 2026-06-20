# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**ldetect2** is a modern reimplementation of [ldetect](https://bitbucket.org/nygcresearch/ldetect), a bioinformatics tool that calculates approximately independent linkage disequilibrium (LD) blocks in the human genome. The algorithm is described in [Berisa & Pickrell, 2016](https://academic.oup.com/bioinformatics/article/32/2/283/1743626).

The complete reference implementation lives in `_reference/ldetect/`. The new implementation is in `src/ldetect2/`.

## Commands

```bash
# Install in editable mode (uv recommended)
uv sync

# Install with heatmap support (matplotlib, required for --generate-heatmap)
uv sync --extra heatmap

# Run unit tests only (fast)
uv run pytest -m "not integration"

# Run a single test
uv run pytest tests/test_filters.py::test_unimodal_single_minimum -v

# Run integration tests (downloads ~5 files from BitBucket on first run, cached to tests/data/)
uv run pytest -m integration

# Run all tests
uv run pytest

# CLI entry point
uv run ldetect2 --help
```

## Architecture

The package is under `src/ldetect2/`. Key modules:

| Module | Role |
|--------|------|
| `shrinkage.py` | Steps 1–2: `partition_chromosome` and `calc_covariance` (Wen/Stephens shrinkage LD estimator). The pairwise LD kernel (`_pairwise_ld_impl`) is JIT-compiled with Numba when available (~50x speedup), falling back to pure Python. |
| `matrix_analysis.py` | Step 3: `MatrixAnalysis` class — reduces covariance matrices to a `[position, diagonal_sum]` vector |
| `filters.py` | Hanning-window convolution (`np.hanning`, symmetric) and `scipy.signal.argrelextrema` minima extraction |
| `find_minima.py` | Step 4 core: `FlexibleBoundedAccessor`, binary search for optimal filter width, `custom_binary_search_with_trackback` |
| `metric.py` | `Metric` class — sum of squared correlations across blocks (50-digit decimal precision) |
| `local_search.py` | `LocalSearch` — refines each breakpoint by local search |
| `pipeline.py` | `find_breakpoints` — orchestrates steps 4–5, outputs JSON with four subsets: `fourier`, `fourier_ls`, `uniform`, `uniform_ls` |
| `interpolate_maps.py` | `interpolate()` — maps SNP positions to genetic positions via linear interpolation |
| `io/partitions.py` | `CovarianceStore` dataclass (replaces reference's global `flat_file_consts.py` dict), `read_partitions`, `relevant_subpartitions`, `first_last` |
| `io/covariance.py` | Covariance matrix I/O: `insert_into_matrix_lean`, `read_partition_into_matrix_lean`, `delete_loci_*` |
| `io/bed.py` | `write_bed` |
| `_util/binary_search.py` | `find_le/ge/lt/gt` and `*_ind` variants wrapping `bisect` |
| `_cli/` | argparse subcommands: `partition-chromosome`, `calc-covariance`, `matrix-to-vector`, `find-minima`, `extract-bpoints`, `interpolate-maps`, `run`; global `-v/--verbosity {debug,info,warning,error}` flag handled in `main.py` |

## Key Design Decisions

- **`CovarianceStore`** (frozen dataclass) replaces the reference's `flat_file_consts.py` global path-config dict. Every function that previously took `input_config: dict` now takes `store: CovarianceStore`.
- **Numba JIT**: `_pairwise_ld_impl` in `shrinkage.py` is decorated with `@_jit` — a decorator defined as `njit(cache=True)` when Numba is available, or a no-op otherwise. The inner k loop was replaced with `np.sum(a * b)` / `np.sum(a)` / `np.sum(b)` to let LLVM vectorize (~50x speedup over pure Python).
- **Parallel covariance**: `ldetect2 run --workers N` uses `concurrent.futures.ProcessPoolExecutor` to calculate covariance matrices for multiple partitions concurrently. Each partition is independent (unique output file, no shared state). The tabix spawn + `calc_covariance` call is extracted into a module-level `_calc_partition` function in `_cli/cmd_run.py` so it is picklable by the process pool.
- **Pickle → JSON**: Intermediate breakpoint output uses `.json` (not `.pickle`) for portability.
- **Symmetric Hann window**: `filters.py` uses `np.hanning(2*width+1)` — the symmetric variant. `scipy.signal.get_window('hann', N)` (default `fftbins=True`) is a periodic DFT window and gives different results.
- **Test data**: Downloaded on first run from BitBucket raw URLs into `tests/data/` (gitignored). Session-scoped fixtures in `tests/conftest.py` handle this.

## Input Data Formats

- **Genetic map**: gzipped TSV, columns: `chr  position  genetic_position_cM`
- **Covariance matrix**: gzipped space-delimited, 8 columns: `i_id  j_id  i_pos  j_pos  i_gpos  j_gpos  naive_ld  shrink_ld`
- **Partition file**: space-delimited `start end` pairs, one per line
- **Reference panel**: phased VCF piped via `tabix`
