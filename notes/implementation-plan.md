# ldetect2 Implementation Plan

## Context

Full refactor of `_reference/ldetect/` into a modern Python package at `src/ldetect2/`, plus the `interpolate_maps.py` script from joepickrell/1000-genomes-genetic-maps. Goals:

- All logic lives under `src/ldetect2/` as importable modules
- Single unified `ldetect2` CLI via stdlib argparse (no extra dependency)
- Replace `commanderline` library
- Replace global config dict (`flat_file_consts.py`) with a `CovarianceStore` dataclass
- Rename cryptic module names (E03→`matrix_analysis`, E05→`find_minima`, E07→`metric`, E08→`local_search`)
- Intermediate breakpoint output switches from `.pickle` → `.json`
- A top-level `ldetect2 run` command chains all five steps end-to-end
- Python 3.10+, type annotations throughout

---

## Module Layout

```
src/ldetect2/
├── __init__.py
├── _cli/
│   ├── __init__.py
│   ├── main.py                  # ArgumentParser + add_subparsers(); main() entry point
│   ├── cmd_partition.py         # ldetect2 partition-chromosome
│   ├── cmd_covariance.py        # ldetect2 calc-covariance
│   ├── cmd_matrix_to_vector.py  # ldetect2 matrix-to-vector
│   ├── cmd_find_minima.py       # ldetect2 find-minima
│   ├── cmd_extract_bpoints.py   # ldetect2 extract-bpoints
│   ├── cmd_interpolate_maps.py  # ldetect2 interpolate-maps
│   └── cmd_run.py               # ldetect2 run  (chains all five steps)
├── io/
│   ├── __init__.py
│   ├── partitions.py            # CovarianceStore, read_partitions, relevant_subpartitions, first_last
│   ├── covariance.py            # insert_into_matrix (lean/full), read_partition_into_matrix, delete_loci_*, write_corr_vector
│   └── bed.py                   # write_bed
├── shrinkage.py                 # Wen/Stephens LD estimator (was P00_01_calc_covariance.py)
├── matrix_analysis.py           # MatrixAnalysis class (was E03_matrix_to_vector.py)
├── filters.py                   # apply_filter, get_minima_loc (was baselib/filters.py)
├── find_minima.py               # custom_binary_search_with_trackback, FlexibleBoundedAccessor (was E05_find_minima.py)
├── metric.py                    # Metric class (was E07_metric.py)
├── local_search.py              # LocalSearch class (was E08_local_search.py)
├── interpolate_maps.py          # interpolate() function (from joepickrell scripts)
└── _util/
    ├── __init__.py
    ├── binary_search.py         # find_le/ge/lt/gt + _ind variants (direct port + type annotations)
    └── logging.py               # log_msg() wrapping stdlib logging (replaces print_log_msg)
```

`_cli/` has a leading underscore because CLI modules are thin wrappers with no public API value. `io/` isolates all disk access from algorithm logic.

---

## Replacing `flat_file_consts.py`

The `return_conf(path)` / `input_config: dict` pattern couples every function to an opaque blob. Replace with:

```python
# src/ldetect2/io/partitions.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class CovarianceStore:
    root: Path

    @property
    def partitions_dir(self) -> Path:
        return self.root / "scripts"

    def partitions_path(self, name: str) -> Path:
        return self.partitions_dir / f"{name}_partitions"

    def partition_path(self, name: str, start: int, end: int) -> Path:
        return self.root / name / f"{name}.{start}.{end}.gz"
```

The 8 column-index constants (`i_id_col=0`, `j_id_col=1`, …) become module-level constants at the top of `io/covariance.py` — they are invariant across all datasets and do not need to travel with the path.

Every function previously taking `input_config: dict` is rewritten to take `store: CovarianceStore`. CLI commands construct `CovarianceStore(root=Path(args.dataset_path))` and pass it down. No global state.

Output delimiter is always `\t` (hardcoded in writer functions). Image output path is derived as `output_path.with_suffix(".png")` when `--generate-heatmap` is set.

---

## CLI Subcommand Signatures

### `partition-chromosome` (was P00_00)
```
ldetect2 partition-chromosome
    --genetic-map PATH       required; gzipped TSV (position, genetic_position)
    --n-individuals INT      required; individuals in reference panel
    --output PATH            required
    --window-size INT        default 5000; target SNPs per partition window
    --ne FLOAT               default 11418.0; effective population size
    --cutoff FLOAT           default 1.5e-8; recombination fraction threshold
```

### `calc-covariance` (was P00_01)
Reads VCF from stdin (pipe from tabix).
```
ldetect2 calc-covariance
    --genetic-map PATH       required
    --individuals PATH       required; one individual ID per line
    --output PATH            required; gzipped 8-column TSV
    --ne FLOAT               default 11418.0
    --cutoff FLOAT           default 1e-7; LD values below this are not written
```

### `matrix-to-vector` (was P01)
```
ldetect2 matrix-to-vector
    --dataset-path PATH      required; root of covariance matrix directory
    --name TEXT              required; chromosome name e.g. chr2
    --output PATH            required; gzipped (position, corr_sum) TSV
    --snp-first INT          optional; auto-detected from partitions if omitted
    --snp-last INT           optional; auto-detected from partitions if omitted
    --mode [diag|vert]       default diag
    --generate-heatmap       flag; write PNG alongside output
```

### `find-minima` (was P02)
```
ldetect2 find-minima
    --input PATH             required; gzipped vector from matrix-to-vector
    --chr-name TEXT          required
    --dataset-path PATH      required
    --n-snps-bw-bpoints INT  required; mean SNPs between breakpoints e.g. 50
    --output PATH            required; .json output
    --snp-first INT          optional
    --snp-last INT           optional
    --trackback-delta INT    default 200
    --trackback-step INT     default 20
    --init-search-loc INT    default 1000
```

### `extract-bpoints` (was P03)
```
ldetect2 extract-bpoints
    --name TEXT              required
    --dataset-path PATH      required
    --breakpoints PATH       required; .json from find-minima
    --subset [fourier|fourier_ls|uniform|uniform_ls]  required
    --output PATH            default stdout; .bed file
```

### `run` (new — chains all five steps)
Uses a temp directory for intermediates.
```
ldetect2 run
    --genetic-map PATH       required
    --reference-panel PATH   required; VCF accessed via tabix
    --individuals PATH       required
    --chromosome TEXT        required; e.g. chr2
    --output-dir PATH        required
    --ne FLOAT               default 11418.0
    --cov-cutoff FLOAT       default 1e-7
    --n-snps-bw-bpoints INT  default 50
    --subset [fourier|fourier_ls|uniform|uniform_ls]  default fourier_ls
```

### `interpolate-maps` (from joepickrell scripts)
```
ldetect2 interpolate-maps
    --snp-file PATH          required; BED file of SNP positions
    --genetic-map PATH       required; gzipped recombination map
    --output PATH            required; gzipped (rs_id, position, genetic_position) TSV
```

---

## `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ldetect2"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov", "ruff", "mypy"]

[project.scripts]
ldetect2 = "ldetect2._cli.main:main"

[tool.hatch.build.targets.wheel]
packages = ["src/ldetect2"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
src = ["src"]
target-version = "py310"
```

No Click/Typer — CLI uses stdlib argparse only.

---

## Implementation Order

### Phase 1 — Package skeleton
1. `pyproject.toml`
2. All `__init__.py` files to establish package structure
3. `_util/binary_search.py` — direct port of `baselib/binary_search.py` + type annotations; zero logic change
4. `_util/logging.py` — `log_msg(msg: str) -> None` wrapping `logging.getLogger("ldetect2")`; replaces `print_log_msg`
5. `io/partitions.py` — `CovarianceStore` dataclass, `read_partitions()`, `relevant_subpartitions()`, `first_last()`

### Phase 2 — I/O layer
6. `io/covariance.py` — `insert_into_matrix_lean()`, `insert_into_matrix()`, `read_partition_into_matrix_lean()`, `read_partition_into_matrix()`, all `delete_loci_*` variants, `write_corr_vector()`; lean path is primary
7. `io/bed.py` — `write_bed(name, loci, snp_first, snp_last, output)`

### Phase 3 — Algorithm modules
8. `filters.py` — direct port of `baselib/filters.py` + type annotations
9. `matrix_analysis.py` — `MatrixAnalysis` class; `calc_diag_lean()` is primary path; `generate_img()` kept
10. `find_minima.py` — `FlexibleBoundedAccessor`, `trackback()`, `custom_binary_search_with_trackback()`
11. `metric.py` — `Metric` class, lean path only; keeps `decimal.Decimal` precision
12. `local_search.py` — `LocalSearch` class; lean path only
13. `shrinkage.py` — Wen/Stephens logic extracted from `P00_01_calc_covariance.py`; `calc_covariance(vcf_stream, ...)` function
14. `interpolate_maps.py` — `interpolate(snp_file, map_file, output)` function; Python 3 port of joepickrell script

### Phase 4 — CLI
15. `_cli/main.py` — `build_parser() -> ArgumentParser` with `add_subparsers()`; dispatch dict mapping subcommand name → handler; `main()` entry point
16. `_cli/cmd_partition.py` — registers subparser, calls `shrinkage.partition_chromosome()`
17. `_cli/cmd_covariance.py` — registers subparser, reads VCF from `sys.stdin`, calls `shrinkage.calc_covariance()`
18. `_cli/cmd_matrix_to_vector.py` — registers subparser, calls `MatrixAnalysis`
19. `_cli/cmd_find_minima.py` — registers subparser, calls pipeline from `find_minima.py`
20. `_cli/cmd_extract_bpoints.py` — registers subparser, loads `.json`, calls `io.bed.write_bed()`
21. `_cli/cmd_interpolate_maps.py` — registers subparser, calls `interpolate_maps.interpolate()`
22. `_cli/cmd_run.py` — registers subparser, uses `tempfile.TemporaryDirectory`, chains steps 16→17→18→19→20

---

## JSON Breakpoint Format

Replaces the `.pickle` output from `find-minima`. Structure:

```json
{
  "n_bpoints": 42,
  "found_width": 1234,
  "fourier":     { "loci": [123456, 234567, ...], "metric": { "sum": "0.123", "N_nonzero": 100, "N_zero": 500 } },
  "fourier_ls":  { "loci": [...], "metric": { ... } },
  "uniform":     { "loci": [...], "metric": { ... } },
  "uniform_ls":  { "loci": [...], "metric": { ... } }
}
```

`metric.sum` is stored as a string to preserve `Decimal` precision across serialisation.

---

## Reference Files

| New file | Ported from |
|----------|-------------|
| `_util/binary_search.py` | `baselib/binary_search.py` |
| `_util/logging.py` | `flat_file.print_log_msg` |
| `io/partitions.py` | `baselib/flat_file.py` (partition funcs) + `flat_file_consts.py` |
| `io/covariance.py` | `baselib/flat_file.py` (matrix funcs) |
| `io/bed.py` | `examples/P03_extract_bpoints.py` (output logic) |
| `filters.py` | `baselib/filters.py` |
| `matrix_analysis.py` | `pipeline_elements/E03_matrix_to_vector.py` |
| `find_minima.py` | `pipeline_elements/E05_find_minima.py` |
| `metric.py` | `pipeline_elements/E07_metric.py` |
| `local_search.py` | `pipeline_elements/E08_local_search.py` |
| `shrinkage.py` | `examples/P00_00_partition_chromosome.py` + `P00_01_calc_covariance.py` |
| `interpolate_maps.py` | joepickrell `scripts/interpolate_maps.py` |

---

## Verification

### Unit tests (`tests/`)
- `test_binary_search.py` — all `find_le/ge/lt/gt` variants, boundary conditions
- `test_partitions.py` — `CovarianceStore` path construction; `read_partitions` against small fixture; `relevant_subpartitions` edge cases
- `test_filters.py` — Hanning filter on synthetic known-peak signal; assert minima at expected positions
- `test_interpolate_maps.py` — hand-computed interpolation cases; clamping before first and after last map marker
- `test_covariance_io.py` — write tiny fake gzipped covariance file → `read_partition_into_matrix_lean` → verify matrix values

### Integration test (`tests/integration/`)
Run all five steps on `_reference/ldetect/examples/example_data/`; diff resulting BED against a committed reference fixture.

### Numeric equivalence
Instantiate reference `Metric` and new `Metric` with identical breakpoints on example data; assert `sum`, `N_nonzero`, `N_zero` are exactly equal (Decimal precision must match).

### CLI smoke tests
Call each subcommand with `--help` via `subprocess.run`; assert exit code 0.
