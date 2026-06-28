# r2-Zarr Exactness and Runtime Workflow

This example runs real 1000 Genomes Phase 1 data through three `ldetect2`
backends and compares their outputs:

- `matrix_hdf5`: compact HDF5 covariance plus matrix-to-vector
- `direct_hdf5`: compact HDF5 covariance plus direct vector fragments
- `r2_zarr`: experimental normalized `r²` Zarr cache plus direct vector fragments

It belongs in its own directory because it is an experimental backend
comparison, not the toy reference fixture in `ldetect_example` and not the
published-block reproduction workflow in `ldetect_original`.

## Run

```bash
cd examples/r2_zarr_exactness

# Dry-run the default EUR chr22 comparison.
uv run snakemake -n

# Run with four cores.
uv run snakemake --cores 4
```

Useful outputs:

- `results/compare/{POP}.{chrom}.exactness.tsv`
- `results/runtime/{POP}.{chrom}.runtime.tsv`
- `results/runs/{mode}/{POP}/{chrom}/`
- `results/benchmarks/{mode}/{POP}/{chrom}.benchmark.tsv`

For a quicker or broader run, override config values:

```bash
uv run snakemake --cores 8 --config chromosomes='[21,22]' cov_workers=8
```

The workflow uses the same population-specific filtering convention as
`examples/ldetect_original`: subset samples first, then keep biallelic records
with `MAC[0] >= 1`.
