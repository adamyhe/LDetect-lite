# ldetect Toy Example

This workflow reproduces the original EUR chr2 toy example distributed with
`ldetect`.

The original VCF used to create the reference covariance matrix is not publicly
archived, so this example starts from the BitBucket covariance fixture:

```text
ref/cov_matrix/{chrom}/{chrom}.{start}.{end}.gz
```

That file is the legacy ldetect 8-column gzipped text covariance format. The
workflow treats it as a reference input only and converts it to the current
ldetect2 NumPy partition format before running `ldetect2 matrix-to-vector`:

```text
work/{chrom}/{chrom}.{start}.{end}.npz
```

This is example-specific compatibility. The optimized core matrix-to-vector
path remains `.npz`-only.

## Run

```bash
uv run snakemake -n
uv run snakemake --cores 1
```

Expected comparison outputs:

```text
results/compare_vector.tsv
results/compare_bpoints.tsv
results/compare_bed.tsv
```
