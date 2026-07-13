# Bitpacked LD Kernel

## Current Status

Bitpacked compact covariance is now the default backend for `ldetect run` and
`ldetect calc-covariance`. The previous `uint8` array-sum backend remains
available with `--ld-kernel uint8` for reference comparisons, diagnostics, and
full-schema debug covariance caches.

The bitpacked backend stores each SNP's haplotype row in `uint64` words and
uses popcount operations to compute pairwise intersection counts. It then feeds
those counts through the same Wen-Stephens shrinkage arithmetic as the `uint8`
backend, so the intended output relationship is exact equality, not a
tolerance-based approximation.

## Evidence

- Unit tests assert bit-exact compact covariance rows, diagonals, and loci for
  bitpacked vs `uint8`, including haplotype counts crossing `uint64` word
  boundaries.
- The toy chr2 example benchmark passes exactness checks when run with
  `--ld-kernel bitpacked`; comparison to original LDetect is at machine
  precision because that is a cross-implementation comparison.
- Serial 1000G diagnostics across EUR, ASN, and AFR show exact downstream
  agreement between bitpacked and `uint8` for all 66 chromosome x population
  runs (`vector_sha256_equal=True`, `vector_max_abs_diff=0.0`, exact loci, and
  `bed_jaccard=1.0`).

## Performance Summary

Across the 66 EUR/ASN/AFR chromosome runs, bitpacked reduced aggregate
covariance time from 32637.68 s to 30238.12 s, an overall speedup of 1.079x.
The median chromosome-level speedup was 1.062x. Bitpacked was faster on 63/66
runs; the only slower chromosomes were AFR chr9 (0.908x), AFR chr12 (0.951x)
and AFR chr21 (0.960x).

Peak RSS was comparable (`uint8`/bitpacked RSS ratio mean 1.016, median 1.002,
range 0.885-1.248). The compact covariance cache size is unchanged
(`covariance_size_ratio=1.0`) because both backends write the same HDF5 schema.
Total wall-clock gains remain modest relative to the row-generation kernel
because VCF/BCF ingestion and HDF5 writes are still substantial parts of the
covariance stage.

## Practical Guidance

- Prefer indexed BCF over `vcf.gz` when users control the reference-panel file
  format; it should be treated as guidance rather than an enforced input
  constraint.
- Keep partition-level parallelism as the main execution model. Any future
  VCF/BCF IO optimization must preserve that parallelism or prove a stronger
  end-to-end speedup.
- Do not downgrade covariance values to `float32`; bitpacking changes pair-count
  representation only, not output precision.
- The direct vector sidecar prototype is abandoned for this merge.
