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
- Completed serial EUR 1000G diagnostics show exact downstream agreement
  between bitpacked and `uint8` (`vector_max_abs_diff=0.0` and exact
  breakpoint/BED agreement).

## Merge Gate

Merge is waiting on the remaining 1000G population diagnostics so the default
backend flip is backed by population-wide validation. Final speed claims should
wait for those runs. Current evidence supports bitpacked as the production
default, but total wall-clock gains are expected to be smaller than the inner
row-generation speedup because VCF/BCF ingestion and HDF5 writes remain
substantial.

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
