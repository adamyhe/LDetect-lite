# Changelog

Notable changes to ldetect-lite. Versions follow the git tags. Performance
details and measurements for each optimization are in
[`docs/optimizations.md`](docs/optimizations.md); planned work is in
[`ROADMAP.md`](ROADMAP.md).

## [Unreleased]

### Performance
- `--generate-ld-neighborhood-plot`: the r² sampler is vectorized (bottom-k
  sampling with NumPy instead of a per-value Python reservoir loop), each
  boundary reads only the covariance rows inside its window, and r² is
  computed once per chunk against the window's slice of the diagonal. About
  4.8x faster on a synthetic 8 Mb cache. The sample is still a uniform random
  subset for a fixed seed, but the drawn values differ from earlier releases.
  `examples/MacDonald2022/scripts/plot_ld_neighborhood_separation.py` uses
  the same helpers; its per-boundary TSV is unchanged.

### Changed
- The reproduction workflows (`examples/ldetect_original`,
  `examples/MacDonald2022`, `examples/ldetect_example`) now write their
  filtered reference panels as GT-only BCF (`bcftools annotate -x
  INFO,^FORMAT/GT -Ob` + `bcftools index`) instead of `.vcf.gz`. Records,
  positions and genotypes are unchanged. Filtered-panel paths changed, so
  existing Snakemake caches recompute.
- The MacDonald2022 legacy runner streams the panel with `bcftools view`
  instead of `tabix -h`, so it reads BCF.
- README and `docs/pipeline-steps.md` examples use a GT-only BCF panel.
- `docs/optimizations.md` notes that BCF no longer reduces peak RSS. With the
  current streaming Step 2, peak memory is the same for both formats; BCF is
  recommended for speed (3.5-4x faster genotype reading).

### Added
- `examples/ldetect_example/scripts/benchmark_legacy_covariance.py` times
  ldetect-lite on both `.vcf.gz` and GT-only BCF, plus the one-time
  conversion (`--no-bcf` to skip).
- `Snakefile.cyvcf2_migration_profiling` compares `.vcf.gz` against a GT-only
  BCF (previously a plain `-Ob` conversion).

### Documentation
- BCF benchmark figures on the exclusive EPYC 7702 node: toy chr2
  `calc-covariance` 3.876 s on `.vcf.gz` vs 3.642 s on GT-only BCF; EUR chr21
  `ldetect run` Step 2 29 s vs 19 s and 56.7 s vs 46.4 s end to end, with
  byte-identical output and no peak-memory change. Raw log:
  `notes/logs/benchmark-figures-bcf-2026-11.md`.
- Benchmark figures re-measured for v0.5.3 on an exclusive AMD EPYC 7702 node
  (#22).

## [0.5.3] - 2026-11-04

### Performance
- Step 2: replace `np.unique` sorts on already-sorted covariance rows with
  O(n) passes (`_util/sorted_unique.py`); the compact writer is 2.4x faster
  per row (#20).
- Step 2: pack haplotypes into `uint64` words while decoding the reference
  panel instead of building per-variant Python lists. Serial Step 2 on EUR
  chr21 went from 77.4 s to 51.6 s (#20).
- Step 4: the streaming metric selects block-crossing pairs before doing
  diagonal lookups. Bitwise-identical metric (#21).

### Documentation
- Recommend converting the reference panel to a GT-only BCF (#20).

## [0.5.2] - 2026-08-07

### Added
- `unrefined_count` in the `fourier_ls`/`uniform_ls` breakpoint output, plus a
  log warning, when local search keeps a breakpoint's unrefined position
  (the legacy "step 4" collapse, mostly in `uniform_ls`) (#19).

## [0.5.1] - 2026-08-07

### Fixed
- Package versioning and `CITATION.cff`.

## [0.5.0] - 2026-08-07

### Changed
- **`--filter-window` now defaults to `symmetric`**, the Hann formula in
  Berisa & Pickrell's supplement. `scipy-periodic` remains available and is
  required to reproduce modern-scipy-era output such as MacDonald et al.
  (2022), which `examples/MacDonald2022` now pins per block set (#18).

### Added
- `EUR_symmetric` block set in `examples/MacDonald2022` to isolate the window
  choice (#18).

## [0.4.0] - 2026-08-05

### Performance
- Fused Step 2/3: the matrix-to-vector fragment is built from the covariance
  stream during Step 2, skipping Step 3's HDF5 re-read. On by default;
  `--no-fused-vector` disables it (#17).
- Per-process genetic-map caching and vectorized parsing (#17).
- Vectorized genotype decoding via cyvcf2's `genotype.array()` (#17).

### Added
- pyrho monotonic maps and LD-neighborhood plots for the corrected
  MacDonald2022 tracks (#16).
- `CONTRIBUTING.md` and citation metadata (#15).

## [0.3.0] - 2026-07-28

### Added
- `ldetect run --generate-ld-neighborhood-plot` / `--ld-neighborhood-window-bp`:
  left/across/right r² box plot around each breakpoint (#14).

### Changed
- `matplotlib` is a required dependency (the `heatmap` extra is removed);
  `psutil` moved to the `dev` group (#14).

### Fixed
- Native thread caps (BLAS/OpenMP/Numba) are managed from the worker budget
  to avoid oversubscription (#13).
- The filter-width search uses row-chunked threaded convolution instead of
  `prange`, which had capped at about 1.2x (#13).

## [0.2.4] - 2026-07-28

### Fixed
- Precision drift in the genetic-distance cutoff (comparison now in exponent
  units) (#12).
- Filter-window mode handling and non-monotonic genetic maps (#12).

### Added
- `--ld-kernel` and `--force-covariance` toggles in the ldetect_original
  Snakefile; a pair-level covariance HDF5 diff script (#12).

## [0.2.3] - 2026-07-26

### Added
- Bitpacked LD kernel (`--ld-kernel bitpacked`), bit-exact against `uint8`
  and validated on 66/66 chromosome × population combinations (#10).
- `examples/ldetect_example` VCF-start toy harness and benchmark figures
  (#10).
- MacDonald2022 reproduction diagnostics: rebuilt pyrho maps,
  MacDonald-compatible interpolation modes, legacy covariance diagnostics,
  forced covariance recomputation, configurable filter parallelism (#11).

### Fixed
- Block-specific SNP thresholds, BED bounds and endpoint loci (#11).
- Reversed reflected convolution kernel (#11).

## [0.2.2] - 2026-07-09

### Changed
- Package metadata and naming cleanup (#9).

## [0.2.1] - 2026-07-08

### Removed
- Unused bcftools-based `read_vcf_samples` helper.

## [0.2.0] - 2026-07-08

### Changed
- VCF/BCF I/O uses cyvcf2 instead of a `tabix` subprocess and a text parser;
  BCF input works with a `.csi` index (#8).

### Performance
- Numba binary search; thread-parallel filter-width search (#8).

### Added
- Option to delete the covariance cache after a run (#8).

### Fixed
- Local search is grouped to avoid running out of memory (#8).

## [0.1.0] - 2026-07-05

Initial release as **ldetect-lite**: a reimplementation of ldetect
(Berisa & Pickrell, 2016) with indexed HDF5 covariance partitions, Numba LD
kernels, parallel covariance and local search, and float arithmetic by default
(`--high-precision` for Decimal).

- Indexed HDF5 covariance caches streamed in chunks (#3); zstd compression,
  12.4% smaller than lzf genome-wide (#6).
- `ldetect interpolate-maps --mode interval`, a port of MacDonald et al.'s
  interval-rate interpolation (#7).
- Reproduction workflows and diagnostics for the Berisa & Pickrell and
  MacDonald et al. (2022) blocks (#1, #2, #4, #5).
- CLI entry points `ldetect-lite`, `ldetect` and `ldl`; HDF5 caches written
  under the old `ldetect2` name still load.

[Unreleased]: https://github.com/adamyhe/LDetect-lite/compare/v0.5.3...HEAD
[0.5.3]: https://github.com/adamyhe/LDetect-lite/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/adamyhe/LDetect-lite/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/adamyhe/LDetect-lite/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/adamyhe/LDetect-lite/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/adamyhe/LDetect-lite/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/adamyhe/LDetect-lite/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/adamyhe/LDetect-lite/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/adamyhe/LDetect-lite/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/adamyhe/LDetect-lite/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/adamyhe/LDetect-lite/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/adamyhe/LDetect-lite/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/adamyhe/LDetect-lite/releases/tag/v0.1.0
