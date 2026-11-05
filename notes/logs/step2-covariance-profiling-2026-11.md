# Step 2 covariance profiling log — 2026-11-04

Raw dated log behind
[`notes/findings/step2-covariance-profiling.md`](../findings/step2-covariance-profiling.md).
Everything here ran on cbsubscb19 (AMD EPYC 9554) through `srun` on the
cbsubscb Slurm cluster. Benchmark scripts are out-of-tree, in
`/home2/ayh8/ldetect-bench/`:

| Script | Purpose |
|---|---|
| `real_chr21.sh` | Two-version real-data A/B |
| `ab3_chr21.sh` | Three-version A/B plus profiles |
| `prof_chr21.sh` / `profile_step2.py` | Per-stage profile |
| `prof2_chr21.sh` / `prof_arrays.py` | Array-build and BCF timing |
| `proto_chrom_cache.py` | Whole-chromosome cache prototype |
| `scaling_chr21.sh` | Worker scaling |
| `compare_runs.py` | Bitwise output comparison |
| `bench_kernel.py`, `bench_write.py`, `bench_unique.py` | Synthetic micro-benchmarks |

## Common setup

- **Data:** 1000G Phase 1 v3 chr21 (`ALL.chr21.phase1_release_v3.20101123...vcf.gz`).
  - EUR = CEU, TSI, FIN, GBR and IBS intersected with the chr22 VCF samples
    (`examples/ldetect_original/scripts/prep_individuals.py`), giving 379
    individuals.
  - Filtered with `bcftools view -S inds | bcftools view -i 'MAC[0]>=1' -m2 -M2`.
  - Map: Pickrell `chr21.interpolated_genetic_map.gz`.
- **Run flags:** `ldetect run --ne 11418 --cov-cutoff 1e-7 --covariance-cache compact --ld-kernel bitpacked --n-snps-bw-bpoints 10000 --subset fourier_ls --filter-window symmetric`.
- **Partitions:** 103. Their lengths sum to 60.0 Mb over a 38.7 Mb span (1.55x overlap).
- **Result:** n_bpoints=23, found_width=4540. The `fourier_ls` BED matches the
  published Bitbucket `ldetect-data` EUR chr21 blocks exactly (24 blocks).

## 1. Synthetic micro-benchmarks (before any code change)

All on a synthetic 8,000-SNP partition with a neutral MAC≥1 frequency spectrum.

- **Kernel:**
  - About 10.9 ns per evaluated pair (246 individuals).
  - An exact frequency-bound pair-skip prototype gave 1.08-1.14x, so it wasn't adopted.
- **Writer:** 50.7 ns/row with 758 haplotypes and about 3M emitted rows.
  - About 28 ns/row of that was one `np.unique(lo, return_counts=True)` per chunk.
  - zstd-1, zstd-3 and uncompressed writes were within 5 ns/row once warm.
    A first run that made zstd-3 look slow was a warm-up artifact.

These led to optimizations #19 (`e997b51`).

## 2. A/B main vs e997b51 (job 4107083, `-c 4`, `--workers 4`)

| Run | Wall time | Peak RSS (KB) | Step 2 |
|---|---:|---:|---:|
| main warm-up | 53.30 s | 644,076 | 25 s |
| main rep 1 | 50.58 s | 626,912 | 24 s |
| main rep 2 | 50.41 s | 623,020 | 25 s |
| branch warm-up | 50.81 s | 639,520 | 23 s |
| branch rep 1 | 48.16 s | 628,040 | 22 s |
| branch rep 2 | 48.12 s | 628,612 | 22 s |

All outputs were bitwise identical.

**Incident:** that job's scratch directory, `/workdir/ayh8/ldetect-realtest-4107083`,
was found empty after the job ended (presumably a Slurm epilog). Later jobs used
`/workdir/ayh8/ldetect-chr21-data`, a name not tied to a job ID. It
survived between back-to-back jobs but was later found wiped too (see
`step4-profiling-2026-11.md`), so scripts must rebuild inputs in the job.

## 3. Per-stage profile (serial `calc_covariance`, all 103 partitions)

Code at `e997b51`. Times are in seconds, summed from the `profile=` dict.

| Stage | Seconds |
|---|---:|
| `vcf_seconds` | 39.81 |
| `array_seconds` | 24.42 |
| `chunk_seconds` (kernel) | 6.27 |
| `write_io_seconds` | 6.56 |
| `pack_seconds` | 0.34 |
| `prepare_seconds` | 0.26 |
| **`total_seconds`** | **77.37** |

Run totals: n_pairs 281,395,582; n_snps 377,790.

On the largest partitions, the full `read_reference_panel` took 0.52-0.72 s.
Iterating records with no genotype access (`iter_only`) was already 0.39-0.54 s
of that, and adding `genotype.array()` brought it to 0.41-0.57 s. So htslib
record parsing is about 75% of the read.

Isolated `np.array(list)` was only 0.07-0.10 s per large partition, about 5 s
for the chromosome. That doesn't account for the 24.4 s `array_seconds` total.
The gap was never isolated, because it disappeared with direct packing (§5).

The LLVM assembly of `_pairwise_ld_compact_chunk_bitpacked_impl` contains a
hardware `popcnt` and no SWAR multiply constant: LLVM recognizes the SWAR
popcount idiom. `/proc/cpuinfo` shows `popcnt` but no `avx512_vpopcntdq`.

## 4. GT-only BCF and chromosome-cache prototype (first pass)

- **GT-only BCF:** `bcftools annotate -x INFO,^FORMAT/GT -Ob` took 15 s and
  produced 12.4 MB, against 334 MB for the filtered `.vcf.gz`.
  - Per large partition, reads took 0.15-0.20 s from the BCF vs 0.55-0.74 s
    from the VCF.
  - Positions and haplotypes were identical on 4 partitions.
- **Cache prototype:**
  - Build from VCF: 20.0 s serial; 10.9 s with 4 tiles on a `-c 4` allocation.
  - Build from BCF: 6.4 s serial; 3.1 s with 4 tiles.
  - Slicing all partitions took 0.03-0.09 s.
  - **Bug:** one mismatching partition (9411243-15153248, one record short)
    and one record lost at the tile seam. Both were a prototype bug: tile 0's
    region started at span start + 1. Fixed in §5.
- **Caveat:** this job ran `uv run` from the repo checkout while I was editing
  `reference_panel.py`. The interpreter had imported the old code before the
  edits; the list-equality check printing `True` confirms the old list-based
  `haplotypes` was in use. So these numbers are for the old path.

## 5. Direct packing (`bfb7bdf`) and three-version A/B (`ab3_chr21.sh`, `-c 4`)

Serial profile at `bfb7bdf`, in seconds:

| Stage | Seconds |
|---|---:|
| `vcf_seconds` | 37.21 |
| `array_seconds` | 0.99 |
| `chunk_seconds` | 6.54 |
| `write_io_seconds` | 6.61 |
| **`total_seconds`** | **51.64** |

Cache prototype after the fix:
- 0/103 mismatches.
- Caches built with 4, 16 and 32 tiles were identical to the serial build.
- Build times were 10.77 s, 10.31 s and 9.94 s for 4, 16 and 32 tiles over
  4 processes; 20.01 s serial.
- Per-partition path, serial: 37.44 s.

`ldetect run --workers 4`. Step 2 is the time from the Step 2 log line to the
Step 3 log line.

| Version | Warm-up | Rep 1 | Rep 2 | Peak RSS |
|---|---|---|---|---|
| main | 72.4 s, step2 36 s | 69.5 s, step2 35 s | 71.3 s, step2 35 s | ~627-644 MB |
| e997b51 | 69.8 s, step2 34 s | 66.9 s, step2 32 s | 70.7 s, step2 32 s | ~626-640 MB |
| bfb7bdf | 66.2 s, step2 30 s | 63.4 s, step2 29 s | 70.3 s, step2 29 s | ~579-595 MB |

Wall times were higher and noisier than §2 (node shared). main vs bfb7bdf was
bitwise identical in both reps.

## 6. Worker scaling (`scaling_chr21.sh`, `-c 8`)

The allocation was CPUs 0-3 and 128-131: 4 physical cores and their
hyperthread siblings. Code at `29fe518`, one rep each after a warm-up per input.

| Input | Workers | Step 2 | Wall | CPU | Peak RSS (KB) |
|---|---:|---:|---:|---:|---:|
| VCF | 1 | 75 s | 2:35.7 | 98% | 544,696 |
| VCF | 2 | 38 s | 1:24.2 | 186% | 535,328 |
| VCF | 4 | 20 s | 0:47.7 | 339% | 530,316 |
| VCF | 8 | 16 s | 0:38.5 | 620% | 532,156 |
| BCF | 1 | 46 s | 2:06.9 | 98% | 539,472 |
| BCF | 2 | 24 s | 1:09.8 | 184% | 541,500 |
| BCF | 4 | 13 s | 0:40.9 | 330% | 538,736 |
| BCF | 8 | 10 s | 0:32.7 | 600% | 538,308 |

The BED had the same md5 across all 8 runs. Node load rose from about 2 to
12-16 during the BCF runs because of other jobs. Scaling was still consistent.

**Conclusion:** the about 1.8x seen at 4 workers in §2/§5 was most likely the
`-c 4` allocation being 2 physical cores plus hyperthreads. The topology of
those allocations wasn't recorded. With proper scaling, the cache's expected
gain is about 4-5 s (VCF) or 1-2 s (BCF) at 4 workers, so it was shelved.
