# Step 2 (covariance) profiling — findings

**Findings summary (current as of 2026-11-04).** Distilled for human review.
Dated raw log with job IDs, commands and every measurement:
[`notes/logs/step2-covariance-profiling-2026-11.md`](../logs/step2-covariance-profiling-2026-11.md).
Human-facing summaries of the two shipped changes: `docs/optimizations.md` #19
and #20.

## Where Step 2 time goes

Serial Step 2 over all 103 EUR chr21 partitions (1000 Genomes Phase 1, 379
individuals = 758 haplotypes, 377,790 partition-SNPs, 281M emitted rows):

| Stage | Before (e997b51) | After (bfb7bdf) |
|---|---:|---:|
| VCF read (`read_reference_panel`) | 39.8 s | 37.2 s |
| Array building (`_build_covariance_inputs`) | 24.4 s | 1.0 s |
| Bitpacked popcount kernel | 6.3 s | 6.5 s |
| HDF5 write | 6.6 s | 6.6 s |
| Packing | 0.3 s | 0 (done during decode) |
| **Total** | **77.4 s** | **51.6 s** |

The LD kernel is not the bottleneck. Getting genotypes out of the VCF is. About
75% of the remaining VCF-read time is htslib iterating and parsing text records
before any genotype is accessed.

## Shipped

1. **Sort-free unique/count** (`e997b51`, optimizations #19). Replaced
   `np.unique` sorts on already-sorted covariance rows with O(n) run-boundary
   passes. Writer 2.4x faster per row. End to end on chr21, `--workers 4`:
   total 50.5 s → 48.1 s.
2. **Pack haplotypes during decode** (`bfb7bdf`, optimizations #20). Removed
   the `list[list[int]]` round-trip. `read_reference_panel(layout="packed")`
   packs bounded `uint8` blocks into `uint64` words with `np.packbits`. Serial
   Step 2 is 33% faster, and peak RSS at 4 workers is about 8% lower.

Both are bitwise exact on chr21: all 103 partitions, the vector, breakpoint
JSON and BED match `main`. The BED also matches the published Berisa & Pickrell
EUR `fourier_ls` blocks for chr21.

## Recommended to users (README)

**Convert the reference panel to a GT-only BCF**
(`bcftools annotate -x INFO,^FORMAT/GT -Ob`). Genotype reading for all chr21
partitions drops from 46.1 s to 11.3 s (serial, 3.5-4x). Step 2 at 4 workers
drops from 20 s to 13 s. The conversion takes 15 s once. The BED was
byte-identical between VCF and BCF input at every worker count tested.

## Ruled out

- **Faster popcount.** LLVM already lowers `_popcount64`'s SWAR code to one
  hardware `popcnt` instruction, as the disassembly confirms. The cbsubscb
  nodes have no AVX-512 VPOPCNTDQ.
- **Frequency-bound pair skipping.** It skips pairs that can't pass the cutoff
  given allele frequencies. It was exact but only 1.08-1.14x on the kernel,
  which is 8% of Step 2.
- **Whole-chromosome packed haplotype cache.** Prototyped, exact, then
  shelved. It decodes each chromosome once instead of about 1.55x, because
  partitions overlap. It reproduces tabix overlap semantics, per-region skip
  counts and per-region dedupe, and matched the per-partition path on all 103
  partitions. It halves decode work (20.0 s vs 37.4 s serial from VCF; 6.4 s vs
  11.3 s from BCF). But Step 2 already scales well across cores, so the
  expected wall-time gain at 4 workers is only about 4-5 s of 20 s on VCF and
  about 1-2 s of 13 s on BCF. That isn't worth restructuring `cmd_run`.
  - **If revisited** (for example, panels with many more samples, where decode
    dominates): share the cache through a read-only memory-mapped `.npy` in
    node-local scratch, not by pickling it to each worker.
  - **Size:** `n_records × ceil(2N/64) × 8` bytes. About 21 MiB for EUR chr21,
    about 200 MB estimated for chr2, and about 3.5 GB for chr2 at N = 10,000.

## Worker scaling

EUR chr21 `ldetect run` on a Slurm `-c 8` allocation of cbsubscb19 (4 physical
AMD EPYC 9554 cores plus their hyperthread siblings):

| Input | 1 | 2 | 4 | 8 (HT) workers |
|---|---:|---:|---:|---:|
| `.vcf.gz` Step 2 | 75 s | 38 s | 20 s | 16 s |
| GT-only `.bcf` Step 2 | 46 s | 24 s | 13 s | 10 s |

Step 2 scales about 3.7x on 4 physical cores. Hyperthreads add about 25%.

**Slurm `-c` counts hardware threads on this cluster.** For N workers on
physical cores, request `-c 2N`. Earlier `-c 4` measurements on this branch
showed about 1.8x at 4 workers, most likely because that allocation was 2
physical cores plus hyperthreads (not verified at the time). Treat those
4-worker numbers as understating real throughput. Ratios between versions in
the same job remain valid.

## Next target

With BCF input and 8 workers, the whole chr21 run takes 33 s, of which Step 2
is 10 s. Most of the rest is Step 4 (metric + local search), which none of this
work touched.
