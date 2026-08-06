# LDetect-lite Covariance I/O Optimization Review

**External review, evaluated and partially implemented.** Not proofread for
external readability. For current, human-readable status of what was
actually verified/shipped versus rejected, see
`notes/findings/covariance-io-optimization.md` and `docs/optimizations.md`
item #18.

_Review date: 2026-08-05_

Polars-Bio is unlikely to improve the current BCF hot path. Its promising use here is as a VCF-Zarr front end, but direct Zarr array access would probably be faster for LDetect's packed-haplotype workload.

## What the code shows

The largest opportunity is eliminating repeated decoding, not replacing `cyvcf2` outright:

- Every partition task reopens the reference panel and rereads/decompresses the complete genetic map in [`cmd_run.py`](https://github.com/adamyhe/LDetect-lite/blob/main/src/ldetect_lite/_cli/cmd_run.py).
- Overlapping partitions consequently fetch and decode some genotypes multiple times.
- [`read_reference_panel()`](https://github.com/adamyhe/LDetect-lite/blob/main/src/ldetect_lite/_util/reference_panel.py) uses `variant.genotypes`, which creates Python lists, followed by a Python loop over every individual.
- Therefore `vcf_seconds` is not pure disk time: it includes GT decoding, Python-object creation, phasing validation, and haplotype-list construction.
- Similarly, `write_io_seconds` includes HDF5 compression, validation, `np.unique`, memory copies, and filesystem buffering—not just physical writes.

### Polars-Bio verdict

Polars-Bio's current format matrix lists plain/BGZF VCF and VCF Zarr, but not BCF. Multisample VCF genotypes are exposed as a nested table column, which adds Arrow/Polars materialization before they can be packed. See the [Polars-Bio reading documentation](https://biodatageeks.org/polars-bio/features/reading/).

VCF Zarr is more relevant because `call_genotype` and `call_genotype_phased` are typed arrays. However, Polars-Bio's own benchmark reports only about 1.2–1.3× for narrow GT-region queries versus BGZF VCF, and does not compare against BCF. See the [Polars-Bio VCZ benchmark](https://biodatageeks.org/polars-bio/blog/2026/05/12/polars-bio-0310-vcf-zarr-support-for-array-native-variant-analytics/).

Recommendations:

- Do not replace `cyvcf2`/BCF with Polars-Bio VCF.
- If testing VCZ, benchmark direct Zarr slicing against Polars-Bio VCZ. Direct access is the stronger candidate for the production path.
- Polars-Bio remains useful for exploratory filtering, joins, and cloud-facing workflows.

## Ranked optimizations

| Priority | Change | Why |
| --- | --- | --- |
| 1 | Stage a packed haplotype cache once per chromosome/population | Eliminates repeated BCF region decoding and directly feeds the existing bitpacked kernel |
| 2 | Replace `variant.genotypes` with `variant.genotype.array()` | Avoids Python list-of-lists and per-sample unpacking |
| 3 | Load the genetic map once per worker/run | Currently the entire gzipped map is parsed for every partition |
| 4 | Change covariance storage from physical-position edge rows to CSR/implicit-band storage | Writes fewer coordinate bytes and matches downstream row-range access |
| 5 | Benchmark faster HDF5 codecs and node-local uncompressed storage | Requires no new dependency because `hdf5plugin` is already installed |
| 6 | Consider PGEN or VCZ for persistent panel storage | Useful when the panel is reused across many populations/runs |
| 7 | Cython/CuPy | Only after the I/O and data-layout changes |

### 1. Packed input cache

The best hot-cache layout is approximately:

```text
positions          int32[n_variants]
genetic_positions  float64[n_variants]
haplotype_sums     uint16/uint32[n_variants]
packed_haplotypes  uint64[n_variants, ceil(n_haplotypes / 64)]
```

Read the chromosome once, write this cache to node-local NVMe, then let partition workers memory-map slices. This avoids:

- repeated BGZF/BCF decoding;
- repeated overlapping-region reads;
- Python genotype objects;
- the intermediate `uint8` haplotype matrix;
- repeated bitpacking.

NumPy supports memory-mapped `.npy` arrays directly, though without compression. See the [NumPy memory-map documentation](https://numpy.org/doc/stable/user/how-to-io.html).

For restart safety, key the cache by hashes of the panel, index, sample list, genetic map, and format version.

A smaller preliminary improvement is a physically sample-subset, GT-only BCF. `samples=` reduces returned samples but cannot avoid reading the bytes already stored in each compressed record.

### 2. Use cyvcf2's array interface

`cyvcf2` already provides a Cython-backed `variant.genotype.array()` returning an `int16` array containing alleles and phase, while `variant.genotypes` explicitly constructs Python lists. See the [cyvcf2 source](https://raw.githubusercontent.com/brentp/cyvcf2/main/cyvcf2/cyvcf2.pyx).

Prototype:

```python
vcf = cyvcf2.VCF(path, samples=individuals, lazy=True, threads=1)

gt = variant.genotype.array()[order]
alleles = gt[:, :2]
if np.any(gt[:, 2] == 0) or np.any(alleles < 0):
    continue

packed = np.packbits(
    (alleles.reshape(-1) != 0),
    bitorder="little",
)
```

Also benchmark:

- `lazy=True`, which delays FORMAT unpacking until a mapped position actually needs GT;
- `threads=2`, but only when the process count is reduced. Reader threads multiplied by partition workers can oversubscribe cores.

These options are documented in the [cyvcf2 API](https://brentp.github.io/cyvcf2/docstrings.html).

The existing [`profile_bitpack_ingestion.py`](https://github.com/adamyhe/LDetect-lite/blob/main/examples/ldetect_example/scripts/profile_bitpack_ingestion.py) is well positioned for this comparison, but its "direct pack" path still uses `variant.genotypes`.

### 3. Covariance cache layout

The current compact representation writes 16 raw bytes per pair:

```text
lo int32 + hi int32 + shrink_ld float64
```

Yet [`covariance_hdf5.py`](https://github.com/adamyhe/LDetect-lite/blob/main/src/ldetect_lite/io/covariance_hdf5.py) already writes `lo_offsets`, effectively most of a CSR index. A more natural schema is:

```text
positions  int32[n]
indptr     uint64[n+1]
indices    uint16[nnz] or uint32[nnz]
shrink_ld  float64[nnz]
diag       float64[n]
```

That is 10 bytes/pair with `uint16` indices or 12 with `uint32`, before compression. Actual disk savings will be smaller because coordinates compress well and existing measurements show `shrink_ld` already accounts for 82% of storage, but it also removes coordinate generation and three-stream writes.

Measure:

$$
\text{density}=\frac{n_{\text{emitted}}}{\sum_i (j_{\text{stop},i}-i)}
$$

If density exceeds roughly 80%, an implicit banded layout containing every candidate value—with filtered pairs represented by zero—may beat CSR because it needs only 8 bytes per candidate and no column indices.

### 4. Keep HDF5 initially, but expand codec choices

The current 65,536-row chunks are already reasonable: 256 KiB for each position dataset and 512 KiB for `shrink_ld`, within h5py's recommended range. See the [h5py chunking guidance](https://docs.h5py.org/en/stable/high/dataset.html).

Before changing containers, add benchmarks for:

- uncompressed;
- Zstd level 1 versus current level 3;
- Blosc2 + LZ4 + bitshuffle;
- Bitshuffle + LZ4.

`hdf5plugin`, already a dependency, supports these filters. See the [hdf5plugin usage guide](https://hdf5plugin.readthedocs.io/en/stable/usage.html).

For runs using `--delete-covariance-cache`, a two-tier design is attractive:

- hot, uncompressed CSR/mmap on `$SLURM_TMPDIR` or node-local NVMe;
- compressed HDF5 only when persistent restartable caching is requested.

Ordinary Zarr is unlikely to beat a single mmap/HDF5 file on local storage. Zarr becomes attractive for object storage or distributed access, and Zarr v3 sharding should be used to avoid thousands of small chunk files. See the [Zarr sharding documentation](https://zarr.readthedocs.io/en/latest/user-guide/arrays/).

### Persistent panel alternatives

- **PGEN/pgenlib:** Strong genotype-specific compression and phased-data support, with fast sample subsetting. It deserves a prototype, but exact haplotype ordering and LDetect's explicit-phasing rejection semantics need validation. See the [PLINK developer documentation](https://www.cog-genomics.org/plink/2.0/dev).
- **VCF Zarr via bio2zarr:** Best general-purpose choice for varying sample subsets, cloud storage, and reuse by other tools. Conversion can run in parallel. See the [bio2zarr documentation](https://sgkit-dev.github.io/bio2zarr/vcf2zarr/overview.html).
- **Custom packed mmap:** Likely fastest for LDetect specifically because it is already in the kernel's native representation.

Cython would help only if it fuses HTSlib genotype extraction, validation, and bitpacking without intermediate arrays. CuPy will not improve BCF reads or HDF5 writes; a GPU path should wait until input is cached and output is compacted, then use a tiled custom kernel rather than an $O(n^2)$ broadcast.

The first implementation to benchmark is: cached genetic map plus `genotype.array()`/`packbits`, followed by a node-local packed-panel mmap. That should establish whether a more invasive PGEN/VCZ or covariance-format rewrite is actually necessary.
