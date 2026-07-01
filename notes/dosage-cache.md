# Dosage caching notes

## Can the normalized dosage vectors be cached in such a way that 1) maintains exactness and 2) correctly handles duplicate positions? i.e., first-retained-pair behavior?

These are two distinct requirements that interact in an important way.

## Exactness

The normalization that matters for r² is:

```python
g_norm = g - g.mean()
g_norm /= np.sqrt((g_norm**2).sum())  # unit vector form
```

With this form, r²(i,j) = (g_norm_i @ g_norm_j)². This is exact in float64 with no approximation, and the normalized vectors are safe to cache because normalization is idempotent — you normalize once at cache-insertion time and never need to renormalize. The only precision concern is the order of operations: compute the mean and norm from the raw integer/float dosages before any other transformation, not from already-rounded values. `cyvcf2`'s `gt_dosages` returns float64 directly from the BCF, so this is fine as long as you don't intermediate through float32.

One subtle exactness issue: if you normalize to unit vectors, you lose the ability to detect zero-variance SNPs (monomorphic sites), which should yield r²=0 against everything. You need to handle this before normalizing:

```python
def normalize_dosages(g):
    g = g - g.mean()
    norm = np.sqrt((g**2).sum())
    if norm < 1e-10:  # monomorphic
        return None   # sentinel for zero-variance
    return g / norm
```

Storing `None` as a sentinel in the cache is cleaner than storing a zero vector, which would silently produce r²=0 rather than signaling that the site is degenerate.

## Duplicate positions

This is the more significant problem. Duplicate positions arise when multiple variants share the same `POS` field — multiallelic sites split into biallelics, indels overlapping SNPs, or genuine multi-allelic SNPs. If your cache is keyed by position, last-write-wins, which is arbitrary. First-retained behavior requires a different keying strategy.

The correct key is not position alone but the **order of encounter in the VCF**, which encodes the implicit first-retained semantics:

```python
dosage_cache = {}  # keyed by (chrom_idx, pos, ref, alt) or by ordinal index

for variant in vcf(region):
    # Use a compound key that's unique per variant record
    key = (variant.POS, variant.REF, tuple(variant.ALT))
    
    # First-retained: skip if position already seen
    if variant.POS in seen_positions:
        continue
    seen_positions.add(variant.POS)
    
    g_norm = normalize_dosages(variant.gt_dosages)
    dosage_cache[key] = g_norm
```

But this raises a deeper question: **what does first-retained mean in the context of the covariance computation?** The covariance pass that produced the vector must have made the same filtering decision — if it retained the first variant at each duplicated position, the local search cache must retain exactly the same variant. If the filtering logic diverges between the two passes, r² values computed during local search will not correspond to the covariance structure encoded in the vector.

This means the deduplication logic needs to be **extracted into a single shared function** called identically in both passes:

```python
def iter_partition_variants(vcf_path, chrom, start, stop, snps_needed=None):
    """
    Single canonical iterator over variants in a partition.
    Enforces first-retained-per-position deduplication consistently
    across both the covariance pass and the local search pass.
    
    snps_needed: if provided, only yield variants at these positions.
    """
    vcf = cyvcf2.VCF(vcf_path)
    seen_positions = set()
    
    for variant in vcf(f"{chrom}:{start}-{stop}"):
        if variant.POS in seen_positions:
            continue
        seen_positions.add(variant.POS)
        
        if snps_needed is not None and variant.POS not in snps_needed:
            continue
        
        g = variant.gt_dosages
        g_norm = normalize_dosages(g)
        yield variant.POS, g_norm
```

Both the covariance computation and the local search cache population call `iter_partition_variants` with the same arguments. Deduplication behavior is guaranteed identical because it lives in one place.

## What to actually store in Zarr

Given exactness and deduplication requirements, the cache entries per SNP are:

```
partitions/
  0/
    .zattrs          # start, stop, band_width, vcf_path
    vector/          # shape (2*n-1,)
    snp_positions/   # shape (n,) int64 — genomic positions in encounter order
    snp_ref/         # shape (n,) variable-length string — for duplicate resolution
    snp_alt/         # shape (n,) variable-length string
    dosages_norm/    # shape (n, n_samples) float64 — normalized, deduped, encounter-ordered
```

Storing `snp_positions` in encounter order (not sorted by position, since VCF order is already position-sorted after deduplication) means the ordinal index into `dosages_norm` is the canonical SNP identifier throughout the pipeline, with position/ref/alt as metadata for cross-referencing. This eliminates position-as-key entirely from the hot path — local search works in index space, not position space.

The `dosages_norm` array is also substantially smaller than you might expect: for n=2000 SNPs and 2500 samples in float64, it's 40MB per partition — larger than the vector but far smaller than any banded covariance matrix, and compressible with Blosc since normalized dosage vectors have smooth structure across samples.

## The complete picture

The two requirements together push toward the same solution: a canonical ordered iterator that enforces first-retained deduplication, normalizes in float64 at point of extraction, and produces an ordinal-indexed dosage matrix that both the covariance pass and the local search treat as the ground truth. Position, ref, and alt are stored as metadata for provenance but never used as cache keys in the hot path.