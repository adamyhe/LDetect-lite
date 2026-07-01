# Tiled vectorized r2 calculation notes

## Would this be able to also work in tandem with the vectorized r2 calculation?

Yes, tiling is a natural fit here and maps directly onto the structure of the metric calculation.

## The memory problem precisely

For w=200 the `G @ G.T` approach allocates:
- `G`: 2w × n_samples × 8 bytes = 400 × 2500 × 8 = **8MB**
- `R`: (2w)² × 8 bytes = 400² × 8 = **1.28MB**

Neither is large for a single candidate, but if you're parallelizing across candidates or chromosomes, these multiply. The more fundamental issue is that for larger w or n_samples, `G` grows linearly in both while `R` grows quadratically in w — and you often don't need all of `R` simultaneously. The metric typically operates on subblocks of `R` (the within-block and between-block submatrices on either side of the candidate breakpoint), not the full 2w×2w matrix.

## Tiled computation

Instead of computing the full `R` at once, compute it in tiles of shape (t, t) and stream each tile through the metric accumulator:

```python
def tiled_r2_metric(dosages_norm, lo, hi, p_local, tile_size=64):
    """
    Compute the block metric for candidate breakpoint at p_local
    (index within window [lo, hi]) using tiled r² computation.
    
    Never materializes the full (hi-lo) x (hi-lo) R matrix.
    """
    n_window = hi - lo
    metric_accumulator = MetricAccumulator(n_window, p_local)
    
    for i_start in range(0, n_window, tile_size):
        i_end = min(i_start + tile_size, n_window)
        G_i = load_dosage_block(dosages_norm, lo + i_start, lo + i_end)
        # shape: (tile_size, n_samples)
        
        for j_start in range(0, n_window, tile_size):
            j_end = min(j_start + tile_size, n_window)
            G_j = load_dosage_block(dosages_norm, lo + j_start, lo + j_end)
            # shape: (tile_size, n_samples)
            
            # Compute r² for this tile only
            R_tile = G_i @ G_j.T   # shape: (tile_size, tile_size)
            
            # Accumulate into metric — tile position tells us which
            # quadrant (within-left, within-right, between) this is
            metric_accumulator.update(R_tile, i_start, j_start)
    
    return metric_accumulator.result()
```

Peak memory is now `2 × tile_size × n_samples × 8` bytes for the two `G` blocks plus `tile_size² × 8` for the tile — for tile_size=64 and n_samples=2500, that's **3.2MB total** regardless of window size.

## Exploiting symmetry to halve the work

`R` is symmetric, so you only need to compute the upper triangle:

```python
for i_start in range(0, n_window, tile_size):
    i_end = min(i_start + tile_size, n_window)
    G_i = load_dosage_block(dosages_norm, lo + i_start, lo + i_end)
    
    for j_start in range(i_start, n_window, tile_size):  # j >= i only
        j_end = min(j_start + tile_size, n_window)
        G_j = load_dosage_block(dosages_norm, lo + j_start, lo + j_end)
        
        R_tile = G_i @ G_j.T
        
        if i_start == j_start:
            # Diagonal tile — only upper triangle is new
            metric_accumulator.update(R_tile, i_start, j_start, diagonal=True)
        else:
            # Off-diagonal — contribute tile and its transpose
            metric_accumulator.update(R_tile, i_start, j_start)
            metric_accumulator.update(R_tile.T, j_start, i_start)
```

This halves both the number of `dgemm` calls and the total data read from `dosages_norm`.

## Tile size selection

Tile size controls the memory/compute trade-off:

```python
def optimal_tile_size(n_samples, target_memory_mb=32, l3_cache_bytes=8*1024*1024):
    # Each G block: tile_size × n_samples × 8 bytes
    # Two blocks fit in target memory
    memory_limited = int((target_memory_mb * 1024**2) / (2 * n_samples * 8))
    
    # For BLAS efficiency, tile should fit two G blocks + R tile in L3
    # 2 * tile * n_samples * 8 + tile² * 8 <= l3_cache_bytes
    # Approximate: dominated by G blocks for typical n_samples
    cache_limited = int(l3_cache_bytes / (2 * n_samples * 8))
    
    # Round down to multiple of BLAS block size (typically 32 or 64)
    size = min(memory_limited, cache_limited)
    return max(32, (size // 32) * 32)
```

For n_samples=2500 and 8MB L3 cache, this gives tile_size≈205, which you'd round down to 192. For a 32MB memory target it gives tile_size≈819, rounded to 800. In practice 64–256 is a good range — large enough for BLAS efficiency, small enough to stay in cache.

## The MetricAccumulator interface

The key design requirement is that the metric can be computed incrementally from tiles. For the LDetect block metric (which compares mean r² within each half of the window against mean r² across the boundary), this is straightforward because it's a sum of r² values partitioned by quadrant:

```python
class MetricAccumulator:
    def __init__(self, n_window, p_local):
        self.p = p_local      # breakpoint position within window
        self.sum_left = 0.0   # sum of r² within left block
        self.sum_right = 0.0  # sum of r² within right block  
        self.sum_cross = 0.0  # sum of r² across boundary
        self.n_left = 0
        self.n_right = 0
        self.n_cross = 0
    
    def update(self, R_tile, i_start, j_start, diagonal=False):
        i_end = i_start + R_tile.shape[0]
        j_end = j_start + R_tile.shape[1]
        p = self.p
        
        # Determine which quadrant(s) this tile falls in
        # and accumulate sums accordingly
        for qi, (i0, i1) in enumerate([(i_start, i_end)]):
            for qj, (j0, j1) in enumerate([(j_start, j_end)]):
                i_left  = (i0 < p) and (i1 <= p)
                i_right = (i0 >= p)
                j_left  = (j0 < p) and (j1 <= p)
                j_right = (j0 >= p)
                
                if i_left and j_left:
                    mask = np.ones_like(R_tile, dtype=bool)
                    if diagonal:
                        np.fill_diagonal(mask, False)
                    self.sum_left += R_tile[mask].sum()
                    self.n_left += mask.sum()
                elif i_right and j_right:
                    mask = np.ones_like(R_tile, dtype=bool)
                    if diagonal:
                        np.fill_diagonal(mask, False)
                    self.sum_right += R_tile[mask].sum()
                    self.n_right += mask.sum()
                else:
                    # Cross-boundary tile
                    self.sum_cross += R_tile.sum()
                    self.n_cross += R_tile.size
    
    def result(self):
        mean_left  = self.sum_left  / self.n_left  if self.n_left  else 0
        mean_right = self.sum_right / self.n_right if self.n_right else 0
        mean_cross = self.sum_cross / self.n_cross if self.n_cross else 0
        return mean_left + mean_right - 2 * mean_cross
```

Tiles that straddle the breakpoint boundary need to be split — the tile is partitioned into left-of-p and right-of-p subregions before accumulation. This adds some index arithmetic but no additional memory allocation.

## Interaction with Zarr chunk shape

The tiled access pattern interacts well with the `dosages_norm` Zarr chunk shape recommendation from earlier (one chunk per SNP row, shape `(1, n_samples)`). Each `load_dosage_block` call reads `tile_size` consecutive chunks, which Zarr will batch into a contiguous read. Aligning `tile_size` to a multiple of the Zarr chunk size (trivial here since chunks are single rows) means no chunk is decompressed more than once per tile pass.

## Net memory profile

| Approach         | Peak memory (w=200, n_samples=2500) |
| ---------------- | ----------------------------------- |
| Full `G @ G.T`   | ~9.3MB per candidate                |
| Tiled (tile=64)  | ~3.3MB regardless of w              |
| Tiled (tile=256) | ~11MB regardless of w               |

The crossover where tiling saves memory is at `tile_size < 2w` — for w=200 any tile smaller than 400 reduces peak memory. The more significant benefit appears at larger w or n_samples, where the full approach scales as O(w × n_samples) for `G` and O(w²) for `R`, while tiling keeps both terms bounded by tile_size.