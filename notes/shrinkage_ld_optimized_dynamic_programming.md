# Shrinkage-Regularized LD Blocks with Optimized Dynamic Programming

## Overview

A promising extension of LDetect-lite is to combine its Wen–Stephens shrinkage LD estimator with a rewritten segmentation algorithm tailored to sparse, banded LD. This would retain LDetect's regularization and end-to-end reference-panel processing while replacing its final heuristic boundary selection with global optimization.

The proposed pipeline is:

```text
Phased VCF/BCF + genetic map
    -> shrinkage covariance estimates
    -> standardized sparse shrinkage correlations
    -> optimized segmentation
    -> BED-formatted LD blocks
```

This is more attractive than adapting the current `snp_ldsplit()` implementation literally because the optimization can operate directly on LDetect-lite's canonical sparse pair representation, avoid large intermediate cost matrices, support physical or genetic size constraints, and compute only the solution that is actually needed.

## Statistical input

Let the Wen–Stephens shrinkage covariance estimate be

$$
\widehat{\Sigma}^{\mathrm{shrink}}.
$$

The splitting objective should normally use standardized correlations rather than raw covariances:

$$
R^{\mathrm{shrink}}_{ij}
=
\frac{\Sigma^{\mathrm{shrink}}_{ij}}
{\sqrt{\Sigma^{\mathrm{shrink}}_{ii}\Sigma^{\mathrm{shrink}}_{jj}}}.
$$

Using covariance directly would implicitly weight the objective by allele-frequency-dependent variances. Standardization preserves the interpretation of minimizing squared correlations between blocks.

Define the nonnegative pair weight

$$
w_{pq}=\left(R^{\mathrm{shrink}}_{pq}\right)^2.
$$

For an interval containing variants $a,\ldots,b$, define the LD retained inside the interval as

$$
W(a,b)=\sum_{a\leq p<q\leq b} w_{pq}.
$$

Because the total LD weight is fixed, minimizing LD between blocks is equivalent to maximizing the LD retained within blocks.

## Exact-$K$ dynamic programming

For exactly $K$ blocks, an interval-segmentation recurrence is

$$
D(k,b)=
\max_{a\in\mathcal A(b)}
\left\{
D(k-1,a-1)+W(a,b)
\right\},
$$

where:

- $D(k,b)$ is the maximum within-block weight for partitioning variants $1,\ldots,b$ into $k$ blocks;
- $a$ is the first variant of the final block;
- $\mathcal A(b)$ contains starts satisfying the block-size constraints;
- $W(a,b)$ is the squared shrinkage-correlation weight retained in the final block.

Backpointers recover the optimal boundaries. This formulation avoids explicitly constructing the cross-block error matrix used in the published `snp_ldsplit()` formulation.

## Primary optimizations

### 1. Consume sparse shrinkage pairs directly

LDetect-lite's compact covariance caches already contain the essential data:

```text
snp_i, snp_j, shrink_ld
```

After covariance-to-correlation standardization, the optimizer can consume canonical triples of the form

```text
i, j, squared_shrinkage_correlation
```

This avoids materializing:

- a dense chromosome-wide matrix;
- duplicate entries from overlapping partitions;
- a large matrix containing every permissible interval cost.

Each pair must appear exactly once. Partition-local indices should be translated to chromosome-wide indices, and the designated canonical copy should be retained in overlap regions.

### 2. Generate interval scores during the DP sweep

For a fixed block start $a$,

$$
W(a,b+1)=W(a,b)+\sum_{p=a}^{b}w_{p,b+1}.
$$

Sparse per-column prefix structures can make the added term inexpensive. Interval scores can therefore be generated during a DP sweep instead of being stored for every possible $(a,b)$ pair.

The target memory footprint becomes approximately:

- sparse pair weights;
- one or two DP score vectors;
- compressed backpointers;
- temporary prefix or sweep state.

### 3. Optimize one required solution instead of every $K$

The existing `snp_ldsplit()` implementation computes solutions for every block count through `max_K`. Most applications ultimately consume one partition.

A penalized formulation removes the explicit $K$ dimension:

$$
F(b)=
\max_a
\left\{
F(a-1)+W(a,b)-\lambda
\right\},
$$

where $\lambda$ is a penalty per block. Larger values produce fewer blocks. Parametric search over $\lambda$ can target:

- a desired block count;
- an acceptable cross-block cost;
- a downstream memory budget;
- a Pareto frontier of block count versus retained LD.

If an exact block count is required, an "aliens trick" or related Lagrangian method may recover it while retaining near-linear DP memory. Correct tie handling is necessary.

### 4. Restrict the optimization to plausible candidate boundaries

LDetect's smoothed diagonal-sum signal can serve as a candidate generator:

1. Calculate the shrinkage LD signal.
2. Retain local minima as candidate cut locations.
3. Include small neighborhoods around each minimum.
4. Add chromosome endpoints and any mandatory boundaries.
5. Run exact DP over the candidate set.

If a chromosome contains 100,000 variants but only a few thousand plausible cut positions, this can reduce the state space dramatically.

The resulting solution is globally optimal over the candidate set, not necessarily over every SNP position. Candidate density should be increased until the selected boundaries and held-out cost stabilize.

This hybrid uses LDetect for statistically motivated candidate generation and dynamic programming for global subset selection. It can replace repeated local-search refinement.

### 5. Support physical and genetic constraints

Constraints expressed only in numbers of variants change meaning when panel density changes. A rewritten optimizer should support:

- minimum and maximum block width in base pairs;
- minimum and maximum block width in centimorgans;
- optional minimum and maximum SNP counts;
- maximum estimated downstream matrix size;
- mandatory or forbidden boundary intervals.

This would improve stability across arrays, imputed panels, and sequencing datasets.

### 6. Investigate Monge or quadrangle structure

Because every pair weight is nonnegative, the interval score has useful additivity and may satisfy the monotonicity conditions required for divide-and-conquer DP optimization or SMAWK.

If the required condition can be proved under the selected block constraints, each exact-$K$ DP layer could potentially be reduced from a full interval search to a substantially smaller search.

This must be established formally and tested on adversarial sparse matrices. Empirically monotone split positions are not sufficient evidence for algorithmic correctness.

## Proposed implementation architecture

```text
1. Reference reader
   - Read phased VCF/BCF and selected samples.
   - Apply variant and allele-frequency filters.

2. Shrinkage estimator
   - Process overlapping local partitions.
   - Produce canonical covariance or LD pairs.

3. Correlation standardizer
   - Normalize covariance by diagonal variances.
   - Remove undefined variants.
   - Handle minor numerical excursions outside [-1, 1].

4. Sparse weight builder
   - Square correlations.
   - Apply an optional small r-squared threshold.
   - Build column-oriented prefix or sweep structures.

5. Candidate generator
   - Calculate the LDetect diagonal-sum signal.
   - Smooth the signal and retain plausible troughs.

6. Segmentation engine
   - Exact-K DP, penalized DP, or candidate-restricted DP.
   - Use physical, genetic, and/or SNP-count constraints.
   - Record compressed backpointers.

7. Output and diagnostics
   - Write BED blocks.
   - Report retained and cross-block LD.
   - Report block sizes, runtime, memory, and parameter provenance.
```

## Parallelization

The most parallelizable components are:

- chromosomes;
- shrinkage covariance partitions;
- sparse weight construction;
- candidate-signal calculation;
- interval-score preparation.

The main DP recurrence retains sequential dependencies across endpoints or block counts, although candidate evaluation within a layer may be vectorized or parallelized. In practice, reducing the state space and memory traffic may be more valuable than aggressively parallelizing the DP itself.

## Validation design

The comparison should separate the effects of the estimator and the splitting algorithm.

| Condition | LD estimator | Boundary algorithm |
|---|---|---|
| A | Raw Pearson correlation | `snp_ldsplit()` |
| B | Wen–Stephens shrinkage correlation | `snp_ldsplit()` |
| C | Wen–Stephens shrinkage correlation | LDetect heuristic/local search |
| D | Wen–Stephens shrinkage correlation | Rewritten exact DP |
| E | Wen–Stephens shrinkage correlation | Candidate-restricted DP |

All methods should use the same:

- ancestry-matched reference population;
- sample and variant quality control;
- variant scaffold;
- target block count or comparable complexity constraint;
- permitted physical or genetic block sizes.

### Held-out evaluation

Boundaries must be evaluated on individuals not used to estimate them. Otherwise, a dynamic-programming method is guaranteed to look best under the same objective it optimized.

Recommended metrics include:

- cross-block squared correlation on held-out individuals;
- percentage of LD weight retained within blocks;
- maximum individual boundary cost;
- boundary stability across reference-panel bootstrap samples;
- distance to recombination hotspots;
- downstream runtime and memory for blockwise matrix operations;
- wall time, aggregate CPU time, and peak memory for block construction.

### Density sensitivity

Repeat the analysis at several common-variant scaffold densities while holding block count or physical/genetic constraints constant. This determines the point at which additional marker density stops materially changing boundaries or held-out independence.

## Expected benefits

The proposed system could provide:

- small-panel robustness from population-genetic LD shrinkage;
- globally optimized boundaries for an explicit objective;
- bounded-memory, sparse computation;
- direct VCF/BCF-to-BED operation;
- density-invariant physical and genetic constraints;
- compatibility with established LDetect coordinate workflows;
- transparent separation of estimator and segmentation effects;
- reproducible runtime, memory, and quality diagnostics.

## Important limitations

- Global optimality applies only to the declared objective and constraints.
- Candidate restriction sacrifices exactness over all SNP positions.
- Shrinkage parameters may affect the selected boundaries.
- Aggressive sparsification can change the optimization target.
- Penalized DP may require careful handling to recover an exact block count.
- A same-matrix evaluation is insufficient; held-out validation is essential.
- Monge-style acceleration must not be used without proof of its assumptions.

## Positioning

The resulting method would be more than a modern reproduction of LDetect. Its defensible description would be:

> An end-to-end, shrinkage-regularized LD block estimator with globally optimized boundaries, sparse bounded-memory execution, density-aware constraints, and direct phased-reference-to-BED operation.

This positioning does not claim that the original LDetect boundary heuristic is superior to `snp_ldsplit()`. Instead, it combines the strongest separable components: LDetect's shrinkage estimation and reference-panel workflow with a modern global segmentation algorithm.

## References

- Berisa, T. and Pickrell, J. K. (2016). [Approximately independent linkage disequilibrium blocks in human populations](https://academic.oup.com/bioinformatics/article/32/2/283/1743626). *Bioinformatics*, 32(2), 283–285.
- Privé, F. (2022). [Optimal linkage disequilibrium splitting](https://academic.oup.com/bioinformatics/article/38/1/255/6321454). *Bioinformatics*, 38(1), 255–256.
- [`snp_ldsplit()` documentation](https://privefl.github.io/bigsnpr/reference/snp_ldsplit.html).
- [LDetect-lite project documentation](https://pypi.org/project/ldetect-lite/).
