# MacDonald2022 Boundary Diagnostics

**Agent-oriented working log.** Raw, dated investigation notes — not proofread for external readability. For current, human-readable status, see `notes/findings/`.

## Focused pyrho stage diagnostics (2026-07-26)

After deCODE/EUR was brought to exact replication except for the tiny chr7
edge-block convention, the remaining question was whether pyrho residual
divergence is a postprocessing artifact or an upstream breakpoint-placement
artifact. A focused Snakemake diagnostic was added for the chromosomes driving
the current pyrho mismatch:

```yaml
pyrho_stage_diagnostics:
  targets:
    pyrho_AFR: [9]
    pyrho_EAS: [4, 9, 17]
    pyrho_EUR: [9, 12, 19]
  tolerance: 50000
  window: 2000000
```

Run:

```bash
cd examples/MacDonald2022
uv run snakemake --cores N pyrho_stage_diagnostics
```

Outputs:

```text
results/compare/diagnostics/raw/{block_set}/chr{chrom}.boundary_diagnostics.tsv
results/compare/diagnostics/final/{block_set}/chr{chrom}.boundary_diagnostics.tsv
```

The per-chromosome rules consume the raw per-chromosome ldetect BED or the
postprocessed final per-chromosome BED, compare against MacDonald's published
reference BED, and pass:

- the block-set genetic map used by the run,
- one-based filtered VCF SNP positions,
- the hg38 centromere intervals.

This matters because the existing all-chromosome diagnostics could classify
boundary mismatches but did not annotate whether a boundary lay on a filtered
SNP/map position or in a local map/SNP desert.

Downloaded diagnostic bundle reviewed locally under:

```text
diagnostics/raw/...
diagnostics/final/...
```

Aggregate result:

| stage | rows | shifted | extra | missing | nonreciprocal | boundaries on SNP | boundaries on map | within 2 Mb centromere | in centromere |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 310 | 262 | 21 | 22 | 5 | 310 | 310 | 11 | 1 |
| final | 309 | 262 | 21 | 16 | 10 | 309 | 309 | 10 | 0 |

Interpretation:

- Raw and final diagnostics are nearly identical, so centromere removal and
  small-block merging are not the source of the remaining pyrho divergence.
- The dominant class is reciprocal `shifted_boundary`, not extra/missing
  segmentation. This is mostly two pipelines choosing different nearby
  candidate boundaries, not one pipeline systematically over- or under-splitting.
- Every diagnostic boundary lies exactly on a filtered SNP and exactly on the
  interpolated map position. This rules out an obvious BED/SNP coordinate
  artifact for these mismatches.
- Centromere/desert effects explain a minority of the largest individual
  offsets, especially around chr9, but they do not explain the broad shifted
  boundary burden across EAS chr4, EAS chr17, EUR chr12, or EUR chr19.

Per-target summaries from the downloaded final diagnostics:

| block set | chrom | rows | dominant class | median offset bp | p90 offset bp | max offset bp | notes |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `pyrho_AFR` | chr9 | 56 | shifted (48/56) | 380,497 | 733,690 | 3,682,809 | largest nonreciprocal near chr9 centromere/desert |
| `pyrho_EAS` | chr4 | 83 | shifted (68/83) | 529,085 | 1,105,988 | 2,113,924 | broad chromosome-wide shifted-boundary burden |
| `pyrho_EAS` | chr9 | 68 | shifted (60/68) | 460,439 | 1,079,673 | 5,230,744 | includes large chr9 centromere/desert nonreciprocal |
| `pyrho_EAS` | chr17 | 21 | shifted (18/21) | 701,634 | 2,497,016 | 4,493,648 | a few very large nonreciprocal/shifted events |
| `pyrho_EUR` | chr12 | 28 | shifted (26/28) | 582,742 | 1,139,876 | 2,390,308 | mostly reciprocal shifts; one near-centromere nonreciprocal |
| `pyrho_EUR` | chr19 | 31 | shifted (26/31) | 349,639 | 975,688 | 1,200,952 | mostly reciprocal shifts; several near centromere |
| `pyrho_EUR` | chr9 | 22 | shifted (16/22) | 612,691 | 1,580,536 | 8,137,388 | two very large nonreciprocal events, one sparse-region event |

Next diagnostic fork:

1. Compare MacDonald-published boundaries against `breakpoints-*.json`'s raw
   `fourier` candidates and refined `fourier_ls` candidates for these same
   chromosomes.
2. If `fourier` is already shifted, the cause is earlier than local search:
   covariance-vector values, filter-window/minima behavior, or input/map
   provenance.
3. If `fourier` is close and `fourier_ls` moves away, use
   `scripts/verify_local_search.py` to replay the local-search metric curve at
   the affected breakpoint(s), as was done previously for EAS chr4.

Run commands from:

```bash
cd /Users/adamhe/github/ldetect-lite/examples/MacDonald2022
export UV_CACHE_DIR=/Users/adamhe/github/ldetect-lite/.uv-cache
```

## Generate EUR diagnostics

The diagnostic Snakemake rules consume the downloaded combined BEDs as
snapshots. They do not rebuild missing per-chromosome pipeline intermediates.

```bash
uv run snakemake --cores 1 \
  results/compare/diagnostics/pyrho_EUR_boundary_diagnostics.tsv \
  results/compare/diagnostics/raw/pyrho_EUR_boundary_diagnostics.tsv \
  --shared-fs-usage input-output persistence software-deployment sources
```

Outputs:

```text
results/compare/diagnostics/pyrho_EUR_boundary_diagnostics.tsv
results/compare/diagnostics/raw/pyrho_EUR_boundary_diagnostics.tsv
```

Each row is a boundary more than 50 kb from its nearest counterpart.
`classification` is one of:

- `shifted_boundary`: the two boundaries are reciprocal nearest neighbors.
- `extra_split`: our internal boundary lies inside a reference block.
- `missing_split`: a reference boundary lies inside one of our blocks.
- `chromosome_edge_mismatch`: chromosome start/end coordinates differ.
- `nonreciprocal_boundary`: the nearest-boundary relationship is ambiguous.
- `unmatched_boundary`: the other BED has no boundaries on the chromosome.

## Inspect chr9

```bash
uv run python - <<'PY'
import pandas as pd

path = "results/compare/diagnostics/pyrho_EUR_boundary_diagnostics.tsv"
df = pd.read_csv(path, sep="\t")
chr9 = df[df["chrom"] == "chr9"]

print(chr9.groupby(["source", "classification"]).size())
print()
print(
    chr9.sort_values("abs_offset_bp", ascending=False)[
        [
            "source",
            "position",
            "nearest_position",
            "abs_offset_bp",
            "classification",
            "query_left_block",
            "query_right_block",
            "nearest_ref_left_block",
            "nearest_ref_right_block",
        ]
    ].to_string(index=False)
)
PY
```

## Add chr9 genetic-map context

Download only the published IBS chr9 map:

```bash
uv run snakemake --cores 1 \
  data/maps/pyrho_interpolated_maps/IBS/chr9.tab.gz \
  --shared-fs-usage input-output persistence software-deployment sources
```

Then generate a chr9-only enriched diagnostic:

```bash
REF=resources/pyrho_EUR_LD_blocks.bed
CENTROMERES=resources/hg38_centromeres.txt.gz
test -f "$REF" || REF=resources/resources/pyrho_EUR_LD_blocks.bed
test -f "$CENTROMERES" || \
  CENTROMERES=resources/resources/hg38_centromeres.txt.gz

uv run python scripts/diagnose_boundaries.py \
  --ours results/pyrho_EUR_LD_blocks.bed \
  --ref "$REF" \
  --centromeres "$CENTROMERES" \
  --genetic-map data/maps/pyrho_interpolated_maps/IBS/chr9.tab.gz \
  --chrom chr9 \
  --tolerance 50000 \
  --window 2000000 \
  --output results/compare/diagnostics/pyrho_EUR_chr9_map_diagnostics.tsv
```

Map-enriched columns report the number of map points and cM span within
plus/minus 2 Mb, plus the nearest map position and its distance.

## Add SNP-density context

This requires the filtered chr9 VCF:

```bash
bcftools query -f '%CHROM\t%POS\n' \
  data/filtered/chr9.vcf.gz \
  > results/compare/diagnostics/pyrho_EUR_chr9_snp_positions.tsv
```

Rerun the chr9 diagnostic with:

```bash
uv run python scripts/diagnose_boundaries.py \
  --ours results/pyrho_EUR_LD_blocks.bed \
  --ref "$REF" \
  --centromeres "$CENTROMERES" \
  --genetic-map data/maps/pyrho_interpolated_maps/IBS/chr9.tab.gz \
  --snp-positions \
    results/compare/diagnostics/pyrho_EUR_chr9_snp_positions.tsv \
  --chrom chr9 \
  --tolerance 50000 \
  --window 2000000 \
  --output results/compare/diagnostics/pyrho_EUR_chr9_enriched.tsv
```

Compare shifted boundaries against local map cM span and SNP count first.
Extra/missing splits should then be inspected as separate segmentation cases.

## EUR chr9 findings (2026-06-11)

The IBS chr9 map and the GRCh38 1000 Genomes chr9 VCF were downloaded and
queried at all mismatched boundary positions.

- All 29 unique candidate positions occur exactly in the IBS genetic map.
- All 29 occur in the VCF and pass MacDonald's global `AF >= 0.01` filter.
- All 29 are polymorphic among the same 417 EUR individuals used here.
- MacDonald's published scripts confirm the same global MAF filter, EUR sample
  composition (TSI, IBS, CEU, GBR; FIN excluded), `Ne=11418`, and covariance
  cutoff `1e-7`.

Generated files:

```text
results/compare/diagnostics/pyrho_EUR_chr9_map_diagnostics.tsv
results/compare/diagnostics/pyrho_EUR_chr9_boundary_variants.vcf.tsv
results/compare/diagnostics/pyrho_EUR_chr9_variant_diagnostics.tsv
results/compare/diagnostics/pyrho_EUR_chr9_selected_samples.tsv
```

These checks rule out missing map positions, the documented global MAF filter,
and simple absence or monomorphism in the selected EUR panel. The next
experiment should therefore be a chr9-only breakpoint parameter/implementation
comparison, beginning with `n_snps_bw_bpoints`, while holding the covariance
inputs fixed.

## Reference repository and paper audit (2026-06-11)

The April 24, 2023 paper version and the reference repository agree on the main
analysis parameters:

- EUR uses the IBS recombination map and 417 TSI/IBS/CEU/GBR samples.
- Variants are filtered at MAF 0.01.
- Chromosomes begin as 5,000-SNP partitions and are extended until the
  first/last shrunken covariance is below `1.5e-8`.
- Covariance uses `Ne=11418` for EUR and cutoff `1e-7`.
- Breakpoints use `fourier-ls` with an average block size of 7,000 SNPs.

The local workflow already uses these values. In particular,
`partition_chromosome()` defaults to 5,000 SNPs and `1.5e-8`, while
`config.yaml` sets `n_snps_bw_bpoints: 7000`.

### Published reference versions differ

There are two distinct published EUR BEDs:

| reference | git state | EUR blocks | centromere-overlap blocks |
| --- | --- | ---: | ---: |
| paper DOI tag / Zenodo | `fa695f7` (2023-02-09) | 1,360 | 24 |
| current GitHub `master` | after `5081b31` (2023-04-21) | 1,336 | 0 |

Commit `5081b31` says that centromere-overlapping blocks were removed. Applying
the current UCSC centromere intervals to the DOI-tagged EUR BED removes exactly
24 blocks and produces the current `master` BED exactly.

This reveals an internal publication mismatch: Table 2 reports 1,360 EUR
blocks, matching the DOI-tagged/Zenodo BED, while the paper text says
centromere-overlapping blocks were excluded, matching the later 1,336-block
GitHub BED.

The local reference currently matches GitHub `master`, not the DOI-tagged
paper table. Reference URLs should ultimately be pinned to an immutable commit
rather than `master`.

### Consequence for the chr9 diagnosis

Reference versioning explains the 24-block whole-genome count difference, but
it does not explain the shifted chr9 boundaries. On chr9 the later commit only
removes the single block spanning `43,269,938-69,277,370`; all non-centromeric
reference boundaries are otherwise unchanged.

The next diagnostic should therefore compare the chr9 breakpoint pipeline
against the reference implementation while keeping these fixed:

1. IBS map and the verified 417-sample filtered VCF.
2. `n_snps_bw_bpoints=7000`.
3. Raw, pre-centromere output.
4. DOI-tagged BED for paper-number replication and current `master` BED only
   for post-centromere replication.

Do not tune `n_snps_bw_bpoints` merely to fit the current reference. The paper
and scripts explicitly establish 7,000; the remaining likely causes are
implementation details in covariance, vectorization, Fourier-width selection,
or local search.

### Legacy LDetect implementation differences

MacDonald's README uses an unpinned `pip install ldetect`. PyPI's final
published release is `ldetect==0.2.5` (September 18, 2015), so that source is
the best available reproducible baseline.

A source comparison found:

- The Hanning filter, extrema detection, breakpoint-count formula, width
  search, and local-search intervals are structurally equivalent.
- Legacy metric and local-search calculations always use 50-digit
  `decimal.Decimal`. `ldetect-lite` defaults to the faster float/array path unless
  `--high-precision` is supplied.
- Legacy chromosome partitioning hardcodes `Ne=11418` in
  `P00_00_partition_chromosome.py`; the covariance script separately accepts
  population-specific `Ne`.
- `ldetect-lite run` currently passes the requested population `Ne` into both
  partitioning and covariance.

For EUR, the partition `Ne` is 11418 either way. The most focused EUR chr9
experiment is therefore a float-versus-`--high-precision` minima/local-search
comparison using identical saved covariance and vector inputs.

For AFR and EAS, partition files should additionally be compared against
legacy-compatible partitions generated with `Ne=11418`. Their covariance runs
should continue to use the population-specific `Ne` values.
