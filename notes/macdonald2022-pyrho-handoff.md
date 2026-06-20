# MacDonald2022 pyrho Handoff

Last updated: 2026-05-19

## Goal

Pick up the MacDonald2022 replication from a clean session and focus first on
improving the published pyrho LD block reproductions:

- `pyrho_AFR`
- `pyrho_EAS`
- `pyrho_EUR`

Set aside `pyrho_SAS` for now because MacDonald et al. do not appear to
document an SAS-specific effective population size. Also set aside the deCODE
`EUR` mismatch initially; it has a different failure mode.

## Current Worktree State

The current session has uncommitted MacDonald2022 workflow/diagnostic changes:

- `examples/MacDonald2022/Snakefile`
- `examples/MacDonald2022/config.yaml`
- `examples/MacDonald2022/scripts/compare_blocks.py`
- `examples/MacDonald2022/scripts/postprocess.py`
- `examples/MacDonald2022/scripts/compare_boundaries.py` (new)

Do not assume these are committed. Before changing behavior in a new session,
check:

```bash
git status --short
git diff -- examples/MacDonald2022
```

The local `uv` environment was synced with all extras/groups from `uv.lock`, and
`pandas 2.3.3`, `snakemake 9.19.0`, and `ruff` were verified importable through
`uv run`.

## Pipeline Changes Already Made

The MacDonald2022 workflow now has an `active_block_sets` layer in
`examples/MacDonald2022/config.yaml`.

Default active targets are:

```yaml
active_block_sets: [EUR, pyrho_AFR, pyrho_EAS, pyrho_EUR]
```

`pyrho_SAS` remains configured, but is excluded from `rule all`.

Centromere filtering is now block-set-specific:

- `EUR` / deCODE: `remove_centromere_blocks: false`
- `pyrho_AFR`: `true`
- `pyrho_EAS`: `true`
- `pyrho_EUR`: `true`
- `pyrho_SAS`: `true`

This was based on diagnostics showing:

- deCODE published blocks appear to retain centromere-spanning blocks relative
  to current UCSC centromere intervals.
- pyrho raw outputs have roughly one extra centromere-associated block per
  chromosome and improve after centromere filtering.

The Snakefile now writes two levels of block comparison:

- Final postprocessed BEDs:
  `results/compare/{block_set}_block_comparison.tsv`
- Raw pre-postprocess BEDs:
  `results/compare/raw/{block_set}_block_comparison.tsv`

It also writes nearest-boundary diagnostics:

- Final boundary offsets:
  `results/compare/boundaries/{block_set}_boundary_offsets.tsv`
- Raw boundary offsets:
  `results/compare/raw/{block_set}_boundary_offsets.tsv`

The new boundary diagnostic script is:

```text
examples/MacDonald2022/scripts/compare_boundaries.py
```

It outputs one row per boundary in both directions:

- `chrom`
- `source` (`ours_to_ref` or `ref_to_ours`)
- `position`
- `nearest_position`
- `signed_offset_bp`
- `abs_offset_bp`
- `within_tolerance`

## Current Comparison Summary

Final postprocessed comparisons downloaded under
`examples/MacDonald2022/results/compare`:

| block set | ours | ref | delta | mean recall | mean bp-Jaccard | mean median offset kb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `EUR` | 1362 | 1361 | +1 | 0.6296 | 1.0000 | 4.5682 |
| `pyrho_AFR` | 1580 | 1580 | 0 | 0.8741 | 0.9881 | 0.0000 |
| `pyrho_EAS` | 1118 | 1121 | -3 | 0.8252 | 0.9832 | 15.7091 |
| `pyrho_EUR` | 1335 | 1336 | -1 | 0.8644 | 0.9792 | 9.8682 |
| `pyrho_SAS` | 1267 | 1267 | 0 | 0.4337 | not in old schema | not in old schema |

Raw pre-postprocess comparisons:

| block set | ours | ref | delta | mean recall | mean bp-Jaccard | mean median offset kb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `EUR` | 1362 | 1361 | +1 | 0.6296 | 1.0000 | 4.5682 |
| `pyrho_AFR` | 1605 | 1580 | +25 | 0.8701 | 0.9216 | 0.0000 |
| `pyrho_EAS` | 1145 | 1121 | +24 | 0.8151 | 0.9165 | 15.7091 |
| `pyrho_EUR` | 1361 | 1336 | +25 | 0.8579 | 0.9193 | 9.8682 |

Final boundary-offset summaries (`ours_to_ref` only):

| block set | n boundaries | exact | <=1 kb | <=10 kb | <=50 kb | <=500 kb | max kb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EUR` | 1384 | 0.1149 | 0.4725 | 0.5838 | 0.6178 | 0.8475 | 2179.761 |
| `pyrho_AFR` | 1621 | 0.8223 | 0.8661 | 0.8748 | 0.8766 | 0.9340 | 7058.424 |
| `pyrho_EAS` | 1157 | 0.7917 | 0.8090 | 0.8176 | 0.8254 | 0.8989 | 7954.210 |
| `pyrho_EUR` | 1375 | 0.8356 | 0.8756 | 0.8815 | 0.8829 | 0.9404 | 8137.388 |

Interpretation:

- pyrho datasets are much closer than deCODE, often >79% exact boundary matches.
- Remaining pyrho disagreement is concentrated in a minority of boundaries and
  a handful of chromosomes.
- The centromere postprocessing mostly fixes the raw `+~1/chromosome` count
  issue for pyrho, but some chromosomes still lose or retain one extra boundary.

## Worst pyrho Chromosomes From Boundary Diagnostics

Worst by fraction of `ours_to_ref` boundaries within 50 kb:

### `pyrho_AFR`

- `chr9`
- `chr18`
- `chr22`
- `chr10`
- `chr11`

### `pyrho_EAS`

- `chr9`
- `chr4`
- `chr17`
- `chr14`
- `chr18`

### `pyrho_EUR`

- `chr19`
- `chr21`
- `chr22`
- `chr9`
- `chr16`

`chr9` is bad across all pyrho sets and is a good first target.

## Working Hypotheses For pyrho

The pyrho results are already close enough that a wholesale algorithm problem is
unlikely. More likely causes:

1. Postprocessing mismatch around centromeres and minimum-SNP block merging.
2. Boundary convention mismatch after removing centromere-spanning blocks.
3. Reference BEDs may be postprocessed from raw pyrho blocks with slightly
   different ordering: centromere removal before/after small-block merging, or
   special handling of blocks adjacent to removed centromeres.
4. A few map-population or population-panel choices may be subtly wrong, but the
   high exact-match rates make this less likely than postprocessing.

## Reference provenance correction (2026-06-11)

The paper/Zenodo release and current GitHub `master` are different reference
sets:

- The repository's `DOI` tag (`fa695f7`, dated 2023-02-09) has 1,360 EUR
  blocks, matching Table 2 in the April 24, 2023 paper.
- Commit `5081b31` (dated 2023-04-21) removed centromere-overlapping blocks,
  leaving 1,336 EUR blocks on current `master`.
- The DOI-tagged EUR BED contains exactly 24 blocks overlapping the current
  UCSC centromere intervals. Removing those blocks reproduces the current
  `master` EUR BED exactly.

The paper is internally inconsistent: its table matches the unfiltered DOI
BED, while its methods/results text says centromere-overlapping blocks were
excluded. Our downloaded reference is the 1,336-block current `master` version.

This provenance issue explains the postprocessing count difference but not the
remaining non-centromeric boundary shifts. For chr9, the two reference versions
differ only by removal of the centromere-spanning block
`43,269,938-69,277,370`.

Future comparisons should name and pin their target explicitly:

- Use the immutable `DOI` tag/Zenodo BED to reproduce the paper's reported
  block counts.
- Use commit `5081b31` or a later immutable commit to reproduce the
  centromere-filtered GitHub maps.
- Do not use a floating `master` URL for a reproducibility benchmark.

The audited paper parameters match the local workflow: 5,000-SNP initial
partitions, extension threshold `1.5e-8`, EUR `Ne=11418`, covariance cutoff
`1e-7`, IBS map, 417 EUR samples, MAF 0.01, `fourier-ls`, and 7,000 SNPs per
block. Further chr9 work should hold these fixed and compare implementation
stages rather than parameter-tuning the published value.

The MacDonald README installs legacy LDetect without a version pin. PyPI's
latest and final release is `ldetect==0.2.5`, uploaded September 18, 2015, and
is the best implementation baseline for direct comparisons.

Two relevant differences from the current port were found:

1. Legacy metric and local search always use 50-digit `Decimal`; `ldetect2`
   defaults to float unless `--high-precision` is requested. Test this on EUR
   chr9 using the same covariance/vector intermediates.
2. Legacy partitioning hardcodes `Ne=11418`, even when covariance later uses a
   population-specific `Ne`. This is identical for EUR but differs from the
   current AFR/EAS workflow, where `ldetect2 run` passes population `Ne` into
   partitioning. Generate legacy-compatible AFR/EAS partitions before
   attributing their remaining boundary shifts to later pipeline stages.

## Suggested Next Steps

### 1. Generate raw and final boundary diagnostics for all pyrho sets

The final files exist locally. Ensure raw boundary files also exist:

```bash
cd examples/MacDonald2022
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache \
uv run snakemake --cores 1 \
  results/compare/raw/pyrho_AFR_boundary_offsets.tsv \
  results/compare/raw/pyrho_EAS_boundary_offsets.tsv \
  results/compare/raw/pyrho_EUR_boundary_offsets.tsv \
  --shared-fs-usage input-output persistence software-deployment sources
```

If Snakemake tries to rerun expensive upstream jobs despite downloaded files,
first check whether the expected raw combined BEDs and reference BEDs exist:

```bash
ls -lh results/pyrho_*_raw_LD_blocks.bed resources/pyrho_*_LD_blocks.bed
```

### 2. Compare raw vs final boundary loss/gain around centromeres

For each pyrho set, inspect boundaries present in raw but absent in final and
compare them to the published reference. The key question is:

- Did centromere removal delete exactly the boundaries missing from reference?
- Or did it also delete one adjacent reference boundary on chromosomes with
  final `delta=-1` or `delta=-2`?

Useful files:

```text
results/{block_set}_raw_LD_blocks.bed
results/{block_set}_LD_blocks.bed
resources/{block_set}_LD_blocks.bed
results/compare/raw/{block_set}_block_comparison.tsv
results/compare/{block_set}_block_comparison.tsv
```

### 3. Test postprocessing order variants on pyrho only

Current order:

1. optionally remove centromere-overlapping blocks
2. merge blocks with fewer than `min_snps_per_block=100` into the left neighbor

Try alternatives in a diagnostic-only rule/script first:

- merge small blocks first, then remove centromere-overlapping blocks
- remove centromere-overlapping blocks, but merge small blocks into nearest
  neighbor by SNP count or genomic distance rather than always left
- when removing centromere-overlapping blocks, merge flanking non-centromere
  blocks if the reference appears to bridge them

Do not change the default pipeline until a variant improves all three pyrho
sets or clearly explains a subset of remaining discrepancies.

### 4. Focus on one shared bad chromosome first

Start with `chr9`, preferably `pyrho_EUR` or `pyrho_AFR`.

Recommended targets:

```bash
cd examples/MacDonald2022
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache \
uv run snakemake --cores 1 \
  results/compare/raw/pyrho_EUR_boundary_offsets.tsv \
  results/compare/boundaries/pyrho_EUR_boundary_offsets.tsv \
  --config chromosomes='[9]' \
  --shared-fs-usage input-output persistence software-deployment sources
```

Then inspect chr9 boundaries with large offsets:

```bash
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache \
uv run python - <<'PY'
import pandas as pd

path = "results/compare/boundaries/pyrho_EUR_boundary_offsets.tsv"
df = pd.read_csv(path, sep="\t")
bad = df[
    (df["chrom"] == "chr9")
    & (df["source"] == "ours_to_ref")
    & (df["abs_offset_bp"] > 50_000)
]
print(bad.sort_values("abs_offset_bp", ascending=False).head(30).to_string(index=False))
PY
```

### 5. Keep deCODE notes separate

deCODE `EUR` raw and final are identical, and its mismatch is not fixed by
postprocessing changes. It likely reflects a deCODE-specific input or map
preprocessing difference. Do not let it drive pyrho postprocessing changes.

## Validation Commands

Static checks:

```bash
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache \
uv run ruff check examples/MacDonald2022/scripts
```

MacDonald dry-run:

```bash
cd examples/MacDonald2022
UV_CACHE_DIR=/Users/adamhe/github/ldetect2/.uv-cache \
uv run snakemake -n \
  --shared-fs-usage input-output persistence software-deployment sources \
  --config chromosomes='[22]'
```

If using explicit Snakemake targets, put targets before
`--shared-fs-usage`; otherwise Snakemake may treat targets as additional
`--shared-fs-usage` values:

```bash
uv run snakemake --cores 1 TARGET1 TARGET2 \
  --shared-fs-usage input-output persistence software-deployment sources
```
