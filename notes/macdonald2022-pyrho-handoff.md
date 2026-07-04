# MacDonald2022 pyrho Handoff

Last updated: 2026-07-03

## Goal

Pick up the MacDonald2022 replication from a clean session and focus first on
improving the published pyrho LD block reproductions:

- `pyrho_AFR`
- `pyrho_EAS`
- `pyrho_EUR`

Set aside `pyrho_SAS` for now because MacDonald et al. do not appear to
document an SAS-specific effective population size. Also set aside the deCODE
`EUR` mismatch initially; it has a different failure mode.

## Current State (2026-07-03)

All MacDonald2022 workflow/diagnostic code referenced in this note is
committed on the `macdonald` branch (rebased onto current `main` as of
2026-07-03). `git status --short` should be clean; if it isn't, check what
changed before trusting this note's file references. `results/` and most of
`resources/`/`data/` are gitignored — cached local intermediates (BEDs, maps,
comparison TSVs) may or may not be present in a fresh checkout. See the "chr9
Investigation" section below for the most recent substantive findings.

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

## chr9 Investigation (2026-07-03) — mechanism found, root cause converges with a parked issue

**Do not re-litigate this without a new lead.** chr9 was picked up as the
first target per the plan above. The mechanism is now well-evidenced; the
remaining root cause converges with the already-parked
`ldetect-original-reproduction` investigation (see
`notes/ldetect-original-handoff.md`), which never resolved it either.

### The mechanism

chr9 has a genuine, large genetic-map dead zone: the pyrho map (IBS/GWD/CHB —
identical positions across all three populations, so it's a map-construction
artifact, not a population-genetic signal) has **zero data points** from
43,387,949 to 60,518,857 (17.1 Mb), plus continued sparse/patchy coverage out
to ~70 Mb. This starts almost exactly at the UCSC-annotated centromere
(43,389,635–45,518,558, only ~2.1 Mb) but extends ~15 Mb further — consistent
with chr9's unusually large classical heterochromatic block (9qh).

**MacDonald's own raw LDetect output treats this entire desert as a single,
unsplit block.** Confirmed directly from their GitHub history (commit
`5081b31`, the centromere-cleanup commit): the diff for `pyrho_EUR` shows
exactly one line removed from the whole chr9 region — `chr9 43269938
69277370` (~26.0 Mb) — with every other line in that diff being pure
tab-formatting noise, not a content change. Same pattern confirmed for
`pyrho_EAS` (`chr9 38157205 64129460` removed as one block) and for
`pyrho_AFR`'s DOI-tagged raw BED (`chr9 42639706 69101139` exists as one
block pre-removal, in `fa695f7`). All three populations: legacy LDetect
never splits inside this desert.

**Our pipeline splits it into two blocks** (e.g. `pyrho_EUR`: raw blocks
`38157205–61139982` and `61139982–70487377`). Only the first overlaps the
narrow UCSC centromere and gets removed by `--remove-centromeres`; the
second survives postprocessing untouched, producing the large boundary
offset (8.1 Mb for EUR, 5.2 Mb for EAS at this chromosome).

**MacDonald's documented postprocessing is *not* a wide heterochromatin
mask** — their README states plainly: "we removed any blocks that overlap
any portion of a centromere, and then combined any small blocks with < 100
SNPs (there were only two, both for AFR) with an adjacent block." That's the
same simple rule we already have. A `--remove-map-gaps` postprocessing patch
(mask blocks overlapping large genetic-map gaps, not just the narrow
centromere) was prototyped and tested this session — it closed most of the
EUR chr9 gap (8.1 Mb → 1.58 Mb) but did nothing for EAS (its spurious block
starts ~400 kb past a reasonably-chosen fixed gap-merge threshold) or AFR
(already clean via the existing narrow centromere check). Given it doesn't
reproduce MacDonald's documented method and doesn't generalize cleanly across
populations, **it was reverted, not landed** — see git history on this
branch if it needs to be resurrected as a starting point.

### Why our pipeline splits where legacy doesn't

Ruled out: a differing total target-breakpoint count. `n_bpoints` is a
single value computed once per chromosome (`ceil(total_snps / 7000) - 1`,
same formula as legacy's `P02_minima_pipeline.py`), and the total chr9 block
count for `pyrho_EUR` is **identical** between our raw output and
MacDonald's DOI-tagged raw output: 60 blocks each. So the global split
budget matches exactly — the divergence is about *where* the same ~59
breakpoints land, not how many there are.

That points to the same mechanism already identified (and left unresolved)
in `notes/ldetect-original-handoff.md`: with a single global Hanning-filter
width applied across the whole chromosome, a small numerical difference
upstream (covariance computation, float-vs-Decimal precision, or a handful
of SNPs differing between our VCF filtering and MacDonald's) can shift which
marginal local minimum survives — and the most likely place for a
"coin-flip" minimum to land is the flattest, most information-starved
stretch of the chromosome, i.e. exactly this desert. The other investigation
already tested Decimal-vs-float precision on a different dataset and ruled
it out as the sole cause, leaving an unidentified upstream input difference
unresolved. Confirming the same here would require the same expensive path
(real pipeline reruns, careful VCF/precision comparisons for chr9
specifically) already tried and shelved there — not attempted this session.

### Other chromosomes: a cheap, reliable test — and a clean bifurcation (2026-07-03, continued)

A first pass cross-referenced worst boundary offsets against generic
genetic-map gaps (≥200 kb, 300 kb buffer) and found no correlation for any
other flagged chromosome. That test was too weak. **The reliable test that
worked for chr9 generalizes and is cheap to repeat**: diff MacDonald's own
BED between the DOI tag (`fa695f7`, pre-centromere-removal) and the
centromere-removal commit (`5081b31`) for the population/chromosome in
question — this directly reveals whether *legacy's own raw output* had a
single large block spanning that chromosome's centromere/desert region.
(Caveat: GitHub's commit-API `.patch` field silently truncates large diffs —
don't trust it for these multi-thousand-line BED files; download both blobs
via `contents?ref=<sha>` and diff the raw file content instead.)

Applying this to every other flagged chromosome (using already-downloaded
maps where relevant, no pipeline reruns):

**Confirmed same mechanism as chr9** (legacy has one clean desert block at/
near the worst-offset position; ours fragments it) — worked out to exactly
match the recorded worst-offset position:

- AFR `chr18` (worst offset 22,415,017, inside legacy's removed
  14,437,172–24,135,791 block)
- AFR `chr22` (worst offset 10,516,173, exactly the *start* of legacy's
  removed 10,516,173–17,574,597 block — ours leaves a stray 6 kb sliver
  10,516,173–10,522,217 and additionally fails to reproduce a real
  reference block 17,574,597–20,217,810 that legacy has)
- EAS `chr9`, `chr17`, `chr18` (worst offsets all land inside their
  respective legacy-removed desert blocks)

**Not explained by this mechanism** — worst-offset position is nowhere near
any centromere-removed block, *and* a direct check of the population's own
genetic map found no local density anomaly there either (normal, densely
mapped region):

- AFR `chr10` (worst offset 111,860,447), `chr11` (95,160,268)
- EAS `chr4` (86,204,416), `chr14` (38,523,730)

So roughly half the flagged chromosomes share chr9's exact mechanism at
smaller scale; the other half have a genuinely different, structurally
unremarkable divergence with no obvious local explanation. EUR chr19/21/22/16
were not re-checked this way yet (no genome-wide EUR boundary data cached
locally — see step 6 below).

### Why almost nothing is exact anywhere (2026-07-03) — this differs sharply from `ldetect_original`

The user asked directly: unlike `ldetect_original` (near-exact genome-wide,
per `notes/ldetect-original-handoff.md`), MacDonald pyrho reproduction has
**no chromosome anywhere close to 100% exact** — per-chromosome exact-match
rates range ~29%–96% for both AFR and EAS, with most clustering around
80–90%. Overall: AFR 17.8% non-exact (288/1621 boundaries), EAS 20.8%
(241/1157).

The offset-magnitude distribution of these non-exact boundaries is the key
diagnostic, and it's **not** what a small numerical/precision-jitter
explanation would predict:

| range | AFR count | EAS count |
| --- | ---: | ---: |
| 1 bp–1 kb | 71 | 20 |
| 1 kb–10 kb | 14 | 10 |
| 10 kb–100 kb | 14 | 15 |
| 100 kb–1 Mb | **158** | **156** |
| >1 Mb | 31 | 40 |

If this were mostly floating-point/precision jitter nudging a breakpoint to
an adjacent SNP, the distribution would be dominated by the sub-kb bucket.
Instead the dominant bucket (>55% of all non-exact boundaries in both
populations) is 100 kb–1 Mb — i.e. most mismatches are cases where our
pipeline and legacy land on two **genuinely different, well-separated**
candidate breakpoints, not a jittered version of the same one. A spot-check
of a *good* chromosome (AFR chr16, 96% exact, only 2 non-exact boundaries)
found one 484 bp nudge and one 558 kb jump — consistent with the same
"coin-flip between distinct candidates" mechanism happening at low frequency
everywhere, not just at the handful of flagged worst chromosomes; the
flagged ones are just where the cumulative/worst-case single offset happens
to be unusually large (typically the centromere/desert mechanism above, or
an unexplained "Category B" case).

**This is the same class of mechanism as the chr9 investigation and the
already-parked `ldetect-original-reproduction` "flat region" finding**, just
shown here to be pervasive across the whole genome rather than confined to
one or two chromosomes. The likely reason `ldetect_original` doesn't show
this pervasively: it replays byte-identical archived historical VCF and
genetic-map files that the original paper's authors themselves used, so
there's very little room for the underlying covariance/correlation-sum
vector to differ from theirs at all. MacDonald's own recipe is inherently
*not* a byte-identical replay — it's "the same documented steps," rerun
independently against a modern, larger VCF release and pyrho/deCODE
recombination maps (a fundamentally different, newer map-estimation method
than the original HapMap maps `ldetect_original` uses), with no published
intermediate (covariance matrices, correlation vectors, partition files) to
byte-compare against — only the final BED. That leaves much more room for
small, genuine numerical differences to accumulate and occasionally tip a
local-minimum choice, and this session's evidence suggests that's happening
routinely, not rarely, for this dataset.

**Not resolved further this session** — doing so would need the same
expensive path as the parked investigation (real pipeline reruns with
careful precision/VCF-parity comparisons), now understood to be necessary
pervasively rather than for a couple of isolated chromosomes.

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
2. ~~Legacy partitioning hardcodes `Ne=11418`, even when covariance later uses
   a population-specific `Ne`... Generate legacy-compatible AFR/EAS
   partitions before attributing their remaining boundary shifts to later
   pipeline stages.~~ **Deprioritized 2026-07-03**: EUR's overall exact-match
   rate (83.6%) is no better than AFR (82.2%) or EAS (79.2%) despite EUR's
   `Ne` already matching legacy's hardcoded value — if Ne-in-partitioning
   were the dominant driver of pervasive divergence, EUR should look much
   closer to `ldetect_original`'s near-exact reproduction and it doesn't.
   Also, `ldetect_original` itself needed population-specific `Ne` (not the
   hardcoded EUR value) to get mostly-correct results for AFR/ASN, so
   replicating legacy's hardcoding here would likely regress AFR/EAS rather
   than help. Not a live lead; see "Why almost nothing is exact anywhere"
   below for the current best explanation.

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

### 3. Test postprocessing order variants on pyrho only — deprioritized

MacDonald's actual documented postprocessing (confirmed from their README,
see the chr9 section above) is just: remove blocks overlapping a centromere,
then merge blocks with <100 SNPs into an adjacent block. No reordering
variant of *that* simple rule is likely to explain further divergence — the
chr9 investigation found the real determinant was upstream (which of the
same total number of breakpoints get placed where), not postprocessing
order. Still current if picked up:

1. optionally remove centromere-overlapping blocks
2. merge blocks with fewer than `min_snps_per_block=100` into the left neighbor

If revisited, do so on one of the *other* bad chromosomes (see step 6 below),
not chr9 — and check the actual raw block structure and MacDonald's git
history first (as in the chr9 section above) before assuming postprocessing
is the lever, since it wasn't for chr9.

### 4. Focus on one shared bad chromosome first — done for chr9, 2026-07-03

See the "chr9 Investigation" section above for the full writeup. Summary: a
genuine ~17-27 Mb genetic-map dead zone on chr9 (identical across
populations) causes legacy LDetect to emit one unsplit block there, which
gets fully removed by centromere-overlap postprocessing; our pipeline splits
the same region into two blocks, only one of which gets removed. The
remaining question — why our pipeline places an extra split inside this
specific desert when the total genome-wide breakpoint budget matches
legacy's exactly — converges with the already-parked, unresolved
`ldetect-original-reproduction` "flat region" finding. Not resolved further
this session; do not re-open without a new lead (e.g. an actual chr9
pipeline rerun with precision/VCF-parity comparisons, mirroring the approach
in `notes/ldetect-original-handoff.md`).

### 5. Keep deCODE notes separate

deCODE `EUR` raw and final are identical, and its mismatch is not fixed by
postprocessing changes. It likely reflects a deCODE-specific input or map
preprocessing difference. Do not let it drive pyrho postprocessing changes.

### 6. Other bad chromosomes — bifurcated 2026-07-03, half explained

Resolved via the DOI-tag-vs-`5081b31` BED diff method (see "Other
chromosomes" above, and use raw blob diffing — the commit API's `.patch`
field truncates on these large files):

- **Same mechanism as chr9** (legacy desert, ours fragments it): AFR chr18,
  chr22; EAS chr9, chr17, chr18. Nothing further to do here beyond what's
  already documented — these are instances of the same parked issue, not a
  separate bug.
- **Genuinely unexplained** ("Category B" — not near any centromere-removed
  desert, no local map-density anomaly either): AFR chr10, chr11; EAS chr4,
  chr14. If picked up again, the next diagnostic step would need real LD
  signal (an actual covariance/correlation-sum vector for that locus, not
  just the map), i.e. a targeted pipeline rerun for that one chromosome —
  not attempted this session.
- EUR chr19, chr21, chr22, chr16 were never checked at all — no genome-wide
  `pyrho_EUR` boundary data is cached locally (only the chr9-only targeted
  run from earlier). Would need either a genome-wide `pyrho_EUR` Snakemake
  run (expensive) or per-chromosome targeted runs for just these four
  (`--config chromosomes='[19,21,22,16]'`, cheaper) to get real per-chromosome
  offsets before applying the same bifurcation test.

Also see "Why almost nothing is exact anywhere" above — the pervasive,
genome-wide ~18-21% non-exact rate (dominated by 100kb-1Mb-scale
mismatches, not tiny jitter) means Category B chromosomes are likely not
individually special; they're just where the same widespread, low-frequency
"different candidate breakpoint chosen" phenomenon happened to produce an
unusually large single worst-case offset. Postprocessing-order hypotheses
(below) are unlikely to explain either category.

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
