# deCODE map interpolation: root cause + fix (2026-07-05)

## Context

The MacDonald2022 pyrho/deCODE investigation (full writeup in
`notes/macdonald2022-pyrho-handoff.md`, which lives on the
`covariance-streaming` branch, not this one) previously found that our
own recomputed deCODE genetic map disagreed with MacDonald's published
`deCODE_interpolated_maps` by a mean absolute error of 0.0013-0.0026 cM
per chromosome (`map_ref_comparison.tsv`). The workaround at the time was
to switch the EUR block set to download MacDonald's already-interpolated
map directly, bypassing our own conversion+interpolation step. That fixed
the EUR reproduction (recall 0.63 -> 0.865) but the root cause of the
divergence itself was never identified.

## Root cause

`src/ldetect2/interpolate_maps.py::interpolate()` is a port of
`joepickrell/1000-genomes-genetic-maps`'s **point**-based linear
interpolator: it expects discrete `(position, cM)` points and interpolates
between the two bracketing points.

MacDonald's actual map-generation pipeline
(https://github.com/jmacdon/LDblocks_GRCh38/blob/master/scripts/interpolate.R)
is **interval**-based:
deCODE's source data gives `Begin, End, cM_per_Mb (rate), cumulative_cM (at
End)` per interval. The R script assigns each SNP to its containing
interval and computes `cM = cumcM_at_End[i-1] + (pos - Begin[i]) *
rate[i] / 1e6` — anchored at the *previous* interval's cumulative endpoint,
advanced using *that interval's own* rate.

Our `convert_decode_map.py` reshapes deCODE's interval data into the
3-column `(position=Begin_i, rate, cM=cumulative_cM_at_End_i)` format that
`interpolate()` expects — pairing each interval's *start* position with
that *same* interval's *end* cM value. Feeding this into point-based
interpolation is a bug: for a SNP inside interval `i`, `interpolate()`
brackets between row `i` and row `i+1`, computing

    gp = cumcM_at_End[i] + frac * (cumcM_at_End[i+1] - cumcM_at_End[i])
       = cumcM_at_End[i] + frac * increment[i+1]

instead of the correct `cumcM_at_End[i-1] + frac * increment[i]`. This is
an off-by-one interval shift: every SNP's genetic position used the
*next* interval's rate anchored at the *current* interval's endpoint. The
resulting error is bounded by roughly one interval's own cM increment
(~0.001-0.01 cM for deCODE's fine-grained map) — matching almost exactly
the previously-measured, previously-unexplained MAE.

## Fix

Added `interpolate_intervals()` in `src/ldetect2/interpolate_maps.py`, a
direct port of the R script's `interpThat`/`follow` logic, as an
alternative to `interpolate()` (not a replacement — `interpolate()` is
still correct for true point-sampled genetic maps). Exposed via
`ldetect2 interpolate-maps --mode {point,interval}` (default `point`,
preserving prior behavior). The diagnostic `interpolate_map` Snakemake
rule in `examples/MacDonald2022/Snakefile` (used by `validate_maps`/
`compare_maps`; the real EUR pipeline downloads MacDonald's published map
directly and doesn't use this rule) now passes `--mode interval`.

New tests in `tests/test_interpolate_maps.py` cover the interval-rate
algorithm's boundary behavior (before-first, mid-interval anchoring,
extrapolation past the last interval instead of clamping) and include a
regression test asserting `interpolate()` and `interpolate_intervals()`
disagree by the predicted amount on the same interval-rate fixture, to
guard against either function silently changing to match the other.

## Local check (2026-07-05) — confirmed on real deCODE data

The user downloaded a real copy of `aau1043_datas3.gz` (deCODE supplementary
S3) and it was used for a local check without running the full Snakemake
pipeline (no VCFs, no bcftools). Recipe: convert the raw file for one
chromosome at a time with `convert_decode_map.py`, build a synthetic SNP
list directly from MacDonald's own published `deCODE_interpolated_maps`
positions (so the comparison lands on identical coordinates), interpolate
with both `--mode interval` and `--mode point`, and diff against the
published reference with `compare_maps.py`.

| chrom | interval-mode MAE | point-mode MAE |
|---|---|---|
| chr1  | 0.000189 cM | 0.001755 cM |
| chr21 | 0.000040 cM | 0.001677 cM |
| chr22 | 0.000038 cM | 0.001967 cM |

Pearson r = 1.0 in all cases. Interval mode is 10-50x closer than point
mode, and point mode's error lands right in the previously-documented
0.0013-0.0026 cM range — confirms the fix on real data, not just the
synthetic unit-test fixtures.

Interval mode is not perfectly zero-error: chr22 has 636/174,293 SNPs
(0.37%) with error > 1e-3 cM, worst case 0.17 cM at position 38911889.
Traced this specific point: our converted deCODE data is internally
consistent (monotonic cM, no corruption) and the interval-rate algorithm
computes exactly what that data implies. The divergent positions sit next
to recombination hotspots (rates of 55-250 cM/Mb over ~1kb, vs. a
~0.5-2 cM/Mb background; 11% of chr22's intervals are hotspot-like by this
measure) — consistent with fine-scale sensitivity to exactly which
snapshot of the deCODE map was used, not an algorithm bug.

Two alternative explanations for this residual were checked and ruled out:

- **R vs. Python semantics.** Couldn't run the actual `interpolate.R`
  locally (this environment's R is broken — fails to evaluate `1+1` due to
  a `libgfortran.5.dylib` load error under `openblas`; not fixed, since
  that means touching the conda env outside this task's scope). Checked
  instead via Bioconductor's documented `follow()`/`precede()` semantics:
  both pick a bracketing interval via exact integer comparison (no
  floating-point tie-breaking difference is possible), and the one
  genuinely ambiguous case (a SNP landing exactly on an interval's `Begin`)
  was hand-traced to produce an identical numeric result under either
  resolution, because the interval's own recorded `cM` value is
  constructed to equal exactly that. Also tested a `unique(inds)`-style
  "skip empty intervals" grouping variant directly against real chr22 data
  — it did not improve the match and made a different position worse, so
  it was discarded rather than adopted.
- **Erratum.** A real erratum exists for Halldorsson et al. 2019
  (Feb 8 2019, DOI `10.1126/science.aaw8705`), but it only corrects a units
  label in Table 4, a locus mislabeling in Table 5B, and augments Data
  S5/S7. It does not mention Data S3 (the recombination-rate file used
  here) — ruled out.

One genuine anomaly surfaced but not resolved: MacDonald's own
`_reference/LDblocks_GRCh38/README.md` states their downloaded
`aau1043_datas3.gz` "has a gz extension, but is not compressed," whereas
our fresh download is legitimately gzip-compressed (`file`/`gzip -l`
confirm a real gzip stream, with an internal timestamp of Aug 2, 2018 —
before the paper's publication, which weakly suggests the underlying bytes
are unchanged and this is a download/transport artifact on MacDonald's
side rather than a server-side content revision). Left open; would need a
working R+Bioconductor install or an archived copy of the file to fully
close out, and both require more invasive steps than this check
warranted.

**Conclusion:** the fix is confirmed correct and matches MacDonald's
reference almost exactly; the tiny remaining residual is not an
algorithmic issue and not worth chasing further without new evidence.

## Not yet done

Cross-referencing this finding into `notes/macdonald2022-pyrho-handoff.md`
on the `covariance-streaming` branch — that branch wasn't touched here per
the "don't port/diff code across branches" guidance; whoever next works on
that branch/notes file should add a pointer back to this one.
