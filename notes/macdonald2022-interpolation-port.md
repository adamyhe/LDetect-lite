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

## Not yet done

Not run in this sandbox (no raw deCODE data / prior pipeline artifacts
present locally): rerunning `compare_maps.py` on the remote host with
`--mode interval` deCODE output vs MacDonald's published reference map,
to empirically confirm the MAE collapses from ~0.0013-0.0026 cM to
near-zero. Also not done: cross-referencing this finding into
`notes/macdonald2022-pyrho-handoff.md` on the `covariance-streaming`
branch — that branch wasn't touched here per the "don't port/diff code
across branches" guidance; whoever next works on that branch/notes file
should add a pointer back to this one.
