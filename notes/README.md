# Notes Index

This directory contains both active plans and historical optimization context.
Use this index first; several older notes are intentionally kept for provenance
but are no longer implementation guidance.

## Active Plans

- `nocache-optimization-findings.md`
  Current plan for `--pair-cache r2-nocache`, including bounded prepared-input
  caching, tiled r2 local-search work, duplicate-position constraints, and
  instrumentation priorities.
- `r2-zarr-cache-optimization-plan.md`
  Current plan for the compact normalized pair-cache path and remaining
  r2-Zarr layout/profiling work.
- `optimizations-handoff.md`
  Agent-facing handoff for the production compact-HDF5 optimization state,
  current bottlenecks, validation commands, and profiling posture.
- `optimizations.md`
  Human-readable summary of the production optimization work.

## Historical Or Superseded Context

- `implementation-plan.md`
  Original package/CLI port plan. Mostly historical; the implemented package is
  now the source of truth.
- `ldetect_optimization_findings.md`
  Early broad optimization survey. Useful background, but newer HDF5, r2-Zarr,
  and nocache notes supersede its active recommendations.
- `post-covariance-optimization-review.md`
  Earlier review after covariance work.
- `local-search-divergence-asn22.md`
  Specific investigation note.
- `ldetect-original-fp64-concordance.md`
  Specific exactness/concordance note.