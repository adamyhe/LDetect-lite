# notes/

Internal working notes — not the polished public documentation (that's `README.md` and `docs/`). Split by purpose:

- **`findings/`** — distilled, current-status summaries of resolved or parked investigations. Read these first; they're written to be reviewed and mined for the paper writeup.
  - `ldetect-original-reproduction.md` — status of reproducing Berisa & Pickrell (2016)'s published LD blocks (ASN exact, AFR/EUR mostly exact, two parked residual divergences).
  - `macdonald2022-reproduction.md` — status of reproducing MacDonald et al. (2022)'s GRCh38 LD blocks (deCODE map interpolation bug + fix, pyrho reproduction mechanisms).
- **`logs/`** — raw, dated, first-person investigation logs kept for provenance and audit. Not meant to be read end-to-end; each findings doc above links to the relevant log(s) for full detail.
  - `implementation-plan.md` — original architecture/porting plan (historical).
  - `ldetect-original-main-pipeline-audit.md` — full dated audit log behind `findings/ldetect-original-reproduction.md`.
  - `local-search-divergence-asn22.md` — investigation and fix for a historical array-vs-Decimal local-search bug.
  - `macdonald2022-boundary-diagnostics.md` — early MacDonald2022 boundary-diagnostic session notes.
  - `macdonald2022-pyrho-handoff.md` — full dated handoff log behind `findings/macdonald2022-reproduction.md`.
  - `macdonald2022-interpolation-port.md` — full root-cause writeup for the deCODE interval-interpolation bug.
  - `post-covariance-optimization-review.md` — post-covariance-optimization performance review and roadmap.
  - `bitpacked-ld-kernel.md` — bitpacked popcount LD kernel (`--ld-kernel bitpacked`), toy-scale validation history, and the genome-scale exactness/timing diagnostic.

The performance-optimization counterpart to this directory, `docs/optimizations.md`, lives outside `notes/` because it's human-facing reference documentation, not an internal working note.
