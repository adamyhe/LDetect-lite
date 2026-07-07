# Multicore utilization — findings

**Findings summary (current as of 2026-07-07).** Distilled for human review. Full investigation detail, diagnostic scripts, and dated process notes: `notes/logs/multicore-utilization-filter-width-search.md`.

## Status: two fixes shipped, one approach tried and rejected, two avenues still open

`ldetect run`'s stage-level parallelism (covariance calc, local search, metric workers) was already well covered. Two real gaps were found and fixed this pass; a tempting third fix was found to be numerically unsafe and reverted.

## Shipped

1. **Local-search OOM fix** (`pipeline.py::_run_local_search`) — the multi-worker process pool previously submitted one task per breakpoint, each independently reloading its own covariance partitions from disk with no sharing across concurrent workers. Peak memory scaled with in-flight *breakpoints*, not workers, and caused a real `BrokenProcessPool` OOM crash on a production run. Fixed by grouping breakpoints that share partition bounds (reusing the existing single-worker grouping logic) and submitting one task per group instead of one per breakpoint. See `docs/optimizations.md` #5.

2. **Filter-width search thread-parallelization** (`find_minima.py`) — profiling a real chr21 run (`plots/EUR-chr21-timeline.pdf`) surfaced an unlabeled, fully single-threaded ~34-of-64-second span that turned out to be `custom_binary_search_with_trackback`: ~41 sequential calls to a direct (non-FFT) Hanning convolution, each ~1s at realistic widths. Two of its three phases (exponential search's doubling, and the coarse/fine trackback refinement sweep) evaluate a boundable, predictable set of candidates per round and are now thread-parallelized — same decision logic, same numerics, computed concurrently instead of one-at-a-time. The core binary search remains sequential (adaptive, shared utility — see below). See `docs/optimizations.md`.

## Tried and rejected: FFT convolution

Swapping the filter's `scipy.ndimage.convolve1d` (O(N·width) direct convolution — the actual root cause of the ~1s/call cost) for `scipy.signal.fftconvolve` (O(N log N)) looked like the highest-leverage fix and was ~100-150x faster in isolation. **It is numerically unsafe for this pipeline and was reverted.**

Direct convolution produces bit-identical output across any flat/constant stretch of the input vector (same shift-invariant arithmetic repeated at every position), so the downstream strict-inequality minima detector (`argrelextrema`) never fires there. FFT convolution's rounding error is *not* shift-invariant — it injects distinct floating-point noise (~1e-15) at every position, which breaks exact ties and manufactures spurious local minima out of nothing (confirmed: 1 real minimum became 23 detected on a synthetic flat-plateau test signal). Since the search is actively hunting for a width producing an exact target minima count, spurious minima anywhere could derail convergence to a materially different breakpoint set — and real covariance-sum vectors can plausibly have flat/tied runs (sparse regions, zero-padding), so this isn't just a synthetic-fixture edge case. Other FFT-family methods (`oaconvolve`) share the same flaw.

## Still open

- **Binary-search phase** (`_util/binary_search.py::find_le_ind`) is not parallelized — each step is adaptive on the previous comparison, so it can't be pre-batched the same way as the other two phases without a fundamentally different (k-ary search) algorithm. It's also a shared utility used elsewhere, so changing its core algorithm needs more care/validation than was in scope this pass.
- **Numba-`prange`-parallel direct convolution** — a bigger structural fix that would preserve flat-region exactness (same summation structure as today, just compiled and parallelized across output positions) while also genuinely speeding up the convolution itself, not just the search's outer scheduling. Not attempted this session; would need careful boundary-condition replication and real-data validation.
- **Cross-stage pipelining** (covariance calc → matrix-to-vector → local search running concurrently across stage boundaries) — considered and deprioritized: step 4's binary search needs the complete chromosome vector before it can start, so the only real overlap opportunity is the "tail" of step 2 (idle workers waiting for the last few partitions), which is a narrower win than it first sounds.
- Real-cluster wall-clock validation of the thread-parallelization shipped above (correctness was validated locally; the actual speedup on the ~34s chr21 phase hasn't been re-profiled on real hardware yet).
