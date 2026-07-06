"""Compare one --fused-vector/--local-search-source vcf-recompute mode
against the default cache-based `ldetect run` path.

Both modes run on identical filtered VCF/map/individuals input and are
exact, deterministic computations of the same thing -- unlike
compare_compression.py's lzf-vs-zstd comparison (which only needs to prove
losslessness), here BED/breakpoints exactness is the pass/fail gate, not
just a reported metric: any mismatch means the fused-vector or vcf-recompute
code path has a real bug. Reports:

  - breakpoints exactness: whether the requested subset's loci list is
    identical, position by position;
  - BED exactness: byte-for-byte file comparison (not tolerance-based --
    these are supposed to be identical, not merely close);
  - performance: wall-clock seconds and peak RSS (MB) from each mode's
    Snakemake `benchmark:` TSV, plus the speedup/ratio -- this is the actual
    point of this diagnostic.

Usage:
    uv run python scripts/compare_priority5_mode.py \
        --chromosome 21 --mode fused_vector \
        --baseline-bed .../baseline/21-ld-blocks.bed \
        --mode-bed .../fused_vector/21-ld-blocks.bed \
        --baseline-breakpoints .../baseline/breakpoints-21.json \
        --mode-breakpoints .../fused_vector/breakpoints-21.json \
        --baseline-benchmark .../logs/baseline.benchmark.tsv \
        --mode-benchmark .../logs/fused_vector.benchmark.tsv \
        --output results/.../compare.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_BENCHMARK_NUMERIC_FIELDS = frozenset(
    {
        "s",
        "max_rss",
        "max_vms",
        "max_uss",
        "max_pss",
        "io_in",
        "io_out",
        "mean_load",
        "cpu_time",
    }
)


def _read_benchmark(path: Path) -> dict[str, float]:
    """Read a Snakemake `benchmark:` TSV, skipping the non-numeric ``h:m:s`` column."""
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        row = next(reader)
    return {
        key: float(value) for key, value in row.items() if key in _BENCHMARK_NUMERIC_FIELDS
    }


def _read_loci(path: Path, subset: str) -> list[int]:
    data = json.loads(path.read_text())
    return [int(x) for x in data[subset]["loci"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--baseline-bed", required=True, type=Path)
    parser.add_argument("--mode-bed", required=True, type=Path)
    parser.add_argument("--baseline-breakpoints", required=True, type=Path)
    parser.add_argument("--mode-breakpoints", required=True, type=Path)
    parser.add_argument("--baseline-benchmark", required=True, type=Path)
    parser.add_argument("--mode-benchmark", required=True, type=Path)
    parser.add_argument("--subset", default="fourier_ls")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline_bed_bytes = args.baseline_bed.read_bytes()
    mode_bed_bytes = args.mode_bed.read_bytes()
    bed_exact = baseline_bed_bytes == mode_bed_bytes

    baseline_loci = _read_loci(args.baseline_breakpoints, args.subset)
    mode_loci = _read_loci(args.mode_breakpoints, args.subset)
    loci_exact = baseline_loci == mode_loci

    baseline_bench = _read_benchmark(args.baseline_benchmark)
    mode_bench = _read_benchmark(args.mode_benchmark)
    speedup = (
        round(baseline_bench["s"] / mode_bench["s"], 4) if mode_bench["s"] > 0 else ""
    )
    rss_ratio = (
        round(mode_bench["max_rss"] / baseline_bench["max_rss"], 4)
        if baseline_bench["max_rss"] > 0
        else ""
    )

    row = {
        "chrom": args.chromosome,
        "mode": args.mode,
        "bed_exact": bed_exact,
        "n_loci_baseline": len(baseline_loci),
        "n_loci_mode": len(mode_loci),
        "loci_exact_match": loci_exact,
        "baseline_seconds": baseline_bench["s"],
        "mode_seconds": mode_bench["s"],
        "speedup_vs_baseline": speedup,
        "baseline_max_rss_mb": baseline_bench["max_rss"],
        "mode_max_rss_mb": mode_bench["max_rss"],
        "mode_peak_rss_ratio": rss_ratio,
    }

    print(
        f"\nPriority-5 profiling ({args.mode}, chr{args.chromosome})\n"
        f"  bed_exact={bed_exact}  loci_exact_match={loci_exact} "
        f"({len(baseline_loci)} vs {len(mode_loci)} loci)\n"
        f"  baseline={row['baseline_seconds']}s  {args.mode}={row['mode_seconds']}s "
        f"speedup={speedup}x\n"
        f"  baseline_peak_rss={row['baseline_max_rss_mb']}MB  "
        f"{args.mode}_peak_rss={row['mode_max_rss_mb']}MB  ratio={rss_ratio}x"
    )
    if not bed_exact or not loci_exact:
        print(
            "\nWARNING: output mismatch -- this indicates a real bug in "
            f"{args.mode}, not expected imprecision."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
