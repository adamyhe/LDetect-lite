"""Compare `ldetect2 run --signal-cache auto` against `--signal-cache off`.

Both modes run on identical filtered VCF/map/individuals input, so the two
outputs are expected to be exact matches (not merely close): the signal
sidecar is only supposed to relocate *when* the per-locus correlation-sum
accumulation happens (Step 2 instead of Step 3), not change *how* it is
computed. This script reports:

  - vector exactness: row count, sha256 digest, and (if the digest differs)
    a numeric max/mean absolute-difference breakdown over shared positions;
  - breakpoints exactness: whether the requested subset's loci list is
    identical, position by position;
  - BED exactness: boundary recall/precision/Jaccard between the two BEDs
    at --tolerance bp (default 0, i.e. exact-position agreement);
  - performance: wall-clock seconds and peak RSS (MB) from each mode's
    Snakemake `benchmark:` TSV, plus the speedup/reduction ratios.

Usage:
    uv run python scripts/compare_signal_cache.py \
        --population EUR --chromosome 22 \
        --baseline-vector .../baseline/vector-22.txt.gz \
        --signal-cache-vector .../signal_cache/vector-22.txt.gz \
        --baseline-breakpoints .../baseline/breakpoints-22.json \
        --signal-cache-breakpoints .../signal_cache/breakpoints-22.json \
        --baseline-bed .../baseline/22-ld-blocks.bed \
        --signal-cache-bed .../signal_cache/22-ld-blocks.bed \
        --baseline-benchmark .../logs/baseline.benchmark.tsv \
        --signal-cache-benchmark .../logs/signal_cache.benchmark.tsv \
        --output results/.../compare/signal_cache_vs_baseline.tsv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from compare_blocks import compare_chrom

from ldetect2.io.bed import read_single_chrom_bed


def _vector_map(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with gzip.open(path, "rt") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            out[int(row[0])] = float(row[1])
    return out


def _vector_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    with gzip.open(path, "rb") as f:
        for raw in f:
            digest.update(raw)
            if raw.strip():
                rows += 1
    return rows, digest.hexdigest()


def _read_loci(path: Path, subset: str) -> list[int]:
    data = json.loads(path.read_text())
    return [int(x) for x in data[subset]["loci"]]


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
        key: float(value)
        for key, value in row.items()
        if key in _BENCHMARK_NUMERIC_FIELDS
    }


def _vector_numeric_diff(baseline: dict[int, float], signal: dict[int, float]) -> dict:
    all_pos = set(baseline) | set(signal)
    shared = [p for p in all_pos if p in baseline and p in signal]
    abs_diffs = [abs(baseline[p] - signal[p]) for p in shared]
    return {
        "n_shared": len(shared),
        "only_in_baseline": sum(1 for p in all_pos if p not in signal),
        "only_in_signal_cache": sum(1 for p in all_pos if p not in baseline),
        "max_abs_diff": max(abs_diffs) if abs_diffs else 0.0,
        "mean_abs_diff": (sum(abs_diffs) / len(abs_diffs)) if abs_diffs else 0.0,
        "exact_matches": sum(1 for d in abs_diffs if d == 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", required=True)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--baseline-vector", required=True, type=Path)
    parser.add_argument("--signal-cache-vector", required=True, type=Path)
    parser.add_argument("--baseline-breakpoints", required=True, type=Path)
    parser.add_argument("--signal-cache-breakpoints", required=True, type=Path)
    parser.add_argument("--baseline-bed", required=True, type=Path)
    parser.add_argument("--signal-cache-bed", required=True, type=Path)
    parser.add_argument("--baseline-benchmark", type=Path, default=None)
    parser.add_argument("--signal-cache-benchmark", type=Path, default=None)
    parser.add_argument("--subset", default="fourier_ls")
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="BED boundary-match tolerance in bp (default: 0, exact match).",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline_rows, baseline_hash = _vector_digest(args.baseline_vector)
    signal_rows, signal_hash = _vector_digest(args.signal_cache_vector)
    vector_digest_equal = baseline_hash == signal_hash
    if vector_digest_equal:
        numeric = {
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "n_shared": baseline_rows,
            "only_in_baseline": 0,
            "only_in_signal_cache": 0,
            "exact_matches": baseline_rows,
        }
    else:
        numeric = _vector_numeric_diff(
            _vector_map(args.baseline_vector), _vector_map(args.signal_cache_vector)
        )

    baseline_loci = _read_loci(args.baseline_breakpoints, args.subset)
    signal_loci = _read_loci(args.signal_cache_breakpoints, args.subset)
    loci_equal = baseline_loci == signal_loci

    _, baseline_blocks = read_single_chrom_bed(args.baseline_bed)
    chrom, signal_blocks = read_single_chrom_bed(args.signal_cache_bed)
    bed_row = compare_chrom(
        chrom or args.chromosome, signal_blocks, baseline_blocks, args.tolerance
    )

    baseline_bench = (
        _read_benchmark(args.baseline_benchmark) if args.baseline_benchmark else None
    )
    signal_bench = (
        _read_benchmark(args.signal_cache_benchmark)
        if args.signal_cache_benchmark
        else None
    )
    speedup = ""
    rss_ratio = ""
    if baseline_bench and signal_bench and signal_bench["s"] > 0:
        speedup = round(baseline_bench["s"] / signal_bench["s"], 4)
    if baseline_bench and signal_bench and signal_bench["max_rss"] > 0:
        rss_ratio = round(baseline_bench["max_rss"] / signal_bench["max_rss"], 4)

    row = {
        "population": args.population,
        "chrom": args.chromosome,
        "vector_rows_baseline": baseline_rows,
        "vector_rows_signal_cache": signal_rows,
        "vector_rows_equal": baseline_rows == signal_rows,
        "vector_sha256_equal": vector_digest_equal,
        "vector_max_abs_diff": numeric["max_abs_diff"],
        "vector_mean_abs_diff": numeric["mean_abs_diff"],
        "vector_exact_matches": numeric["exact_matches"],
        "vector_only_in_baseline": numeric["only_in_baseline"],
        "vector_only_in_signal_cache": numeric["only_in_signal_cache"],
        "n_loci_baseline": len(baseline_loci),
        "n_loci_signal_cache": len(signal_loci),
        "loci_exact_match": loci_equal,
        "bed_recall": bed_row["recall"],
        "bed_precision": bed_row["precision"],
        "bed_jaccard": bed_row["jaccard"],
        "bed_bp_jaccard": bed_row["bp_jaccard"],
        "baseline_seconds": baseline_bench["s"] if baseline_bench else "",
        "signal_cache_seconds": signal_bench["s"] if signal_bench else "",
        "speedup": speedup,
        "baseline_max_rss_mb": baseline_bench["max_rss"] if baseline_bench else "",
        "signal_cache_max_rss_mb": signal_bench["max_rss"] if signal_bench else "",
        "max_rss_ratio": rss_ratio,
    }

    print(
        f"\nSignal-cache diagnostic ({args.population} chr{args.chromosome})\n"
        f"  vector: rows_equal={row['vector_rows_equal']} "
        f"sha256_equal={row['vector_sha256_equal']} "
        f"max_abs_diff={row['vector_max_abs_diff']:.3e}\n"
        f"  breakpoints ({args.subset}): loci_exact_match={loci_equal} "
        f"({len(baseline_loci)} vs {len(signal_loci)} loci)\n"
        f"  bed: recall={row['bed_recall']} precision={row['bed_precision']} "
        f"jaccard={row['bed_jaccard']}\n"
        f"  performance: baseline={row['baseline_seconds']}s "
        f"signal_cache={row['signal_cache_seconds']}s speedup={speedup}x "
        f"max_rss_ratio={rss_ratio}x"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
