"""Compare a candidate `ldetect2 run` covariance-storage mode against `baseline`.

Both modes run on identical filtered VCF/map/individuals input. For a
lossless candidate (codec-only, e.g. `--covariance-compression zstd`) the two
outputs are expected to be exact matches -- the codec only changes how
covariance rows are stored on disk, not the values themselves. For a lossy
candidate (e.g. `--shrink-ld-precision float32`) exact matches are NOT
guaranteed -- that is exactly what this script is for: measuring whether
precision truncation shifts any real breakpoints. This script reports:

  - vector exactness: row count, sha256 digest, and (if the digest differs)
    a numeric max/mean absolute-difference breakdown over shared positions;
  - breakpoints exactness: whether the requested subset's loci list is
    identical, position by position;
  - BED exactness: boundary recall/precision/Jaccard between the two BEDs
    at --tolerance bp (default 0, i.e. exact-position agreement);
  - covariance directory size: total bytes of all .h5 partitions under each
    mode's covariance directory, plus the size ratio/reduction -- this is
    the actual point of this diagnostic;
  - performance: wall-clock seconds and peak RSS (MB) from each mode's
    Snakemake `benchmark:` TSV, plus the speedup/reduction ratios.

Usage:
    uv run python scripts/compare_compression.py \
        --population EUR --chromosome 22 --candidate-mode zstd \
        --baseline-vector .../baseline/vector-22.txt.gz \
        --candidate-vector .../zstd/vector-22.txt.gz \
        --baseline-breakpoints .../baseline/breakpoints-22.json \
        --candidate-breakpoints .../zstd/breakpoints-22.json \
        --baseline-bed .../baseline/22-ld-blocks.bed \
        --candidate-bed .../zstd/22-ld-blocks.bed \
        --baseline-covariance-dir .../baseline/22 \
        --candidate-covariance-dir .../zstd/22 \
        --baseline-benchmark .../logs/baseline.benchmark.tsv \
        --candidate-benchmark .../logs/zstd.benchmark.tsv \
        --output results/.../compare/compression_vs_baseline.zstd.tsv
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


def _vector_numeric_diff(
    baseline: dict[int, float], candidate: dict[int, float]
) -> dict:
    all_pos = set(baseline) | set(candidate)
    shared = [p for p in all_pos if p in baseline and p in candidate]
    abs_diffs = [abs(baseline[p] - candidate[p]) for p in shared]
    return {
        "n_shared": len(shared),
        "only_in_baseline": sum(1 for p in all_pos if p not in candidate),
        "only_in_candidate": sum(1 for p in all_pos if p not in baseline),
        "max_abs_diff": max(abs_diffs) if abs_diffs else 0.0,
        "mean_abs_diff": (sum(abs_diffs) / len(abs_diffs)) if abs_diffs else 0.0,
        "exact_matches": sum(1 for d in abs_diffs if d == 0.0),
    }


def _covariance_dir_bytes(covariance_dir: Path) -> int:
    return sum(p.stat().st_size for p in covariance_dir.glob("*.h5"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", required=True)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument(
        "--candidate-mode",
        required=True,
        help="Label for the non-baseline mode being compared (e.g. zstd, zstd_f32).",
    )
    parser.add_argument("--baseline-vector", required=True, type=Path)
    parser.add_argument("--candidate-vector", required=True, type=Path)
    parser.add_argument("--baseline-breakpoints", required=True, type=Path)
    parser.add_argument("--candidate-breakpoints", required=True, type=Path)
    parser.add_argument("--baseline-bed", required=True, type=Path)
    parser.add_argument("--candidate-bed", required=True, type=Path)
    parser.add_argument("--baseline-covariance-dir", required=True, type=Path)
    parser.add_argument("--candidate-covariance-dir", required=True, type=Path)
    parser.add_argument("--baseline-benchmark", type=Path, default=None)
    parser.add_argument("--candidate-benchmark", type=Path, default=None)
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
    candidate_rows, candidate_hash = _vector_digest(args.candidate_vector)
    vector_digest_equal = baseline_hash == candidate_hash
    if vector_digest_equal:
        numeric = {
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "n_shared": baseline_rows,
            "only_in_baseline": 0,
            "only_in_candidate": 0,
            "exact_matches": baseline_rows,
        }
    else:
        numeric = _vector_numeric_diff(
            _vector_map(args.baseline_vector), _vector_map(args.candidate_vector)
        )

    baseline_loci = _read_loci(args.baseline_breakpoints, args.subset)
    candidate_loci = _read_loci(args.candidate_breakpoints, args.subset)
    loci_equal = baseline_loci == candidate_loci

    _, baseline_blocks = read_single_chrom_bed(args.baseline_bed)
    chrom, candidate_blocks = read_single_chrom_bed(args.candidate_bed)
    bed_row = compare_chrom(
        chrom or args.chromosome, candidate_blocks, baseline_blocks, args.tolerance
    )

    baseline_bytes = _covariance_dir_bytes(args.baseline_covariance_dir)
    candidate_bytes = _covariance_dir_bytes(args.candidate_covariance_dir)
    size_ratio = (candidate_bytes / baseline_bytes) if baseline_bytes > 0 else ""
    size_reduction_pct = (
        round((1 - candidate_bytes / baseline_bytes) * 100, 2)
        if baseline_bytes > 0
        else ""
    )

    baseline_bench = (
        _read_benchmark(args.baseline_benchmark) if args.baseline_benchmark else None
    )
    candidate_bench = (
        _read_benchmark(args.candidate_benchmark) if args.candidate_benchmark else None
    )
    speedup = ""
    rss_ratio = ""
    if baseline_bench and candidate_bench and candidate_bench["s"] > 0:
        speedup = round(baseline_bench["s"] / candidate_bench["s"], 4)
    if baseline_bench and candidate_bench and candidate_bench["max_rss"] > 0:
        rss_ratio = round(baseline_bench["max_rss"] / candidate_bench["max_rss"], 4)

    row = {
        "population": args.population,
        "chrom": args.chromosome,
        "candidate_mode": args.candidate_mode,
        "vector_rows_baseline": baseline_rows,
        "vector_rows_candidate": candidate_rows,
        "vector_rows_equal": baseline_rows == candidate_rows,
        "vector_sha256_equal": vector_digest_equal,
        "vector_max_abs_diff": numeric["max_abs_diff"],
        "vector_mean_abs_diff": numeric["mean_abs_diff"],
        "vector_exact_matches": numeric["exact_matches"],
        "vector_only_in_baseline": numeric["only_in_baseline"],
        "vector_only_in_candidate": numeric["only_in_candidate"],
        "n_loci_baseline": len(baseline_loci),
        "n_loci_candidate": len(candidate_loci),
        "loci_exact_match": loci_equal,
        "bed_recall": bed_row["recall"],
        "bed_precision": bed_row["precision"],
        "bed_jaccard": bed_row["jaccard"],
        "bed_bp_jaccard": bed_row["bp_jaccard"],
        "baseline_covariance_bytes": baseline_bytes,
        "candidate_covariance_bytes": candidate_bytes,
        "covariance_size_ratio": size_ratio,
        "covariance_size_reduction_pct": size_reduction_pct,
        "baseline_seconds": baseline_bench["s"] if baseline_bench else "",
        "candidate_seconds": candidate_bench["s"] if candidate_bench else "",
        "speedup": speedup,
        "baseline_max_rss_mb": baseline_bench["max_rss"] if baseline_bench else "",
        "candidate_max_rss_mb": candidate_bench["max_rss"] if candidate_bench else "",
        "max_rss_ratio": rss_ratio,
    }

    print(
        f"\nCompression diagnostic ({args.population} chr{args.chromosome}, "
        f"candidate={args.candidate_mode})\n"
        f"  vector: rows_equal={row['vector_rows_equal']} "
        f"sha256_equal={row['vector_sha256_equal']} "
        f"max_abs_diff={row['vector_max_abs_diff']:.3e}\n"
        f"  breakpoints ({args.subset}): loci_exact_match={loci_equal} "
        f"({len(baseline_loci)} vs {len(candidate_loci)} loci)\n"
        f"  bed: recall={row['bed_recall']} precision={row['bed_precision']} "
        f"jaccard={row['bed_jaccard']}\n"
        f"  covariance size: baseline={baseline_bytes:,}B "
        f"candidate={candidate_bytes:,}B "
        f"ratio={size_ratio} reduction={size_reduction_pct}%\n"
        f"  performance: baseline={row['baseline_seconds']}s "
        f"candidate={row['candidate_seconds']}s speedup={speedup}x "
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
