"""Compare `ldetect run --covariance-compression zstd` against `lzf`.

Both modes run on identical filtered VCF/map/individuals input and
compression is lossless, so the two outputs are expected to be exact matches
(not merely close) -- the codec only changes how covariance rows are stored
on disk, not the values themselves. This script reports:

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
        --population EUR --chromosome 22 \
        --baseline-vector .../baseline/vector-22.txt.gz \
        --zstd-vector .../zstd/vector-22.txt.gz \
        --baseline-breakpoints .../baseline/breakpoints-22.json \
        --zstd-breakpoints .../zstd/breakpoints-22.json \
        --baseline-bed .../baseline/22-ld-blocks.bed \
        --zstd-bed .../zstd/22-ld-blocks.bed \
        --baseline-covariance-dir .../baseline/22 \
        --zstd-covariance-dir .../zstd/22 \
        --baseline-benchmark .../logs/baseline.benchmark.tsv \
        --zstd-benchmark .../logs/zstd.benchmark.tsv \
        --output results/.../compare/compression_vs_baseline.tsv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from compare_blocks import compare_chrom

from ldetect_lite.io.bed import read_single_chrom_bed


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


def _vector_numeric_diff(baseline: dict[int, float], zstd: dict[int, float]) -> dict:
    all_pos = set(baseline) | set(zstd)
    shared = [p for p in all_pos if p in baseline and p in zstd]
    abs_diffs = [abs(baseline[p] - zstd[p]) for p in shared]
    return {
        "n_shared": len(shared),
        "only_in_baseline": sum(1 for p in all_pos if p not in zstd),
        "only_in_zstd": sum(1 for p in all_pos if p not in baseline),
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
    parser.add_argument("--baseline-vector", required=True, type=Path)
    parser.add_argument("--zstd-vector", required=True, type=Path)
    parser.add_argument("--baseline-breakpoints", required=True, type=Path)
    parser.add_argument("--zstd-breakpoints", required=True, type=Path)
    parser.add_argument("--baseline-bed", required=True, type=Path)
    parser.add_argument("--zstd-bed", required=True, type=Path)
    parser.add_argument("--baseline-covariance-dir", required=True, type=Path)
    parser.add_argument("--zstd-covariance-dir", required=True, type=Path)
    parser.add_argument("--baseline-benchmark", type=Path, default=None)
    parser.add_argument("--zstd-benchmark", type=Path, default=None)
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
    zstd_rows, zstd_hash = _vector_digest(args.zstd_vector)
    vector_digest_equal = baseline_hash == zstd_hash
    if vector_digest_equal:
        numeric = {
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "n_shared": baseline_rows,
            "only_in_baseline": 0,
            "only_in_zstd": 0,
            "exact_matches": baseline_rows,
        }
    else:
        numeric = _vector_numeric_diff(
            _vector_map(args.baseline_vector), _vector_map(args.zstd_vector)
        )

    baseline_loci = _read_loci(args.baseline_breakpoints, args.subset)
    zstd_loci = _read_loci(args.zstd_breakpoints, args.subset)
    loci_equal = baseline_loci == zstd_loci

    _, baseline_blocks = read_single_chrom_bed(args.baseline_bed)
    chrom, zstd_blocks = read_single_chrom_bed(args.zstd_bed)
    bed_row = compare_chrom(
        chrom or args.chromosome, zstd_blocks, baseline_blocks, args.tolerance
    )

    baseline_bytes = _covariance_dir_bytes(args.baseline_covariance_dir)
    zstd_bytes = _covariance_dir_bytes(args.zstd_covariance_dir)
    size_ratio = (zstd_bytes / baseline_bytes) if baseline_bytes > 0 else ""
    size_reduction_pct = (
        round((1 - zstd_bytes / baseline_bytes) * 100, 2) if baseline_bytes > 0 else ""
    )

    baseline_bench = (
        _read_benchmark(args.baseline_benchmark) if args.baseline_benchmark else None
    )
    zstd_bench = _read_benchmark(args.zstd_benchmark) if args.zstd_benchmark else None
    speedup = ""
    rss_ratio = ""
    if baseline_bench and zstd_bench and zstd_bench["s"] > 0:
        speedup = round(baseline_bench["s"] / zstd_bench["s"], 4)
    if baseline_bench and zstd_bench and zstd_bench["max_rss"] > 0:
        rss_ratio = round(baseline_bench["max_rss"] / zstd_bench["max_rss"], 4)

    row = {
        "population": args.population,
        "chrom": args.chromosome,
        "vector_rows_baseline": baseline_rows,
        "vector_rows_zstd": zstd_rows,
        "vector_rows_equal": baseline_rows == zstd_rows,
        "vector_sha256_equal": vector_digest_equal,
        "vector_max_abs_diff": numeric["max_abs_diff"],
        "vector_mean_abs_diff": numeric["mean_abs_diff"],
        "vector_exact_matches": numeric["exact_matches"],
        "vector_only_in_baseline": numeric["only_in_baseline"],
        "vector_only_in_zstd": numeric["only_in_zstd"],
        "n_loci_baseline": len(baseline_loci),
        "n_loci_zstd": len(zstd_loci),
        "loci_exact_match": loci_equal,
        "bed_recall": bed_row["recall"],
        "bed_precision": bed_row["precision"],
        "bed_jaccard": bed_row["jaccard"],
        "bed_bp_jaccard": bed_row["bp_jaccard"],
        "baseline_covariance_bytes": baseline_bytes,
        "zstd_covariance_bytes": zstd_bytes,
        "covariance_size_ratio": size_ratio,
        "covariance_size_reduction_pct": size_reduction_pct,
        "baseline_seconds": baseline_bench["s"] if baseline_bench else "",
        "zstd_seconds": zstd_bench["s"] if zstd_bench else "",
        "speedup": speedup,
        "baseline_max_rss_mb": baseline_bench["max_rss"] if baseline_bench else "",
        "zstd_max_rss_mb": zstd_bench["max_rss"] if zstd_bench else "",
        "max_rss_ratio": rss_ratio,
    }

    print(
        f"\nCompression diagnostic ({args.population} chr{args.chromosome})\n"
        f"  vector: rows_equal={row['vector_rows_equal']} "
        f"sha256_equal={row['vector_sha256_equal']} "
        f"max_abs_diff={row['vector_max_abs_diff']:.3e}\n"
        f"  breakpoints ({args.subset}): loci_exact_match={loci_equal} "
        f"({len(baseline_loci)} vs {len(zstd_loci)} loci)\n"
        f"  bed: recall={row['bed_recall']} precision={row['bed_precision']} "
        f"jaccard={row['bed_jaccard']}\n"
        f"  covariance size: baseline={baseline_bytes:,}B zstd={zstd_bytes:,}B "
        f"ratio={size_ratio} reduction={size_reduction_pct}%\n"
        f"  performance: baseline={row['baseline_seconds']}s "
        f"zstd={row['zstd_seconds']}s speedup={speedup}x "
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
