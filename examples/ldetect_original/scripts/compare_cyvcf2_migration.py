"""Compare the cyvcf2-based `ldetect run` against the pre-migration baseline.

Both runs are the *same* commit's config against *identical* filtered VCF
input, but executed via two separately installed `ldetect` entry points (see
cyvcf2_migration_profiling.yaml's docstring) -- one built before the VCF I/O
migration (tabix subprocess + naive text parser), one built after (cyvcf2).
There is no CLI flag to toggle between them; the old parser was deleted, not
kept side by side. Both are exact, deterministic computations of the same
thing, so output must match exactly, not just approximately. Reports:

  - vector exactness: row count and sha256 digest (bit-exact expected);
  - breakpoints exactness: whether the requested subset's loci list is
    identical, position by position;
  - BED exactness: byte-for-byte file comparison;
  - performance: wall-clock seconds and peak RSS (MB) from each run's
    Snakemake `benchmark:` TSV, plus the speedup ratio -- the actual point
    of this diagnostic, now that the harness for the *parsing/subprocess*
    change is properly isolated from the profiling-flag confound found
    during earlier priority-5 profiling work.

Usage:
    uv run python scripts/compare_cyvcf2_migration.py \
        --chromosome 21 \
        --baseline-vector .../baseline/vector-21.txt.gz \
        --new-vector .../cyvcf2/vector-21.txt.gz \
        --baseline-breakpoints .../baseline/breakpoints-21.json \
        --new-breakpoints .../cyvcf2/breakpoints-21.json \
        --baseline-bed .../baseline/21-ld-blocks.bed \
        --new-bed .../cyvcf2/21-ld-blocks.bed \
        --baseline-benchmark .../logs/baseline.benchmark.tsv \
        --new-benchmark .../logs/cyvcf2.benchmark.tsv \
        --output results/.../compare.tsv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--baseline-vector", required=True, type=Path)
    parser.add_argument("--new-vector", required=True, type=Path)
    parser.add_argument("--baseline-breakpoints", required=True, type=Path)
    parser.add_argument("--new-breakpoints", required=True, type=Path)
    parser.add_argument("--baseline-bed", required=True, type=Path)
    parser.add_argument("--new-bed", required=True, type=Path)
    parser.add_argument("--baseline-benchmark", required=True, type=Path)
    parser.add_argument("--new-benchmark", required=True, type=Path)
    parser.add_argument("--subset", default="fourier_ls")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline_rows, baseline_hash = _vector_digest(args.baseline_vector)
    new_rows, new_hash = _vector_digest(args.new_vector)
    vector_sha256_equal = baseline_hash == new_hash

    baseline_loci = _read_loci(args.baseline_breakpoints, args.subset)
    new_loci = _read_loci(args.new_breakpoints, args.subset)
    loci_equal = baseline_loci == new_loci

    bed_exact = args.baseline_bed.read_bytes() == args.new_bed.read_bytes()

    baseline_bench = _read_benchmark(args.baseline_benchmark)
    new_bench = _read_benchmark(args.new_benchmark)
    speedup = (
        round(baseline_bench["s"] / new_bench["s"], 4) if new_bench["s"] > 0 else ""
    )
    rss_ratio = (
        round(new_bench["max_rss"] / baseline_bench["max_rss"], 4)
        if baseline_bench["max_rss"] > 0
        else ""
    )

    row = {
        "chrom": args.chromosome,
        "vector_rows_baseline": baseline_rows,
        "vector_rows_new": new_rows,
        "vector_rows_equal": baseline_rows == new_rows,
        "vector_sha256_equal": vector_sha256_equal,
        "n_loci_baseline": len(baseline_loci),
        "n_loci_new": len(new_loci),
        "loci_exact_match": loci_equal,
        "bed_exact": bed_exact,
        "baseline_seconds": baseline_bench["s"],
        "new_seconds": new_bench["s"],
        "speedup_vs_baseline": speedup,
        "baseline_max_rss_mb": baseline_bench["max_rss"],
        "new_max_rss_mb": new_bench["max_rss"],
        "new_peak_rss_ratio": rss_ratio,
    }

    print(
        f"\ncyvcf2 migration profiling (chr{args.chromosome})\n"
        f"  vector_sha256_equal={vector_sha256_equal} "
        f"loci_exact_match={loci_equal} bed_exact={bed_exact}\n"
        f"  baseline={row['baseline_seconds']}s new={row['new_seconds']}s "
        f"speedup={speedup}x\n"
        f"  baseline_peak_rss={row['baseline_max_rss_mb']}MB "
        f"new_peak_rss={row['new_max_rss_mb']}MB ratio={rss_ratio}x"
    )
    if not vector_sha256_equal or not loci_equal or not bed_exact:
        print(
            "\nWARNING: output mismatch -- this indicates a real bug in the "
            "cyvcf2 migration, not expected imprecision."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
