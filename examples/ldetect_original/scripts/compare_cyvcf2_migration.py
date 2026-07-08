"""Cross-check `ldetect run` against VCF vs. BCF input, same install.

Both runs are the same commit's config against identical content -- one as
`.vcf.gz`/`.tbi`, one converted to `.bcf`/`.csi` -- executed through the same
`ldetect` install (cyvcf2 handles both formats via the same code path since
the cyvcf2-vcf-io-migration branch). Both are exact, deterministic
computations of the same thing, so output must match exactly, not just
approximately. Reports:

  - vector exactness: row count and sha256 digest (bit-exact expected);
  - breakpoints exactness: whether the requested subset's loci list is
    identical, position by position;
  - BED exactness: byte-for-byte file comparison;
  - performance: wall-clock seconds and peak RSS (MB) from each run's
    Snakemake `benchmark:` TSV, plus the bcf/vcf ratio -- characterizing any
    format-level read overhead, not a pre-/post-migration speedup (the
    parsing-level speedup was already measured on covariance-cache-redesign).

Usage:
    uv run python scripts/compare_cyvcf2_migration.py \
        --chromosome 21 \
        --vcf-vector .../vcf/vector-21.txt.gz \
        --bcf-vector .../bcf/vector-21.txt.gz \
        --vcf-breakpoints .../vcf/breakpoints-21.json \
        --bcf-breakpoints .../bcf/breakpoints-21.json \
        --vcf-bed .../vcf/21-ld-blocks.bed \
        --bcf-bed .../bcf/21-ld-blocks.bed \
        --vcf-benchmark .../logs/vcf.benchmark.tsv \
        --bcf-benchmark .../logs/bcf.benchmark.tsv \
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
    parser.add_argument("--vcf-vector", required=True, type=Path)
    parser.add_argument("--bcf-vector", required=True, type=Path)
    parser.add_argument("--vcf-breakpoints", required=True, type=Path)
    parser.add_argument("--bcf-breakpoints", required=True, type=Path)
    parser.add_argument("--vcf-bed", required=True, type=Path)
    parser.add_argument("--bcf-bed", required=True, type=Path)
    parser.add_argument("--vcf-benchmark", required=True, type=Path)
    parser.add_argument("--bcf-benchmark", required=True, type=Path)
    parser.add_argument("--subset", default="fourier_ls")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    vcf_rows, vcf_hash = _vector_digest(args.vcf_vector)
    bcf_rows, bcf_hash = _vector_digest(args.bcf_vector)
    vector_sha256_equal = vcf_hash == bcf_hash

    vcf_loci = _read_loci(args.vcf_breakpoints, args.subset)
    bcf_loci = _read_loci(args.bcf_breakpoints, args.subset)
    loci_equal = vcf_loci == bcf_loci

    bed_exact = args.vcf_bed.read_bytes() == args.bcf_bed.read_bytes()

    vcf_bench = _read_benchmark(args.vcf_benchmark)
    bcf_bench = _read_benchmark(args.bcf_benchmark)
    bcf_vs_vcf_ratio = (
        round(bcf_bench["s"] / vcf_bench["s"], 4) if vcf_bench["s"] > 0 else ""
    )
    rss_ratio = (
        round(bcf_bench["max_rss"] / vcf_bench["max_rss"], 4)
        if vcf_bench["max_rss"] > 0
        else ""
    )

    row = {
        "chrom": args.chromosome,
        "vector_rows_vcf": vcf_rows,
        "vector_rows_bcf": bcf_rows,
        "vector_rows_equal": vcf_rows == bcf_rows,
        "vector_sha256_equal": vector_sha256_equal,
        "n_loci_vcf": len(vcf_loci),
        "n_loci_bcf": len(bcf_loci),
        "loci_exact_match": loci_equal,
        "bed_exact": bed_exact,
        "vcf_seconds": vcf_bench["s"],
        "bcf_seconds": bcf_bench["s"],
        "bcf_vs_vcf_seconds_ratio": bcf_vs_vcf_ratio,
        "vcf_max_rss_mb": vcf_bench["max_rss"],
        "bcf_max_rss_mb": bcf_bench["max_rss"],
        "bcf_vs_vcf_rss_ratio": rss_ratio,
    }

    print(
        f"\ncyvcf2 vcf-vs-bcf profiling (chr{args.chromosome})\n"
        f"  vector_sha256_equal={vector_sha256_equal} "
        f"loci_exact_match={loci_equal} bed_exact={bed_exact}\n"
        f"  vcf={row['vcf_seconds']}s bcf={row['bcf_seconds']}s "
        f"bcf_vs_vcf_ratio={bcf_vs_vcf_ratio}x\n"
        f"  vcf_peak_rss={row['vcf_max_rss_mb']}MB "
        f"bcf_peak_rss={row['bcf_max_rss_mb']}MB ratio={rss_ratio}x"
    )
    if not vector_sha256_equal or not loci_equal or not bed_exact:
        print(
            "\nWARNING: VCF and BCF input produced different output for "
            "identical content -- this indicates a real bug, not expected "
            "imprecision."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
