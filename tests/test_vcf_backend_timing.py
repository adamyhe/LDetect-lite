"""VCF backend equivalence + timing: naive parsing vs. pysam vs. cyvcf2.

Answers the question the redesign log raised alongside priority 5: is
switching `calc_covariance`'s VCF parsing to a C-accelerated library worth
it? This only measures -- it doesn't switch the production parser (see
`notes/logs/covariance-cache-redesign-plan.md`).

Uses the same real, gitignored chr9 VCF as
`test_local_search_vcf_recompute.py`; skips gracefully if it's absent.
`pysam`/`cyvcf2` are optional (`vcf-benchmark` extra) -- skips those specific
backends if not installed, so the equivalence/naive-only parts of this file
still run without them.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ldetect_lite._util.vcf_backends import (
    read_genotypes_naive,
    read_genotypes_pysam,
)

_CHR9_VCF = Path(
    "examples/MacDonald2022/data/raw/"
    "ALL.chr9.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz"
)
_EUR_INDS = Path("examples/MacDonald2022/resources/EUR_inds.txt")
_N_INDIVIDUALS = 20
_CHROM = "chr9"
# ~20,500 variants -- matches local search's typical ~10,000-20,000 SNP
# window scale (notes/logs/covariance-cache-redesign-plan.md's research).
_BENCH_START, _BENCH_END = 100_000, 700_000
# Small region for the equivalence check (fast, still exercises real data).
_EQUIV_START, _EQUIV_END = 100_000, 400_000


def _require_real_data() -> None:
    for path in (_CHR9_VCF, _EUR_INDS):
        if not path.exists():
            pytest.skip(
                f"Real chr9 fixture is missing (gitignored, local-only): {path}"
            )


@pytest.fixture(scope="module")
def individuals() -> list[str]:
    _require_real_data()
    return _EUR_INDS.read_text().splitlines()[:_N_INDIVIDUALS]


def _try_import(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def test_pysam_matches_naive_genotype_extraction(individuals: list[str]) -> None:
    if not _try_import("pysam"):
        pytest.skip("pysam not installed (uv sync --extra vcf-benchmark)")

    naive_pos, naive_haps = read_genotypes_naive(
        _CHR9_VCF, _CHROM, _EQUIV_START, _EQUIV_END, individuals
    )
    pysam_pos, pysam_haps = read_genotypes_pysam(
        _CHR9_VCF, _CHROM, _EQUIV_START, _EQUIV_END, individuals
    )
    assert len(naive_pos) > 0, "fixture window produced no rows"
    assert naive_pos == pysam_pos
    assert naive_haps == pysam_haps


def test_cyvcf2_matches_naive_genotype_extraction(individuals: list[str]) -> None:
    if not _try_import("cyvcf2"):
        pytest.skip("cyvcf2 not installed (uv sync --extra vcf-benchmark)")
    from ldetect_lite._util.vcf_backends import read_genotypes_cyvcf2

    naive_pos, naive_haps = read_genotypes_naive(
        _CHR9_VCF, _CHROM, _EQUIV_START, _EQUIV_END, individuals
    )
    cyvcf2_pos, cyvcf2_haps = read_genotypes_cyvcf2(
        _CHR9_VCF, _CHROM, _EQUIV_START, _EQUIV_END, individuals
    )
    assert naive_pos == cyvcf2_pos
    assert naive_haps == cyvcf2_haps


def test_vcf_backend_timing_benchmark() -> None:
    """Report-only: prints wall-clock per backend, swept over individual count.

    Individual count matters more than region size here: pysam's per-sample
    dict-like accessor (`record.samples[ind]["GT"]`) has real per-call
    Python overhead that scales *worse* than the naive parser as individual
    count grows, while cyvcf2's flat `variant.genotypes` list stays ahead at
    every size tested -- a single-point benchmark at one individual count
    would have missed this.
    """
    _require_real_data()
    all_individuals = _EUR_INDS.read_text().splitlines()

    backends = {"naive": read_genotypes_naive}
    if _try_import("pysam"):
        backends["pysam"] = read_genotypes_pysam
    if _try_import("cyvcf2"):
        from ldetect_lite._util.vcf_backends import read_genotypes_cyvcf2

        backends["cyvcf2"] = read_genotypes_cyvcf2

    print(f"\n[vcf backend benchmark] region={_CHROM}:{_BENCH_START}-{_BENCH_END}")
    for n_individuals in (20, 100, len(all_individuals)):
        individuals = all_individuals[:n_individuals]
        results: dict[str, tuple[float, int]] = {}
        for label, fn in backends.items():
            start = time.perf_counter()
            positions, _haps = fn(
                _CHR9_VCF, _CHROM, _BENCH_START, _BENCH_END, individuals
            )
            elapsed = time.perf_counter() - start
            results[label] = (elapsed, len(positions))

        n_rows = next(iter(results.values()))[1]
        naive_seconds = results["naive"][0]
        for label, (seconds, rows) in results.items():
            assert rows == n_rows, f"{label} produced a different row count"
            print(
                f"[vcf backend benchmark] n_individuals={n_individuals} "
                f"n_rows={n_rows} backend={label} seconds={seconds:.4f} "
                f"speedup_vs_naive={naive_seconds / seconds:.2f}x"
            )
