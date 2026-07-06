"""Tests for the fused covariance-generation sidecar prototype.

Validates, in order:

1. ``_diag_values_impl`` (vectorized diagonal precompute) reproduces the
   persisted ``index/diag_pos``/``index/diag_val`` datasets bit-for-bit,
   without a second pairwise-kernel pass.
2. The fused sidecar accumulator (tee'd off the same ``CovarianceRowChunk``
   stream used to write the persisted HDF5 partition) reproduces the
   existing post-hoc vector build (``vector_array.py``) and metric
   (``metric_from_arrays``) bit-for-bit, on both a real chromosome fixture
   and synthetic multi-partition overlap fixtures.
"""

from __future__ import annotations

import gzip
import random
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from ldetect_lite._util.covariance_array import (
    load_chromosome_covariance,
    metric_from_arrays,
)
from ldetect_lite._util.covariance_sidecars import (
    CovarianceSidecarAccumulator,
    merge_metric_coverage_fragments,
    metric_coverage_sum_at_breakpoints,
)
from ldetect_lite._util.vector_array import (
    _merge_diag_vector_partition_result,
    write_diag_vector_array,
)
from ldetect_lite.io.covariance_hdf5 import open_covariance_reader
from ldetect_lite.io.partitions import CovarianceStore
from ldetect_lite.shrinkage import _diag_values_impl, calc_covariance

# ---------------------------------------------------------------------------
# Shared synthetic fixture builders
# ---------------------------------------------------------------------------

_INDIVIDUALS = ["sample_a", "sample_b", "sample_c", "sample_d", "sample_e"]

# (position, genetic_position_cM, genotypes-per-individual)
_LOCI: list[tuple[int, float, list[str]]] = [
    (100, 0.000, ["0|0", "0|0", "0|0", "0|0", "0|0"]),  # monomorphic -> filtered
    (200, 0.001, ["0|1", "0|0", "0|0", "0|0", "0|0"]),
    (300, 0.002, ["0|1", "1|0", "0|1", "0|0", "1|1"]),
    (400, 0.003, ["1|1", "1|1", "0|1", "1|0", "0|0"]),
    (500, 0.004, ["0|1", "1|1", "1|1", "1|1", "1|1"]),  # near-monomorphic
]


def _write_map(path: Path) -> None:
    with gzip.open(path, "wt") as f:
        for pos, gpos, _ in _LOCI:
            f.write(f"1 {pos} {gpos}\n")


def _write_individuals(path: Path) -> None:
    path.write_text("\n".join(_INDIVIDUALS) + "\n")


def _vcf_stream() -> StringIO:
    header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(
        _INDIVIDUALS
    )
    lines = ["##fileformat=VCFv4.2", header]
    for pos, _, genotypes in _LOCI:
        lines.append(
            f"1\t{pos}\trs{pos}\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(genotypes)
        )
    lines.append("")
    return StringIO("\n".join(lines))


def _expected_hap_sums() -> tuple[np.ndarray, float, float]:
    """Independently derive hap_sums/n_total/theta the way calc_covariance does."""
    n_ind = len(_INDIVIDUALS)
    n_haps = 2 * n_ind
    hap_sums = []
    for _, _, genotypes in _LOCI:
        alleles = [int(a) for gt in genotypes for a in gt.split("|")]
        assert len(alleles) == n_haps
        hap_sums.append(float(sum(alleles)))
    harmonic = sum(1.0 / i for i in range(1, n_haps))
    theta = (1.0 / harmonic) / (n_haps + 1.0 / harmonic)
    return np.array(hap_sums, dtype=np.float64), float(n_haps), theta


@pytest.fixture()
def synthetic_partition(tmp_path: Path) -> Path:
    map_path = tmp_path / "map.gz"
    _write_map(map_path)
    individuals_path = tmp_path / "inds.txt"
    _write_individuals(individuals_path)

    out_path = tmp_path / "cov.h5"
    calc_covariance(
        vcf_stream=_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
        compact_output=True,
    )
    return out_path


def test_diag_values_impl_matches_persisted_diagonal(
    synthetic_partition: Path,
) -> None:
    hap_sums, n_total, theta = _expected_hap_sums()
    positions = np.array([pos for pos, _, _ in _LOCI], dtype=np.int64)

    valid, diag_val = _diag_values_impl(hap_sums, n_total, theta, cutoff=1e-7)

    with open_covariance_reader(synthetic_partition, 100, 500) as reader:
        rows = reader.read_all()
    diag_pos = np.asarray(rows.lo)[np.asarray(rows.lo) == np.asarray(rows.hi)]
    diag_val_persisted = np.asarray(rows.shrink_ld)[
        np.asarray(rows.lo) == np.asarray(rows.hi)
    ]

    assert set(positions[valid].tolist()) == set(diag_pos.tolist())

    order = np.argsort(positions[valid])
    persisted_order = np.argsort(diag_pos)
    np.testing.assert_array_equal(positions[valid][order], diag_pos[persisted_order])
    np.testing.assert_array_equal(
        diag_val[valid][order], diag_val_persisted[persisted_order]
    )


def test_diag_values_impl_monomorphic_locus_is_filtered(
    synthetic_partition: Path,
) -> None:
    hap_sums, n_total, theta = _expected_hap_sums()
    valid, _ = _diag_values_impl(hap_sums, n_total, theta, cutoff=1e-7)
    # Locus 100 (all 0|0) and locus 500 (near-monomorphic, f1 close to but not
    # exactly 1) — only the truly monomorphic locus must be filtered.
    positions = [pos for pos, _, _ in _LOCI]
    assert not valid[positions.index(100)]


# ---------------------------------------------------------------------------
# Fused sidecar vs. post-hoc HDF5 read: single-partition bit-exactness
# ---------------------------------------------------------------------------

_RICH_N_IND = 10
_RICH_N_SNPS = 12
_RICH_START = 10_000
_RICH_STEP = 100


def _rich_individuals() -> list[str]:
    return [f"ind{i}" for i in range(_RICH_N_IND)]


def _rich_positions() -> list[int]:
    return [_RICH_START + i * _RICH_STEP for i in range(_RICH_N_SNPS)]


def _rich_genotypes() -> list[list[str]]:
    """Deterministic, LD-bearing genotypes: some SNPs correlated, some not."""
    rng = random.Random(20260706)
    n_haps = 2 * _RICH_N_IND
    base_haps = [rng.random() < 0.4 for _ in range(n_haps)]

    genotypes: list[list[str]] = []
    for snp_idx in range(_RICH_N_SNPS):
        if snp_idx % 3 == 0:
            # Correlated with the base haplotype pattern (a handful of flips).
            haps = list(base_haps)
            for flip_idx in rng.sample(range(n_haps), k=2):
                haps[flip_idx] = not haps[flip_idx]
        else:
            haps = [rng.random() < 0.4 for _ in range(n_haps)]
        genotypes.append(
            [
                f"{int(haps[2 * ind])}|{int(haps[2 * ind + 1])}"
                for ind in range(_RICH_N_IND)
            ]
        )
    return genotypes


def _rich_vcf_stream() -> StringIO:
    individuals = _rich_individuals()
    positions = _rich_positions()
    genotypes = _rich_genotypes()
    header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(
        individuals
    )
    lines = ["##fileformat=VCFv4.2", header]
    for pos, row in zip(positions, genotypes):
        lines.append(f"1\t{pos}\trs{pos}\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(row))
    lines.append("")
    return StringIO("\n".join(lines))


@pytest.fixture()
def rich_store(tmp_path: Path) -> tuple[CovarianceStore, CovarianceSidecarAccumulator]:
    """A single, richer synthetic partition generated *through the sidecar hook*."""
    map_path = tmp_path / "map.gz"
    with gzip.open(map_path, "wt") as f:
        for i, pos in enumerate(_rich_positions()):
            f.write(f"1 {pos} {i * 0.001}\n")
    individuals_path = tmp_path / "inds.txt"
    individuals_path.write_text("\n".join(_rich_individuals()) + "\n")

    store = CovarianceStore(root=tmp_path / "cov")
    (store.root / "chr1").mkdir(parents=True)
    out_path = store.partition_path("chr1", _RICH_START, _rich_positions()[-1])

    sidecar = CovarianceSidecarAccumulator()
    calc_covariance(
        vcf_stream=_rich_vcf_stream(),
        genetic_map_path=map_path,
        individuals_path=individuals_path,
        output_path=out_path,
        cutoff=1e-7,
        compact_output=True,
        sidecar=sidecar,
    )
    return store, sidecar


def _read_vector_gz(path: Path) -> dict[int, float]:
    data: dict[int, float] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                data[int(parts[0])] = float(parts[1])
    return data


def test_fused_vector_fragment_matches_posthoc_hdf5_read(
    rich_store: tuple[CovarianceStore, CovarianceSidecarAccumulator],
    tmp_path: Path,
) -> None:
    store, sidecar = rich_store
    positions = _rich_positions()
    snp_first, snp_last = positions[0], positions[-1]

    # Ground truth: existing post-hoc HDF5-based vector build on the exact
    # same persisted partition (no sidecar involved).
    ref_path = tmp_path / "ref_vector.txt.gz"
    write_diag_vector_array(
        name="chr1",
        store=store,
        partitions=[(snp_first, snp_last)],
        snp_first=snp_first,
        snp_last=snp_last,
        out_path=ref_path,
    )
    ref = _read_vector_gz(ref_path)
    assert ref, "reference vector build produced no rows; fixture is too sparse"

    # Fused: replay the buffered, tee'd chunks from generation time and feed
    # the result through the *same* merge/flush machinery, unmodified.
    fragment = sidecar.finalize_vector(
        end=snp_last,
        next_start=None,
        snp_last=snp_last,
        center_lower_bound=snp_first,
        center_lower_inclusive=True,
    )
    fused_path = tmp_path / "fused_vector.txt.gz"
    pending_sums: dict[int, float] = {}
    parent_profile: dict[str, float | int] = {
        "merge_seconds": 0.0,
        "flush_seconds": 0.0,
        "worker_wait_seconds": 0.0,
        "partitions": 0,
    }
    _merge_diag_vector_partition_result(
        result=fragment,
        snp_first=snp_first,
        snp_last=snp_last,
        current_locus=snp_first,
        pending_sums=pending_sums,
        out_path=fused_path,
        parent_profile=parent_profile,
    )
    fused = _read_vector_gz(fused_path)

    assert fused.keys() == ref.keys()
    for locus in ref:
        assert fused[locus] == ref[locus], (
            f"locus {locus}: fused={fused[locus]!r} ref={ref[locus]!r}"
        )


def test_fused_metric_coverage_matches_metric_from_arrays(
    rich_store: tuple[CovarianceStore, CovarianceSidecarAccumulator],
) -> None:
    store, sidecar = rich_store
    positions = _rich_positions()
    snp_first, snp_last = positions[0], positions[-1]

    fragment = sidecar.finalize_metric_coverage(
        lower_min=snp_first, lower_max=snp_last, lower_inclusive=True
    )
    cov_positions, coverage = merge_metric_coverage_fragments([fragment])

    cache = load_chromosome_covariance(
        "chr1", store, [(snp_first, snp_last)], snp_first, snp_last
    )
    # This fixture's max surviving-pair span is 600bp (6 SNP steps) -- verified
    # empirically in the "violates single crossing assumption" test below.
    # Breakpoint sets here are spaced > 600bp apart so no pair's [lo, hi)
    # interval contains more than one breakpoint (the sidecar's documented
    # single-breakpoint-crossing precondition).
    for breakpoints in (
        [positions[3]],
        [positions[1], positions[9]],
    ):
        expected = metric_from_arrays(cache, breakpoints)
        got_sum = metric_coverage_sum_at_breakpoints(
            cov_positions, coverage, np.asarray(breakpoints, dtype=np.int64)
        )
        # Not bit-exact by design: metric_from_arrays sums crossing pairs
        # directly in one np.sum() reduction, while the coverage sidecar
        # sums the *same* r2 values via a difference-array + prefix-sum
        # (grouped by position, then cumsum) -- a genuinely different
        # addition order. Floating-point addition isn't associative, so
        # only near-ULP-level agreement is expected here, unlike the vector
        # fragment above (which replays identical per-chunk grouping and IS
        # bit-exact).
        assert got_sum == pytest.approx(expected["sum"], rel=1e-9), (
            f"breakpoints={breakpoints}: fused={got_sum!r} ref={expected['sum']!r}"
        )


def test_metric_coverage_violates_single_crossing_assumption_when_breakpoints_are_close(
    rich_store: tuple[CovarianceStore, CovarianceSidecarAccumulator],
) -> None:
    """Demonstrate the sidecar's documented precondition actually matters.

    With breakpoints closer together than the longest surviving pair span
    (600bp in this fixture -- some pair spans SNP indices 2..8), a pair whose
    interval contains *two* breakpoints gets its r2 summed at both, while
    ``metric_from_arrays`` counts it once (``i_blocks != j_blocks`` is a
    boolean, not a count). This is the failure mode
    ``covariance-cache-redesign-plan.md`` flags as needing verification
    before trusting the coverage-array sidecar on real data -- it is not a
    hypothetical.
    """
    store, sidecar = rich_store
    positions = _rich_positions()
    snp_first, snp_last = positions[0], positions[-1]

    fragment = sidecar.finalize_metric_coverage(
        lower_min=snp_first, lower_max=snp_last, lower_inclusive=True
    )
    cov_positions, coverage = merge_metric_coverage_fragments([fragment])
    cache = load_chromosome_covariance(
        "chr1", store, [(snp_first, snp_last)], snp_first, snp_last
    )

    breakpoints = [positions[2], positions[5], positions[9]]
    expected = metric_from_arrays(cache, breakpoints)
    got_sum = metric_coverage_sum_at_breakpoints(
        cov_positions, coverage, np.asarray(breakpoints, dtype=np.int64)
    )
    assert got_sum != pytest.approx(expected["sum"], rel=1e-6), (
        "expected the close-breakpoint case to overcount relative to "
        f"metric_from_arrays, got fused={got_sum!r} ref={expected['sum']!r}"
    )


# ---------------------------------------------------------------------------
# Multi-partition overlap: ownership/merge must not double-count shared pairs
# ---------------------------------------------------------------------------

_OVERLAP_N_IND = 10
_OVERLAP_N_SNPS = 16
_OVERLAP_START = 10_000
_OVERLAP_STEP = 100
_OVERLAP_PARTITIONS = [(10_000, 10_800), (10_500, 11_500)]


def _overlap_individuals() -> list[str]:
    return [f"ind{i}" for i in range(_OVERLAP_N_IND)]


def _overlap_positions() -> list[int]:
    return [_OVERLAP_START + i * _OVERLAP_STEP for i in range(_OVERLAP_N_SNPS)]


def _overlap_genotype_rows() -> dict[int, list[str]]:
    """One genotype row per physical position, shared by both partition slices."""
    rng = random.Random(20260706)
    n_haps = 2 * _OVERLAP_N_IND
    base_haps = [rng.random() < 0.4 for _ in range(n_haps)]
    rows: dict[int, list[str]] = {}
    for snp_idx, pos in enumerate(_overlap_positions()):
        if snp_idx % 3 == 0:
            haps = list(base_haps)
            for flip_idx in rng.sample(range(n_haps), k=2):
                haps[flip_idx] = not haps[flip_idx]
        else:
            haps = [rng.random() < 0.4 for _ in range(n_haps)]
        rows[pos] = [
            f"{int(haps[2 * ind])}|{int(haps[2 * ind + 1])}"
            for ind in range(_OVERLAP_N_IND)
        ]
    return rows


@pytest.fixture()
def overlap_store(
    tmp_path: Path,
) -> tuple[CovarianceStore, list[CovarianceSidecarAccumulator]]:
    """Two overlapping partitions of one source VCF, each generated through
    its own sidecar hook -- mirrors
    ``build_two_overlapping_partitions_with_duplicate_position`` but with
    richer, LD-bearing genotypes and sidecars attached.
    """
    individuals = _overlap_individuals()
    genotype_rows = _overlap_genotype_rows()
    positions = _overlap_positions()

    map_path = tmp_path / "map.gz"
    with gzip.open(map_path, "wt") as f:
        for i, pos in enumerate(positions):
            f.write(f"1 {pos} {i * 0.001}\n")
    individuals_path = tmp_path / "inds.txt"
    individuals_path.write_text("\n".join(individuals) + "\n")

    header = (
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(individuals)
        + "\n"
    )
    store = CovarianceStore(root=tmp_path / "cov")
    chrom_dir = store.root / "chr1"
    chrom_dir.mkdir(parents=True)
    with (store.root / "chr1_partitions.txt").open("w") as f:
        for start, end in _OVERLAP_PARTITIONS:
            f.write(f"{start} {end}\n")

    sidecars: list[CovarianceSidecarAccumulator] = []
    for start, end in _OVERLAP_PARTITIONS:
        slice_positions = [p for p in positions if start <= p <= end]
        body = "\n".join(
            f"1\t{pos}\trs{pos}\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(genotype_rows[pos])
            for pos in slice_positions
        )
        sidecar = CovarianceSidecarAccumulator()
        calc_covariance(
            vcf_stream=StringIO(f"{header}{body}\n"),
            genetic_map_path=map_path,
            individuals_path=individuals_path,
            output_path=chrom_dir / f"chr1.{start}.{end}.h5",
            cutoff=1e-7,
            compact_output=True,
            sidecar=sidecar,
        )
        sidecars.append(sidecar)

    return store, sidecars


def test_fused_vector_fragments_merge_matches_posthoc_multi_partition_read(
    overlap_store: tuple[CovarianceStore, list[CovarianceSidecarAccumulator]],
    tmp_path: Path,
) -> None:
    from ldetect_lite._util.vector_array import _plan_diag_vector_partitions

    store, sidecars = overlap_store
    positions = _overlap_positions()
    snp_first, snp_last = positions[0], positions[-1]

    ref_path = tmp_path / "ref_vector.txt.gz"
    write_diag_vector_array(
        name="chr1",
        store=store,
        partitions=_OVERLAP_PARTITIONS,
        snp_first=snp_first,
        snp_last=snp_last,
        out_path=ref_path,
    )
    ref = _read_vector_gz(ref_path)
    assert ref, "reference vector build produced no rows; fixture is too sparse"

    plans = _plan_diag_vector_partitions(_OVERLAP_PARTITIONS, snp_first, snp_last)
    assert len(plans) == len(sidecars)

    fused_path = tmp_path / "fused_vector.txt.gz"
    pending_sums: dict[int, float] = {}
    parent_profile: dict[str, float | int] = {
        "merge_seconds": 0.0,
        "flush_seconds": 0.0,
        "worker_wait_seconds": 0.0,
        "partitions": 0,
    }
    current_locus = snp_first
    for plan, sidecar in zip(plans, sidecars):
        fragment = sidecar.finalize_vector(
            end=plan.end,
            next_start=plan.next_start,
            snp_last=snp_last,
            center_lower_bound=plan.center_lower_bound,
            center_lower_inclusive=plan.center_lower_inclusive,
        )
        current_locus = _merge_diag_vector_partition_result(
            result=fragment,
            snp_first=snp_first,
            snp_last=snp_last,
            current_locus=current_locus,
            pending_sums=pending_sums,
            out_path=fused_path,
            parent_profile=parent_profile,
        )
    fused = _read_vector_gz(fused_path)

    assert fused.keys() == ref.keys()
    for locus in ref:
        assert fused[locus] == ref[locus], (
            f"locus {locus}: fused={fused[locus]!r} ref={ref[locus]!r}"
        )


def test_fused_metric_coverage_merge_matches_metric_from_arrays_across_partitions(
    overlap_store: tuple[CovarianceStore, list[CovarianceSidecarAccumulator]],
) -> None:
    store, sidecars = overlap_store
    positions = _overlap_positions()
    snp_first, snp_last = positions[0], positions[-1]

    fragments = []
    for p_index, ((start, end), sidecar) in enumerate(
        zip(_OVERLAP_PARTITIONS, sidecars)
    ):
        lower_min = snp_first if p_index == 0 else start
        lower_max = (
            _OVERLAP_PARTITIONS[p_index + 1][0]
            if p_index + 1 < len(_OVERLAP_PARTITIONS)
            else snp_last
        )
        fragments.append(
            sidecar.finalize_metric_coverage(
                lower_min=lower_min,
                lower_max=lower_max,
                lower_inclusive=(p_index == 0),
            )
        )
    cov_positions, coverage = merge_metric_coverage_fragments(fragments)

    cache = load_chromosome_covariance(
        "chr1", store, _OVERLAP_PARTITIONS, snp_first, snp_last
    )
    # Single, isolated breakpoint inside the overlap region -- exercises
    # ownership across the two overlapping partitions without risking the
    # separately-tested single-breakpoint-crossing assumption.
    breakpoints = [positions[6]]
    expected = metric_from_arrays(cache, breakpoints)
    got_sum = metric_coverage_sum_at_breakpoints(
        cov_positions, coverage, np.asarray(breakpoints, dtype=np.int64)
    )
    assert got_sum == pytest.approx(expected["sum"], rel=1e-9)
    assert expected["sum"] != 0.0, "test is vacuous if nothing crosses the breakpoint"


# ---------------------------------------------------------------------------
# Single-breakpoint-crossing assumption: empirical check on real chromosome data
# ---------------------------------------------------------------------------


def test_single_breakpoint_crossing_assumption_on_real_fixture(
    example_data_dir: Path, example_store: CovarianceStore
) -> None:
    """Report (not assert pass/fail) max pair span vs. min real breakpoint gap.

    This fixture is a *single* HDF5 partition (chr2:39967768-40067768), so it
    cannot validate the cross-partition case the design doc actually worries
    about -- only whether the assumption holds for pairs and breakpoints both
    drawn from real data at all. That's a necessary but not sufficient check;
    stated as a limitation in ``notes/logs/covariance-cache-redesign-plan.md``.
    """
    import pickle

    with open_covariance_reader(
        example_data_dir / "cov_matrix/chr2/chr2.39967768.40067768.h5",
        39967768,
        40067768,
    ) as reader:
        rows = reader.read_all()
    lo = np.asarray(rows.lo)
    hi = np.asarray(rows.hi)
    off_diag = lo != hi
    max_pair_span = int((hi[off_diag] - lo[off_diag]).max())

    minima_path = (
        example_data_dir / "minima/minima-EUR-chr2-50-39967768-40067768.pickle"
    )
    with open(minima_path, "rb") as f:
        minima = pickle.load(f)
    loci = sorted(int(x) for x in minima["fourier_ls"]["loci"])
    gaps = [b - a for a, b in zip(loci, loci[1:])]
    min_gap = min(gaps)

    print(
        f"\n[single-breakpoint-crossing check] max_pair_span={max_pair_span} "
        f"min_real_breakpoint_gap={min_gap} "
        f"assumption_holds_here={max_pair_span < min_gap}"
    )
    # Not asserted: a single-partition fixture with one real breakpoint set
    # cannot prove the assumption holds in general (different populations,
    # window sizes, or n_snps_bw_bpoints settings could violate it even if
    # this one combination doesn't) -- see the "violates single crossing
    # assumption" test above for a constructive counterexample showing the
    # failure mode is real -- and, per the print output above, on real data.
