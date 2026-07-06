"""Priority 5 prototype validation: on-demand VCF recompute vs. persisted cache.

Uses a small real slice of the MacDonald2022 chr9 VCF (gitignored, local-only
data -- these tests skip gracefully if it's absent) to prove the two things
that actually matter for
``notes/logs/covariance-cache-redesign-plan.md``'s priority 5:

1. ``recompute_partition_to_hdf5`` is deterministic: calling it twice for the
   same ``(start, end)`` bounds against the same VCF/map/individuals produces
   bit-identical HDF5 partitions -- the invariant on-demand recompute depends
   on (nothing about the source data changes between generation time and
   query time).
2. ``LocalSearch`` makes identical decisions whether it reads a "cache" store
   (generated once) or a "recompute" store (generated independently, later,
   simulating on-demand recompute), across a window spanning two partitions
   -- the multi-partition boundary path
   ``notes/logs/local-search-divergence-asn22.md`` documents as historically
   fragile. This is the test that actually matters for priority 5's
   correctness claim, not just matching HDF5 bytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ldetect_lite._util.local_search_vcf_recompute import recompute_partition_to_hdf5
from ldetect_lite.io.covariance_hdf5 import open_covariance_reader
from ldetect_lite.io.partitions import CovarianceStore
from ldetect_lite.local_search import LocalSearch
from ldetect_lite.metric import Metric

_CHR9_VCF = Path(
    "examples/MacDonald2022/data/raw/"
    "ALL.chr9.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz"
)
_CHR9_MAP = Path(
    "examples/MacDonald2022/data/maps/pyrho_interpolated_maps/GWD/chr9.tab.gz"
)
_EUR_INDS = Path("examples/MacDonald2022/resources/EUR_inds.txt")
_N_INDIVIDUALS = 20
_CHROM = "chr9"
# Two overlapping windows, matching how real chromosome partitioning
# overlaps -- exercises get_final_partitions() spanning both.
_PARTITIONS = [(100_000, 250_000), (200_000, 400_000)]


def _require_real_data() -> None:
    for path in (_CHR9_VCF, _CHR9_MAP, _EUR_INDS):
        if not path.exists():
            pytest.skip(
                f"Real chr9 fixture is missing (gitignored, local-only): {path}"
            )


@pytest.fixture(scope="module")
def individuals_subset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _require_real_data()
    lines = _EUR_INDS.read_text().splitlines()[:_N_INDIVIDUALS]
    path = tmp_path_factory.mktemp("recompute_inds") / "inds.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def _build_store(root: Path, individuals_path: Path) -> CovarianceStore:
    store = CovarianceStore(root=root)
    chrom_dir = root / _CHROM
    chrom_dir.mkdir(parents=True)
    with (root / f"{_CHROM}_partitions.txt").open("w") as f:
        for start, end in _PARTITIONS:
            f.write(f"{start} {end}\n")
    for start, end in _PARTITIONS:
        recompute_partition_to_hdf5(
            vcf_path=_CHR9_VCF,
            genetic_map_path=_CHR9_MAP,
            individuals_path=individuals_path,
            chrom=_CHROM,
            start=start,
            end=end,
            output_path=chrom_dir / f"{_CHROM}.{start}.{end}.h5",
        )
    return store


def test_recompute_is_deterministic_across_independent_invocations(
    individuals_subset: Path, tmp_path: Path
) -> None:
    """Same (start, end) bounds, same VCF/map/individuals -> bit-identical rows."""
    cache_store = _build_store(tmp_path / "cache", individuals_subset)
    recompute_store = _build_store(tmp_path / "recompute", individuals_subset)

    for start, end in _PARTITIONS:
        cache_path = cache_store.partition_path(_CHROM, start, end)
        recompute_path = recompute_store.partition_path(_CHROM, start, end)
        with open_covariance_reader(cache_path, start, end) as reader:
            cache_rows = reader.read_all()
            cache_diag_pos, cache_diag_val = reader.read_diagonal()
        with open_covariance_reader(recompute_path, start, end) as reader:
            recompute_rows = reader.read_all()
            recompute_diag_pos, recompute_diag_val = reader.read_diagonal()

        assert cache_rows.lo.size > 0, "fixture window produced no rows"
        np.testing.assert_array_equal(cache_rows.lo, recompute_rows.lo)
        np.testing.assert_array_equal(cache_rows.hi, recompute_rows.hi)
        np.testing.assert_array_equal(cache_rows.shrink_ld, recompute_rows.shrink_ld)
        np.testing.assert_array_equal(cache_diag_pos, recompute_diag_pos)
        np.testing.assert_array_equal(cache_diag_val, recompute_diag_val)


def test_local_search_matches_across_cache_and_recompute_stores(
    individuals_subset: Path, tmp_path: Path
) -> None:
    cache_store = _build_store(tmp_path / "cache", individuals_subset)
    recompute_store = _build_store(tmp_path / "recompute", individuals_subset)

    breakpoints = [110_000, 230_000, 380_000]
    idx = 1
    start_search, stop_search = 150_000, 300_000

    metric = Metric(_CHROM, cache_store, breakpoints, 100_000, 400_000).calc_metric()

    def _run(
        store: CovarianceStore, use_decimal: bool
    ) -> tuple[int | None, dict | None]:
        search = LocalSearch(
            _CHROM,
            start_search,
            stop_search,
            idx,
            breakpoints,
            metric["sum"],
            metric["N_zero"],
            store,
            use_decimal=use_decimal,
        )
        return search.search()

    for use_decimal in (False, True):
        cache_bp, cache_metric = _run(cache_store, use_decimal)
        recompute_bp, recompute_metric = _run(recompute_store, use_decimal)

        assert cache_bp == recompute_bp, f"use_decimal={use_decimal}"
        if cache_metric is None:
            assert recompute_metric is None
        else:
            assert recompute_metric is not None
            assert float(recompute_metric["sum"]) == pytest.approx(
                float(cache_metric["sum"])
            )
            assert float(recompute_metric["N_zero"]) == float(cache_metric["N_zero"])


def test_find_breakpoints_matches_across_cache_and_recompute_source(
    individuals_subset: Path, tmp_path: Path
) -> None:
    """Exercises find_breakpoints's own local_search_source="vcf-recompute"
    wiring (temp CovarianceStore lifecycle, _load_or_recompute_partition
    memoization, args threading) -- not just LocalSearch called directly, as
    in the tests above.
    """
    import json

    from ldetect_lite._util.vector_array import write_diag_vector_array
    from ldetect_lite.pipeline import find_breakpoints

    cache_store = _build_store(tmp_path / "cache", individuals_subset)
    snp_first, snp_last = _PARTITIONS[0][0], _PARTITIONS[-1][1]

    vector_path = tmp_path / "vector.txt.gz"
    write_diag_vector_array(
        name=_CHROM,
        store=cache_store,
        partitions=_PARTITIONS,
        snp_first=snp_first,
        snp_last=snp_last,
        out_path=vector_path,
    )

    def _run_find_breakpoints(local_search_source: str, out_name: str) -> dict:
        out_path = tmp_path / out_name
        find_breakpoints(
            input_path=vector_path,
            chr_name=_CHROM,
            store=cache_store,
            n_snps_bw_bpoints=50,
            output_path=out_path,
            snp_first=snp_first,
            snp_last=snp_last,
            local_search_source=local_search_source,
            vcf_path=str(_CHR9_VCF),
            genetic_map_path=_CHR9_MAP,
            individuals_path=individuals_subset,
            subsets={"fourier_ls"},
        )
        return json.loads(out_path.read_text())

    cache_result = _run_find_breakpoints("cache", "cache.json")
    recompute_result = _run_find_breakpoints("vcf-recompute", "recompute.json")

    assert cache_result["fourier_ls"]["loci"] == recompute_result["fourier_ls"]["loci"]
    assert cache_result["fourier_ls"]["metric"]["sum"] == pytest.approx(
        recompute_result["fourier_ls"]["metric"]["sum"]
    )
    assert (
        cache_result["fourier_ls"]["metric"]["N_zero"]
        == recompute_result["fourier_ls"]["metric"]["N_zero"]
    )


def test_find_breakpoints_rejects_vcf_recompute_with_multiple_workers(
    individuals_subset: Path, tmp_path: Path
) -> None:
    from ldetect_lite._util.vector_array import write_diag_vector_array
    from ldetect_lite.pipeline import find_breakpoints

    cache_store = _build_store(tmp_path / "cache", individuals_subset)
    snp_first, snp_last = _PARTITIONS[0][0], _PARTITIONS[-1][1]
    vector_path = tmp_path / "vector.txt.gz"
    write_diag_vector_array(
        name=_CHROM,
        store=cache_store,
        partitions=_PARTITIONS,
        snp_first=snp_first,
        snp_last=snp_last,
        out_path=vector_path,
    )

    with pytest.raises(ValueError, match="vcf-recompute"):
        find_breakpoints(
            input_path=vector_path,
            chr_name=_CHROM,
            store=cache_store,
            n_snps_bw_bpoints=50,
            output_path=tmp_path / "out.json",
            snp_first=snp_first,
            snp_last=snp_last,
            workers=2,
            local_search_source="vcf-recompute",
            vcf_path=str(_CHR9_VCF),
            genetic_map_path=_CHR9_MAP,
            individuals_path=individuals_subset,
            subsets={"fourier_ls"},
        )
