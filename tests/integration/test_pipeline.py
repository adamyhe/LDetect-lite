"""Integration tests for the example pipeline against reference outputs."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _read_vector_gz(path: Path) -> dict[int, float]:
    data: dict[int, float] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                data[int(parts[0])] = float(parts[1])
    return data


def _parse_bed(path: Path) -> list[tuple[str, int, int]]:
    """Parse a BED file into (chrom, start, stop) tuples, skipping the header."""
    regions: list[tuple[str, int, int]] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            parts = line.strip().split()
            if parts:
                regions.append((parts[0], int(parts[1]), int(parts[2])))
    return regions


# ---------------------------------------------------------------------------
# Step 3: matrix-to-vector
# ---------------------------------------------------------------------------

def test_matrix_to_vector_positions(example_data_dir, example_store, tmp_path):
    """Output positions should match the reference vector positions exactly."""
    from ldetect2.matrix_analysis import MatrixAnalysis

    out_path = tmp_path / "vector.txt.gz"
    MatrixAnalysis("chr2", example_store).calc_diag_lean(out_path)

    ref_vector = example_data_dir / "vector/vector-EUR-chr2-39967768-40067768.txt.gz"
    ref = _read_vector_gz(ref_vector)
    out = _read_vector_gz(out_path)

    assert set(out.keys()) == set(ref.keys()), "Position sets differ"


def test_matrix_to_vector_values(example_data_dir, example_store, tmp_path):
    """Output values should agree with the reference vector within tolerance."""
    from ldetect2.matrix_analysis import MatrixAnalysis

    out_path = tmp_path / "vector.txt.gz"
    MatrixAnalysis("chr2", example_store).calc_diag_lean(out_path)

    ref_vector = example_data_dir / "vector/vector-EUR-chr2-39967768-40067768.txt.gz"
    ref = _read_vector_gz(ref_vector)
    out = _read_vector_gz(out_path)

    mismatches = [
        pos for pos in ref
        if abs(out.get(pos, 0.0) - ref[pos]) > 1e-8
    ]
    assert not mismatches, (
        f"{len(mismatches)} positions differ by more than 1e-8: "
        f"{mismatches[:5]}"
    )


# ---------------------------------------------------------------------------
# Steps 4–5: find-breakpoints → BED
# ---------------------------------------------------------------------------

def test_find_breakpoints_json_structure(example_data_dir, example_store, tmp_path):
    """find_breakpoints must produce valid JSON with expected subsets."""
    from ldetect2.pipeline import find_breakpoints

    out_json = tmp_path / "breakpoints.json"
    ref_vector = example_data_dir / "vector/vector-EUR-chr2-39967768-40067768.txt.gz"
    find_breakpoints(
        input_path=ref_vector,
        chr_name="chr2",
        store=example_store,
        n_snps_bw_bpoints=50,
        output_path=out_json,
    )

    data = json.loads(out_json.read_text())
    for subset in ("fourier", "fourier_ls", "uniform", "uniform_ls"):
        assert subset in data, f"Missing subset {subset!r}"
        assert "loci" in data[subset]
        assert "metric" in data[subset]


def test_find_breakpoints_loci_in_range(example_data_dir, example_store, tmp_path):
    """All breakpoint loci must fall within the chromosome range."""
    from ldetect2.pipeline import find_breakpoints

    out_json = tmp_path / "breakpoints.json"
    ref_vector = example_data_dir / "vector/vector-EUR-chr2-39967768-40067768.txt.gz"
    find_breakpoints(
        input_path=ref_vector,
        chr_name="chr2",
        store=example_store,
        n_snps_bw_bpoints=50,
        output_path=out_json,
    )

    data = json.loads(out_json.read_text())
    loci = data["fourier_ls"]["loci"]
    out_of_range = [loc for loc in loci if not (39967768 <= loc <= 40067768)]
    assert not out_of_range, f"Loci out of range: {out_of_range}"


def test_find_breakpoints_uses_supplied_covariance_cache(
    example_data_dir,
    example_store,
    tmp_path,
    monkeypatch,
):
    """Cached normal metrics should avoid reloading chromosome arrays."""
    import ldetect2.local_search as local_search_mod
    import ldetect2.pipeline as pipeline_mod
    from ldetect2._util.covariance_array import load_chromosome_covariance
    from ldetect2.io.partitions import get_final_partitions

    partitions = get_final_partitions(
        example_store,
        "chr2",
        39967768,
        40067768,
    )
    cache = load_chromosome_covariance(
        "chr2",
        example_store,
        partitions,
        39967768,
        40067768,
    )

    def fail_reload(*args, **kwargs):
        raise AssertionError("find_breakpoints should reuse supplied covariance cache")

    monkeypatch.setattr(pipeline_mod, "load_covariance_arrays", fail_reload)
    monkeypatch.setattr(local_search_mod, "load_covariance_arrays", fail_reload)

    out_json = tmp_path / "breakpoints.json"
    ref_vector = example_data_dir / "vector/vector-EUR-chr2-39967768-40067768.txt.gz"
    pipeline_mod.find_breakpoints(
        input_path=ref_vector,
        chr_name="chr2",
        store=example_store,
        n_snps_bw_bpoints=50,
        output_path=out_json,
        snp_first=39967768,
        snp_last=40067768,
        covariance_cache=cache,
    )

    data = json.loads(out_json.read_text())
    for subset in ("fourier", "fourier_ls", "uniform", "uniform_ls"):
        assert data[subset]["loci"]
        assert "metric" in data[subset]


# ---------------------------------------------------------------------------
# Full pipeline: covariance → BED comparison against reference
# ---------------------------------------------------------------------------

def test_full_pipeline_bed_structure(example_store, tmp_path):
    """End-to-end pipeline must produce a BED file with correct structure."""
    from ldetect2.io.bed import write_bed
    from ldetect2.matrix_analysis import MatrixAnalysis
    from ldetect2.pipeline import find_breakpoints

    vector_path = tmp_path / "vector.txt.gz"
    MatrixAnalysis("chr2", example_store).calc_diag_lean(vector_path)

    bp_path = tmp_path / "breakpoints.json"
    find_breakpoints(
        input_path=vector_path,
        chr_name="chr2",
        store=example_store,
        n_snps_bw_bpoints=50,
        output_path=bp_path,
    )

    data = json.loads(bp_path.read_text())
    loci = data["fourier_ls"]["loci"]

    bed_path = tmp_path / "out.bed"
    write_bed("chr2", loci, snp_first=39967768, snp_last=40067768, output=bed_path)

    regions = _parse_bed(bed_path)
    assert len(regions) > 0

    # Regions are contiguous
    for i in range(len(regions) - 1):
        assert regions[i][2] == regions[i + 1][1], "Regions must be contiguous"

    # Coverage starts and ends at the correct positions
    assert regions[0][1] == 39967768
    assert regions[-1][2] == 40067769  # snp_last + 1


def test_full_pipeline_bed_matches_reference(example_data_dir, example_store, tmp_path):
    """End-to-end pipeline BED output should match the reference BED file.

    NOTE: This test verifies numeric reproducibility.  If the algorithm diverges
    from the reference implementation, update the expected values accordingly.
    """
    from ldetect2.io.bed import write_bed
    from ldetect2.matrix_analysis import MatrixAnalysis
    from ldetect2.pipeline import find_breakpoints

    vector_path = tmp_path / "vector.txt.gz"
    MatrixAnalysis("chr2", example_store).calc_diag_lean(vector_path)

    bp_path = tmp_path / "breakpoints.json"
    find_breakpoints(
        input_path=vector_path,
        chr_name="chr2",
        store=example_store,
        n_snps_bw_bpoints=50,
        output_path=bp_path,
    )

    data = json.loads(bp_path.read_text())
    loci = data["fourier_ls"]["loci"]

    bed_path = tmp_path / "out.bed"
    write_bed("chr2", loci, snp_first=39967768, snp_last=40067768, output=bed_path)

    ref_regions = _parse_bed(example_data_dir / "bed/EUR-chr2-50-39967768-40067768.bed")
    out_regions = _parse_bed(bed_path)

    assert len(out_regions) == len(ref_regions), (
        f"Region count mismatch: got {len(out_regions)}, expected {len(ref_regions)}"
    )
    for i, (ref, out) in enumerate(zip(ref_regions, out_regions)):
        assert out[1] == ref[1] and out[2] == ref[2], (
            f"Region {i}: got ({out[1]}, {out[2]}), expected ({ref[1]}, {ref[2]})"
        )
