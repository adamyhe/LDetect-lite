"""Tests for reference_panel.py: genetic map caching and haplotype decoding."""

from __future__ import annotations

import gzip
import shutil
import subprocess
from pathlib import Path

import cyvcf2
import numpy as np
import pytest

from ldetect_lite._util import reference_panel as rp
from ldetect_lite._util.reference_panel import read_genetic_map
from ldetect_lite.shrinkage import _pack_haplotypes_impl


def _write_map(path: Path, rows: list[tuple[int, float]]) -> None:
    with gzip.open(path, "wt") as f:
        for pos, gpos in rows:
            f.write(f"1 {pos} {gpos}\n")


def test_read_genetic_map_parses_position_to_gpos(tmp_path: Path) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path, [(100, 0.0), (200, 0.001), (300, 0.0025)])

    result = read_genetic_map(map_path)

    assert result == {100: 0.0, 200: 0.001, 300: 0.0025}


def test_read_genetic_map_caches_per_path(tmp_path: Path) -> None:
    map_path = tmp_path / "map.gz"
    _write_map(map_path, [(100, 0.0), (200, 0.001)])

    first = read_genetic_map(map_path)
    second = read_genetic_map(map_path)

    assert first is second  # same object: parsed once, not reparsed


def test_read_genetic_map_distinguishes_different_paths(tmp_path: Path) -> None:
    map_a = tmp_path / "a.gz"
    map_b = tmp_path / "b.gz"
    _write_map(map_a, [(100, 0.0)])
    _write_map(map_b, [(200, 0.5)])

    assert read_genetic_map(map_a) == {100: 0.0}
    assert read_genetic_map(map_b) == {200: 0.5}


# ---------------------------------------------------------------------------
# Direct-to-array / direct-to-packed haplotype decoding
# ---------------------------------------------------------------------------

requires_htslib_tools = pytest.mark.skipif(
    shutil.which("bgzip") is None or shutil.which("tabix") is None,
    reason="bgzip/tabix not found on PATH",
)


@pytest.mark.parametrize("n_haps", [1, 2, 63, 64, 65, 128, 130, 758])
def test_pack_haplotype_rows_matches_numba_packer(n_haps: int) -> None:
    rng = np.random.default_rng(n_haps)
    rows = rng.integers(0, 3, size=(37, n_haps)).astype(np.uint8)  # incl. 2s
    packed = rp.pack_haplotype_rows(rows)
    assert packed.dtype == np.uint64
    np.testing.assert_array_equal(packed, _pack_haplotypes_impl(rows))
    binary = (rows != 0).astype(np.uint8)
    np.testing.assert_array_equal(rp.unpack_haplotypes(packed, n_haps), binary)


def test_pack_haplotype_rows_empty() -> None:
    assert rp.pack_haplotype_rows(np.zeros((0, 70), np.uint8)).shape == (0, 2)
    assert rp.unpack_haplotypes(np.zeros((0, 2), np.uint64), 70).shape == (0, 70)


def _oracle_panel(
    vcf_path: Path, region: str | None, individuals: list[str], pos2gpos: dict
) -> tuple[list[int], list[str], np.ndarray, int, int]:
    """Independent re-statement of the original list-based decode + dedupe."""
    vcf = cyvcf2.VCF(str(vcf_path), samples=individuals)
    col = {s: i for i, s in enumerate(vcf.samples)}
    rows, pos, ids, skipped = [], [], [], 0
    for v in vcf(region) if region else vcf:
        if v.POS not in pos2gpos:
            continue
        g = [v.genotypes[col[s]] for s in individuals]
        if any((not ph) or a < 0 or b < 0 for a, b, ph in g):
            skipped += 1
            continue
        rows.append([x for a, b, _ in g for x in (a, b)])
        pos.append(v.POS)
        ids.append(v.ID or ".")
    vcf.close()
    seen: set[int] = set()
    keep = []
    for k, p in enumerate(pos):
        if p not in seen:
            seen.add(p)
            keep.append(k)
    hm = np.array([rows[k] for k in keep], dtype=np.uint8).reshape(
        len(keep), 2 * len(individuals)
    )
    return (
        [pos[k] for k in keep],
        [ids[k] for k in keep],
        hm,
        skipped,
        len(pos) - len(keep),
    )


def _random_vcf(tmp_path: Path, n_samples: int, n_records: int) -> Path:
    rng = np.random.default_rng(11)
    samples = [f"s{i}" for i in range(n_samples)]
    lines = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "##contig=<ID=1>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    pos = 1000
    for r in range(n_records):
        kind = rng.random()
        if kind > 0.12:  # new position; else duplicate of the previous one
            pos += int(rng.integers(1, 50))
        gts = [f"{a}|{b}" for a, b in rng.integers(0, 2, size=(n_samples, 2))]
        bad = rng.random()
        if bad < 0.05:
            gts[int(rng.integers(n_samples))] = "0/1"  # unphased
        elif bad < 0.08:
            gts[int(rng.integers(n_samples))] = ".|."  # missing
        rid = f"rs{r}" if rng.random() < 0.9 else "."
        lines.append(f"1\t{pos}\t{rid}\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(gts))
    raw = tmp_path / "rand.vcf"
    raw.write_text("\n".join(lines) + "\n")
    subprocess.run(["bgzip", "-f", str(raw)], check=True)
    gz = tmp_path / "rand.vcf.gz"
    subprocess.run(["tabix", "-f", "-p", "vcf", str(gz)], check=True)
    return gz


@requires_htslib_tools
@pytest.mark.parametrize("block_rows", [3, 4096])
@pytest.mark.parametrize("region", [None, "1:1500-4000"])
def test_read_reference_panel_layouts_match_list_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block_rows: int, region: str | None
) -> None:
    monkeypatch.setattr(rp, "_PACK_BLOCK_ROWS", block_rows)
    n_samples = 70  # 140 haplotypes: crosses two uint64 word boundaries
    vcf = _random_vcf(tmp_path, n_samples, 300)
    v = cyvcf2.VCF(str(vcf))
    all_pos = sorted({rec.POS for rec in v})
    v.close()
    pos2gpos = {p: i * 1e-4 for i, p in enumerate(all_pos) if i % 7 != 3}  # unmapped
    rng = np.random.default_rng(3)
    individuals = [f"s{i}" for i in rng.permutation(n_samples)[:60]]  # non-VCF order

    exp_pos, exp_ids, exp_hm, exp_skip, exp_dup = _oracle_panel(
        vcf, region, individuals, pos2gpos
    )
    assert exp_skip > 0 and exp_dup > 0  # fixture exercises both skip paths
    if block_rows < 100:
        assert exp_hm.shape[0] > 3 * block_rows  # several packed flushes / grows

    for layout in ("uint8", "packed"):
        panel = rp.read_reference_panel(vcf, region, individuals, pos2gpos, layout)
        assert panel.positions == exp_pos
        assert panel.rs_ids == exp_ids
        assert panel.skipped_unphased == exp_skip
        assert panel.duplicate_positions == exp_dup
        assert panel.n_haps == 2 * len(individuals)
        np.testing.assert_array_equal(
            panel.hap_sums, exp_hm.sum(axis=1).astype(np.float64)
        )
        np.testing.assert_array_equal(panel.haplotype_matrix(), exp_hm)
        if layout == "packed":
            assert panel.haplotypes is None
            np.testing.assert_array_equal(
                panel.packed_haplotypes, _pack_haplotypes_impl(exp_hm)
            )
        else:
            assert panel.packed_haplotypes is None
            assert panel.haplotypes is not None and panel.haplotypes.dtype == np.uint8


@requires_htslib_tools
def test_read_reference_panel_rejects_unknown_layout(tmp_path: Path) -> None:
    vcf = _random_vcf(tmp_path, 2, 3)
    with pytest.raises(ValueError, match="layout"):
        rp.read_reference_panel(vcf, None, ["s0"], {}, "bits")  # type: ignore[arg-type]
