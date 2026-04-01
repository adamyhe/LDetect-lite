"""Tests for ldetect2.io.bed."""

from __future__ import annotations

from ldetect2.io.bed import write_bed


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def test_write_bed_header(tmp_path):
    out = tmp_path / "out.bed"
    write_bed("chr1", [200, 400], snp_first=100, snp_last=500, output=out)
    lines = out.read_text().splitlines()
    assert lines[0] == "chr\tstart\tstop"


def test_write_bed_line_count(tmp_path):
    loci = [200, 400, 600]
    out = tmp_path / "out.bed"
    write_bed("chr1", loci, snp_first=100, snp_last=700, output=out)
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    # header + (len(loci) + 1) regions
    assert len(lines) == len(loci) + 2


def test_write_bed_first_region(tmp_path):
    out = tmp_path / "out.bed"
    write_bed("chr2", [300, 500], snp_first=100, snp_last=700, output=out)
    lines = out.read_text().splitlines()
    fields = lines[1].split("\t")
    assert fields[0] == "chr2"
    assert int(fields[1]) == 100
    assert int(fields[2]) == 300


def test_write_bed_last_region(tmp_path):
    out = tmp_path / "out.bed"
    write_bed("chr2", [300, 500], snp_first=100, snp_last=700, output=out)
    lines = out.read_text().splitlines()
    fields = lines[-1].split("\t")
    assert int(fields[1]) == 500
    assert int(fields[2]) == 701  # snp_last + 1


def test_write_bed_contiguous_regions(tmp_path):
    loci = [200, 400, 600]
    out = tmp_path / "out.bed"
    write_bed("chr1", loci, snp_first=100, snp_last=700, output=out)
    lines = out.read_text().splitlines()[1:]  # skip header
    regions = [(int(l.split("\t")[1]), int(l.split("\t")[2])) for l in lines if l.strip()]
    for i in range(len(regions) - 1):
        assert regions[i][1] == regions[i + 1][0], "Regions must be contiguous"


def test_write_bed_single_locus(tmp_path):
    out = tmp_path / "out.bed"
    write_bed("chrX", [500], snp_first=100, snp_last=800, output=out)
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 3  # header + 2 regions
    fields_1 = lines[1].split("\t")
    assert int(fields_1[1]) == 100
    assert int(fields_1[2]) == 500
    fields_2 = lines[2].split("\t")
    assert int(fields_2[1]) == 500
    assert int(fields_2[2]) == 801


# ---------------------------------------------------------------------------
# stdout output
# ---------------------------------------------------------------------------

def test_write_bed_stdout(capsys):
    write_bed("chr3", [300, 500], snp_first=100, snp_last=700, output=None)
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0] == "chr\tstart\tstop"
    assert len(lines) == 4  # header + 3 regions


def test_write_bed_stdout_matches_file(tmp_path, capsys):
    loci = [300, 500]
    out = tmp_path / "out.bed"
    write_bed("chr3", loci, snp_first=100, snp_last=700, output=out)
    file_content = out.read_text()

    write_bed("chr3", loci, snp_first=100, snp_last=700, output=None)
    stdout_content = capsys.readouterr().out

    assert file_content == stdout_content
