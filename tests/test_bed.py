"""Tests for ldetect_lite.io.bed."""

from __future__ import annotations

import gzip
import subprocess

import pytest

from ldetect_lite.io.bed import (
    read_genome_bed,
    read_single_chrom_bed,
    write_bed,
    write_block_bed,
)

# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def test_write_bed_header(tmp_path):
    out = tmp_path / "out.bed"
    write_bed("chr1", [200, 400], snp_first=100, snp_last=500, output=out)
    lines = out.read_text().splitlines()
    assert lines[0] == "#chr\tstart\tstop"


def test_write_bed_line_count(tmp_path):
    loci = [200, 400, 600]
    out = tmp_path / "out.bed"
    write_bed("chr1", loci, snp_first=100, snp_last=700, output=out)
    lines = [line for line in out.read_text().splitlines() if line.strip()]
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
    regions = [
        (int(line.split("\t")[1]), int(line.split("\t")[2]))
        for line in lines
        if line.strip()
    ]
    for i in range(len(regions) - 1):
        assert regions[i][1] == regions[i + 1][0], "Regions must be contiguous"


def test_write_bed_single_locus(tmp_path):
    out = tmp_path / "out.bed"
    write_bed("chrX", [500], snp_first=100, snp_last=800, output=out)
    lines = [line for line in out.read_text().splitlines() if line.strip()]
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
    assert lines[0] == "#chr\tstart\tstop"
    assert len(lines) == 4  # header + 3 regions


def test_write_bed_stdout_matches_file(tmp_path, capsys):
    loci = [300, 500]
    out = tmp_path / "out.bed"
    write_bed("chr3", loci, snp_first=100, snp_last=700, output=out)
    file_content = out.read_text()

    write_bed("chr3", loci, snp_first=100, snp_last=700, output=None)
    stdout_content = capsys.readouterr().out

    assert file_content == stdout_content


def test_read_genome_bed_skips_headers_and_groups_chromosomes(tmp_path):
    bed = tmp_path / "blocks.bed"
    bed.write_text(
        "#chr\tstart\tstop\n"
        "chr1\t100\t200\n"
        "chr1\t200\t300\n"
        "chr2\t50\t75\n"
        "track name=ignored\n"
        "chr3\tstart\tstop\n"
    )

    assert read_genome_bed(bed) == {
        "chr1": [(100, 200), (200, 300)],
        "chr2": [(50, 75)],
    }


def test_read_single_chrom_bed_returns_first_chrom_and_blocks(tmp_path):
    bed = tmp_path / "blocks.bed"
    bed.write_text("chr\tstart\tstop\nchr7\t10\t20\nchr7\t20\t30\n")

    assert read_single_chrom_bed(bed) == ("chr7", [(10, 20), (20, 30)])


def test_read_genome_bed_reads_gzip_file(tmp_path):
    bed = tmp_path / "blocks.bed.gz"
    with gzip.open(bed, "wt") as f:
        f.write("#chr\tstart\tstop\nchr1\t100\t200\nchr2\t50\t75\n")

    assert read_genome_bed(bed) == {
        "chr1": [(100, 200)],
        "chr2": [(50, 75)],
    }


def test_read_single_chrom_bed_reads_gzip_file(tmp_path):
    bed = tmp_path / "blocks.bed.gz"
    with gzip.open(bed, "wt") as f:
        f.write("chr\tstart\tstop\nchr7\t10\t20\nchr7\t20\t30\n")

    assert read_single_chrom_bed(bed) == ("chr7", [(10, 20), (20, 30)])


def test_write_block_bed_writes_gzip_file(tmp_path):
    bed = tmp_path / "blocks.bed.gz"
    calls = []

    def fake_run(cmd, input, capture_output, check):
        calls.append((cmd, input, capture_output, check))
        return subprocess.CompletedProcess(cmd, 0, stdout=gzip.compress(input))

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        write_block_bed("chr9", [(10, 20), (20, 30)], bed)
    finally:
        monkeypatch.undo()

    assert calls == [
        (
            ["bgzip", "-c"],
            b"chr\tstart\tstop\nchr9\t10\t20\nchr9\t20\t30\n",
            True,
            True,
        ),
    ]
    with gzip.open(bed, "rt") as f:
        assert f.read() == "chr\tstart\tstop\nchr9\t10\t20\nchr9\t20\t30\n"
    assert read_single_chrom_bed(bed) == ("chr9", [(10, 20), (20, 30)])


def test_write_block_bed_compressed_requires_bgzip(tmp_path, monkeypatch):
    bed = tmp_path / "blocks.bed.gz"

    def fake_run(cmd, input, capture_output, check):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="bgzip was not found"):
        write_block_bed("chr9", [(10, 20)], bed)


def test_write_block_bed_plain_file(tmp_path):
    bed = tmp_path / "blocks.bed"
    write_block_bed("chr9", [(10, 20), (20, 30)], bed)

    assert bed.read_text() == "chr\tstart\tstop\nchr9\t10\t20\nchr9\t20\t30\n"
    assert read_single_chrom_bed(bed) == ("chr9", [(10, 20), (20, 30)])
