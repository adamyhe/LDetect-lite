"""Tests for ldetect2.io.vcf."""

from __future__ import annotations

import subprocess

from ldetect2.io.vcf import read_vcf_samples


def test_read_vcf_samples_invokes_bcftools_query(monkeypatch, tmp_path):
    vcf = tmp_path / "panel.vcf.gz"
    calls = []

    def fake_run(cmd, capture_output, text, check):
        calls.append((cmd, capture_output, text, check))
        return subprocess.CompletedProcess(cmd, 0, stdout="HG00096\nHG00097\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert read_vcf_samples(vcf) == {"HG00096", "HG00097"}
    assert calls == [
        (["bcftools", "query", "-l", str(vcf)], True, True, True),
    ]
