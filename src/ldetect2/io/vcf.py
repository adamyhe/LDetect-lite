"""Small VCF I/O helpers used by command-line workflows."""

from __future__ import annotations

import subprocess
from pathlib import Path


def read_vcf_samples(vcf_path: Path) -> set[str]:
    """Return sample IDs present in a VCF header using ``bcftools query -l``."""
    result = subprocess.run(
        ["bcftools", "query", "-l", str(vcf_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.strip().splitlines())
