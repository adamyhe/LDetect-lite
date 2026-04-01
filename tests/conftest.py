"""Shared pytest fixtures for ldetect2 tests."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from ldetect2.io.partitions import CovarianceStore

# Local cache directory (gitignored).
DATA_DIR = Path(__file__).parent / "data"

_BITBUCKET_RAW = (
    "https://bitbucket.org/nygcresearch/ldetect"
    "/raw/master/ldetect/examples/example_data"
)

# All files to download, relative to the example_data root.
_EXAMPLE_FILES = [
    "bed/EUR-chr2-50-39967768-40067768.bed",
    "chr2.interpolated_genetic_map.gz",
    "cov_matrix/chr2/chr2.39967768.40067768.gz",
    "cov_matrix/scripts/chr2_partitions",
    "eurinds.txt",
    "minima/minima-EUR-chr2-50-39967768-40067768.pickle",
    "vector/vector-EUR-chr2-39967768-40067768.txt.gz",
]


def _download_example_data(dest: Path) -> None:
    """Download example data from BitBucket into *dest* if not already present."""
    for rel in _EXAMPLE_FILES:
        target = dest / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{_BITBUCKET_RAW}/{rel}"
        print(f"  Downloading {rel} ...")
        urllib.request.urlretrieve(url, target)


@pytest.fixture(scope="session")
def example_data_dir() -> Path:
    """Return the path to the example data, downloading it if necessary."""
    _download_example_data(DATA_DIR)
    return DATA_DIR


@pytest.fixture(scope="session")
def example_store(example_data_dir: Path) -> CovarianceStore:
    return CovarianceStore(root=example_data_dir / "cov_matrix")
