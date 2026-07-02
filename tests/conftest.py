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

# All files to download: (local_rel, url_rel) pairs.
# local_rel is the path under *dest*; url_rel is the path on BitBucket.
_EXAMPLE_FILES: list[tuple[str, str]] = [
    ("bed/EUR-chr2-50-39967768-40067768.bed",
     "bed/EUR-chr2-50-39967768-40067768.bed"),
    ("chr2.interpolated_genetic_map.gz",
     "chr2.interpolated_genetic_map.gz"),
    # BitBucket stores this under scripts/; new convention uses the root.
    ("cov_matrix/chr2_partitions.txt",
     "cov_matrix/scripts/chr2_partitions"),
    ("eurinds.txt",
     "eurinds.txt"),
    ("minima/minima-EUR-chr2-50-39967768-40067768.pickle",
     "minima/minima-EUR-chr2-50-39967768-40067768.pickle"),
    ("vector/vector-EUR-chr2-39967768-40067768.txt.gz",
     "vector/vector-EUR-chr2-39967768-40067768.txt.gz"),
]


def _download_example_data(dest: Path) -> None:
    """Download example data from BitBucket into *dest* if not already present."""
    for local_rel, url_rel in _EXAMPLE_FILES:
        target = dest / local_rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{_BITBUCKET_RAW}/{url_rel}"
        print(f"  Downloading {local_rel} ...")
        urllib.request.urlretrieve(url, target)


@pytest.fixture(scope="session")
def example_data_dir() -> Path:
    """Return the path to the example data, downloading it if necessary."""
    _download_example_data(DATA_DIR)
    h5_path = DATA_DIR / "cov_matrix/chr2/chr2.39967768.40067768.h5"
    if not h5_path.exists():
        pytest.skip(f"Example HDF5 covariance fixture is missing: {h5_path}")
    return DATA_DIR


@pytest.fixture(scope="session")
def example_store(example_data_dir: Path) -> CovarianceStore:
    return CovarianceStore(root=example_data_dir / "cov_matrix")
