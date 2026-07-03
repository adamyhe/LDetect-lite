"""Download optional legacy ldetect toy example data.

The legacy diagnostics pipeline stages its own inputs from ldetect2 outputs and
does not require these files. They are useful only when inspecting or manually
running the vendored legacy example scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = (
    "https://bitbucket.org/nygcresearch/ldetect/raw/master/"
    "ldetect/examples/example_data"
)

FILES = (
    "eurinds.txt",
    "chr2.interpolated_genetic_map.gz",
    "vector/vector-EUR-chr2-39967768-40067768.txt.gz",
    "minima/minima-EUR-chr2-50-39967768-40067768.pickle",
    "bed/EUR-chr2-50-39967768-40067768.bed",
    "cov_matrix/scripts/chr2_partitions",
    "cov_matrix/chr2/chr2.39967768.40067768.gz",
)


def default_output_dir() -> Path:
    return (
        Path(__file__).resolve().parent
        / "legacy_ldetect"
        / "ldetect"
        / "examples"
        / "example_data"
    )


def download_file(base_url: str, relative_path: str, output_dir: Path, force: bool) -> None:
    output_path = output_dir / relative_path
    if output_path.exists() and not force:
        print(f"exists: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    url = f"{base_url.rstrip('/')}/{relative_path}"
    print(f"download: {url} -> {output_path}")
    urlretrieve(url, tmp_path)
    tmp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory where the legacy example_data tree should be written.",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="Base URL containing the legacy example_data files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files that already exist.",
    )
    args = parser.parse_args()

    for relative_path in FILES:
        download_file(args.base_url, relative_path, args.output_dir, args.force)


if __name__ == "__main__":
    main()
