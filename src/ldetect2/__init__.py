"""ldetect2: approximately independent LD blocks in the human genome."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ldetect2")
except PackageNotFoundError:
    __version__ = "unknown"
