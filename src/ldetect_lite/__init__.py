"""ldetect-lite: approximately independent LD blocks in the human genome."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ldetect-lite")
except PackageNotFoundError:
    __version__ = "unknown"
