"""Compatibility shim for the legacy ldetect example entrypoints.

The diagnostic wrappers import legacy modules and call their pipeline functions
directly, so the full third-party commanderline package is not needed.
"""

from __future__ import annotations


def commander_line(*_args, **_kwargs) -> None:
    raise RuntimeError(
        "The vendored ldetect scripts are intended to be called through "
        "scripts/run_legacy_ldetect.py in this repository."
    )

