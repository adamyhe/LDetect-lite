"""Small process-memory diagnostics used by profiling logs."""

from __future__ import annotations

import sys
from pathlib import Path

from ldetect_lite._util.logging import log_debug, log_msg


def current_rss_mib() -> float | None:
    """Return current resident set size in MiB when the platform exposes it."""
    status_path = Path("/proc/self/status")
    if status_path.exists():
        try:
            for line in status_path.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
        except Exception:
            return None
    return None


def max_rss_mib() -> float | None:
    """Return process lifetime maximum RSS in MiB when available."""
    try:
        import resource
    except Exception:
        return None

    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if rss <= 0:
        return None
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def log_memory_checkpoint(label: str, *, debug: bool = False) -> None:
    """Log current and lifetime RSS for a named pipeline checkpoint."""
    current = current_rss_mib()
    maximum = max_rss_mib()
    current_text = (
        f"current_rss_mib={current:.1f}"
        if current is not None
        else "current_rss_mib=None"
    )
    max_text = (
        f"max_rss_mib={maximum:.1f}" if maximum is not None else "max_rss_mib=None"
    )
    message = f"Memory checkpoint {label}: {current_text} {max_text}"
    if debug:
        log_debug(message)
    else:
        log_msg(message)
