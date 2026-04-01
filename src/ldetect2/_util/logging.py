"""Logging utilities for ldetect2."""

import logging
import sys

_logger = logging.getLogger("ldetect2")


def configure_logging(verbose: bool = False) -> None:
    """Configure root ldetect2 logger. Call once from CLI entry point."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    _logger.setLevel(level)
    _logger.addHandler(handler)


def log_msg(msg: str) -> None:
    """Log an informational message (drop-in for the original print_log_msg)."""
    _logger.info(msg)
