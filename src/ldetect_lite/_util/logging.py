"""Logging utilities for ldetect_lite."""

import logging
import sys

_logger = logging.getLogger("ldetect-lite")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root ldetect-lite logger. Call once from CLI entry point."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    )
    _logger.setLevel(level)
    _logger.addHandler(handler)


def log_msg(msg: str) -> None:
    """Log an informational message (drop-in for the original print_log_msg)."""
    _logger.info(msg)


def log_debug(msg: str) -> None:
    """Log a debug-level message."""
    _logger.debug(msg)
