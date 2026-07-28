"""ldetect-lite unified CLI entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import cast

from ldetect_lite import __version__
from ldetect_lite._util.logging import configure_logging

_NATIVE_THREAD_CAP_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_NUMBA_THREAD_CAP_ENV_VAR = "NUMBA_NUM_THREADS"
_THREAD_MANAGED_COMMANDS = {
    "calc-covariance",
    "find-minima",
    "matrix-to-vector",
    "run",
}
_COMMAND_WORKER_DEFAULTS = {
    "calc-covariance": 1,
    "find-minima": 1,
    "matrix-to-vector": 1,
    "run": 1,
}


def _option_int(argv: list[str], option: str, default: int) -> int:
    prefix = f"{option}="
    for i, arg in enumerate(argv):
        if arg.startswith(prefix):
            try:
                return int(arg[len(prefix) :])
            except ValueError:
                return default
        if arg == option and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return default
    return default


def _selected_command(argv: list[str]) -> str | None:
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-v", "--verbosity"}:
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        return arg
    return None


def _configure_native_thread_caps(argv: list[str]) -> None:
    """Set native thread-pool caps before importing numpy/scipy/numba users.

    The CLI's public parallelism knobs are process/thread counts such as
    ``--workers`` and ``--filter-workers``. Native libraries otherwise inherit
    ambient machine-wide defaults, which can multiply those explicit worker
    counts inside each process-pool child.
    """
    command = _selected_command(argv)
    if command not in _THREAD_MANAGED_COMMANDS:
        return

    for name in _NATIVE_THREAD_CAP_ENV_VARS:
        os.environ[name] = "1"

    worker_cap = _option_int(
        argv,
        "--workers",
        _COMMAND_WORKER_DEFAULTS.get(command, 1),
    )
    filter_workers = _option_int(argv, "--filter-workers", 1)
    os.environ[_NUMBA_THREAD_CAP_ENV_VAR] = str(max(1, worker_cap, filter_workers))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    _configure_native_thread_caps(argv)

    parser = argparse.ArgumentParser(
        prog="ldetect",
        description="Compute approximately independent LD blocks in the human genome.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        choices=["debug", "info", "warning", "error"],
        default="info",
        metavar="LEVEL",
        help="Logging verbosity: debug, info (default), warning, error.",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # Register all subcommands
    from ldetect_lite._cli import (
        cmd_covariance,
        cmd_covariance_summary,
        cmd_extract_bpoints,
        cmd_find_minima,
        cmd_interpolate_maps,
        cmd_matrix_to_vector,
        cmd_partition,
        cmd_run,
    )

    cmd_partition.register(subparsers)
    cmd_covariance.register(subparsers)
    cmd_covariance_summary.register(subparsers)
    cmd_matrix_to_vector.register(subparsers)
    cmd_find_minima.register(subparsers)
    cmd_extract_bpoints.register(subparsers)
    cmd_interpolate_maps.register(subparsers)
    cmd_run.register(subparsers)

    args = parser.parse_args(argv)

    configure_logging(level=getattr(logging, args.verbosity.upper()))

    return cast(int, args.func(args))


if __name__ == "__main__":
    sys.exit(main())
