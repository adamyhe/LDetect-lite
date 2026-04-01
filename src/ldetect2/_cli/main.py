"""ldetect2 unified CLI entry point."""

from __future__ import annotations

import argparse
import sys

from ldetect2 import __version__
from ldetect2._util.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ldetect2",
        description="Compute approximately independent LD blocks in the human genome.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # Register all subcommands
    from ldetect2._cli import (
        cmd_covariance,
        cmd_extract_bpoints,
        cmd_find_minima,
        cmd_interpolate_maps,
        cmd_matrix_to_vector,
        cmd_partition,
        cmd_run,
    )

    cmd_partition.register(subparsers)
    cmd_covariance.register(subparsers)
    cmd_matrix_to_vector.register(subparsers)
    cmd_find_minima.register(subparsers)
    cmd_extract_bpoints.register(subparsers)
    cmd_interpolate_maps.register(subparsers)
    cmd_run.register(subparsers)

    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
