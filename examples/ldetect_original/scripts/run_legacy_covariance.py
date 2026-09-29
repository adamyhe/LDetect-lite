"""Run the vendored original LDetect covariance script over a whole chromosome.

The original pipeline invokes `ldetect/examples/P00_01_calc_covariance.py` once
per partition, as an independent process fed a VCF region on stdin. That is how
Berisa & Pickrell actually ran it (generated shell scripts under `scripts/`),
and it is the only part of legacy LDetect that parallelizes: the downstream
`P01`/`P02`/`P03` stages are single-threaded per chromosome.

This driver reproduces that structure so the legacy arm of the runtime
benchmark measures legacy end-to-end rather than reusing ldetect-lite
covariance. It records per-partition wall clock, user/system CPU, and peak RSS
via `os.wait4`, so the totals are real CPU-seconds rather than a wall-clock
number that only reflects the chosen worker count.

Two details matter for comparability with ldetect-lite:

- Variants absent from the genetic map are dropped before the stream reaches
  `P00_01`. ldetect-lite does the same (`_util/reference_panel.py` retains only
  variants present in `pos2gpos`); legacy instead does a bare dict lookup and
  would raise `KeyError`. Filtering here keeps the two arms on an identical
  variant set. The chr2 toy interval never exposed this because every position
  in it happens to be in the map.
- Each `P00_01` process reloads the entire chromosome genetic map into a dict.
  That is genuine legacy overhead paid once per partition and is deliberately
  left in.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TIMING_COLUMNS = (
    "population",
    "chromosome",
    "partition_start",
    "partition_end",
    "status",
    "wall_seconds",
    "user_seconds",
    "sys_seconds",
    "cpu_seconds",
    "max_rss_mib",
    "output_bytes",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--partitions", required=True, type=Path)
    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--individuals", required=True, type=Path)
    parser.add_argument("--ne", required=True, type=float)
    parser.add_argument("--cutoff", required=True, type=float)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Legacy dataset root; partitions land in <root>/<chrom>/.",
    )
    parser.add_argument("--timings", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--covariance-script",
        type=Path,
        default=Path(
            "scripts/legacy_ldetect/ldetect/examples/P00_01_calc_covariance.py"
        ),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Leave already-written partition files alone (for resuming).",
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    partitions = read_partitions(args.partitions)
    if not partitions:
        raise SystemExit(f"No partitions found in {args.partitions}")

    chrom_dir = args.dataset_dir / args.chromosome
    chrom_dir.mkdir(parents=True, exist_ok=True)
    (args.dataset_dir / "scripts").mkdir(parents=True, exist_ok=True)
    stage_partitions_file(args.partitions, args.dataset_dir, args.chromosome)

    # Kept outside --dataset-dir so that directory stays a faithful replica of
    # the layout legacy's flat-file loader expects.
    targets = write_map_targets(
        args.genetic_map,
        args.chromosome,
        args.dataset_dir.parent / f"{args.chromosome}.map_positions.tsv",
    )

    args.timings.parent.mkdir(parents=True, exist_ok=True)
    writer_lock = threading.Lock()
    with args.timings.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_COLUMNS, delimiter="\t")
        writer.writeheader()
        handle.flush()

        def work(partition: tuple[int, int]) -> dict[str, object]:
            row = run_partition(args, partition, chrom_dir, targets)
            with writer_lock:
                writer.writerow(row)
                handle.flush()
            return row

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(work, partitions))

    report(rows, args)


def read_partitions(path: Path) -> list[tuple[int, int]]:
    partitions: list[tuple[int, int]] = []
    with path.open() as f:
        for line in f:
            fields = line.split()
            if not fields:
                continue
            partitions.append((int(fields[0]), int(fields[1])))
    return partitions


def stage_partitions_file(source: Path, dataset_dir: Path, chromosome: str) -> None:
    """Copy the partition list to the filename the legacy loader expects."""
    destination = dataset_dir / "scripts" / f"{chromosome}_partitions"
    destination.write_text(source.read_text())


def write_map_targets(genetic_map: Path, chromosome: str, output: Path) -> Path:
    """Write a bcftools targets file of every position in the genetic map.

    ldetect-lite drops variants that have no genetic-map entry; legacy would
    raise `KeyError` on them. Restricting the stream up front keeps both arms
    on the same variant set.
    """
    opener = gzip.open if genetic_map.suffix == ".gz" else open
    with opener(genetic_map, "rt") as src, output.open("w") as dst:
        for line in src:
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                position = int(fields[1])
            except ValueError:
                continue  # header row
            dst.write(f"{chromosome}\t{position}\n")
    return output


def run_partition(
    args: argparse.Namespace,
    partition: tuple[int, int],
    chrom_dir: Path,
    targets: Path,
) -> dict[str, object]:
    start, end = partition
    output = chrom_dir / f"{args.chromosome}.{start}.{end}.gz"

    if args.skip_existing and output.exists() and output.stat().st_size > 0:
        return timing_row(args, partition, "skipped", 0.0, None, output)

    partial = output.parent / (output.name + ".partial")
    error_log = output.parent / (output.name + ".stderr")
    command = (
        "set -o pipefail; "
        f"bcftools view -r {shell_quote(f'{args.chromosome}:{start}-{end}')} "
        f"-T {shell_quote(str(targets))} {shell_quote(str(args.vcf))} "
        f"| {shell_quote(str(args.python))} {shell_quote(str(args.covariance_script))} "
        f"{shell_quote(str(args.genetic_map))} {shell_quote(str(args.individuals))} "
        f"{args.ne} {args.cutoff} {shell_quote(str(partial))} "
        f"> /dev/null 2> {shell_quote(str(error_log))}"
    )

    # posix_spawn rather than subprocess, so os.wait4 can attribute rusage to
    # this exact child without racing the subprocess module's own reaping. The
    # shell reaps both pipeline stages, so its rusage covers bcftools too.
    began = time.monotonic()
    pid = os.posix_spawn("/bin/bash", ["bash", "-c", command], os.environ)
    _, status, usage = os.wait4(pid, 0)
    wall = time.monotonic() - began

    exit_code = os.waitstatus_to_exitcode(status)
    if exit_code != 0:
        partial.unlink(missing_ok=True)
        detail = error_log.read_text(errors="replace") if error_log.exists() else ""
        raise SystemExit(
            f"Legacy covariance failed for {args.chromosome}:{start}-{end} "
            f"(exit code {exit_code}):\n{detail}"
        )

    partial.replace(output)
    error_log.unlink(missing_ok=True)
    return timing_row(args, partition, "ok", wall, usage, output)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def timing_row(
    args: argparse.Namespace,
    partition: tuple[int, int],
    status: str,
    wall: float,
    usage: object | None,
    output: Path,
) -> dict[str, object]:
    start, end = partition
    user = getattr(usage, "ru_utime", 0.0)
    system = getattr(usage, "ru_stime", 0.0)
    return {
        "population": args.population,
        "chromosome": args.chromosome,
        "partition_start": start,
        "partition_end": end,
        "status": status,
        "wall_seconds": f"{wall:.6f}",
        "user_seconds": f"{user:.6f}",
        "sys_seconds": f"{system:.6f}",
        "cpu_seconds": f"{user + system:.6f}",
        "max_rss_mib": f"{max_rss_mib(usage):.3f}",
        "output_bytes": output.stat().st_size if output.exists() else 0,
    }


def max_rss_mib(usage: object | None) -> float:
    """Convert `ru_maxrss` to MiB, accounting for the platform's unit."""
    if usage is None:
        return 0.0
    value = float(getattr(usage, "ru_maxrss", 0.0))
    # Linux reports kilobytes; macOS reports bytes.
    return value / 1024.0 if sys.platform.startswith("linux") else value / (1024.0**2)


def report(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    cpu = sum(float(row["cpu_seconds"]) for row in rows)
    wall = sum(float(row["wall_seconds"]) for row in rows)
    peak = max((float(row["max_rss_mib"]) for row in rows), default=0.0)
    skipped = sum(1 for row in rows if row["status"] == "skipped")
    print(
        f"legacy covariance {args.population} chr{args.chromosome}: "
        f"{len(rows)} partitions ({skipped} skipped), "
        f"cpu={cpu / 3600:.3f} h, summed_wall={wall / 3600:.3f} h, "
        f"peak_partition_rss={peak:.1f} MiB, workers={args.workers}"
    )
    print(f"timings: {args.timings}")


if __name__ == "__main__":
    main()
