"""Parse ldetect2 diagnostics logs into profiling TSVs and optional plots."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

RUN_COLS = [
    "profile_name",
    "population",
    "chrom",
    "elapsed_seconds",
    "user_seconds",
    "system_seconds",
    "max_rss_kb",
    "major_page_faults",
    "minor_page_faults",
    "swaps",
    "filesystem_inputs",
    "filesystem_outputs",
    "exit_status",
]

BREAKPOINT_COLS = [
    "profile_name",
    "population",
    "chrom",
    "subset",
    "idx",
    "start",
    "stop",
    "partitions",
    "rows",
    "precompute_seconds",
    "search_seconds",
    "total_seconds",
    "max_rss_mib",
    "partition_load_seconds",
    "canonicalize_seconds",
    "append_seconds",
    "diagonal_seconds",
    "slice_seconds",
    "normalize_seconds",
    "vertical_seconds",
    "horizontal_seconds",
    "hdf5_read_seconds",
    "chunk_filter_seconds",
    "dedup_seconds",
    "dedup_merge_seconds",
    "accumulator_seconds",
    "candidate_rows",
    "eligible_rows",
    "normalized_rows",
    "rows_read",
    "rows_after_filter",
    "rows_after_dedup",
    "duplicate_rows_skipped",
    "chunks",
    "segments",
    "active_rows_peak",
    "peak_chunk_rows",
    "hdf5_reader_open_count",
    "hdf5_reader_reuse_count",
]

GROUP_COLS = [
    "profile_name",
    "population",
    "chrom",
    "subset",
    "group_index",
    "breakpoints",
    "partitions",
    "rows",
    "load_seconds",
    "canonicalize_seconds",
    "total_seconds",
]

BY_CHROM_COLS = [
    "profile_name",
    "population",
    "chrom",
    "subset",
    "breakpoints",
    "set_elapsed_seconds",
    "breakpoint_count",
    "precompute_seconds",
    "search_seconds",
    "total_seconds",
    "max_rss_mib",
    "rows",
    "partitions",
    "partition_load_seconds",
    "canonicalize_seconds",
    "append_seconds",
    "diagonal_seconds",
    "slice_seconds",
    "normalize_seconds",
    "vertical_seconds",
    "horizontal_seconds",
    "hdf5_read_seconds",
    "chunk_filter_seconds",
    "dedup_seconds",
    "dedup_merge_seconds",
    "accumulator_seconds",
    "candidate_rows",
    "eligible_rows",
    "normalized_rows",
    "rows_read",
    "rows_after_filter",
    "rows_after_dedup",
    "duplicate_rows_skipped",
    "chunks",
    "segments",
    "active_rows_peak",
    "peak_chunk_rows",
    "hdf5_reader_open_count",
    "hdf5_reader_reuse_count",
    "group_count",
    "group_breakpoints",
    "group_rows",
    "group_partitions",
    "group_load_seconds",
    "group_canonicalize_seconds",
    "group_total_seconds",
    "local_search_unaccounted_seconds",
]

_BREAKPOINT_EXTRA_FLOAT_COLS = [
    "partition_load_seconds",
    "canonicalize_seconds",
    "append_seconds",
    "diagonal_seconds",
    "slice_seconds",
    "normalize_seconds",
    "vertical_seconds",
    "horizontal_seconds",
    "hdf5_read_seconds",
    "chunk_filter_seconds",
    "dedup_seconds",
    "dedup_merge_seconds",
    "accumulator_seconds",
]

_BREAKPOINT_EXTRA_INT_COLS = [
    "candidate_rows",
    "eligible_rows",
    "normalized_rows",
    "rows_read",
    "rows_after_filter",
    "rows_after_dedup",
    "duplicate_rows_skipped",
    "chunks",
    "segments",
    "hdf5_reader_open_count",
    "hdf5_reader_reuse_count",
]

_MAC_TIME_RE = re.compile(r"^\s*(?P<value>[0-9.]+)\s+(?P<key>.+?)\s*$")
_SET_RE = re.compile(
    r"Local search (?P<subset>\S+) done: breakpoints=(?P<breakpoints>\d+) "
    r"elapsed_seconds=(?P<elapsed>[0-9.]+)"
)
_GROUP_RE = re.compile(r"Local search (?P<subset>\S+) group loaded: ")
_BP_PREFIX_RE = re.compile(r"(?P<subset>\S+) breakpoint idx=")
_KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>None|-?[0-9.]+)")


def parse_elapsed(value: str) -> float | None:
    """Parse GNU time elapsed values like h:mm:ss, m:ss, or seconds."""
    value = value.strip()
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(value)
    except ValueError:
        return None


def parse_time_log(path: Path) -> dict[str, str]:
    """Parse GNU ``time -v`` or macOS ``time -l`` output."""
    row = {
        col: ""
        for col in RUN_COLS
        if col not in {"profile_name", "population", "chrom"}
    }
    if not path.exists():
        return row

    for line in path.read_text(errors="replace").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            key = key.strip().lower()
            value = value.strip()
            _apply_gnu_time_value(row, key, value)
            continue

        match = _MAC_TIME_RE.match(line)
        if match:
            key = match.group("key").strip().lower()
            value = match.group("value").strip()
            _apply_mac_time_value(row, key, value)

    return row


def _apply_gnu_time_value(row: dict[str, str], key: str, value: str) -> None:
    if key.startswith("user time"):
        row["user_seconds"] = value
    elif key.startswith("system time"):
        row["system_seconds"] = value
    elif key.startswith("elapsed"):
        parsed = parse_elapsed(value)
        row["elapsed_seconds"] = "" if parsed is None else f"{parsed:.6f}"
    elif key.startswith("maximum resident set size"):
        row["max_rss_kb"] = value
    elif key.startswith("major"):
        row["major_page_faults"] = value
    elif key.startswith("minor"):
        row["minor_page_faults"] = value
    elif key == "swaps":
        row["swaps"] = value
    elif key.startswith("file system inputs"):
        row["filesystem_inputs"] = value
    elif key.startswith("file system outputs"):
        row["filesystem_outputs"] = value
    elif key.startswith("exit status"):
        row["exit_status"] = value


def _apply_mac_time_value(row: dict[str, str], key: str, value: str) -> None:
    if key == "real":
        parsed = parse_elapsed(value)
        row["elapsed_seconds"] = "" if parsed is None else f"{parsed:.6f}"
    elif key == "user":
        row["user_seconds"] = value
    elif key == "sys":
        row["system_seconds"] = value
    elif key == "maximum resident set size":
        try:
            row["max_rss_kb"] = str(int(float(value) / 1024.0))
        except ValueError:
            row["max_rss_kb"] = ""
    elif key == "page reclaims":
        row["minor_page_faults"] = value
    elif key == "page faults":
        row["major_page_faults"] = value
    elif key == "swaps":
        row["swaps"] = value
    elif key == "file system inputs":
        row["filesystem_inputs"] = value
    elif key == "file system outputs":
        row["filesystem_outputs"] = value


def parse_ldetect2_log(
    path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Parse local-search set and breakpoint diagnostics from ldetect2 logs."""
    sets: list[dict[str, str]] = []
    groups: list[dict[str, str]] = []
    breakpoints: list[dict[str, str]] = []
    if not path.exists():
        return sets, groups, breakpoints

    for line in path.read_text(errors="replace").splitlines():
        set_match = _SET_RE.search(line)
        if set_match:
            sets.append(
                {
                    "subset": set_match.group("subset"),
                    "breakpoints": set_match.group("breakpoints"),
                    "set_elapsed_seconds": set_match.group("elapsed"),
                }
            )

        group_row = _parse_group_line(line, len(groups))
        if group_row is not None:
            groups.append(group_row)

        breakpoint_row = _parse_breakpoint_line(line)
        if breakpoint_row is not None:
            breakpoints.append(breakpoint_row)

    return sets, groups, breakpoints


def _parse_group_line(line: str, group_index: int) -> dict[str, str] | None:
    """Parse one local-search partition-group timing line into TSV columns."""
    group_match = _GROUP_RE.search(line)
    if not group_match:
        return None
    values = {
        key: _none_to_empty(value)
        for key, value in _KV_RE.findall(line)
        if key in GROUP_COLS
    }
    required = {
        "breakpoints",
        "partitions",
        "rows",
        "load_seconds",
        "canonicalize_seconds",
    }
    if not required <= set(values):
        return None

    metadata_cols = {"profile_name", "population", "chrom"}
    row = {col: "" for col in GROUP_COLS if col not in metadata_cols}
    row.update(values)
    row["subset"] = group_match.group("subset")
    row["group_index"] = str(group_index)
    total_seconds = _float(row["load_seconds"]) + _float(
        row["canonicalize_seconds"]
    )
    row["total_seconds"] = f"{total_seconds:.6f}"
    return row


def _parse_breakpoint_line(line: str) -> dict[str, str] | None:
    """Parse one local-search breakpoint debug line into TSV columns."""
    prefix_match = _BP_PREFIX_RE.search(line)
    if not prefix_match:
        return None
    values = {
        key: _none_to_empty(value)
        for key, value in _KV_RE.findall(line)
        if key in BREAKPOINT_COLS
    }
    required = {
        "idx",
        "start",
        "stop",
        "partitions",
        "rows",
        "precompute_seconds",
        "search_seconds",
        "total_seconds",
    }
    if not required <= set(values):
        return None

    metadata_cols = {"profile_name", "population", "chrom"}
    row = {col: "" for col in BREAKPOINT_COLS if col not in metadata_cols}
    row.update(values)
    row["subset"] = prefix_match.group("subset")
    return row


def _none_to_empty(value: str | None) -> str:
    return "" if value in {None, "None"} else value


def _chrom_log_path(log_dir: Path, chrom: str, suffix: str) -> Path:
    """Return the chromosome-prefixed log path, falling back to legacy names."""
    prefixed = log_dir / f"{chrom}.{suffix}"
    if prefixed.exists():
        return prefixed
    return log_dir / suffix


def profile_chromosome(
    diagnostic_root: Path,
    population: str,
    chrom: str,
    profile_name: str,
) -> tuple[
    dict[str, str],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    """Parse timing and ldetect2 logs for one diagnostic chromosome."""
    log_dir = diagnostic_root / str(chrom) / "logs"
    run_row = parse_time_log(_chrom_log_path(log_dir, str(chrom), "timing.log"))
    run_row.update(
        {"profile_name": profile_name, "population": population, "chrom": f"chr{chrom}"}
    )

    set_rows, group_rows, breakpoint_rows = parse_ldetect2_log(
        _chrom_log_path(log_dir, str(chrom), "ldetect2.log")
    )
    for row in set_rows + group_rows + breakpoint_rows:
        row.update(
            {
                "profile_name": profile_name,
                "population": population,
                "chrom": f"chr{chrom}",
            }
        )
    return run_row, set_rows, group_rows, breakpoint_rows


def aggregate_by_chrom(
    set_rows: list[dict[str, str]],
    group_rows: list[dict[str, str]],
    breakpoint_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Aggregate per-breakpoint local-search profiling rows by chrom/subset."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    grouped_groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = (
        defaultdict(list)
    )
    set_elapsed: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in breakpoint_rows:
        key = (row["profile_name"], row["population"], row["chrom"], row["subset"])
        grouped[key].append(row)
    for row in group_rows:
        key = (row["profile_name"], row["population"], row["chrom"], row["subset"])
        grouped_groups[key].append(row)
    for row in set_rows:
        key = (row["profile_name"], row["population"], row["chrom"], row["subset"])
        set_elapsed[key] = row

    keys = sorted(set(grouped) | set(grouped_groups) | set(set_elapsed))
    out: list[dict[str, str]] = []
    for key in keys:
        profile_name, population, chrom, subset = key
        rows = grouped.get(key, [])
        groups = grouped_groups.get(key, [])
        set_row = set_elapsed.get(key, {})
        set_seconds = set_row.get("set_elapsed_seconds", "")
        breakpoint_total = _sum_field(rows, "total_seconds")
        group_total = _sum_field(groups, "total_seconds")
        out_row = {
            "profile_name": profile_name,
            "population": population,
            "chrom": chrom,
            "subset": subset,
            "breakpoints": set_row.get("breakpoints", ""),
            "set_elapsed_seconds": set_row.get("set_elapsed_seconds", ""),
            "breakpoint_count": str(len(rows)),
            "precompute_seconds": _sum_field(rows, "precompute_seconds"),
            "search_seconds": _sum_field(rows, "search_seconds"),
            "total_seconds": breakpoint_total,
            "max_rss_mib": _max_field(rows, "max_rss_mib"),
            "rows": _sum_int_field(rows, "rows"),
            "partitions": _sum_int_field(rows, "partitions"),
        }
        for field in _BREAKPOINT_EXTRA_FLOAT_COLS:
            out_row[field] = _sum_field(rows, field)
        for field in _BREAKPOINT_EXTRA_INT_COLS:
            out_row[field] = _sum_int_field(rows, field)
        out_row["active_rows_peak"] = _max_int_field(rows, "active_rows_peak")
        out_row["peak_chunk_rows"] = _max_int_field(rows, "peak_chunk_rows")
        out_row["group_count"] = str(len(groups)) if groups else ""
        out_row["group_breakpoints"] = _sum_int_field(groups, "breakpoints")
        out_row["group_rows"] = _sum_int_field(groups, "rows")
        out_row["group_partitions"] = _sum_int_field(groups, "partitions")
        out_row["group_load_seconds"] = _sum_field(groups, "load_seconds")
        out_row["group_canonicalize_seconds"] = _sum_field(
            groups, "canonicalize_seconds"
        )
        out_row["group_total_seconds"] = group_total
        accounted = _optional_sum(breakpoint_total, group_total)
        if set_seconds and accounted:
            out_row["local_search_unaccounted_seconds"] = (
                f"{_float(set_seconds) - accounted:.6f}"
            )
        else:
            out_row["local_search_unaccounted_seconds"] = ""
        out.append(out_row)
    return out


def _sum_field(rows: list[dict[str, str]], field: str) -> str:
    values = [float(row[field]) for row in rows if row.get(field)]
    return f"{sum(values):.6f}" if values else ""


def _sum_int_field(rows: list[dict[str, str]], field: str) -> str:
    values = [int(row[field]) for row in rows if row.get(field)]
    return str(sum(values)) if values else ""


def _max_field(rows: list[dict[str, str]], field: str) -> str:
    values = [float(row[field]) for row in rows if row.get(field)]
    return f"{max(values):.6f}" if values else ""


def _max_int_field(rows: list[dict[str, str]], field: str) -> str:
    values = [int(row[field]) for row in rows if row.get(field)]
    return str(max(values)) if values else ""


def _float(value: str) -> float:
    return float(value) if value else 0.0


def _optional_sum(*values: str) -> float | None:
    present = [value for value in values if value]
    if not present:
        return None
    return sum(float(value) for value in present)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write rows to a tab-delimited file with a stable header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_plots(
    plot_dir: Path,
    run_rows: list[dict[str, str]],
    breakpoint_rows: list[dict[str, str]],
    by_chrom_rows: list[dict[str, str]],
    enabled: bool,
) -> None:
    """Write profiling plots, or a skip marker when plotting is unavailable."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not enabled:
        (plot_dir / "SKIPPED.txt").write_text("Plots disabled by configuration.\n")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional matplotlib
        (plot_dir / "SKIPPED.txt").write_text(
            f"Plots skipped because matplotlib is unavailable: {exc}\n"
        )
        return

    _plot_bar(
        plt,
        plot_dir / "max_rss_by_chrom.png",
        run_rows,
        "chrom",
        "max_rss_kb",
        "Max RSS by chromosome",
        "Max RSS (KB)",
    )
    _plot_bar(
        plt,
        plot_dir / "wall_time_by_chrom.png",
        run_rows,
        "chrom",
        "elapsed_seconds",
        "Wall time by chromosome",
        "Elapsed seconds",
    )
    _plot_stacked_local_search(
        plt, plot_dir / "local_search_time_by_chrom.png", by_chrom_rows
    )
    _plot_stacked_precompute_phases(
        plt, plot_dir / "local_search_precompute_phases_by_chrom.png", by_chrom_rows
    )
    _plot_scatter(
        plt,
        plot_dir / "rows_vs_precompute_seconds.png",
        breakpoint_rows,
        "rows",
        "precompute_seconds",
        "Rows loaded vs precompute seconds",
        "Rows loaded",
        "Precompute seconds",
    )
    _plot_scatter(
        plt,
        plot_dir / "partitions_vs_precompute_seconds.png",
        breakpoint_rows,
        "partitions",
        "precompute_seconds",
        "Partitions loaded vs precompute seconds",
        "Partitions loaded",
        "Precompute seconds",
    )


def _plot_bar(plt, path: Path, rows, x_field, y_field, title, ylabel) -> None:
    plot_rows = [row for row in rows if row.get(y_field)]
    labels = [row[x_field] for row in plot_rows]
    values = [float(row[y_field]) for row in plot_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Chromosome")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_stacked_local_search(plt, path: Path, rows) -> None:
    plot_rows = [row for row in rows if row.get("precompute_seconds")]
    labels = [f"{row['chrom']} {row['subset']}" for row in plot_rows]
    precompute = [float(row["precompute_seconds"]) for row in plot_rows]
    group = [float(row["group_total_seconds"] or 0.0) for row in plot_rows]
    search = [float(row["search_seconds"] or 0.0) for row in plot_rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, precompute, label="precompute")
    ax.bar(labels, group, bottom=precompute, label="group load/canonicalize")
    group_bottom = [pre + grp for pre, grp in zip(precompute, group)]
    ax.bar(labels, search, bottom=group_bottom, label="search")
    ax.set_title("Local-search time by chromosome")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_stacked_precompute_phases(plt, path: Path, rows) -> None:
    phase_fields = [
        "partition_load_seconds",
        "canonicalize_seconds",
        "append_seconds",
        "diagonal_seconds",
        "slice_seconds",
        "normalize_seconds",
        "vertical_seconds",
        "horizontal_seconds",
        "hdf5_read_seconds",
        "chunk_filter_seconds",
        "dedup_seconds",
        "accumulator_seconds",
    ]
    plot_rows = [row for row in rows if any(row.get(field) for field in phase_fields)]
    if not plot_rows:
        return

    labels = [f"{row['chrom']} {row['subset']}" for row in plot_rows]
    bottoms = [0.0 for _ in plot_rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for field in phase_fields:
        values = [float(row.get(field) or 0.0) for row in plot_rows]
        ax.bar(labels, values, bottom=bottoms, label=field.removesuffix("_seconds"))
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_title("Local-search precompute phases by chromosome")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_scatter(
    plt, path: Path, rows, x_field, y_field, title, xlabel, ylabel
) -> None:
    plot_rows = [row for row in rows if row.get(x_field) and row.get(y_field)]
    x_values = [float(row[x_field]) for row in plot_rows]
    y_values = [float(row[y_field]) for row in plot_rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x_values, y_values, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-root", required=True, type=Path)
    parser.add_argument("--population", required=True)
    parser.add_argument("--profile-name", default="default")
    parser.add_argument("--chromosomes", nargs="+", required=True)
    parser.add_argument("--run-summary", required=True, type=Path)
    parser.add_argument("--local-search-groups", required=True, type=Path)
    parser.add_argument("--local-search-breakpoints", required=True, type=Path)
    parser.add_argument("--local-search-by-chrom", required=True, type=Path)
    parser.add_argument("--plot-dir", required=True, type=Path)
    parser.add_argument("--plots", dest="plots", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    args = parser.parse_args()

    run_rows: list[dict[str, str]] = []
    set_rows: list[dict[str, str]] = []
    group_rows: list[dict[str, str]] = []
    breakpoint_rows: list[dict[str, str]] = []
    for chrom in args.chromosomes:
        (
            run_row,
            chrom_set_rows,
            chrom_group_rows,
            chrom_breakpoint_rows,
        ) = profile_chromosome(
            args.diagnostic_root,
            args.population,
            chrom,
            args.profile_name,
        )
        run_rows.append(run_row)
        set_rows.extend(chrom_set_rows)
        group_rows.extend(chrom_group_rows)
        breakpoint_rows.extend(chrom_breakpoint_rows)

    by_chrom_rows = aggregate_by_chrom(set_rows, group_rows, breakpoint_rows)
    write_tsv(args.run_summary, RUN_COLS, run_rows)
    write_tsv(args.local_search_groups, GROUP_COLS, group_rows)
    write_tsv(args.local_search_breakpoints, BREAKPOINT_COLS, breakpoint_rows)
    write_tsv(args.local_search_by_chrom, BY_CHROM_COLS, by_chrom_rows)
    write_plots(args.plot_dir, run_rows, breakpoint_rows, by_chrom_rows, args.plots)

    if args.plots and not any(args.plot_dir.glob("*.png")):
        print("Profiling TSVs written; plots skipped.", file=sys.stderr)


if __name__ == "__main__":
    main()
