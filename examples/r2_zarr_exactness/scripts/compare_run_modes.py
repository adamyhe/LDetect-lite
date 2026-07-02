"""Compare ldetect2 backend outputs and Snakemake benchmark files."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path


def read_vector(path: Path) -> dict[int, float]:
    opener = gzip.open(path, "rt") if path.suffix in (".gz", ".gzip") else open(path)
    out: dict[int, float] = {}
    with opener as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                out[int(row[0])] = float(row[1])
    return out


def read_bed(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                rows.append(line)
    return rows


def diff_stats(a: dict[int, float], b: dict[int, float]) -> dict[str, str]:
    all_keys = sorted(set(a) | set(b))
    shared = [key for key in all_keys if key in a and key in b]
    abs_diffs = [abs(a[key] - b[key]) for key in shared]
    rel_diffs = [
        abs(a[key] - b[key]) / max(abs(a[key]), 1e-30)
        for key in shared
    ]
    max_abs = max(abs_diffs) if abs_diffs else 0.0
    mean_abs = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
    max_rel = max(rel_diffs) if rel_diffs else 0.0
    return {
        "n_baseline": str(len(a)),
        "n_mode": str(len(b)),
        "n_shared": str(len(shared)),
        "only_baseline": str(sum(1 for key in all_keys if key not in b)),
        "only_mode": str(sum(1 for key in all_keys if key not in a)),
        "exact": str(set(a) == set(b) and max_abs == 0.0),
        "max_abs_diff": f"{max_abs:.17g}",
        "mean_abs_diff": f"{mean_abs:.17g}",
        "max_rel_diff": f"{max_rel:.17g}",
    }


def loci_stats(a: list[int], b: list[int]) -> dict[str, str]:
    n = min(len(a), len(b))
    diffs = [abs(a[idx] - b[idx]) for idx in range(n)]
    max_abs = max(diffs) if diffs else 0
    mean_abs = sum(diffs) / len(diffs) if diffs else 0.0
    return {
        "n_baseline": str(len(a)),
        "n_mode": str(len(b)),
        "n_shared": str(n),
        "only_baseline": str(max(0, len(a) - len(b))),
        "only_mode": str(max(0, len(b) - len(a))),
        "exact": str(a == b),
        "max_abs_diff": str(max_abs),
        "mean_abs_diff": f"{mean_abs:.17g}",
        "max_rel_diff": "0",
    }


def scalar_stats(a: float, b: float) -> dict[str, str]:
    abs_diff = abs(a - b)
    rel_diff = abs_diff / max(abs(a), 1e-30)
    return {
        "n_baseline": "1",
        "n_mode": "1",
        "n_shared": "1",
        "only_baseline": "0",
        "only_mode": "0",
        "exact": str(abs_diff == 0.0),
        "max_abs_diff": f"{abs_diff:.17g}",
        "mean_abs_diff": f"{abs_diff:.17g}",
        "max_rel_diff": f"{rel_diff:.17g}",
    }


def parse_benchmark(path: Path) -> dict[str, str]:
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows[0] if rows else {}


def read_partitions(path: Path | None) -> list[tuple[int, int]]:
    if path is None or not path.exists():
        return []
    partitions: list[tuple[int, int]] = []
    with path.open() as f:
        for raw in f:
            fields = raw.split()
            if len(fields) >= 2:
                partitions.append((int(fields[0]), int(fields[1])))
    return partitions


def nearest_partition_boundary(
    position: int,
    partitions: list[tuple[int, int]],
) -> tuple[str, str, str, str, str]:
    if not partitions:
        return "", "", "", "", ""

    best_boundary = 0
    best_distance: int | None = None
    best_role = ""
    best_partition = partitions[0]
    for start, end in partitions:
        for boundary, role in ((start, "start"), (end, "end")):
            distance = abs(position - boundary)
            if best_distance is None or distance < best_distance:
                best_boundary = boundary
                best_distance = distance
                best_role = role
                best_partition = (start, end)
    return (
        str(best_partition[0]),
        str(best_partition[1]),
        str(best_boundary),
        str(best_distance if best_distance is not None else ""),
        best_role,
    )


def vector_diff_rows(
    *,
    comparison: str,
    baseline: dict[int, float],
    mode: dict[int, float],
    partitions: list[tuple[int, int]],
    limit: int,
) -> list[dict[str, str]]:
    rows: list[tuple[float, int, dict[str, str]]] = []
    for position in set(baseline) | set(mode):
        baseline_present = position in baseline
        mode_present = position in mode
        baseline_value = baseline.get(position)
        mode_value = mode.get(position)
        if baseline_present and mode_present:
            abs_diff = abs(float(baseline_value) - float(mode_value))
            sort_abs_diff = abs_diff
            rel_diff = abs_diff / max(abs(float(baseline_value)), 1e-30)
            baseline_text = f"{float(baseline_value):.17g}"
            mode_text = f"{float(mode_value):.17g}"
            abs_text = f"{abs_diff:.17g}"
            rel_text = f"{rel_diff:.17g}"
        else:
            sort_abs_diff = math.inf
            baseline_text = "" if baseline_value is None else f"{baseline_value:.17g}"
            mode_text = "" if mode_value is None else f"{mode_value:.17g}"
            abs_text = ""
            rel_text = ""

        (
            partition_start,
            partition_end,
            nearest_boundary,
            distance_to_boundary,
            boundary_role,
        ) = nearest_partition_boundary(position, partitions)
        rows.append(
            (
                sort_abs_diff,
                position,
                {
                    "comparison": comparison,
                    "rank": "",
                    "position": str(position),
                    "baseline_present": str(baseline_present),
                    "mode_present": str(mode_present),
                    "baseline_value": baseline_text,
                    "mode_value": mode_text,
                    "abs_diff": abs_text,
                    "rel_diff": rel_text,
                    "nearest_partition_start": partition_start,
                    "nearest_partition_end": partition_end,
                    "nearest_boundary": nearest_boundary,
                    "distance_to_boundary": distance_to_boundary,
                    "boundary_role": boundary_role,
                },
            )
        )

    rows.sort(key=lambda item: (-item[0], item[1]))
    out = [row for _, _, row in rows[:limit]]
    for rank, row in enumerate(out, start=1):
        row["rank"] = str(rank)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--vectors", nargs="+", required=True, type=Path)
    parser.add_argument("--breakpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--beds", nargs="+", required=True, type=Path)
    parser.add_argument("--benchmarks", nargs="+", required=True, type=Path)
    parser.add_argument("--exactness-output", required=True, type=Path)
    parser.add_argument("--runtime-output", required=True, type=Path)
    parser.add_argument("--vector-diff-output", required=True, type=Path)
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--top-vector-diffs", type=int, default=1000)
    args = parser.parse_args()

    n_modes = len(args.modes)
    for name, values in (
        ("vectors", args.vectors),
        ("breakpoints", args.breakpoints),
        ("beds", args.beds),
        ("benchmarks", args.benchmarks),
    ):
        if len(values) != n_modes:
            raise SystemExit(f"{name} count does not match --modes")
    if args.baseline not in args.modes:
        raise SystemExit("--baseline must be present in --modes")

    vector_by_mode = {
        mode: read_vector(path) for mode, path in zip(args.modes, args.vectors)
    }
    breakpoint_by_mode = {
        mode: json.loads(path.read_text())
        for mode, path in zip(args.modes, args.breakpoints)
    }
    bed_by_mode = {mode: read_bed(path) for mode, path in zip(args.modes, args.beds)}
    partitions = read_partitions(args.partitions)

    baseline = args.baseline
    exact_rows: list[dict[str, str]] = []
    fields = [
        "comparison",
        "artifact",
        "subset",
        "n_baseline",
        "n_mode",
        "n_shared",
        "only_baseline",
        "only_mode",
        "exact",
        "max_abs_diff",
        "mean_abs_diff",
        "max_rel_diff",
    ]

    for mode in args.modes:
        if mode == baseline:
            continue
        comparison = f"{mode}_vs_{baseline}"
        row = {
            "comparison": comparison,
            "artifact": "vector",
            "subset": "",
            **diff_stats(vector_by_mode[baseline], vector_by_mode[mode]),
        }
        exact_rows.append(row)

        base_data = breakpoint_by_mode[baseline]
        mode_data = breakpoint_by_mode[mode]
        subsets = sorted(
            key
            for key in set(base_data) | set(mode_data)
            if isinstance(base_data.get(key, mode_data.get(key)), dict)
            and "loci" in base_data.get(key, mode_data.get(key, {}))
        )
        for subset in subsets:
            exact_rows.append(
                {
                    "comparison": comparison,
                    "artifact": "breakpoint_loci",
                    "subset": subset,
                    **loci_stats(
                        list(base_data.get(subset, {}).get("loci", [])),
                        list(mode_data.get(subset, {}).get("loci", [])),
                    ),
                }
            )
            for metric_name in ("sum", "N_zero"):
                base_metric = base_data.get(subset, {}).get("metric", {})
                mode_metric = mode_data.get(subset, {}).get("metric", {})
                if metric_name in base_metric and metric_name in mode_metric:
                    exact_rows.append(
                        {
                            "comparison": comparison,
                            "artifact": f"metric_{metric_name}",
                            "subset": subset,
                            **scalar_stats(
                                float(base_metric[metric_name]),
                                float(mode_metric[metric_name]),
                            ),
                        }
                    )

        base_bed = bed_by_mode[baseline]
        mode_bed = bed_by_mode[mode]
        n = min(len(base_bed), len(mode_bed))
        mismatch = sum(1 for idx in range(n) if base_bed[idx] != mode_bed[idx])
        mismatch += abs(len(base_bed) - len(mode_bed))
        exact_rows.append(
            {
                "comparison": comparison,
                "artifact": "bed",
                "subset": args.subset,
                "n_baseline": str(len(base_bed)),
                "n_mode": str(len(mode_bed)),
                "n_shared": str(n),
                "only_baseline": str(max(0, len(base_bed) - len(mode_bed))),
                "only_mode": str(max(0, len(mode_bed) - len(base_bed))),
                "exact": str(mismatch == 0),
                "max_abs_diff": str(mismatch),
                "mean_abs_diff": str(mismatch / max(len(base_bed), 1)),
                "max_rel_diff": "0",
            }
        )

    vector_diff_fields = [
        "comparison",
        "rank",
        "position",
        "baseline_present",
        "mode_present",
        "baseline_value",
        "mode_value",
        "abs_diff",
        "rel_diff",
        "nearest_partition_start",
        "nearest_partition_end",
        "nearest_boundary",
        "distance_to_boundary",
        "boundary_role",
    ]
    vector_diff_rows_out: list[dict[str, str]] = []
    for mode in args.modes:
        if mode == baseline:
            continue
        vector_diff_rows_out.extend(
            vector_diff_rows(
                comparison=f"{mode}_vs_{baseline}",
                baseline=vector_by_mode[baseline],
                mode=vector_by_mode[mode],
                partitions=partitions,
                limit=max(0, int(args.top_vector_diffs)),
            )
        )

    args.vector_diff_output.parent.mkdir(parents=True, exist_ok=True)
    with args.vector_diff_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=vector_diff_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(vector_diff_rows_out)

    args.exactness_output.parent.mkdir(parents=True, exist_ok=True)
    with args.exactness_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(exact_rows)

    runtime_rows = []
    benchmark_fields: list[str] = []
    for mode, path in zip(args.modes, args.benchmarks):
        row = parse_benchmark(path)
        if not benchmark_fields:
            benchmark_fields = list(row)
        runtime_rows.append({"mode": mode, **row})

    args.runtime_output.parent.mkdir(parents=True, exist_ok=True)
    with args.runtime_output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", *benchmark_fields],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(runtime_rows)


if __name__ == "__main__":
    main()
