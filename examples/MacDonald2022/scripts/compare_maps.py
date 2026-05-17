"""Compare our interpolated genetic maps against MacDonald et al. (2022) reference maps.

MacDonald's maps (deCODE_interpolated_maps/chr{N}.tab.gz) were produced by an R
interpolation script and represent the ground-truth target for our pipeline.

Both maps are joined on physical position. For SNPs present in both, we compute:
  - Pearson correlation of cM values
  - Mean absolute error (MAE) in cM
  - Max absolute error
  - Fraction of SNPs with |error| > 0.001 cM and > 0.01 cM

Also reports:
  - SNPs only in ours (positions absent from MacDonald's map)
  - SNPs only in MacDonald's (positions absent from ours)

Usage:
    python scripts/compare_maps_ref.py \
        --ours    data/maps/interpolated/chr{1..22}.tab.gz \
        --ref-dir resources/macdonalds_maps/ \
        --output  results/compare/map_ref_comparison.tsv
"""

from __future__ import annotations

import argparse
import gzip
import math
import statistics
from pathlib import Path

MACDONALDS_MAP_URL = (
    "https://raw.githubusercontent.com/jmacdon/LDblocks_GRCh38"
    "/master/data/deCODE_interpolated_maps/chr{chrom}.tab.gz"
)


def _chrom_num(path: Path) -> int:
    stem = path.name.split(".")[0]  # chr22.tab.gz → chr22
    return int(stem.lstrip("chr"))


def read_our_map(path: Path) -> dict[int, float]:
    """Return {position: cM} from our interpolate-maps output.

    Format: ``rs_id  position  cM`` (space-delimited, no header).
    """
    result: dict[int, float] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                result[int(parts[1])] = float(parts[2])
            except ValueError:
                pass  # skip header-like lines
    return result


def read_ref_map(path: Path) -> dict[int, float]:
    """Return {position: cM} from MacDonald's reference map.

    Format: ``chrom  position  cM`` (tab-delimited, no header).
    """
    result: dict[int, float] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                result[int(parts[1])] = float(parts[2])
            except ValueError:
                pass
    return result


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx == 0 or sy == 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / (sx * sy)


def compare_chrom(
    chrom: str,
    ours: dict[int, float],
    ref: dict[int, float],
) -> dict:
    shared_pos = sorted(set(ours) & set(ref))
    only_ours  = len(set(ours) - set(ref))
    only_ref   = len(set(ref) - set(ours))

    if not shared_pos:
        return {
            "chrom": chrom, "n_shared": 0,
            "only_ours": only_ours, "only_ref": only_ref,
        }

    our_vals = [ours[p] for p in shared_pos]
    ref_vals = [ref[p]  for p in shared_pos]
    errors   = [abs(o - r) for o, r in zip(our_vals, ref_vals)]

    return {
        "chrom": chrom,
        "n_shared": len(shared_pos),
        "only_ours": only_ours,
        "only_ref": only_ref,
        "pearson_r": round(pearson(our_vals, ref_vals), 6),
        "mae_cM": round(statistics.mean(errors), 6),
        "max_err_cM": round(max(errors), 6),
        "frac_gt_1e3": round(sum(1 for e in errors if e > 1e-3) / len(errors), 4),
        "frac_gt_1e2": round(sum(1 for e in errors if e > 1e-2) / len(errors), 4),
    }


def main() -> None:
    import urllib.request

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", nargs="+", required=True, type=Path,
                        metavar="PATH", help="Our interpolated map files.")
    parser.add_argument("--ref-dir", required=True, type=Path,
                        help="Directory containing MacDonald's reference maps "
                             "(downloaded here if absent).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output TSV with per-chromosome comparison metrics.")
    args = parser.parse_args()

    args.ref_dir.mkdir(parents=True, exist_ok=True)
    our_files = sorted(args.ours, key=_chrom_num)

    cols = ["chrom", "n_shared", "only_ours", "only_ref",
            "pearson_r", "mae_cM", "max_err_cM", "frac_gt_1e3", "frac_gt_1e2"]

    rows: list[dict] = []
    for our_path in our_files:
        chrom_n = _chrom_num(our_path)
        chrom   = f"chr{chrom_n}"
        ref_path = args.ref_dir / f"chr{chrom_n}.tab.gz"

        if not ref_path.exists():
            url = MACDONALDS_MAP_URL.format(chrom=chrom_n)
            print(f"  Downloading {chrom} reference map...")
            urllib.request.urlretrieve(url, ref_path)

        ours_map = read_our_map(our_path)
        ref_map  = read_ref_map(ref_path)
        row = compare_chrom(chrom, ours_map, ref_map)
        rows.append(row)

        r = row.get("pearson_r", "")
        mae = row.get("mae_cM", "")
        print(f"  {chrom}: {row.get('n_shared', 0):>7,} shared SNPs  "
              f"r={r}  MAE={mae} cM")

    # Print summary
    shared_rows = [r for r in rows if r.get("n_shared", 0) > 0]
    if shared_rows:
        mean_r   = statistics.mean(r["pearson_r"] for r in shared_rows)
        mean_mae = statistics.mean(r["mae_cM"]    for r in shared_rows)
        print(f"\nMean Pearson r: {mean_r:.6f}")
        print(f"Mean MAE:       {mean_mae:.6f} cM")

    # Write TSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
    print(f"\nComparison written to {args.output}")


if __name__ == "__main__":
    main()
