"""Compare regenerated ldetect-lite covariance against the original ldetect fixture."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np


def read_reference(path: Path) -> dict[str, np.ndarray]:
    i_id: list[str] = []
    j_id: list[str] = []
    i_pos: list[int] = []
    j_pos: list[int] = []
    i_gpos: list[float] = []
    j_gpos: list[float] = []
    naive: list[float] = []
    shrink: list[float] = []
    with gzip.open(path, "rt") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            i_id.append(row[0])
            j_id.append(row[1])
            i_pos.append(int(row[2]))
            j_pos.append(int(row[3]))
            i_gpos.append(float(row[4]))
            j_gpos.append(float(row[5]))
            naive.append(float(row[6]))
            shrink.append(float(row[7]))
    return canonicalize(
        {
            "lo": np.minimum(i_pos, j_pos).astype(np.int64),
            "hi": np.maximum(i_pos, j_pos).astype(np.int64),
            "i_gpos": np.asarray(i_gpos, dtype=np.float64),
            "j_gpos": np.asarray(j_gpos, dtype=np.float64),
            "naive_ld": np.asarray(naive, dtype=np.float64),
            "shrink_ld": np.asarray(shrink, dtype=np.float64),
            "i_id": np.asarray(i_id, dtype=str),
            "j_id": np.asarray(j_id, dtype=str),
        }
    )


def read_hdf5(path: Path) -> dict[str, np.ndarray]:
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(path, "r") as h5:
        return canonicalize(
            {
                "lo": np.asarray(h5["covariance/lo"][:], dtype=np.int64),
                "hi": np.asarray(h5["covariance/hi"][:], dtype=np.int64),
                "i_gpos": np.asarray(h5["metadata/i_gpos"][:], dtype=np.float64),
                "j_gpos": np.asarray(h5["metadata/j_gpos"][:], dtype=np.float64),
                "naive_ld": np.asarray(h5["covariance/naive_ld"][:], dtype=np.float64),
                "shrink_ld": np.asarray(
                    h5["covariance/shrink_ld"][:], dtype=np.float64
                ),
                "i_id": decode_string_array(h5["metadata/i_id"][:]),
                "j_id": decode_string_array(h5["metadata/j_id"][:]),
            }
        )


def decode_string_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values
        ],
        dtype=str,
    )


def canonicalize(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if data["lo"].size == 0:
        return data
    order = np.lexsort((data["hi"], data["lo"]))
    ordered = {key: value[order] for key, value in data.items()}
    keep = np.ones(ordered["lo"].size, dtype=bool)
    keep[1:] = (ordered["lo"][1:] != ordered["lo"][:-1]) | (
        ordered["hi"][1:] != ordered["hi"][:-1]
    )
    return {key: value[keep] for key, value in ordered.items()}


def write_plot(
    ref: dict[str, np.ndarray],
    ours: dict[str, np.ndarray],
    path: Path,
) -> None:
    configure_matplotlib_cache(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.8, 2.9),
        width_ratios=(1.0, 1.15),
        constrained_layout=True,
    )
    lo = min(float(ref["shrink_ld"].min()), float(ours["shrink_ld"].min()))
    hi = max(float(ref["shrink_ld"].max()), float(ours["shrink_ld"].max()))
    pad = (hi - lo) * 0.03 if hi > lo else 1e-6
    extent = (lo - pad, hi + pad, lo - pad, hi + pad)
    counts, xedges, yedges = np.histogram2d(
        ref["shrink_ld"],
        ours["shrink_ld"],
        bins=180,
        range=((extent[0], extent[1]), (extent[2], extent[3])),
    )
    masked = np.ma.masked_where(counts.T == 0, counts.T)
    image = axes[0].imshow(
        masked,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="magma",
        norm=LogNorm(vmin=1, vmax=max(1, int(counts.max()))),
        rasterized=True,
    )
    axes[0].set_xlim(extent[0], extent[1])
    axes[0].set_ylim(extent[2], extent[3])
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("original LDetect shrinkage")
    axes[0].set_ylabel("ldetect-lite regenerated shrinkage")
    axes[0].set_title("Covariance density from chr2 VCF")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04, label="rows")

    diff = ours["shrink_ld"] - ref["shrink_ld"]
    axes[1].plot(np.arange(diff.size), diff, linewidth=1)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("canonical row")
    axes[1].set_ylabel("difference")
    axes[1].set_title("Row-wise difference")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def configure_matplotlib_cache(path: Path) -> None:
    import os

    root = path.parent.parent if path.parent.name == "plots" else path.parent
    mpl_config = root / ".mplconfig"
    xdg_cache = root / ".cache"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--require-equivalent", action="store_true")
    args = parser.parse_args()

    ours = read_hdf5(args.ours)
    ref = read_reference(args.ref)
    shared_shape = ours["lo"].shape == ref["lo"].shape
    keys_exact = (
        shared_shape
        and np.array_equal(ours["lo"], ref["lo"])
        and np.array_equal(ours["hi"], ref["hi"])
    )

    rows = [("metric", "value")]
    rows.append(("n_ours", ours["lo"].size))
    rows.append(("n_ref", ref["lo"].size))
    rows.append(("keys_exact", "yes" if keys_exact else "no"))
    all_exact = keys_exact
    all_equivalent = keys_exact
    for field in ("naive_ld", "shrink_ld", "i_gpos", "j_gpos", "i_id", "j_id"):
        exact = shared_shape and np.array_equal(ours[field], ref[field])
        rows.append((f"{field}_exact", "yes" if exact else "no"))
        all_exact = all_exact and exact
        if np.issubdtype(ours[field].dtype, np.number) and shared_shape:
            max_abs = (
                float(np.max(np.abs(ours[field] - ref[field])))
                if ours[field].size
                else 0.0
            )
            rows.append((f"{field}_max_abs_diff", f"{max_abs:.6e}"))
            equivalent = max_abs <= args.atol
        else:
            equivalent = exact
        rows.append((f"{field}_equivalent", "yes" if equivalent else "no"))
        all_equivalent = all_equivalent and equivalent
    rows.append(("all_exact", "yes" if all_exact else "no"))
    rows.append(("all_equivalent", "yes" if all_equivalent else "no"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(rows)
    if args.plot is not None:
        write_plot(ref, ours, args.plot)
    if args.require_exact and not all_exact:
        raise SystemExit("covariance conversion is not exact")
    if args.require_equivalent and not all_equivalent:
        raise SystemExit("covariance output is not numerically equivalent")


if __name__ == "__main__":
    main()
