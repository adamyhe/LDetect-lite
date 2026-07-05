"""HDF5 signal-sidecar storage: partition-local correlation-sum contributions.

This is the Phase 1 cache described in
``notes/covariance-streaming-cache-implementation-note.md``: a small
per-partition sidecar written alongside a compact covariance HDF5 partition,
holding the same per-locus correlation-sum contributions that
``matrix_analysis.MatrixAnalysis.calc_diag_array`` would otherwise have to
recompute by rereading and renormalizing pair-level covariance rows.

Sums stored here are partition-local: they cover every locus discovered
within the partition's own ``[start, end]`` bounds, without the
cross-partition ownership trimming that the final chromosome-wide vector
needs. Assembly-time code (``_util.vector_array.write_diag_vector_signal``)
applies that trimming using the same partition plan as the direct HDF5
vector path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_FORMAT = "ldetect2-signal-h5"
_VERSION = 1
_REQUIRED_DATASETS = frozenset({"signal/loci", "signal/sum_r2"})


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "h5py is required for HDF5 signal partitions. Install ldetect2 "
            "with its project dependencies before running covariance workflows."
        ) from exc
    return h5py


def _position_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64, copy=False)
    if arr.size == 0:
        return arr.astype(np.int32, copy=False)
    int32_info = np.iinfo(np.int32)
    if int(arr.min()) >= int32_info.min and int(arr.max()) <= int32_info.max:
        return arr.astype(np.int32, copy=False)
    return arr.astype(np.int64, copy=False)


def write_signal_partition_hdf5(
    path: Path,
    *,
    loci: np.ndarray,
    sum_r2: np.ndarray,
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
    compression: str | None = "lzf",
) -> None:
    """Write one partition-local signal sidecar.

    ``loci`` must be sorted ascending with no duplicates; ``sum_r2`` is the
    aligned partition-local correlation-sum contribution for each locus
    (zero where a locus was discovered but never a center of a retained
    pair).
    """
    loci = _position_array(loci)
    sum_r2 = np.asarray(sum_r2, dtype=np.float64)
    if loci.shape != sum_r2.shape:
        raise ValueError("signal loci and sum_r2 arrays must have identical shapes")
    if loci.size > 1 and np.any(loci[1:] <= loci[:-1]):
        raise ValueError("signal loci must be sorted ascending with no duplicates")

    h5py = _h5py()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = _FORMAT
        h5.attrs["version"] = _VERSION
        if chrom is not None:
            h5.attrs["chrom"] = chrom
        if start is not None:
            h5.attrs["start"] = int(start)
        if end is not None:
            h5.attrs["end"] = int(end)
        h5.attrs["position_dtype"] = str(loci.dtype)

        sig = h5.create_group("signal")
        kwargs = {"compression": compression, "shuffle": True}
        sig.create_dataset("loci", data=loci, **kwargs)
        sig.create_dataset("sum_r2", data=sum_r2, **kwargs)


def validate_signal_hdf5(path: Path) -> bool:
    """Return whether *path* is a readable ldetect2 HDF5 signal sidecar."""
    if not path.exists():
        return False
    try:
        h5py = _h5py()
        with h5py.File(path, "r") as h5:
            if h5.attrs.get("format") != _FORMAT:
                return False
            if int(h5.attrs.get("version", -1)) != _VERSION:
                return False
            return all(dataset in h5 for dataset in _REQUIRED_DATASETS)
    except Exception:
        return False


def read_signal_partition_hdf5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the full ``(loci, sum_r2)`` arrays from one signal sidecar."""
    if not path.exists():
        raise FileNotFoundError(
            f"Signal partition {path} is missing. Regenerate covariance with "
            "`ldetect2 calc-covariance --signal-output` or `ldetect2 run` "
            "signal-cache mode."
        )
    h5py = _h5py()
    with h5py.File(path, "r") as h5:
        missing = sorted(dataset for dataset in _REQUIRED_DATASETS if dataset not in h5)
        if missing:
            raise ValueError(
                f"Signal partition {path} is missing required dataset(s): "
                f"{', '.join(missing)}"
            )
        loci = _position_array(h5["signal/loci"][:]).astype(np.int64, copy=False)
        sum_r2 = np.asarray(h5["signal/sum_r2"][:], dtype=np.float64)
    return loci, sum_r2
