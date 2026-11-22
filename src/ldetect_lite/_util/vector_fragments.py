"""Saved matrix-to-vector fragments, so cached reruns can skip Step 3.

The fused Step 2/3 path (``_util/covariance_sidecars.py``) builds each
partition's vector fragment while the partition's covariance is written.
Those fragments used to live only in memory, so any rerun that reused even
one cached partition fell back to re-reading every partition in Step 3.

Each fragment is now also saved next to its partition as
``<chrom>.<start>.<end>.vector.npz``. A saved fragment is only reused when it
provably describes the current partition file and the current vector plan:

- the partition ``.h5`` has the same size and ``st_mtime_ns`` as when the
  fragment was saved (any rewrite of the partition invalidates it);
- the plan fields that shape a fragment (``end``, ``next_start``, the
  center-locus lower bound and its inclusiveness, ``snp_first`` and
  ``snp_last``) are unchanged;
- the file format version matches.

Anything else -- a missing, unreadable, stale or mismatched file -- is a
cache miss, and the caller recomputes that one partition's fragment from its
``.h5`` (``_compute_diag_vector_partition_hdf5``), which is byte-identical to
the fused fragment. Saving is atomic (write to a temporary file, then
``os.replace``), so an interrupted save never leaves a partial fragment.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from ldetect_lite._util.vector_array import (
    _DiagVectorPartitionPlan,
    _DiagVectorPartitionResult,
)

FRAGMENT_FORMAT_VERSION = 1
_NO_NEXT_START = -1


def fragment_path(partition_path: Path) -> Path:
    """Saved-fragment path for a covariance partition ``.h5`` path."""
    return partition_path.with_suffix(".vector.npz")


def _plan_key(
    plan: _DiagVectorPartitionPlan, snp_first: int, snp_last: int
) -> np.ndarray:
    return np.array(
        [
            FRAGMENT_FORMAT_VERSION,
            plan.start,
            plan.end,
            _NO_NEXT_START if plan.next_start is None else plan.next_start,
            plan.center_lower_bound,
            int(plan.center_lower_inclusive),
            snp_first,
            snp_last,
        ],
        dtype=np.int64,
    )


def _source_key(partition_path: Path) -> np.ndarray:
    stat = partition_path.stat()
    return np.array([stat.st_size, stat.st_mtime_ns], dtype=np.int64)


def save_vector_fragment(
    partition_path: Path,
    result: _DiagVectorPartitionResult,
    *,
    plan: _DiagVectorPartitionPlan,
    snp_first: int,
    snp_last: int,
) -> Path:
    """Atomically save *result* next to *partition_path*; return its path.

    Must be called after the partition ``.h5`` is fully written and closed,
    since the fragment records the file's size and modification time.
    """
    path = fragment_path(partition_path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(
                handle,
                plan_key=_plan_key(plan, snp_first, snp_last),
                source_key=_source_key(partition_path),
                # Original dtypes are kept, so a loaded fragment merges
                # exactly like the in-memory one.
                loci=np.asarray(result.loci),
                sum_loci=np.asarray(result.sum_loci),
                sum_values=np.asarray(result.sum_values),
                bounds=np.array(
                    [result.end_locus, result.write_cutoff], dtype=np.int64
                ),
            )
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def load_vector_fragment(
    partition_path: Path,
    *,
    plan: _DiagVectorPartitionPlan,
    snp_first: int,
    snp_last: int,
) -> _DiagVectorPartitionResult | None:
    """Return the saved fragment if it is valid for this partition and plan.

    Returns ``None`` (a cache miss) when the fragment is missing, unreadable,
    from another format version, saved for a different plan, or older than
    the partition file's current contents.
    """
    path = fragment_path(partition_path)
    if not path.exists() or not partition_path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(
                data["plan_key"], _plan_key(plan, snp_first, snp_last)
            ):
                return None
            if not np.array_equal(data["source_key"], _source_key(partition_path)):
                return None
            end_locus, write_cutoff = (int(v) for v in data["bounds"])
            return _DiagVectorPartitionResult(
                loci=data["loci"],
                sum_loci=data["sum_loci"],
                sum_values=data["sum_values"],
                end_locus=end_locus,
                write_cutoff=write_cutoff,
                profile={"checkpoint": plan.checkpoint},
            )
    except (OSError, KeyError, ValueError):
        return None


def remove_vector_fragment(partition_path: Path) -> None:
    """Delete a partition's saved fragment, if any."""
    fragment_path(partition_path).unlink(missing_ok=True)
