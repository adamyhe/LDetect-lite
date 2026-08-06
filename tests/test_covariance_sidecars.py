"""Tests for CovarianceSidecarAccumulator's own safety/lifecycle behavior.

Bit-exactness of its vector fragment against the post-hoc read is covered by
tests/test_cmd_run_fused_vector.py; this file covers the accumulator's
contract in isolation, including the single-pass-fallback safety net.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ldetect_lite._util.covariance_sidecars import CovarianceSidecarAccumulator
from ldetect_lite.io.covariance_hdf5 import CovarianceRowChunk
from ldetect_lite.shrinkage import calc_covariance

_TABIX_TOOLS_AVAILABLE = all(
    shutil.which(tool) is not None for tool in ("bgzip", "tabix")
)
requires_htslib_tools = pytest.mark.skipif(
    not _TABIX_TOOLS_AVAILABLE, reason="bgzip/tabix not found on PATH"
)


def test_finalize_vector_before_set_diagonals_raises() -> None:
    sidecar = CovarianceSidecarAccumulator()
    with pytest.raises(ValueError, match="set_diagonals"):
        sidecar.finalize_vector(
            end=100,
            next_start=None,
            snp_last=100,
            center_lower_bound=0,
            center_lower_inclusive=True,
        )


def test_finalize_vector_after_mark_invalid_raises() -> None:
    sidecar = CovarianceSidecarAccumulator()
    sidecar.set_diagonals(np.array([100], dtype=np.int64), np.array([0.5]))
    sidecar.mark_invalid()

    with pytest.raises(RuntimeError, match="invalid"):
        sidecar.finalize_vector(
            end=100,
            next_start=None,
            snp_last=100,
            center_lower_bound=0,
            center_lower_inclusive=True,
        )


def test_wrap_yields_chunks_unchanged_while_buffering() -> None:
    sidecar = CovarianceSidecarAccumulator()
    chunk = CovarianceRowChunk(
        lo=np.array([100], dtype=np.int32),
        hi=np.array([200], dtype=np.int32),
        shrink_ld=np.array([0.5]),
    )

    wrapped = list(sidecar.wrap(iter([chunk])))

    assert wrapped == [chunk]
    assert sidecar._buffered == [chunk]


def test_empty_stream_produces_empty_fragment() -> None:
    sidecar = CovarianceSidecarAccumulator()
    sidecar.set_diagonals(np.array([], dtype=np.int64), np.array([], dtype=np.float64))

    result = sidecar.finalize_vector(
        end=100,
        next_start=None,
        snp_last=100,
        center_lower_bound=0,
        center_lower_inclusive=True,
    )

    assert result.loci.size == 0
    assert result.sum_loci.size == 0
    assert result.sum_values.size == 0


@requires_htslib_tools
def test_single_pass_fallback_marks_sidecar_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rare mid-stream single-pass failure must not silently produce a
    wrong fragment: calc_covariance still completes via the two-pass
    fallback (existing behavior, unaffected), but the sidecar it was given
    must come back marked invalid rather than holding a partial buffer."""
    samples = [f"s{i}" for i in range(4)]
    header = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "##contig=<ID=1>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    rows = [
        "1\t100\trs100\tA\tG\t.\t.\t.\tGT\t0|1\t1|0\t0|0\t1|1",
        "1\t200\trs200\tA\tG\t.\t.\t.\tGT\t1|1\t0|0\t0|1\t1|0",
        "1\t300\trs300\tA\tG\t.\t.\t.\tGT\t0|0\t1|1\t1|0\t0|1",
    ]
    raw_path = tmp_path / "panel.vcf"
    raw_path.write_text("\n".join(header + rows) + "\n")
    subprocess.run(["bgzip", "-f", str(raw_path)], check=True)
    vcf_path = tmp_path / "panel.vcf.gz"
    subprocess.run(["tabix", "-f", "-p", "vcf", str(vcf_path)], check=True)

    genetic_map_path = tmp_path / "map.txt.gz"
    with gzip.open(genetic_map_path, "wt") as f:
        f.write("1 100 0.000\n1 200 0.001\n1 300 0.002\n")
    individuals_path = tmp_path / "individuals.txt"
    individuals_path.write_text("\n".join(samples) + "\n")

    import ldetect_lite.shrinkage as shrinkage_mod

    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced single-pass failure for testing")

    monkeypatch.setattr(
        shrinkage_mod, "_compact_pair_chunks_single_pass_bitpacked", _boom
    )

    sidecar = CovarianceSidecarAccumulator()
    output_path = tmp_path / "partition.h5"
    calc_covariance(
        vcf_path=vcf_path,
        region="1:100-300",
        genetic_map_path=genetic_map_path,
        individuals_path=individuals_path,
        output_path=output_path,
        compact_output=True,
        ld_kernel="bitpacked",
        sidecar=sidecar,
    )

    assert output_path.exists()  # two-pass fallback still wrote the partition
    assert sidecar.invalid
    with pytest.raises(RuntimeError, match="invalid"):
        sidecar.finalize_vector(
            end=300,
            next_start=None,
            snp_last=300,
            center_lower_bound=100,
            center_lower_inclusive=True,
        )
