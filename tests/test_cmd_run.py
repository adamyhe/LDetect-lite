"""Tests for the run subcommand helpers."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ldetect2._util.run as run_util
import ldetect2.shrinkage as shrinkage
from ldetect2._cli.cmd_run import _r2_nocache_config, _run
from ldetect2._util.run import (
    breakpoint_subsets_for_run,
    calc_partition,
    is_valid_covariance_partition,
)
from ldetect2.io.covariance_hdf5 import write_covariance_partition_hdf5


def _minimal_run_args(tmp_path: Path, **overrides):
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        pair_cache="hdf5",
        high_precision=False,
        r2_zarr_compressor="default",
        r2_nocache_cache_mib=512,
        r2_nocache_tile_size=128,
        r2_nocache_disable_tiled_local_search=False,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_full_covariance_partition_validates_against_full_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "valid.h5"
    write_covariance_partition_hdf5(
        path,
        i_pos=np.array([100], dtype=np.int32),
        j_pos=np.array([100], dtype=np.int32),
        shrink_ld=np.array([0.1]),
        i_gpos=np.array([0.1]),
        j_gpos=np.array([0.1]),
        naive_ld=np.array([0.1]),
        i_id=np.array(["rs1"]),
        j_id=np.array(["rs1"]),
    )

    assert is_valid_covariance_partition(path, require_full=True)
    assert is_valid_covariance_partition(path, require_full=False)


def test_compact_covariance_partition_validates_against_compact_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact.h5"
    write_covariance_partition_hdf5(
        path,
        i_pos=np.array([100], dtype=np.int32),
        j_pos=np.array([100], dtype=np.int32),
        shrink_ld=np.array([0.1]),
    )

    assert is_valid_covariance_partition(path, require_full=False)
    assert not is_valid_covariance_partition(path, require_full=True)


def test_invalid_covariance_partition_missing_shrink_ld(tmp_path: Path) -> None:
    path = tmp_path / "invalid.h5"
    path.write_text("not hdf5")

    assert not is_valid_covariance_partition(path, require_full=False)


def test_run_subset_requests_only_final_breakpoint_subset() -> None:
    assert breakpoint_subsets_for_run("fourier_ls", False) == {"fourier_ls"}


def test_run_all_breakpoint_subsets_preserves_full_output() -> None:
    assert breakpoint_subsets_for_run("fourier_ls", True) is None


def test_r2_nocache_config_accepts_new_flags(tmp_path: Path) -> None:
    args = SimpleNamespace(
        reference_panel="panel.vcf.gz",
        genetic_map=tmp_path / "map.gz",
        individuals=tmp_path / "individuals.txt",
        ne=123.0,
        cov_cutoff=1e-6,
        r2_nocache_cache_mib=1024,
        r2_nocache_tile_size=32,
        r2_nocache_disable_tiled_local_search=True,
    )

    config = _r2_nocache_config(args, "chr1")

    assert config.prepared_cache_mib == 1024
    assert config.tile_size == 32
    assert not config.use_tiled_local_search


def test_r2_nocache_flags_rejected_for_other_pair_caches(
    tmp_path: Path,
    capsys,
) -> None:
    args = _minimal_run_args(tmp_path, r2_nocache_cache_mib=1024)

    assert _run(args) == 1
    assert "--r2-nocache-* options" in capsys.readouterr().err


def test_r2_nocache_tile_size_must_be_positive(tmp_path: Path, capsys) -> None:
    args = _minimal_run_args(
        tmp_path,
        pair_cache="r2-nocache",
        r2_nocache_tile_size=0,
    )

    assert _run(args) == 1
    assert "--r2-nocache-tile-size" in capsys.readouterr().err


def test_direct_hdf5_partition_generation_streams_tabix_twice(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class NoReadStream(io.StringIO):
        def read(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("direct_hdf5 must not buffer the full VCF stream")

    popen_calls = []
    wait_calls = []
    calc_calls = []

    def fake_popen(cmd, stdout, text):  # noqa: ANN001
        stream = NoReadStream("##fileformat=VCFv4.2\n")
        popen_calls.append((cmd, stdout, text, stream))

        def wait() -> int:
            wait_calls.append(stream)
            return 0

        return SimpleNamespace(stdout=stream, wait=wait)

    def fake_calc_covariance(**kwargs) -> None:  # noqa: ANN003
        calc_calls.append(("covariance", kwargs["vcf_stream"]))

    def fake_calc_covariance_vector(**kwargs) -> None:  # noqa: ANN003
        calc_calls.append(("vector", kwargs["vcf_stream"]))

    monkeypatch.setattr(run_util.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(shrinkage, "calc_covariance", fake_calc_covariance)
    monkeypatch.setattr(
        shrinkage,
        "calc_covariance_vector",
        fake_calc_covariance_vector,
    )

    calc_partition(
        100,
        200,
        "chr1",
        "panel.vcf.gz",
        tmp_path / "map.gz",
        tmp_path / "individuals.txt",
        tmp_path / "chr1_100_200.h5",
        ne=11418.0,
        cutoff=1e-7,
        compact_output=True,
        pair_cache="hdf5",
        vector_output_path=tmp_path / "vector.txt.gz",
    )

    assert len(popen_calls) == 2
    assert len(wait_calls) == 2
    assert [name for name, _ in calc_calls] == ["covariance", "vector"]
    assert calc_calls[0][1] is popen_calls[0][3]
    assert calc_calls[1][1] is popen_calls[1][3]
