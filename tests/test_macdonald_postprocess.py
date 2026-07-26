"""Tests for MacDonald2022 BED postprocessing helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / "examples"
    / "MacDonald2022"
    / "scripts"
    / "postprocess.py"
)
SPEC = importlib.util.spec_from_file_location("macdonald_postprocess", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
postprocess = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(postprocess)


def test_drop_nonpositive_blocks_removes_zero_and_negative_width_blocks() -> None:
    assert postprocess.drop_nonpositive_blocks(
        [(10, 10), (10, 20), (30, 25), (20, 30)]
    ) == [(10, 20), (20, 30)]


def test_drop_empty_blocks_removes_zero_snp_blocks_only() -> None:
    assert postprocess.drop_empty_blocks(
        blocks=[(10, 14), (14, 100), (100, 110)],
        counts=[0, 1, 10],
    ) == ([(14, 100), (100, 110)], [1, 10])


def test_trim_blocks_to_snp_span_drops_leading_block_ending_at_first_snp() -> None:
    assert postprocess.trim_blocks_to_snp_span(
        blocks=[(31439, 31443), (31443, 1196882)],
        first_snp=31443,
        last_snp=1196882,
    ) == [(31443, 1196882)]


def test_trim_blocks_to_snp_span_keeps_interior_small_blocks() -> None:
    assert postprocess.trim_blocks_to_snp_span(
        blocks=[
            (55438332, 59565357),
            (59565357, 59582067),
            (59582067, 62477349),
        ],
        first_snp=55438332,
        last_snp=62477349,
    ) == [
        (55438332, 59565357),
        (59565357, 59582067),
        (59582067, 62477349),
    ]


def test_merge_small_blocks_merges_leading_small_block_right() -> None:
    assert postprocess.merge_small_blocks(
        blocks=[(10, 14), (14, 100), (100, 200)],
        counts=[1, 200, 200],
        min_snps=100,
    ) == [(10, 100), (100, 200)]


def test_merge_small_blocks_merges_interior_small_block_left() -> None:
    assert postprocess.merge_small_blocks(
        blocks=[(10, 100), (100, 110), (110, 200)],
        counts=[200, 1, 200],
        min_snps=100,
    ) == [(10, 110), (110, 200)]


def test_merge_small_blocks_disabled_when_min_snps_is_zero() -> None:
    blocks = [(10, 100), (100, 110), (110, 200)]
    assert postprocess.merge_small_blocks(
        blocks=blocks,
        counts=[200, 1, 200],
        min_snps=0,
    ) == blocks
