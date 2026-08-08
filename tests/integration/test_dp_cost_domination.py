"""Cost-domination check: the DP-optimal splitter should never do worse than
the LDetect heuristic at matching a given block count, since it exactly
minimizes the same objective LDetect's `fourier_ls` only hill-climbs toward.

This intentionally does not compare BED regions to a reference fixture (see
`tests/integration/test_pipeline.py`) -- `dp` is a distinct algorithm, not a
reproduction of the original LDetect method, so the correctness property that
matters here is cost domination, not boundary agreement.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration


def test_dp_cost_at_matching_k_beats_or_matches_fourier_ls(
    example_data_dir, example_store, tmp_path
):
    from ldetect_lite.pipeline import find_breakpoints

    ref_vector = example_data_dir / "vector/vector-EUR-chr2-39967768-40067768.txt.gz"
    out_json = tmp_path / "breakpoints.json"

    find_breakpoints(
        input_path=ref_vector,
        chr_name="chr2",
        store=example_store,
        n_snps_bw_bpoints=50,
        output_path=out_json,
        subsets={"fourier_ls", "dp"},
        dp_max_k=40,
        dp_candidate_width=10,
    )

    data = json.loads(out_json.read_text())
    fourier_ls_cost = float(data["fourier_ls"]["metric"]["sum"])
    fourier_ls_k = len(data["fourier_ls"]["loci"]) + 1

    candidates = {c["n_block"]: c["cost"] for c in data["dp"]["candidates"]}
    assert fourier_ls_k in candidates, (
        f"DP has no solution at fourier_ls's block count ({fourier_ls_k}); "
        f"available: {sorted(candidates)}"
    )
    assert candidates[fourier_ls_k] <= fourier_ls_cost + 1e-9
