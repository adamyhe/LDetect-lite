"""Profile VCF/BCF ingestion and bitpack construction costs.

This diagnostic separates the costs that are currently folded into
``calc_covariance``'s broad ``vcf_seconds``/``array_seconds``/``pack_seconds``
timing buckets:

* cyvcf2 region iteration without genotype access;
* genotype materialization and phased-sample validation;
* current list-backed reference-panel construction;
* conversion of that panel to arrays plus uint64 bitpacking;
* direct one-pass packing from cyvcf2 genotypes into uint64 rows;
* writing the directly-packed arrays to a small HDF5 sidecar.

The script is meant for optimization diagnosis, not production pipeline use.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cyvcf2
import h5py
import hdf5plugin  # noqa: F401 - registers bundled HDF5 filters
import numpy as np

from ldetect_lite._util.reference_panel import (
    read_genetic_map,
    read_individuals,
    read_reference_panel,
    watterson_theta,
)
from ldetect_lite.shrinkage import _build_covariance_inputs, _pack_haplotypes_impl


def _open_vcf(path: Path, individuals: list[str]) -> cyvcf2.VCF:
    vcf = cyvcf2.VCF(str(path), samples=individuals)
    missing = [ind for ind in individuals if ind not in vcf.samples]
    if missing:
        vcf.close()
        raise ValueError(
            f"individuals not found in VCF/BCF header: {', '.join(missing)}"
        )
    return vcf


def _sample_order(vcf: cyvcf2.VCF, individuals: list[str]) -> list[int]:
    sample_index = {ind: idx for idx, ind in enumerate(vcf.samples)}
    return [sample_index[ind] for ind in individuals]


def _scan_positions(
    vcf_path: Path,
    region: str,
    individuals: list[str],
    pos2gpos: dict[int, float],
) -> dict[str, int]:
    vcf = _open_vcf(vcf_path, individuals)
    n_variants = 0
    n_mapped = 0
    for variant in vcf(region):
        n_variants += 1
        if variant.POS in pos2gpos:
            n_mapped += 1
    vcf.close()
    return {"n_variants": n_variants, "n_mapped": n_mapped}


def _decode_genotypes(
    vcf_path: Path,
    region: str,
    individuals: list[str],
    pos2gpos: dict[int, float],
) -> dict[str, int]:
    vcf = _open_vcf(vcf_path, individuals)
    order = _sample_order(vcf, individuals)
    n_variants = 0
    n_mapped = 0
    n_valid = 0
    skipped_genotypes = 0
    checksum = 0
    for variant in vcf(region):
        n_variants += 1
        if variant.POS not in pos2gpos:
            continue
        n_mapped += 1
        genotypes = variant.genotypes
        valid = True
        for col in order:
            allele1, allele2, phased = genotypes[col]
            if not phased or allele1 < 0 or allele2 < 0:
                skipped_genotypes += 1
                valid = False
                break
            checksum += allele1 + allele2
        if valid:
            n_valid += 1
    vcf.close()
    return {
        "n_variants": n_variants,
        "n_mapped": n_mapped,
        "n_valid": n_valid,
        "skipped_genotypes": skipped_genotypes,
        "checksum": checksum,
    }


def _current_panel(
    vcf_path: Path,
    region: str,
    individuals: list[str],
    pos2gpos: dict[int, float],
    n_haps: int,
) -> dict[str, int]:
    panel = read_reference_panel(vcf_path, region, individuals, pos2gpos, n_haps)
    return {
        "n_snps": len(panel.positions),
        "skipped_unphased": panel.skipped_unphased,
        "duplicate_positions": panel.duplicate_positions,
    }


def _arrays_and_pack(
    vcf_path: Path,
    region: str,
    individuals: list[str],
    pos2gpos: dict[int, float],
    n_ind: int,
    cutoff: float,
    ne: float,
) -> dict[str, int]:
    n_haps = 2 * n_ind
    panel = read_reference_panel(vcf_path, region, individuals, pos2gpos, n_haps)
    inputs = _build_covariance_inputs(panel, pos2gpos, ne, n_ind, cutoff)
    packed = _pack_haplotypes_impl(inputs.hap_mat)
    return {
        "n_snps": int(inputs.pos_arr.size),
        "n_haps": int(inputs.hap_mat.shape[1]) if inputs.hap_mat.size else n_haps,
        "n_words": int(packed.shape[1]) if packed.size else (n_haps + 63) // 64,
        "packed_bytes": int(packed.nbytes),
    }


def _direct_pack(
    vcf_path: Path,
    region: str,
    individuals: list[str],
    pos2gpos: dict[int, float],
) -> dict[str, Any]:
    n_haps = 2 * len(individuals)
    n_words = (n_haps + 63) // 64
    vcf = _open_vcf(vcf_path, individuals)
    order = _sample_order(vcf, individuals)

    positions: list[int] = []
    gpos: list[float] = []
    hap_sums: list[float] = []
    packed_rows: list[np.ndarray] = []
    seen_positions: set[int] = set()
    duplicate_positions = 0
    skipped_unphased = 0

    one = np.uint64(1)
    for variant in vcf(region):
        pos = variant.POS
        genetic_pos = pos2gpos.get(pos)
        if genetic_pos is None:
            continue

        row = np.zeros(n_words, dtype=np.uint64)
        row_sum = 0
        hap_idx = 0
        genotypes = variant.genotypes
        skip = False
        for col in order:
            allele1, allele2, phased = genotypes[col]
            if not phased or allele1 < 0 or allele2 < 0:
                skipped_unphased += 1
                skip = True
                break
            if allele1:
                row[hap_idx // 64] |= one << np.uint64(hap_idx % 64)
                row_sum += 1
            hap_idx += 1
            if allele2:
                row[hap_idx // 64] |= one << np.uint64(hap_idx % 64)
                row_sum += 1
            hap_idx += 1

        if skip:
            continue
        if pos in seen_positions:
            duplicate_positions += 1
            continue
        seen_positions.add(pos)
        positions.append(pos)
        gpos.append(genetic_pos)
        hap_sums.append(float(row_sum))
        packed_rows.append(row)

    vcf.close()

    if packed_rows:
        packed = np.vstack(packed_rows).astype(np.uint64, copy=False)
    else:
        packed = np.empty((0, n_words), dtype=np.uint64)
    return {
        "positions": np.asarray(positions, dtype=np.int32),
        "gpos": np.asarray(gpos, dtype=np.float64),
        "hap_sums": np.asarray(hap_sums, dtype=np.float64),
        "packed": packed,
        "n_snps": len(positions),
        "n_haps": n_haps,
        "n_words": n_words,
        "packed_bytes": int(packed.nbytes),
        "skipped_unphased": skipped_unphased,
        "duplicate_positions": duplicate_positions,
    }


def _direct_pack_summary(
    vcf_path: Path,
    region: str,
    individuals: list[str],
    pos2gpos: dict[int, float],
) -> dict[str, int]:
    packed = _direct_pack(vcf_path, region, individuals, pos2gpos)
    return {
        "n_snps": int(packed["n_snps"]),
        "n_haps": int(packed["n_haps"]),
        "n_words": int(packed["n_words"]),
        "packed_bytes": int(packed["packed_bytes"]),
        "skipped_unphased": int(packed["skipped_unphased"]),
        "duplicate_positions": int(packed["duplicate_positions"]),
    }


def _write_packed_h5(packed: dict[str, Any], output_path: Path) -> dict[str, int]:
    with h5py.File(output_path, "w") as h5:
        h5.attrs["format"] = "ldetect-lite-packed-panel-profile"
        h5.attrs["n_haps"] = int(packed["n_haps"])
        kwargs = {**hdf5plugin.Zstd(clevel=3), "shuffle": True}
        h5.create_dataset("positions", data=packed["positions"], **kwargs)
        h5.create_dataset("gpos", data=packed["gpos"], **kwargs)
        h5.create_dataset("hap_sums", data=packed["hap_sums"], **kwargs)
        h5.create_dataset("packed_haplotypes", data=packed["packed"], **kwargs)
    return {"output_bytes": output_path.stat().st_size}


def _time_call(
    fn: Callable[[], dict[str, Any]],
    repeats: int,
) -> tuple[float, dict[str, Any]]:
    timings: list[float] = []
    last: dict[str, Any] = {}
    for _ in range(repeats):
        start = time.perf_counter()
        last = fn()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-panel", required=True, type=Path)
    parser.add_argument("--region", required=True)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--individuals", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ne", type=float, default=11418.0)
    parser.add_argument("--cutoff", type=float, default=1e-7)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    individuals = read_individuals(args.individuals)
    pos2gpos = read_genetic_map(args.genetic_map)
    n_ind = len(individuals)
    n_haps = 2 * n_ind
    theta = watterson_theta(n_haps)

    rows: list[dict[str, Any]] = []

    def add_step(
        name: str,
        fn: Callable[[], dict[str, Any]],
        repeats: int | None = None,
    ) -> None:
        median_s, meta = _time_call(fn, args.repeats if repeats is None else repeats)
        row = {
            "step": name,
            "median_seconds": f"{median_s:.6f}",
            "repeats": args.repeats if repeats is None else repeats,
            "reference_panel": str(args.reference_panel),
            "region": args.region,
            "n_individuals": n_ind,
            "n_haps": n_haps,
            "theta": f"{theta:.12g}",
        }
        row.update(meta)
        rows.append(row)

    add_step(
        "cyvcf2_scan_positions",
        lambda: _scan_positions(
            args.reference_panel, args.region, individuals, pos2gpos
        ),
    )
    add_step(
        "cyvcf2_decode_genotypes",
        lambda: _decode_genotypes(
            args.reference_panel, args.region, individuals, pos2gpos
        ),
    )
    add_step(
        "current_read_reference_panel",
        lambda: _current_panel(
            args.reference_panel,
            args.region,
            individuals,
            pos2gpos,
            n_haps,
        ),
    )
    add_step(
        "current_read_panel_arrays_pack",
        lambda: _arrays_and_pack(
            args.reference_panel,
            args.region,
            individuals,
            pos2gpos,
            n_ind,
            args.cutoff,
            args.ne,
        ),
    )
    add_step(
        "direct_pack_from_cyvcf2",
        lambda: _direct_pack_summary(
            args.reference_panel, args.region, individuals, pos2gpos
        ),
    )

    direct = _direct_pack(args.reference_panel, args.region, individuals, pos2gpos)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "packed-panel-profile.h5"
        add_step("packed_h5_write", lambda: _write_packed_h5(direct, tmp_path))

    all_keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    for row in rows:
        extras = " ".join(
            f"{key}={value}"
            for key, value in row.items()
            if key not in {"step", "median_seconds", "reference_panel", "region"}
        )
        print(f"{row['step']}: {row['median_seconds']}s {extras}")


if __name__ == "__main__":
    main()
