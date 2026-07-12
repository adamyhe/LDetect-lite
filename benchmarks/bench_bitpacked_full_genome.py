"""Full-genome speed and exactness benchmark for compact LD kernels.

This benchmark downloads 1000 Genomes Phase 1 VCFs for one population,
generates the same chromosome partitions used by ldetect-lite, and runs each
partition twice:

1. the established compact ``uint8`` backend
2. the experimental compact ``bitpacked`` backend

For every partition it compares the compact HDF5 row keys, shrinkage values,
diagonal index, and lower-locus index before recording timing results.
It also samples process RSS during each covariance call; with
``--include-chromosome-mode`` it reports chromosome-load peak RSS, retained
packed-cache bytes, and per-partition chromosome-mode RSS.

**Superseded for full-dataset exactness validation.**
``examples/ldetect_original/Snakefile.ld_kernel_diagnostics`` is now the
canonical genome-scale ``uint8``-vs-``bitpacked`` exactness diagnostic: it
runs the actual ``ldetect run`` CLI end to end (not just ``calc_covariance``)
across all three 1000G populations by default and compares the final
vector/breakpoints/BED, not just covariance-partition rows. Prefer that
Snakefile when the goal is "does the bitpacked kernel change pipeline
output." This script remains useful for what the Snakefile doesn't do:
fine-grained per-stage timing/RSS profiling (prepare/VCF/array/pack/write
seconds) at the covariance layer, and quick single-population/small-scale
smoke checks without a Snakemake setup. Its ``--include-chromosome-mode``
path only ever exercises chromosome mode with ``ld_kernel="bitpacked"`` (see
``time_calc_covariance_from_genotypes``) -- it does not, and has never,
covered the chromosome-mode + ``uint8`` combination that
``notes/logs/covariance-bitpacked-kernel-and-chromosome-mode.md`` documents
as inexact.

Examples:

    uv run python benchmarks/bench_bitpacked_full_genome.py --population EUR

    uv run python benchmarks/bench_bitpacked_full_genome.py \\
      --population EUR --chromosomes 22 --max-partitions-per-chrom 2

    uv run python benchmarks/bench_bitpacked_full_genome.py \\
      --population EUR --chromosomes 21 22 --include-chromosome-mode
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ldetect_lite._util.memory import current_rss_mib, max_rss_mib  # noqa: E402
from ldetect_lite.io.covariance_hdf5 import open_covariance_reader  # noqa: E402
from ldetect_lite.shrinkage import (  # noqa: E402
    ChromosomeGenotypes,
    _compact_pair_chunks_single_pass,
    _compact_pair_chunks_single_pass_bitpacked,
    _genetic_stop_bounds_impl,
    _pack_haplotypes_impl,
    calc_covariance,
    calc_covariance_from_genotypes,
    load_chromosome_genotypes,
    partition_chromosome,
)

DEFAULT_DATA_DIR = REPO_ROOT / "benchmarks/data/bitpacked_full_genome"
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks/results/bitpacked_full_genome"

VCF_BASE_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20110521"
VCF_TEMPLATE = (
    "ALL.chr{chrom}.phase1_release_v3.20101123."
    "snps_indels_svs.genotypes.vcf.gz"
)
PANEL_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20110521/"
    "phase1_integrated_calls.20101123.ALL.panel"
)
MAP_BASE_URL = (
    "https://raw.githubusercontent.com/joepickrell/"
    "1000-genomes-genetic-maps/master/interpolated_from_hapmap"
)
MAP_TEMPLATE = "chr{chrom}.interpolated_genetic_map.gz"

POPULATIONS: dict[str, dict[str, object]] = {
    "EUR": {"subpops": {"CEU", "TSI", "FIN", "GBR", "IBS"}, "ne": 11418.0},
    "AFR": {"subpops": {"YRI", "LWK", "ASW"}, "ne": 17469.0},
    "ASN": {"subpops": {"CHB", "JPT", "CHS"}, "ne": 14269.0},
}


@dataclass(frozen=True)
class PartitionResult:
    population: str
    chrom: str
    start: int
    end: int
    n_rows: int
    uint8_seconds: float
    bitpacked_seconds: float
    speedup: float
    chromosome_bitpacked_seconds: float
    chromosome_load_seconds: float
    chromosome_speedup_vs_partition_bitpacked: float
    uint8_peak_rss_mib: float
    bitpacked_peak_rss_mib: float
    chromosome_load_peak_rss_mib: float
    chromosome_peak_rss_mib: float
    chromosome_uint8_bytes: int
    chromosome_packed_bytes: int
    uint8_prepare_seconds: float
    uint8_vcf_seconds: float
    uint8_array_seconds: float
    uint8_pack_seconds: float
    uint8_chunk_seconds: float
    uint8_write_io_seconds: float
    uint8_write_total_seconds: float
    bitpacked_prepare_seconds: float
    bitpacked_vcf_seconds: float
    bitpacked_array_seconds: float
    bitpacked_pack_seconds: float
    bitpacked_chunk_seconds: float
    bitpacked_write_io_seconds: float
    bitpacked_write_total_seconds: float
    chromosome_slice_seconds: float
    chromosome_bounds_seconds: float
    chromosome_chunk_seconds: float
    chromosome_write_io_seconds: float
    chromosome_write_total_seconds: float
    uint8_bytes: int
    bitpacked_bytes: int
    chromosome_bitpacked_bytes: int
    byte_ratio: float
    exact: bool
    chromosome_exact: bool
    max_abs_diff: float
    chromosome_max_abs_diff: float


class RssSampler:
    """Sample current RSS in a background thread during one benchmark section."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_mib: float = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> RssSampler:
        self._sample_once()
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample_once()

    def _sample_once(self) -> None:
        rss = current_rss_mib()
        if rss is None:
            rss = max_rss_mib()
        if rss is not None:
            self.peak_mib = max(self.peak_mib, rss)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample_once()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        choices=sorted(POPULATIONS),
        default="EUR",
        help="1000 Genomes population grouping to benchmark (default: EUR).",
    )
    parser.add_argument(
        "--chromosomes",
        nargs="+",
        default=[str(i) for i in range(1, 23)],
        help="Chromosomes to run, without the 'chr' prefix (default: 1..22).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Download/preparation cache (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Benchmark output directory (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5000,
        help="Target SNPs per partition passed to partition_chromosome.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=1e-7,
        help="Wen/Stephens covariance cutoff (default: 1e-7).",
    )
    parser.add_argument(
        "--ne",
        type=float,
        default=None,
        help="Effective population size override; defaults by population.",
    )
    parser.add_argument(
        "--compact-chunk-rows",
        type=int,
        default=1_000_000,
        help="Compact HDF5 write chunk target passed to calc_covariance.",
    )
    parser.add_argument(
        "--compression",
        choices=("zstd", "lzf"),
        default="zstd",
        help="HDF5 compression for both outputs (default: zstd).",
    )
    parser.add_argument(
        "--max-partitions-per-chrom",
        type=int,
        default=None,
        help="Smoke-test limiter; omit for full chromosome coverage.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Refresh cached panel, maps, raw VCFs, and indexes.",
    )
    parser.add_argument(
        "--force-filter",
        action="store_true",
        help="Regenerate filtered population VCFs even when cached.",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep per-partition uint8/bitpacked HDF5 files after comparison.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the synthetic Numba warmup before partition timing.",
    )
    parser.add_argument(
        "--include-chromosome-mode",
        action="store_true",
        help=(
            "Also load each chromosome once and benchmark bitpacked compact "
            "partition writes from prepared chromosome arrays."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_tools(["bcftools", "tabix"])
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    population = args.population
    pop_meta = POPULATIONS[population]
    ne = float(args.ne if args.ne is not None else pop_meta["ne"])

    if not args.no_warmup:
        warmup_ld_kernels()

    panel_path = args.data_dir / "resources" / Path(PANEL_URL).name
    download(PANEL_URL, panel_path, force=args.force_download)

    results: list[PartitionResult] = []
    for chrom in args.chromosomes:
        chrom_results = run_chromosome(
            chrom=str(chrom),
            population=population,
            subpops=set(pop_meta["subpops"]),
            panel_path=panel_path,
            ne=ne,
            args=args,
        )
        results.extend(chrom_results)
        write_outputs(args.results_dir, population, results)

    write_outputs(args.results_dir, population, results)
    print_summary(results)
    return 0 if all(result.exact for result in results) else 2


def require_tools(names: list[str]) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"Missing required tool(s) on PATH: {', '.join(missing)}")


def warmup_ld_kernels() -> None:
    rng = np.random.default_rng(20260711)
    hap_mat = rng.integers(0, 2, size=(16, 256), dtype=np.uint8)
    gpos_arr = np.cumsum(rng.uniform(0.0005, 0.01, size=hap_mat.shape[0]))
    hap_sums = np.asarray(hap_mat.sum(axis=1), dtype=np.float64)
    pos_arr = np.arange(100, 100 + hap_mat.shape[0] * 10, 10, dtype=np.int32)
    n_ind = hap_mat.shape[1] / 2.0
    theta = 0.01
    cutoff = 1e-7
    j_stop_by_i = _genetic_stop_bounds_impl(gpos_arr, 11418.0, n_ind, cutoff)
    list(
        _compact_pair_chunks_single_pass(
            hap_mat,
            gpos_arr,
            hap_sums,
            j_stop_by_i,
            pos_arr,
            11418.0,
            n_ind,
            theta,
            cutoff,
            1000,
        )
    )
    packed = _pack_haplotypes_impl(hap_mat)
    list(
        _compact_pair_chunks_single_pass_bitpacked(
            packed,
            gpos_arr,
            hap_sums,
            j_stop_by_i,
            pos_arr,
            hap_mat.shape[1],
            11418.0,
            n_ind,
            theta,
            cutoff,
            1000,
        )
    )


def run_chromosome(
    *,
    chrom: str,
    population: str,
    subpops: set[str],
    panel_path: Path,
    ne: float,
    args: argparse.Namespace,
) -> list[PartitionResult]:
    print(f"Preparing chr{chrom} {population}", flush=True)
    raw_vcf = ensure_raw_vcf(args.data_dir, chrom, force=args.force_download)
    map_path = ensure_map(args.data_dir, chrom, force=args.force_download)
    individuals_path = ensure_population_individuals(
        args.data_dir,
        population,
        chrom,
        subpops,
        panel_path,
        raw_vcf,
    )
    filtered_vcf = ensure_filtered_vcf(
        args.data_dir,
        population,
        chrom,
        raw_vcf,
        individuals_path,
        force=args.force_filter,
    )
    partitions_path = ensure_partitions(
        args.data_dir,
        population,
        chrom,
        map_path,
        individuals_path,
        ne,
        args.window_size,
    )
    partitions = read_partitions(partitions_path)
    if args.max_partitions_per_chrom is not None:
        partitions = partitions[: args.max_partitions_per_chrom]

    chromosome_genotypes: ChromosomeGenotypes | None = None
    chromosome_load_seconds = 0.0
    chromosome_load_peak_rss_mib = 0.0
    chromosome_uint8_bytes = 0
    chromosome_packed_bytes = 0
    if args.include_chromosome_mode:
        print(f"  Loading chr{chrom} {population} once for chromosome mode", flush=True)
        load_profile: dict[str, float] = {}
        load_start = time.perf_counter()
        with RssSampler() as sampler:
            chromosome_genotypes = load_chromosome_genotypes(
                vcf_path=filtered_vcf,
                chrom=chrom,
                genetic_map_path=map_path,
                individuals_path=individuals_path,
                storage="packed",
                profile=load_profile,
            )
        chromosome_load_seconds = time.perf_counter() - load_start
        chromosome_load_peak_rss_mib = sampler.peak_mib
        chromosome_uint8_bytes = int(load_profile.get("uint8_bytes", 0.0))
        chromosome_packed_bytes = int(load_profile.get("packed_bytes", 0.0))
        print(
            "  chromosome load "
            f"n_snps={int(load_profile.get('n_snps', 0.0))} "
            f"seconds={chromosome_load_seconds:.3f} "
            f"peak_rss_mib={chromosome_load_peak_rss_mib:.1f}",
            flush=True,
        )

    results: list[PartitionResult] = []
    for idx, (start, end) in enumerate(partitions, start=1):
        print(
            f"  chr{chrom} partition {idx}/{len(partitions)} {start}-{end}",
            flush=True,
        )
        results.append(
            benchmark_partition(
                population=population,
                chrom=chrom,
                start=start,
                end=end,
                filtered_vcf=filtered_vcf,
                map_path=map_path,
                individuals_path=individuals_path,
                ne=ne,
                args=args,
                chromosome_genotypes=chromosome_genotypes,
                chromosome_load_seconds=(
                    chromosome_load_seconds if idx == 1 else 0.0
                ),
                chromosome_load_peak_rss_mib=(
                    chromosome_load_peak_rss_mib if idx == 1 else 0.0
                ),
                chromosome_uint8_bytes=chromosome_uint8_bytes if idx == 1 else 0,
                chromosome_packed_bytes=chromosome_packed_bytes if idx == 1 else 0,
            )
        )
    return results


def ensure_raw_vcf(data_dir: Path, chrom: str, *, force: bool) -> Path:
    raw_dir = data_dir / "raw"
    filename = VCF_TEMPLATE.format(chrom=chrom)
    raw_vcf = raw_dir / filename
    download(f"{VCF_BASE_URL}/{filename}", raw_vcf, force=force)
    download(
        f"{VCF_BASE_URL}/{filename}.tbi",
        raw_vcf.with_suffix(raw_vcf.suffix + ".tbi"),
        force=force,
    )
    return raw_vcf


def ensure_map(data_dir: Path, chrom: str, *, force: bool) -> Path:
    maps_dir = data_dir / "maps"
    filename = MAP_TEMPLATE.format(chrom=chrom)
    path = maps_dir / filename
    download(f"{MAP_BASE_URL}/{filename}", path, force=force)
    return path


def download(url: str, path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except urllib.error.URLError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url} -> {path}: {exc}") from exc
    tmp_path.replace(path)


def ensure_population_individuals(
    data_dir: Path,
    population: str,
    chrom: str,
    subpops: set[str],
    panel_path: Path,
    raw_vcf: Path,
) -> Path:
    resources_dir = data_dir / "resources" / population
    resources_dir.mkdir(parents=True, exist_ok=True)
    individuals_path = resources_dir / f"{population}.chr{chrom}.individuals.txt"

    panel_samples = read_panel_samples(panel_path, subpops)
    vcf_samples = set(query_vcf_samples(raw_vcf))
    samples = [sample for sample in panel_samples if sample in vcf_samples]
    if not samples:
        raise RuntimeError(f"No {population} panel samples found in {raw_vcf}")
    individuals_path.write_text("\n".join(samples) + "\n")
    return individuals_path


def read_panel_samples(panel_path: Path, subpops: set[str]) -> list[str]:
    samples: list[str] = []
    with open(panel_path) as f:
        for raw in f:
            parts = raw.strip().split()
            if not parts or parts[0].lower() == "sample":
                continue
            if len(parts) < 2:
                continue
            if parts[1] in subpops:
                samples.append(parts[0])
    return samples


def query_vcf_samples(vcf_path: Path) -> list[str]:
    proc = subprocess.run(
        ["bcftools", "query", "-l", str(vcf_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in proc.stdout.splitlines() if line]


def ensure_filtered_vcf(
    data_dir: Path,
    population: str,
    chrom: str,
    raw_vcf: Path,
    individuals_path: Path,
    *,
    force: bool,
) -> Path:
    filtered_dir = data_dir / "filtered" / population
    filtered_dir.mkdir(parents=True, exist_ok=True)
    filtered_vcf = filtered_dir / f"chr{chrom}.{population}.biallelic.mac1.vcf.gz"
    if (
        filtered_vcf.exists()
        and filtered_vcf.with_suffix(filtered_vcf.suffix + ".tbi").exists()
        and not force
    ):
        return filtered_vcf

    tmp_path = filtered_vcf.with_suffix(filtered_vcf.suffix + ".tmp")
    p1 = subprocess.Popen(
        ["bcftools", "view", "-S", str(individuals_path), "-Ou", str(raw_vcf)],
        stdout=subprocess.PIPE,
    )
    assert p1.stdout is not None
    p2 = subprocess.run(
        [
            "bcftools",
            "view",
            "-i",
            "MAC[0]>=1",
            "-m2",
            "-M2",
            "-Oz",
            "-o",
            str(tmp_path),
            "-",
        ],
        stdin=p1.stdout,
    )
    p1.stdout.close()
    p1_return = p1.wait()
    if p1_return != 0 or p2.returncode != 0:
        raise RuntimeError(f"bcftools filtering failed for chr{chrom} {population}")
    tmp_path.replace(filtered_vcf)
    subprocess.run(["tabix", "-f", "-p", "vcf", str(filtered_vcf)], check=True)
    return filtered_vcf


def ensure_partitions(
    data_dir: Path,
    population: str,
    chrom: str,
    map_path: Path,
    individuals_path: Path,
    ne: float,
    window_size: int,
) -> Path:
    partitions_dir = data_dir / "partitions" / population
    partitions_dir.mkdir(parents=True, exist_ok=True)
    partitions_path = partitions_dir / f"chr{chrom}.partitions.txt"
    n_individuals = sum(1 for line in individuals_path.read_text().splitlines() if line)
    partition_chromosome(
        genetic_map_path=map_path,
        n_individuals=n_individuals,
        output_path=partitions_path,
        window_size=window_size,
        ne=ne,
    )
    return partitions_path


def read_partitions(path: Path) -> list[tuple[int, int]]:
    partitions: list[tuple[int, int]] = []
    with open(path) as f:
        for raw in f:
            parts = raw.strip().split()
            if parts:
                partitions.append((int(parts[0]), int(parts[1])))
    return partitions


def benchmark_partition(
    *,
    population: str,
    chrom: str,
    start: int,
    end: int,
    filtered_vcf: Path,
    map_path: Path,
    individuals_path: Path,
    ne: float,
    args: argparse.Namespace,
    chromosome_genotypes: ChromosomeGenotypes | None,
    chromosome_load_seconds: float,
    chromosome_load_peak_rss_mib: float,
    chromosome_uint8_bytes: int,
    chromosome_packed_bytes: int,
) -> PartitionResult:
    out_dir = args.results_dir / population / f"chr{chrom}" / "partitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    uint8_path = out_dir / f"chr{chrom}.{start}.{end}.uint8.h5"
    bitpacked_path = out_dir / f"chr{chrom}.{start}.{end}.bitpacked.h5"
    uint8_path.unlink(missing_ok=True)
    bitpacked_path.unlink(missing_ok=True)

    uint8_seconds, uint8_profile, uint8_peak_rss_mib = time_calc_covariance(
        vcf_path=filtered_vcf,
        region=f"{chrom}:{start}-{end}",
        map_path=map_path,
        individuals_path=individuals_path,
        output_path=uint8_path,
        ne=ne,
        cutoff=args.cutoff,
        compact_chunk_rows=args.compact_chunk_rows,
        compression=args.compression,
        ld_kernel="uint8",
    )
    bitpacked_seconds, bitpacked_profile, bitpacked_peak_rss_mib = time_calc_covariance(
        vcf_path=filtered_vcf,
        region=f"{chrom}:{start}-{end}",
        map_path=map_path,
        individuals_path=individuals_path,
        output_path=bitpacked_path,
        ne=ne,
        cutoff=args.cutoff,
        compact_chunk_rows=args.compact_chunk_rows,
        compression=args.compression,
        ld_kernel="bitpacked",
    )
    chromosome_bitpacked_seconds = 0.0
    chromosome_peak_rss_mib = 0.0
    chromosome_profile: dict[str, float] = {}
    chromosome_bitpacked_path = out_dir / f"chr{chrom}.{start}.{end}.chromosome.h5"
    chromosome_bitpacked_path.unlink(missing_ok=True)
    if chromosome_genotypes is not None:
        chromosome_bitpacked_seconds, chromosome_profile, chromosome_peak_rss_mib = (
            time_calc_covariance_from_genotypes(
                genotypes=chromosome_genotypes,
                start=start,
                end=end,
                output_path=chromosome_bitpacked_path,
                ne=ne,
                cutoff=args.cutoff,
                compact_chunk_rows=args.compact_chunk_rows,
                compression=args.compression,
            )
        )
    exact, n_rows, max_abs_diff = compare_outputs(
        uint8_path,
        bitpacked_path,
        start,
        end,
    )
    if chromosome_genotypes is not None:
        chromosome_exact, _, chromosome_max_abs_diff = compare_outputs(
            bitpacked_path,
            chromosome_bitpacked_path,
            start,
            end,
        )
    else:
        chromosome_exact = True
        chromosome_max_abs_diff = 0.0
    uint8_bytes = uint8_path.stat().st_size if uint8_path.exists() else 0
    bitpacked_bytes = bitpacked_path.stat().st_size if bitpacked_path.exists() else 0
    chromosome_bitpacked_bytes = (
        chromosome_bitpacked_path.stat().st_size
        if chromosome_bitpacked_path.exists()
        else 0
    )

    if not args.keep_outputs:
        uint8_path.unlink(missing_ok=True)
        bitpacked_path.unlink(missing_ok=True)
        chromosome_bitpacked_path.unlink(missing_ok=True)

    return PartitionResult(
        population=population,
        chrom=chrom,
        start=start,
        end=end,
        n_rows=n_rows,
        uint8_seconds=uint8_seconds,
        bitpacked_seconds=bitpacked_seconds,
        speedup=(
            uint8_seconds / bitpacked_seconds if bitpacked_seconds else float("inf")
        ),
        chromosome_bitpacked_seconds=chromosome_bitpacked_seconds,
        chromosome_load_seconds=chromosome_load_seconds,
        chromosome_speedup_vs_partition_bitpacked=(
            bitpacked_seconds / chromosome_bitpacked_seconds
            if chromosome_bitpacked_seconds
            else 0.0
        ),
        uint8_peak_rss_mib=uint8_peak_rss_mib,
        bitpacked_peak_rss_mib=bitpacked_peak_rss_mib,
        chromosome_load_peak_rss_mib=chromosome_load_peak_rss_mib,
        chromosome_peak_rss_mib=chromosome_peak_rss_mib,
        chromosome_uint8_bytes=chromosome_uint8_bytes,
        chromosome_packed_bytes=chromosome_packed_bytes,
        uint8_prepare_seconds=profile_value(uint8_profile, "prepare_seconds"),
        uint8_vcf_seconds=profile_value(uint8_profile, "vcf_seconds"),
        uint8_array_seconds=profile_value(uint8_profile, "array_seconds"),
        uint8_pack_seconds=profile_value(uint8_profile, "pack_seconds"),
        uint8_chunk_seconds=profile_value(uint8_profile, "chunk_seconds"),
        uint8_write_io_seconds=profile_value(uint8_profile, "write_io_seconds"),
        uint8_write_total_seconds=profile_value(
            uint8_profile, "write_total_seconds"
        ),
        bitpacked_prepare_seconds=profile_value(bitpacked_profile, "prepare_seconds"),
        bitpacked_vcf_seconds=profile_value(bitpacked_profile, "vcf_seconds"),
        bitpacked_array_seconds=profile_value(bitpacked_profile, "array_seconds"),
        bitpacked_pack_seconds=profile_value(bitpacked_profile, "pack_seconds"),
        bitpacked_chunk_seconds=profile_value(bitpacked_profile, "chunk_seconds"),
        bitpacked_write_io_seconds=profile_value(
            bitpacked_profile, "write_io_seconds"
        ),
        bitpacked_write_total_seconds=profile_value(
            bitpacked_profile, "write_total_seconds"
        ),
        chromosome_slice_seconds=profile_value(chromosome_profile, "slice_seconds"),
        chromosome_bounds_seconds=profile_value(chromosome_profile, "bounds_seconds"),
        chromosome_chunk_seconds=profile_value(chromosome_profile, "chunk_seconds"),
        chromosome_write_io_seconds=profile_value(
            chromosome_profile, "write_io_seconds"
        ),
        chromosome_write_total_seconds=profile_value(
            chromosome_profile, "write_total_seconds"
        ),
        uint8_bytes=uint8_bytes,
        bitpacked_bytes=bitpacked_bytes,
        chromosome_bitpacked_bytes=chromosome_bitpacked_bytes,
        byte_ratio=bitpacked_bytes / uint8_bytes if uint8_bytes else float("nan"),
        exact=exact,
        chromosome_exact=chromosome_exact,
        max_abs_diff=max_abs_diff,
        chromosome_max_abs_diff=chromosome_max_abs_diff,
    )


def profile_value(profile: dict[str, float], key: str) -> float:
    return float(profile.get(key, 0.0))


def time_calc_covariance(
    *,
    vcf_path: Path,
    region: str,
    map_path: Path,
    individuals_path: Path,
    output_path: Path,
    ne: float,
    cutoff: float,
    compact_chunk_rows: int,
    compression: str,
    ld_kernel: str,
) -> tuple[float, dict[str, float], float]:
    profile: dict[str, float] = {}
    start_time = time.perf_counter()
    with RssSampler() as sampler:
        calc_covariance(
            vcf_path=vcf_path,
            region=region,
            genetic_map_path=map_path,
            individuals_path=individuals_path,
            output_path=output_path,
            ne=ne,
            cutoff=cutoff,
            compact_output=True,
            compact_chunk_rows=compact_chunk_rows,
            compression=compression,
            ld_kernel=ld_kernel,
            profile=profile,
        )
    seconds = time.perf_counter() - start_time
    profile.setdefault("total_seconds", seconds)
    return seconds, profile, sampler.peak_mib


def time_calc_covariance_from_genotypes(
    *,
    genotypes: ChromosomeGenotypes,
    start: int,
    end: int,
    output_path: Path,
    ne: float,
    cutoff: float,
    compact_chunk_rows: int,
    compression: str,
) -> tuple[float, dict[str, float], float]:
    profile: dict[str, float] = {}
    start_time = time.perf_counter()
    with RssSampler() as sampler:
        calc_covariance_from_genotypes(
            genotypes,
            start,
            end,
            output_path,
            ne=ne,
            cutoff=cutoff,
            compact_chunk_rows=compact_chunk_rows,
            compression=compression,
            ld_kernel="bitpacked",
            profile=profile,
        )
    seconds = time.perf_counter() - start_time
    profile.setdefault("total_seconds", seconds)
    return seconds, profile, sampler.peak_mib


def compare_outputs(
    uint8_path: Path,
    bitpacked_path: Path,
    start: int,
    end: int,
) -> tuple[bool, int, float]:
    if not uint8_path.exists() or not bitpacked_path.exists():
        return (
            not uint8_path.exists() and not bitpacked_path.exists(),
            0,
            float("nan"),
        )

    with open_covariance_reader(uint8_path, start, end) as reader:
        uint8_rows = reader.read_all()
        uint8_diag = reader.read_diagonal()
        uint8_loci = reader.read_loci()
    with open_covariance_reader(bitpacked_path, start, end) as reader:
        bitpacked_rows = reader.read_all()
        bitpacked_diag = reader.read_diagonal()
        bitpacked_loci = reader.read_loci()

    same_keys = np.array_equal(uint8_rows.lo, bitpacked_rows.lo) and np.array_equal(
        uint8_rows.hi, bitpacked_rows.hi
    )
    same_values = np.array_equal(uint8_rows.shrink_ld, bitpacked_rows.shrink_ld)
    same_diag = np.array_equal(uint8_diag[0], bitpacked_diag[0]) and np.array_equal(
        uint8_diag[1], bitpacked_diag[1]
    )
    same_loci = np.array_equal(uint8_loci, bitpacked_loci)
    if uint8_rows.shrink_ld.shape == bitpacked_rows.shrink_ld.shape:
        max_abs_diff = float(
            np.max(np.abs(uint8_rows.shrink_ld - bitpacked_rows.shrink_ld))
        ) if uint8_rows.shrink_ld.size else 0.0
    else:
        max_abs_diff = float("inf")
    return (
        bool(same_keys and same_values and same_diag and same_loci),
        int(uint8_rows.shrink_ld.size),
        max_abs_diff,
    )


def write_outputs(
    results_dir: Path,
    population: str,
    results: list[PartitionResult],
) -> None:
    tsv_path = results_dir / f"{population}.bitpacked_full_genome.tsv"
    json_path = results_dir / f"{population}.bitpacked_full_genome.summary.json"
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PartitionResult.__dataclass_fields__)
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)
    json_path.write_text(json.dumps(summary_dict(results), indent=2) + "\n")


def summary_dict(results: list[PartitionResult]) -> dict[str, object]:
    total_uint8 = sum(result.uint8_seconds for result in results)
    total_bitpacked = sum(result.bitpacked_seconds for result in results)
    total_chromosome_bitpacked = sum(
        result.chromosome_bitpacked_seconds for result in results
    )
    total_chromosome_load = sum(result.chromosome_load_seconds for result in results)
    total_uint8_bytes = sum(result.uint8_bytes for result in results)
    total_bitpacked_bytes = sum(result.bitpacked_bytes for result in results)
    total_chromosome_bitpacked_bytes = sum(
        result.chromosome_bitpacked_bytes for result in results
    )
    total_chromosome_uint8_bytes = sum(
        result.chromosome_uint8_bytes for result in results
    )
    total_chromosome_packed_bytes = sum(
        result.chromosome_packed_bytes for result in results
    )
    total_uint8_prepare = sum(result.uint8_prepare_seconds for result in results)
    total_bitpacked_prepare = sum(
        result.bitpacked_prepare_seconds for result in results
    )
    total_uint8_vcf = sum(result.uint8_vcf_seconds for result in results)
    total_bitpacked_vcf = sum(result.bitpacked_vcf_seconds for result in results)
    total_uint8_array = sum(result.uint8_array_seconds for result in results)
    total_bitpacked_array = sum(result.bitpacked_array_seconds for result in results)
    total_uint8_pack = sum(result.uint8_pack_seconds for result in results)
    total_bitpacked_pack = sum(result.bitpacked_pack_seconds for result in results)
    total_uint8_chunk = sum(result.uint8_chunk_seconds for result in results)
    total_bitpacked_chunk = sum(
        result.bitpacked_chunk_seconds for result in results
    )
    total_uint8_write_io = sum(result.uint8_write_io_seconds for result in results)
    total_bitpacked_write_io = sum(
        result.bitpacked_write_io_seconds for result in results
    )
    total_chromosome_chunk = sum(
        result.chromosome_chunk_seconds for result in results
    )
    total_chromosome_write_io = sum(
        result.chromosome_write_io_seconds for result in results
    )
    peak_uint8_rss = max((result.uint8_peak_rss_mib for result in results), default=0.0)
    peak_bitpacked_rss = max(
        (result.bitpacked_peak_rss_mib for result in results), default=0.0
    )
    peak_chromosome_load_rss = max(
        (result.chromosome_load_peak_rss_mib for result in results), default=0.0
    )
    peak_chromosome_rss = max(
        (result.chromosome_peak_rss_mib for result in results), default=0.0
    )
    return {
        "n_partitions": len(results),
        "all_exact": all(result.exact for result in results),
        "chromosome_all_exact": all(result.chromosome_exact for result in results),
        "total_rows": sum(result.n_rows for result in results),
        "total_uint8_seconds": total_uint8,
        "total_bitpacked_seconds": total_bitpacked,
        "overall_speedup": total_uint8 / total_bitpacked if total_bitpacked else None,
        "total_chromosome_bitpacked_seconds": total_chromosome_bitpacked,
        "total_chromosome_load_seconds": total_chromosome_load,
        "chromosome_speedup_vs_partition_bitpacked": (
            total_bitpacked / (total_chromosome_load + total_chromosome_bitpacked)
            if total_chromosome_load + total_chromosome_bitpacked
            else None
        ),
        "total_uint8_bytes": total_uint8_bytes,
        "total_bitpacked_bytes": total_bitpacked_bytes,
        "total_chromosome_bitpacked_bytes": total_chromosome_bitpacked_bytes,
        "total_chromosome_uint8_cache_bytes": total_chromosome_uint8_bytes,
        "total_chromosome_packed_cache_bytes": total_chromosome_packed_bytes,
        "overall_byte_ratio": (
            total_bitpacked_bytes / total_uint8_bytes if total_uint8_bytes else None
        ),
        "peak_uint8_rss_mib": peak_uint8_rss,
        "peak_bitpacked_rss_mib": peak_bitpacked_rss,
        "peak_chromosome_load_rss_mib": peak_chromosome_load_rss,
        "peak_chromosome_partition_rss_mib": peak_chromosome_rss,
        "process_max_rss_mib": max_rss_mib(),
        "total_uint8_prepare_seconds": total_uint8_prepare,
        "total_bitpacked_prepare_seconds": total_bitpacked_prepare,
        "total_uint8_vcf_seconds": total_uint8_vcf,
        "total_bitpacked_vcf_seconds": total_bitpacked_vcf,
        "total_uint8_array_seconds": total_uint8_array,
        "total_bitpacked_array_seconds": total_bitpacked_array,
        "total_uint8_pack_seconds": total_uint8_pack,
        "total_bitpacked_pack_seconds": total_bitpacked_pack,
        "total_uint8_chunk_seconds": total_uint8_chunk,
        "total_bitpacked_chunk_seconds": total_bitpacked_chunk,
        "chunk_speedup": (
            total_uint8_chunk / total_bitpacked_chunk
            if total_bitpacked_chunk
            else None
        ),
        "total_uint8_write_io_seconds": total_uint8_write_io,
        "total_bitpacked_write_io_seconds": total_bitpacked_write_io,
        "total_chromosome_chunk_seconds": total_chromosome_chunk,
        "total_chromosome_write_io_seconds": total_chromosome_write_io,
        "max_abs_diff": max((result.max_abs_diff for result in results), default=0.0),
        "chromosome_max_abs_diff": max(
            (result.chromosome_max_abs_diff for result in results),
            default=0.0,
        ),
    }


def print_summary(results: list[PartitionResult]) -> None:
    summary = summary_dict(results)
    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
