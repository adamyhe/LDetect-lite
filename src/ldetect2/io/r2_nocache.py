"""On-demand normalized r2 row streams without a persisted pair cache."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ldetect2.io.r2_zarr import R2RowChunk


@dataclass(frozen=True)
class R2NoCacheConfig:
    """Inputs needed to recompute normalized r2 rows for one chromosome."""

    reference_panel: str
    genetic_map_path: Path
    individuals_path: Path
    chrom: str
    ne: float = 11418.0
    cutoff: float = 1e-7


@dataclass
class R2NoCacheProfile:
    """Diagnostics for no-cache VCF decode, prep, and row recomputation."""

    vcf_query_count: int = 0
    vcf_query_bp: int = 0
    vcf_records_seen: int = 0
    vcf_records_retained: int = 0
    vcf_records_skipped: int = 0
    cyvcf2_query_count: int = 0
    tabix_query_count: int = 0
    cyvcf2_fallback_count: int = 0
    vcf_fetch_seconds: float = 0.0
    vcf_decode_seconds: float = 0.0
    array_prep_seconds: float = 0.0
    row_generation_seconds: float = 0.0
    duplicate_fallback_seconds: float = 0.0
    duplicate_fallback_partitions: int = 0
    dosage_cache_hits: int = 0
    dosage_cache_misses: int = 0
    dosage_cache_evictions: int = 0
    dosage_cache_bytes: int = 0
    ld_compute_seconds: float = 0.0
    tile_count: int = 0
    max_tile_snps: int = 0
    pair_candidates: int = 0
    pairs_after_cutoff: int = 0

    def absorb(self, other: R2NoCacheProfile) -> None:
        """Add another profile into this profile."""
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def delta(self, before: R2NoCacheProfile) -> R2NoCacheProfile:
        """Return the difference between this profile and an earlier snapshot."""
        out = R2NoCacheProfile()
        for name in self.__dataclass_fields__:
            setattr(out, name, getattr(self, name) - getattr(before, name))
        return out

    def copy(self) -> R2NoCacheProfile:
        """Return a detached snapshot of this profile."""
        out = R2NoCacheProfile()
        out.absorb(self)
        return out

    def log_fields(self, prefix: str = "nocache_") -> str:
        """Return stable key/value fields for debug profiling logs."""
        parts = []
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, float):
                parts.append(f"{prefix}{name}={value:.6f}")
            else:
                parts.append(f"{prefix}{name}={value}")
        return " ".join(parts)


@dataclass(frozen=True)
class R2NoCachePreparedPartition:
    """Decoded partition inputs reused by no-cache r2 row generation."""

    config: R2NoCacheConfig
    start: int
    end: int
    hap_mat: np.ndarray
    gpos_arr: np.ndarray
    hap_sums: np.ndarray
    pos_arr: np.ndarray
    j_stop_by_i: np.ndarray
    diag_shrink: np.ndarray
    theta: float
    n_ind: int
    has_duplicate_positions: bool
    profile: R2NoCacheProfile = field(default_factory=R2NoCacheProfile)


class R2NoCachePartitionReader:
    """Reader-shaped wrapper that recomputes normalized r2 rows on demand."""

    def __init__(
        self,
        config: R2NoCacheConfig,
        start: int,
        end: int,
        *,
        use_cyvcf2: bool = True,
        prepared: R2NoCachePreparedPartition | None = None,
    ) -> None:
        self.config = config
        self.start = int(start)
        self.end = int(end)
        self.use_cyvcf2 = use_cyvcf2
        self._prepared = prepared
        self._loci: np.ndarray | None = None
        self._row_count: int | None = None

    def __enter__(self) -> R2NoCachePartitionReader:
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    @property
    def row_count(self) -> int:
        if self._row_count is None:
            self._ensure_index()
        return int(self._row_count)

    def read_loci(self) -> np.ndarray:
        """Return sorted unique lower-endpoint loci for this partition."""
        if self._loci is None:
            self._ensure_index()
        return np.asarray(self._loci, dtype=np.int64)

    def iter_rows(
        self,
        lo_min: int,
        lo_max: int,
        chunk_rows: int,
    ) -> Iterator[R2RowChunk]:
        """Yield bounded row chunks whose ``lo`` values are in range."""
        for chunk in self._iter_all_rows(chunk_rows):
            mask = (chunk.lo >= lo_min) & (chunk.lo <= lo_max)
            if np.any(mask):
                yield R2RowChunk(
                    lo=chunk.lo[mask],
                    hi=chunk.hi[mask],
                    r2=chunk.r2[mask],
                )

    def iter_owned_rows(
        self,
        lower_min: int,
        lower_max: int,
        snp_first: int,
        snp_last: int,
        chunk_rows: int,
        include_lower_min: bool = True,
    ) -> Iterator[R2RowChunk]:
        """Yield chunks filtered to the metric/vector ownership window."""
        for chunk in self.iter_rows(lower_min, lower_max, chunk_rows):
            lower_owned = (
                chunk.lo >= lower_min if include_lower_min else chunk.lo > lower_min
            )
            mask = (
                (chunk.lo >= snp_first)
                & (chunk.lo <= snp_last)
                & (chunk.hi >= snp_first)
                & (chunk.hi <= snp_last)
                & lower_owned
                & (chunk.lo <= lower_max)
            )
            if np.any(mask):
                yield R2RowChunk(
                    lo=chunk.lo[mask],
                    hi=chunk.hi[mask],
                    r2=chunk.r2[mask],
                )

    def _ensure_index(self) -> None:
        loci_parts: list[np.ndarray] = []
        row_count = 0
        for chunk in self._iter_all_rows(1_000_000):
            row_count += int(chunk.lo.size)
            if chunk.lo.size:
                loci_parts.append(chunk.lo)
        if loci_parts:
            self._loci = np.unique(np.concatenate(loci_parts)).astype(
                np.int64, copy=False
            )
        else:
            self._loci = np.array([], dtype=np.int64)
        self._row_count = row_count

    def _iter_all_rows(self, chunk_rows: int) -> Iterator[R2RowChunk]:
        if self._prepared is None:
            self._prepared = prepare_r2_nocache_partition(
                self.config,
                self.start,
                self.end,
                use_cyvcf2=self.use_cyvcf2,
            )
        return iter_prepared_r2_nocache_partition_rows(
            self._prepared,
            chunk_rows=chunk_rows,
        )


def open_r2_nocache_reader(
    config: R2NoCacheConfig,
    start: int,
    end: int,
    *,
    use_cyvcf2: bool = True,
    prepared: R2NoCachePreparedPartition | None = None,
) -> R2NoCachePartitionReader:
    """Return an on-demand normalized r2 reader for one partition."""
    return R2NoCachePartitionReader(
        config,
        start,
        end,
        use_cyvcf2=use_cyvcf2,
        prepared=prepared,
    )


def iter_r2_nocache_partition_rows(
    config: R2NoCacheConfig,
    start: int,
    end: int,
    *,
    chunk_rows: int = 1_000_000,
    use_cyvcf2: bool = True,
) -> Iterator[R2RowChunk]:
    """Recompute normalized r2 rows for one partition and yield row chunks."""
    prepared = prepare_r2_nocache_partition(
        config,
        start,
        end,
        use_cyvcf2=use_cyvcf2,
    )
    yield from iter_prepared_r2_nocache_partition_rows(
        prepared,
        chunk_rows=chunk_rows,
    )


def prepare_r2_nocache_partition(
    config: R2NoCacheConfig,
    start: int,
    end: int,
    *,
    use_cyvcf2: bool = True,
) -> R2NoCachePreparedPartition:
    """Decode VCF/BCF and precompute partition-local LD inputs once."""
    from ldetect2._util.logging import log_debug
    from ldetect2.shrinkage import (
        _diag_shrink_values_impl,
        _genetic_stop_bounds_impl,
        _has_duplicate_positions,
        _prepare_ld_arrays,
        _read_genetic_map,
        _read_individuals,
    )

    profile = R2NoCacheProfile()
    individuals = _read_individuals(config.individuals_path)
    pos2gpos = _read_genetic_map(config.genetic_map_path)
    if use_cyvcf2:
        try:
            all_pos, haps = _parse_cyvcf2_haplotypes(
                config.reference_panel,
                config.chrom,
                start,
                end,
                individuals,
                pos2gpos,
                profile,
            )
        except Exception:
            profile.cyvcf2_fallback_count += 1
            all_pos, haps = _parse_tabix_haplotypes(config, start, end, profile)
    else:
        all_pos, haps = _parse_tabix_haplotypes(config, start, end, profile)

    n_ind = len(individuals)
    n_haps = 2 * n_ind
    harmonic = sum(1.0 / i for i in range(1, n_haps))
    theta = (1.0 / harmonic) / (n_haps + 1.0 / harmonic)
    if not all_pos:
        return R2NoCachePreparedPartition(
            config=config,
            start=int(start),
            end=int(end),
            hap_mat=np.empty((0, 2 * n_ind), dtype=np.uint8),
            gpos_arr=np.array([], dtype=np.float64),
            hap_sums=np.array([], dtype=np.float64),
            pos_arr=np.array([], dtype=np.int32),
            j_stop_by_i=np.array([], dtype=np.int32),
            diag_shrink=np.array([], dtype=np.float64),
            theta=theta,
            n_ind=n_ind,
            has_duplicate_positions=False,
            profile=profile,
        )

    array_start = time.perf_counter()
    hap_mat, gpos_arr, hap_sums, pos_arr = _prepare_ld_arrays(
        all_pos,
        haps,
        pos2gpos,
    )
    j_stop_by_i = _genetic_stop_bounds_impl(
        gpos_arr, config.ne, float(n_ind), config.cutoff
    )
    diag_shrink = _diag_shrink_values_impl(hap_mat, hap_sums, theta, config.cutoff)
    profile.array_prep_seconds += time.perf_counter() - array_start
    has_duplicate_positions = _has_duplicate_positions(pos_arr)
    log_debug(
        "prepare_r2_nocache_partition profile "
        f"chrom={config.chrom} start={int(start)} end={int(end)} "
        f"n_snps={int(pos_arr.size)} n_haps={int(hap_mat.shape[1])} "
        f"positive_diagonal={int(np.count_nonzero(diag_shrink > 0.0))} "
        f"duplicate_positions={has_duplicate_positions} "
        f"{profile.log_fields()}"
    )
    return R2NoCachePreparedPartition(
        config=config,
        start=int(start),
        end=int(end),
        hap_mat=hap_mat,
        gpos_arr=gpos_arr,
        hap_sums=hap_sums,
        pos_arr=pos_arr,
        j_stop_by_i=j_stop_by_i,
        diag_shrink=diag_shrink,
        theta=theta,
        n_ind=n_ind,
        has_duplicate_positions=has_duplicate_positions,
        profile=profile,
    )


def iter_prepared_r2_nocache_partition_rows(
    prepared: R2NoCachePreparedPartition,
    *,
    chunk_rows: int = 1_000_000,
) -> Iterator[R2RowChunk]:
    """Yield normalized r2 rows from decoded partition inputs."""
    from ldetect2.shrinkage import (
        _compact_pair_chunks_single_pass,
        _compact_pair_chunks_by_physical_lo,
        _count_pairwise_ld_by_i_impl,
        _diag_from_first_physical_position,
        _r2_pair_chunks_from_canonical_stream,
        _r2_pair_chunks_from_covariance,
    )

    profile = prepared.profile
    if prepared.pos_arr.size == 0:
        return

    if prepared.has_duplicate_positions:
        row_start = time.perf_counter()
        row_counts = _count_pairwise_ld_by_i_impl(
            prepared.hap_mat,
            prepared.gpos_arr,
            prepared.hap_sums,
            prepared.j_stop_by_i,
            prepared.config.ne,
            float(prepared.n_ind),
            prepared.theta,
            prepared.config.cutoff,
        )
        diag_pos, diag_val = _diag_from_first_physical_position(
            prepared.pos_arr,
            prepared.diag_shrink,
        )
        positive_diag = diag_val > 0.0
        row_chunks = _r2_pair_chunks_from_canonical_stream(
            _compact_pair_chunks_by_physical_lo(
                prepared.hap_mat,
                prepared.gpos_arr,
                prepared.hap_sums,
                prepared.j_stop_by_i,
                prepared.pos_arr,
                row_counts,
                prepared.config.ne,
                float(prepared.n_ind),
                prepared.theta,
                prepared.config.cutoff,
                chunk_rows,
            ),
            diag_pos[positive_diag],
            diag_val[positive_diag],
        )
        profile.duplicate_fallback_partitions += 1
        while True:
            try:
                chunk = next(row_chunks)
            except StopIteration:
                elapsed = time.perf_counter() - row_start
                profile.duplicate_fallback_seconds += elapsed
                profile.row_generation_seconds += elapsed
                break
            profile.tile_count += 1
            profile.max_tile_snps = max(profile.max_tile_snps, int(chunk.lo.size))
            profile.pair_candidates += int(chunk.lo.size)
            profile.pairs_after_cutoff += int(np.count_nonzero(chunk.lo < chunk.hi))
            yield chunk
        return

    row_chunks = _r2_pair_chunks_from_covariance(
        _compact_pair_chunks_single_pass(
            prepared.hap_mat,
            prepared.gpos_arr,
            prepared.hap_sums,
            prepared.j_stop_by_i,
            prepared.pos_arr,
            prepared.config.ne,
            float(prepared.n_ind),
            prepared.theta,
            prepared.config.cutoff,
            chunk_rows,
        ),
        prepared.pos_arr,
        prepared.diag_shrink,
    )
    while True:
        row_start = time.perf_counter()
        try:
            chunk = next(row_chunks)
        except StopIteration:
            profile.row_generation_seconds += time.perf_counter() - row_start
            break
        elapsed = time.perf_counter() - row_start
        profile.row_generation_seconds += elapsed
        profile.ld_compute_seconds += elapsed
        profile.tile_count += 1
        profile.max_tile_snps = max(profile.max_tile_snps, int(chunk.lo.size))
        profile.pair_candidates += int(chunk.lo.size)
        profile.pairs_after_cutoff += int(np.count_nonzero(chunk.lo < chunk.hi))
        yield chunk


def _parse_cyvcf2_haplotypes(
    reference_panel: str,
    chrom: str,
    start: int,
    end: int,
    individuals: list[str],
    pos2gpos: dict[int, float],
    profile: R2NoCacheProfile,
) -> tuple[list[int], list[list[int]]]:
    try:
        from cyvcf2 import VCF
    except ModuleNotFoundError as exc:
        raise RuntimeError("cyvcf2 is not installed") from exc

    query_start = time.perf_counter()
    vcf = VCF(reference_panel, samples=individuals)
    all_pos: list[int] = []
    haps: list[list[int]] = []
    decode_seconds = 0.0
    profile.vcf_query_count += 1
    profile.cyvcf2_query_count += 1
    profile.vcf_query_bp += max(0, int(end) - int(start) + 1)
    try:
        if _has_vcf_index(reference_panel):
            variants = vcf(f"{chrom}:{start}-{end}")
        else:
            variants = vcf
        for variant in variants:
            profile.vcf_records_seen += 1
            if str(variant.CHROM) != str(chrom):
                profile.vcf_records_skipped += 1
                continue
            pos = int(variant.POS)
            if pos < start or pos > end or pos not in pos2gpos:
                profile.vcf_records_skipped += 1
                continue
            decode_start = time.perf_counter()
            row_haps = _cyvcf2_row_haps(variant.genotypes)
            decode_seconds += time.perf_counter() - decode_start
            if row_haps is None:
                profile.vcf_records_skipped += 1
                continue
            all_pos.append(pos)
            haps.append(row_haps)
            profile.vcf_records_retained += 1
    finally:
        vcf.close()
    profile.vcf_decode_seconds += decode_seconds
    profile.vcf_fetch_seconds += time.perf_counter() - query_start - decode_seconds
    return all_pos, haps


def _has_vcf_index(reference_panel: str) -> bool:
    path = Path(reference_panel)
    return (
        Path(f"{reference_panel}.tbi").exists()
        or Path(f"{reference_panel}.csi").exists()
        or path.with_suffix(path.suffix + ".tbi").exists()
        or path.with_suffix(path.suffix + ".csi").exists()
    )


def _cyvcf2_row_haps(genotypes) -> list[int] | None:
    row_haps: list[int] = []
    for gt in genotypes:
        if len(gt) < 3 or not gt[2] or gt[0] < 0 or gt[1] < 0:
            return None
        row_haps.append(int(gt[0]))
        row_haps.append(int(gt[1]))
    return row_haps


def _parse_tabix_haplotypes(
    config: R2NoCacheConfig,
    start: int,
    end: int,
    profile: R2NoCacheProfile,
) -> tuple[list[int], list[list[int]]]:
    from ldetect2.shrinkage import (
        _parse_vcf_haplotypes,
        _read_genetic_map,
        _read_individuals,
    )

    region = f"{config.chrom}:{start}-{end}"
    profile.vcf_query_count += 1
    profile.tabix_query_count += 1
    profile.vcf_query_bp += max(0, int(end) - int(start) + 1)
    query_start = time.perf_counter()
    try:
        proc = subprocess.Popen(
            ["tabix", "-h", config.reference_panel, region],
            stdout=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "tabix not found and cyvcf2 VCF reading was unavailable"
        ) from exc
    if proc.stdout is None:
        raise RuntimeError("tabix did not provide stdout")
    with proc.stdout:
        all_pos, _, haps = _parse_vcf_haplotypes(
            proc.stdout,
            _read_individuals(config.individuals_path),
            _read_genetic_map(config.genetic_map_path),
        )
    proc.wait()
    elapsed = time.perf_counter() - query_start
    profile.vcf_decode_seconds += elapsed
    profile.vcf_records_retained += len(all_pos)
    return all_pos, haps


def _split_r2_chunk(chunk: R2RowChunk, chunk_rows: int) -> Iterator[R2RowChunk]:
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, chunk.lo.size, chunk_rows):
        stop = min(start + chunk_rows, chunk.lo.size)
        yield R2RowChunk(
            lo=chunk.lo[start:stop],
            hi=chunk.hi[start:stop],
            r2=chunk.r2[start:stop],
        )
