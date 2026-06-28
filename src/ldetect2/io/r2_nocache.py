"""On-demand normalized r2 row streams without a persisted pair cache."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
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
    from ldetect2.shrinkage import (
        _diag_shrink_values_impl,
        _genetic_stop_bounds_impl,
        _has_duplicate_positions,
        _prepare_ld_arrays,
        _read_genetic_map,
        _read_individuals,
    )

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
            )
        except Exception:
            all_pos, haps = _parse_tabix_haplotypes(config, start, end)
    else:
        all_pos, haps = _parse_tabix_haplotypes(config, start, end)

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
        )

    hap_mat, gpos_arr, hap_sums, pos_arr = _prepare_ld_arrays(
        all_pos,
        haps,
        pos2gpos,
    )
    j_stop_by_i = _genetic_stop_bounds_impl(
        gpos_arr, config.ne, float(n_ind), config.cutoff
    )
    diag_shrink = _diag_shrink_values_impl(hap_mat, hap_sums, theta, config.cutoff)
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
        has_duplicate_positions=_has_duplicate_positions(pos_arr),
    )


def iter_prepared_r2_nocache_partition_rows(
    prepared: R2NoCachePreparedPartition,
    *,
    chunk_rows: int = 1_000_000,
) -> Iterator[R2RowChunk]:
    """Yield normalized r2 rows from decoded partition inputs."""
    from ldetect2.shrinkage import (
        _compact_pair_chunks_single_pass,
        _duplicate_compatible_pair_rows,
        _r2_pair_chunk_from_canonical_rows,
        _r2_pair_chunks_from_covariance,
    )

    if prepared.pos_arr.size == 0:
        return

    if prepared.has_duplicate_positions:
        rows = _duplicate_compatible_pair_rows(
            prepared.hap_mat,
            prepared.gpos_arr,
            prepared.hap_sums,
            prepared.j_stop_by_i,
            prepared.pos_arr,
            prepared.config.ne,
            float(prepared.n_ind),
            prepared.theta,
            prepared.config.cutoff,
        )
        chunk = _r2_pair_chunk_from_canonical_rows(rows)
        if chunk.lo.size:
            yield from _split_r2_chunk(chunk, chunk_rows)
        return

    yield from _r2_pair_chunks_from_covariance(
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


def _parse_cyvcf2_haplotypes(
    reference_panel: str,
    chrom: str,
    start: int,
    end: int,
    individuals: list[str],
    pos2gpos: dict[int, float],
) -> tuple[list[int], list[list[int]]]:
    try:
        from cyvcf2 import VCF
    except ModuleNotFoundError as exc:
        raise RuntimeError("cyvcf2 is not installed") from exc

    vcf = VCF(reference_panel, samples=individuals)
    all_pos: list[int] = []
    haps: list[list[int]] = []
    try:
        if _has_vcf_index(reference_panel):
            variants = vcf(f"{chrom}:{start}-{end}")
        else:
            variants = vcf
        for variant in variants:
            if str(variant.CHROM) != str(chrom):
                continue
            pos = int(variant.POS)
            if pos < start or pos > end or pos not in pos2gpos:
                continue
            row_haps = _cyvcf2_row_haps(variant.genotypes)
            if row_haps is None:
                continue
            all_pos.append(pos)
            haps.append(row_haps)
    finally:
        vcf.close()
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
) -> tuple[list[int], list[list[int]]]:
    from ldetect2.shrinkage import (
        _parse_vcf_haplotypes,
        _read_genetic_map,
        _read_individuals,
    )

    region = f"{config.chrom}:{start}-{end}"
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
