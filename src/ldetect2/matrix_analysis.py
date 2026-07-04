"""MatrixAnalysis: convert a covariance matrix dataset into a correlation-sum vector."""

from __future__ import annotations

import math
from bisect import bisect_left
from pathlib import Path

from ldetect2._util.covariance_array import ChromosomeCovariance
from ldetect2._util.logging import log_debug, log_msg
from ldetect2._util.vector_array import write_diag_vector_array
from ldetect2.io.covariance import (
    Matrix,
    delete_loci_smaller_than,
    delete_loci_smaller_than_lean,
    read_partition_into_matrix_lean,
    write_corr_vector,
)
from ldetect2.io.partitions import CovarianceStore, first_last, get_final_partitions

_USE_ARRAY_DIAG = True


class MatrixAnalysis:
    """Loads covariance partitions and computes sum-of-squared correlations per locus.

    The primary computation path is :meth:`calc_diag_lean`, which streams data
    through memory incrementally.  :meth:`calc_diag` keeps everything in RAM
    (required only if a heatmap image is needed).
    """

    def __init__(
        self,
        name: str,
        store: CovarianceStore,
        snp_first: int = -1,
        snp_last: int = -1,
    ) -> None:
        self.name = name
        self.store = store
        self.matrix: Matrix = {}
        self.locus_list: list[int] = []
        self.vert_sum: dict[int, float] = {}
        self.vert_sum_len: dict[int, int] = {}
        self.locus_list_deleted: list[int] = []

        self.snp_first, self.snp_last = first_last(name, store, snp_first, snp_last)
        self.partitions = get_final_partitions(
            store, name, self.snp_first, self.snp_last
        )

        self.dynamic_delete = False
        self.calculation_complete = False
        self.start_locus = -1
        self.start_locus_index = -1
        self.end_locus = -1
        self.end_locus_index = -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_r2(self, r2: float, locus: int) -> None:
        if locus not in self.vert_sum:
            self.vert_sum[locus] = r2
            self.vert_sum_len[locus] = 1
        else:
            self.vert_sum[locus] += r2
            self.vert_sum_len[locus] += 1

    def _find_first_locus(self, curr_locus: int) -> tuple[int, int, int, int]:
        """Locate the first locus >= snp_first.

        Returns (curr_locus, index, start_locus, start_index).
        """
        for i, locus in enumerate(self.locus_list):
            if locus >= self.snp_first:
                return locus, i, locus, i
        raise RuntimeError("locus_list is empty or contains no locus >= snp_first")

    # ------------------------------------------------------------------
    # Primary computation path (lean — streams to disk)
    # ------------------------------------------------------------------

    def calc_diag_lean(
        self,
        out_path: Path,
        covariance_cache: ChromosomeCovariance | None = None,
        matrix_workers: int = 1,
    ) -> None:
        """Compute the diagonal correlation-sum vector, writing output incrementally.

        This is the memory-efficient path.  Results are appended to *out_path*
        (gzipped TSV: position \\t corr_sum) as each partition is processed.
        """
        if _USE_ARRAY_DIAG:
            self.calc_diag_array(
                out_path,
                covariance_cache=covariance_cache,
                matrix_workers=matrix_workers,
            )
            return
        self._calc_diag_lean_legacy(out_path)

    def calc_diag_array(
        self,
        out_path: Path,
        covariance_cache: ChromosomeCovariance | None = None,
        matrix_workers: int = 1,
    ) -> None:
        """Compute the diagonal correlation-sum vector from partition arrays."""
        self.dynamic_delete = True
        log_msg("calc_diag_array: start")
        write_diag_vector_array(
            name=self.name,
            store=self.store,
            partitions=self.partitions,
            snp_first=self.snp_first,
            snp_last=self.snp_last,
            out_path=out_path,
            covariance_cache=covariance_cache,
            matrix_workers=matrix_workers,
        )
        self.calculation_complete = True

    def _calc_diag_lean_legacy(self, out_path: Path) -> None:
        """Dictionary-backed implementation retained as a correctness fallback."""
        self.dynamic_delete = True

        log_msg("calc_diag_lean: start")

        # Truncate output file so reruns don't append to stale data
        out_path.unlink(missing_ok=True)

        # Pre-read all partitions whose end is before snp_first
        last_p_num = -1
        for p_num_init in range(len(self.partitions) - 1):
            if self.snp_first >= self.partitions[p_num_init + 1][0]:
                log_debug(f"Pre-reading partition: {self.partitions[p_num_init]}")
                read_partition_into_matrix_lean(
                    self.partitions,
                    p_num_init,
                    self.matrix,
                    self.locus_list,
                    self.name,
                    self.store,
                    self.snp_first,
                    self.snp_last,
                )
                last_p_num = p_num_init
            else:
                break

        curr_locus = -1
        start_locus = -1
        start_locus_index = -1
        end_locus = -1
        end_locus_index = -1

        for p_num in range(last_p_num + 1, len(self.partitions)):
            p = self.partitions[p_num]
            log_debug(f"Reading partition: {p}")
            read_partition_into_matrix_lean(
                self.partitions,
                p_num,
                self.matrix,
                self.locus_list,
                self.name,
                self.store,
                self.snp_first,
                self.snp_last,
            )

            # Locate curr_locus in the updated list
            if curr_locus < 0:
                if not self.locus_list:
                    raise RuntimeError("locus_list is empty")
                for i, locus in enumerate(self.locus_list):
                    if locus >= self.snp_first:
                        curr_locus = locus
                        start_locus = locus
                        curr_locus_index = i
                        start_locus_index = i
                        break
            else:
                i = bisect_left(self.locus_list, curr_locus)
                if i < len(self.locus_list) and self.locus_list[i] == curr_locus:
                    curr_locus_index = i
                else:
                    if self.locus_list:
                        curr_locus = self.locus_list[0]
                        curr_locus_index = 0
                    else:
                        raise RuntimeError("locus_list is empty")

            if curr_locus < 0:
                log_debug(
                    f"Warning: curr_locus not found in partition {p} "
                    f"(snp_first={self.snp_first}); skipping"
                )
                continue

            # Determine end locus for this partition
            if p_num + 1 < len(self.partitions):
                end_locus = int(
                    (self.partitions[p_num][1] + self.partitions[p_num + 1][0]) / 2
                )
            else:
                end_locus_found = False
                for i in reversed(range(len(self.locus_list))):
                    if self.locus_list[i] <= self.snp_last:
                        end_locus = self.locus_list[i]
                        end_locus_index = i
                        end_locus_found = True
                        break
                if not end_locus_found:
                    end_locus_index = 0
                    end_locus = self.locus_list[0]

            log_debug(f"Running for partition: {p}")

            p_start = self.partitions[p_num][0]
            p_end = self.partitions[p_num][1]
            matrix = self.matrix
            locus_list = self.locus_list

            while curr_locus <= end_locus:
                x = locus_list[curr_locus_index]
                y = locus_list[curr_locus_index]
                delta = 0

                while x >= p_start and y <= p_end:
                    if x in matrix and y in matrix[x]:
                        diag_x = matrix[x].get(x, 0.0)
                        diag_y = matrix[y].get(y, 0.0)
                        if diag_x > 0 and diag_y > 0:
                            cov = matrix[x][y]
                            self._add_r2(cov * cov / (diag_x * diag_y), curr_locus)

                    if delta != 0:
                        x2 = locus_list[curr_locus_index - delta + 1]
                        if x2 in matrix and y in matrix[x2]:
                            diag_x2 = matrix[x2].get(x2, 0.0)
                            diag_y = matrix[y].get(y, 0.0)
                            if diag_x2 > 0 and diag_y > 0:
                                cov2 = matrix[x2][y]
                                self._add_r2(
                                    cov2 * cov2 / (diag_x2 * diag_y),
                                    curr_locus,
                                )

                    delta += 1
                    if curr_locus_index - delta >= 0:
                        x = locus_list[curr_locus_index - delta]
                    else:
                        break
                    if curr_locus_index + delta < len(locus_list):
                        y = locus_list[curr_locus_index + delta]
                    else:
                        break

                if curr_locus_index + 1 < len(locus_list):
                    curr_locus_index += 1
                    curr_locus = locus_list[curr_locus_index]
                else:
                    log_debug("curr_locus_index out of bounds")
                    break

            # Stream completed loci to disk and free memory
            if p_num + 1 < len(self.partitions):
                delete_loc = self.partitions[p_num + 1][0]
            else:
                delete_loc = end_locus

            delete_loci_smaller_than_lean(
                delete_loc,
                self.matrix,
                self.locus_list,
                self.locus_list_deleted,
                out_path,
                self.vert_sum,
            )

        self.start_locus = start_locus
        self.start_locus_index = start_locus_index
        self.end_locus = end_locus
        self.end_locus_index = end_locus_index
        self.calculation_complete = True

    # ------------------------------------------------------------------
    # Full computation path (keeps matrix in RAM — needed for heatmap)
    # ------------------------------------------------------------------

    def calc_diag(self) -> None:
        """Compute the diagonal correlation-sum vector, keeping matrix in RAM."""
        from ldetect2.io.covariance import (
            read_partition_into_matrix,
        )

        self.dynamic_delete = True
        log_msg("calc_diag: start")

        last_p_num = -1
        for p_num_init in range(len(self.partitions) - 1):
            if self.snp_first >= self.partitions[p_num_init + 1][0]:
                log_debug(f"Pre-reading partition: {self.partitions[p_num_init]}")
                read_partition_into_matrix(
                    self.partitions,
                    p_num_init,
                    self.matrix,
                    self.locus_list,
                    self.name,
                    self.store,
                    self.snp_first,
                    self.snp_last,
                )
                last_p_num = p_num_init
            else:
                break

        curr_locus = -1
        start_locus = -1
        start_locus_index = -1
        end_locus = -1
        end_locus_index = -1

        for p_num in range(last_p_num + 1, len(self.partitions)):
            p = self.partitions[p_num]
            log_debug(f"Reading partition: {p}")
            read_partition_into_matrix(
                self.partitions,
                p_num,
                self.matrix,
                self.locus_list,
                self.name,
                self.store,
                self.snp_first,
                self.snp_last,
            )

            if curr_locus < 0:
                if not self.locus_list:
                    raise RuntimeError("locus_list is empty")
                for i, locus in enumerate(self.locus_list):
                    if locus >= self.snp_first:
                        curr_locus = locus
                        start_locus = locus
                        curr_locus_index = i
                        start_locus_index = i
                        break
            else:
                i = bisect_left(self.locus_list, curr_locus)
                if i < len(self.locus_list) and self.locus_list[i] == curr_locus:
                    curr_locus_index = i
                else:
                    if self.locus_list:
                        curr_locus = self.locus_list[0]
                        curr_locus_index = 0
                    else:
                        raise RuntimeError("locus_list is empty")

            if curr_locus < 0:
                log_debug(f"Warning: curr_locus not found in partition {p}; skipping")
                continue

            if p_num + 1 < len(self.partitions):
                end_locus = int(
                    (self.partitions[p_num][1] + self.partitions[p_num + 1][0]) / 2
                )
            else:
                end_locus_found = False
                for i in reversed(range(len(self.locus_list))):
                    if self.locus_list[i] <= self.snp_last:
                        end_locus = self.locus_list[i]
                        end_locus_index = i
                        end_locus_found = True
                        break
                if not end_locus_found:
                    end_locus_index = 0
                    end_locus = self.locus_list[0]

            log_debug(f"Running for partition: {p}")

            p_start = self.partitions[p_num][0]
            p_end = self.partitions[p_num][1]
            matrix = self.matrix
            locus_list = self.locus_list

            while curr_locus <= end_locus:
                x = locus_list[curr_locus_index]
                y = locus_list[curr_locus_index]
                delta = 0

                while x >= p_start and y <= p_end:
                    if x in matrix and y in matrix[x]["data"]:
                        sx = matrix[x]["data"][x]["shrink"]
                        sy = matrix[y]["data"][y]["shrink"]
                        if sx > 0 and sy > 0:
                            corr = matrix[x]["data"][y]["shrink"] / math.sqrt(sx * sy)
                            self._add_r2(corr * corr, curr_locus)
                            matrix[x]["data"][y]["corr_coeff"] = corr

                    if delta != 0:
                        x2 = locus_list[curr_locus_index - delta + 1]
                        if x2 in matrix and y in matrix[x2]["data"]:
                            sx2 = matrix[x2]["data"][x2]["shrink"]
                            sy = matrix[y]["data"][y]["shrink"]
                            if sx2 > 0 and sy > 0:
                                corr = matrix[x2]["data"][y]["shrink"] / math.sqrt(
                                    sx2 * sy
                                )
                                self._add_r2(corr * corr, curr_locus)
                                matrix[x2]["data"][y]["corr_coeff"] = corr

                    delta += 1
                    if curr_locus_index - delta >= 0:
                        x = locus_list[curr_locus_index - delta]
                    else:
                        break
                    if curr_locus_index + delta < len(locus_list):
                        y = locus_list[curr_locus_index + delta]
                    else:
                        break

                if curr_locus_index + 1 < len(locus_list):
                    curr_locus_index += 1
                    curr_locus = locus_list[curr_locus_index]
                else:
                    log_debug("curr_locus_index out of bounds")
                    break

            if p_num + 1 < len(self.partitions):
                delete_loc = self.partitions[p_num + 1][0]
            else:
                delete_loc = end_locus

            delete_loci_smaller_than(
                delete_loc, self.matrix, self.locus_list, self.locus_list_deleted
            )

        self.start_locus = start_locus
        self.start_locus_index = start_locus_index
        self.end_locus = end_locus
        self.end_locus_index = end_locus_index
        self.calculation_complete = True

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_output_to_file(self, out_path: Path, avg: bool = False) -> None:
        if not self.calculation_complete:
            raise RuntimeError("Calculation must complete before writing output")
        log_msg("Writing output to file")
        write_corr_vector(
            out_path,
            self.locus_list,
            self.locus_list_deleted,
            self.vert_sum,
            self.vert_sum_len if avg else None,
        )

    def generate_img(self, img_path: Path, marked_snp: int | None = None) -> None:
        """Write a PNG heatmap of the correlation matrix."""
        if not self.calculation_complete:
            raise RuntimeError("Calculation must complete before generating image")
        if self.dynamic_delete:
            raise RuntimeError("Matrix was dynamically deleted; cannot generate image")
        if not self.matrix:
            raise RuntimeError("Matrix is empty")

        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as pt
        import numpy as np

        log_msg("Generating heatmap image")
        size = self.end_locus_index - self.start_locus_index + 1
        plot_mtrx = [[0.0] * size for _ in range(size)]
        x_values = [0] * size

        for loc_i, row_data in self.matrix.items():
            if self.snp_first <= loc_i <= self.snp_last:
                idx_i = self.locus_list.index(loc_i) - self.start_locus_index
                x_values[idx_i] = loc_i
                for loc_j, cell in row_data["data"].items():
                    if (
                        self.snp_first <= loc_j <= self.snp_last
                        and "corr_coeff" in cell
                    ):
                        idx_j = self.locus_list.index(loc_j) - self.start_locus_index
                        try:
                            plot_mtrx[idx_i][idx_j] = cell["corr_coeff"] ** 2
                        except IndexError:
                            pass

        fig = pt.figure(figsize=(40, 30))
        pt.pcolormesh(np.array(plot_mtrx), cmap="binary", vmin=0, vmax=1)
        pt.colorbar()

        if marked_snp is not None and marked_snp in x_values:
            bpt = x_values.index(marked_snp)
            pt.scatter(bpt, bpt, marker="x", color="green")

        pt.xlabel("SNP #")
        pt.ylabel("SNP #")
        pt.title("Correlation coefficient squared matrix")
        fig.savefig(img_path)
        pt.close(fig)
        log_msg(f"Image saved to {img_path}")
