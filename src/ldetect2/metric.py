"""Metric: quality measure for a set of LD block breakpoints."""

from __future__ import annotations

import decimal
import math

from ldetect2._util.logging import log_msg
from ldetect2.io.covariance import (
    delete_loci_smaller_than_leanest,
    read_partition_into_matrix_lean,
)
from ldetect2.io.partitions import CovarianceStore, first_last, get_final_partitions

_PREC = 50


class Metric:
    """Calculates the sum-of-squared-correlations quality metric.

    The metric is defined as::

        sum(r²) / N_zero

    where *N_zero* is the expected number of non-zero entries (the total block
    area minus the diagonal triangles).  A lower value indicates better-
    separated LD blocks.

    All arithmetic uses :class:`decimal.Decimal` with 50-digit precision.
    """

    def __init__(
        self,
        name: str,
        store: CovarianceStore,
        breakpoints: list[int],
        snp_first: int = -1,
        snp_last: int = -1,
    ) -> None:
        decimal.getcontext().prec = _PREC

        self.name = name
        self.store = store
        self.breakpoints = breakpoints

        self.matrix: dict = {}
        self.locus_list: list[int] = []
        self.locus_list_deleted: list[int] = []

        self.metric: dict[str, decimal.Decimal] = {
            "sum": decimal.Decimal("0"),
            "N_nonzero": decimal.Decimal("0"),
            "N_zero": decimal.Decimal("0"),
        }

        self.snp_first, self.snp_last = first_last(name, store, snp_first, snp_last)
        self.partitions = get_final_partitions(store, name, self.snp_first, self.snp_last)

        self.dynamic_delete = True
        self.calculation_complete = False
        self.start_locus = -1
        self.start_locus_index = -1
        self.end_locus = -1
        self.end_locus_index = -1

    def calc_metric(self) -> dict[str, decimal.Decimal]:
        """Run the lean metric calculation and return the result dict."""
        return self._calc_metric_lean()

    def _calc_metric_lean(self) -> dict[str, decimal.Decimal]:
        decimal.getcontext().prec = _PREC

        log_msg("Start metric (lean)")

        curr_breakpoint_index = 0
        block_width = 0
        total_n_snps = decimal.Decimal("0")
        block_width_sum = decimal.Decimal("0")

        # Pre-read partitions whose range is entirely before snp_first
        last_p_num = -1
        for p_num_init in range(len(self.partitions) - 1):
            if self.snp_first >= self.partitions[p_num_init + 1][0]:
                log_msg(f"Pre-reading partition: {self.partitions[p_num_init]}")
                read_partition_into_matrix_lean(
                    self.partitions, p_num_init,
                    self.matrix, self.locus_list,
                    self.name, self.store,
                    self.snp_first, self.snp_last,
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
            log_msg(f"Reading partition: {p}")
            read_partition_into_matrix_lean(
                self.partitions, p_num,
                self.matrix, self.locus_list,
                self.name, self.store,
                self.snp_first, self.snp_last,
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
                try:
                    curr_locus_index = self.locus_list.index(curr_locus)
                except ValueError:
                    if self.locus_list:
                        curr_locus = self.locus_list[0]
                        curr_locus_index = 0
                    else:
                        raise RuntimeError("locus_list is empty")

            if curr_locus < 0:
                log_msg(
                    f"Warning: curr_locus not found in partition {p} "
                    f"(snp_first={self.snp_first}); skipping"
                )
                continue

            if p_num + 1 < len(self.partitions):
                end_locus = self.partitions[p_num + 1][0]
                end_locus_index = -1
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

            log_msg(f"Running metric for partition: {p}")

            while curr_locus <= end_locus:
                # Advance breakpoint if we've passed it
                if curr_breakpoint_index < len(self.breakpoints):
                    if curr_locus > self.breakpoints[curr_breakpoint_index]:
                        block_height = decimal.Decimal(0) - total_n_snps
                        self.metric["N_zero"] += block_height * block_width
                        block_width_sum += block_width
                        curr_breakpoint_index += 1
                        block_width = 0

                if curr_breakpoint_index >= len(self.breakpoints):
                    break

                try:
                    for key in self.matrix.get(curr_locus, {}):
                        if key > self.breakpoints[curr_breakpoint_index]:
                            diag_curr = self.matrix[curr_locus].get(curr_locus, 0.0)
                            diag_key = self.matrix.get(key, {}).get(key, 0.0)
                            if diag_curr > 0 and diag_key > 0:
                                corr = (
                                    self.matrix[curr_locus][key]
                                    / math.sqrt(diag_curr * diag_key)
                                )
                                self.metric["sum"] += decimal.Decimal(corr ** 2)
                                self.metric["N_nonzero"] += decimal.Decimal(1)
                except IndexError as exc:
                    log_msg(f"IndexError at locus {curr_locus}: {exc}")

                block_width += 1

                if curr_locus_index + 1 < len(self.locus_list):
                    curr_locus_index += 1
                    curr_locus = self.locus_list[curr_locus_index]
                    total_n_snps += decimal.Decimal(1)
                else:
                    log_msg("curr_locus_index out of bounds")
                    break

            delete_loci_smaller_than_leanest(end_locus, self.matrix, self.locus_list)

        self.start_locus = start_locus
        self.start_locus_index = start_locus_index
        self.end_locus = end_locus
        self.end_locus_index = end_locus_index

        self.metric["N_zero"] += total_n_snps * block_width_sum
        log_msg(
            f"Metric done: sum={self.metric['sum']}, "
            f"N_zero={self.metric['N_zero']}"
        )

        self.calculation_complete = True
        return self.metric
