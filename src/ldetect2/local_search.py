"""LocalSearch: greedy local refinement of a single breakpoint position."""

from __future__ import annotations

import decimal
import math

from ldetect2._util.binary_search import find_ge_ind, find_le_ind
from ldetect2._util.logging import log_msg
from ldetect2.io.covariance import (
    delete_loci_smaller_than_leanest,
    read_partition_into_matrix_lean,
)
from ldetect2.io.partitions import CovarianceStore, get_final_partitions

_PREC = 50


class LocalSearch:
    """Precomputes per-locus LD sums and searches for the locally-optimal breakpoint.

    The search evaluates each locus within [start_search, stop_search] as a
    candidate breakpoint and returns the one that minimises
    ``sum(r²) / N_zero``.
    """

    def __init__(
        self,
        name: str,
        start_search: int,
        stop_search: int,
        initial_breakpoint_index: int,
        breakpoints: list[int],
        total_sum: decimal.Decimal,
        total_n: decimal.Decimal,
        store: CovarianceStore,
    ) -> None:
        decimal.getcontext().prec = _PREC

        self.name = name
        self.start_search = start_search
        self.stop_search = stop_search
        self.initial_breakpoint_index = initial_breakpoint_index
        self.breakpoints = breakpoints
        self.total_sum = total_sum
        self.total_n = total_n
        self.store = store

        self.matrix: dict = {}
        self.locus_list: list[int] = []
        self.locus_list_deleted: list[int] = []

        self.precomputed: dict = {
            "locus_list": [],
            "data": {},
        }

        self.dynamic_delete = True
        self.init_complete = False
        self.search_complete = False

        # --- validation ---
        if start_search >= stop_search:
            raise ValueError(f"start_search ({start_search}) >= stop_search ({stop_search})")
        if not (0 <= initial_breakpoint_index < len(breakpoints)):
            raise ValueError("initial_breakpoint_index out of bounds")
        if breakpoints[initial_breakpoint_index] >= stop_search:
            raise ValueError("breakpoint >= stop_search")
        if breakpoints[initial_breakpoint_index] <= start_search:
            raise ValueError("breakpoint <= start_search")

        tmp_partitions = get_final_partitions(store, name, start_search, stop_search)

        if not (tmp_partitions[0][0] <= start_search <= tmp_partitions[-1][1]):
            raise ValueError("start_search is out of partition bounds")
        if not (tmp_partitions[0][0] <= stop_search <= tmp_partitions[-1][1]):
            raise ValueError("stop_search is out of partition bounds")

        if initial_breakpoint_index > 0:
            if start_search < breakpoints[initial_breakpoint_index - 1]:
                raise ValueError("start_search cannot be further than a neighbouring breakpoint")
        if initial_breakpoint_index < len(breakpoints) - 1:
            if stop_search > breakpoints[initial_breakpoint_index + 1]:
                raise ValueError("stop_search cannot be further than a neighbouring breakpoint")

        self.snp_first = start_search
        self.snp_last = stop_search

        if initial_breakpoint_index + 1 < len(breakpoints):
            self.snp_top = breakpoints[initial_breakpoint_index + 1]
        else:
            self.snp_top = tmp_partitions[-1][1]

        if initial_breakpoint_index - 1 >= 0:
            self.snp_bottom = breakpoints[initial_breakpoint_index - 1]
        else:
            self.snp_bottom = tmp_partitions[0][0]

        log_msg(f"LocalSearch: snp_first={self.snp_first} snp_last={self.snp_last} "
                f"snp_bottom={self.snp_bottom} snp_top={self.snp_top}")

        self.partitions = get_final_partitions(store, name, self.snp_bottom, self.snp_top)

        self.start_locus = -1
        self.start_locus_index = -1
        self.end_locus = -1
        self.end_locus_index = -1

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def init_search(self) -> None:
        """Precompute per-locus vertical and horizontal LD sums (lean path)."""
        decimal.getcontext().prec = _PREC
        log_msg("Start local search init (lean)")

        last_p_num = -1
        for p_num_init in range(len(self.partitions) - 1):
            if self.snp_bottom >= self.partitions[p_num_init + 1][0]:
                log_msg(f"Pre-reading partition: {self.partitions[p_num_init]}")
                read_partition_into_matrix_lean(
                    self.partitions, p_num_init,
                    self.matrix, self.locus_list,
                    self.name, self.store,
                    self.snp_bottom, self.snp_top,
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
                self.snp_bottom, self.snp_top,
            )

            if curr_locus < 0:
                if not self.locus_list:
                    raise RuntimeError("locus_list is empty")
                for i, locus in enumerate(self.locus_list):
                    if locus >= self.snp_bottom:
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
                    f"(snp_bottom={self.snp_bottom}); skipping"
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

            log_msg(f"Precomputing for partition: {p}")

            while curr_locus <= end_locus:
                self._add_locus(curr_locus)

                in_range = (
                    (curr_locus > self.snp_first or self.initial_breakpoint_index == 0)
                    and curr_locus <= self.snp_last
                )
                if in_range:
                    for key in self.matrix.get(curr_locus, {}):
                        if key <= self.snp_top:
                            diag_curr = self.matrix[curr_locus].get(curr_locus, 0.0)
                            diag_key = self.matrix.get(key, {}).get(key, 0.0)
                            if diag_curr > 0 and diag_key > 0:
                                corr = (
                                    self.matrix[curr_locus][key]
                                    / math.sqrt(diag_curr * diag_key)
                                )
                                self._add_val(decimal.Decimal(corr ** 2), curr_locus, key)
                else:
                    self._add_val(decimal.Decimal(0), curr_locus, curr_locus)

                if curr_locus_index + 1 < len(self.locus_list):
                    curr_locus_index += 1
                    curr_locus = self.locus_list[curr_locus_index]
                else:
                    log_msg("curr_locus_index out of bounds")
                    break

            delete_loci_smaller_than_leanest(end_locus, self.matrix, self.locus_list)

        self.start_locus = start_locus
        self.start_locus_index = start_locus_index
        self.end_locus = end_locus
        self.end_locus_index = end_locus_index
        self.init_complete = True

    def _add_val(self, val: decimal.Decimal, curr_locus: int, key: int) -> None:
        for loc in (curr_locus, key):
            if loc not in self.precomputed["data"]:
                self.precomputed["data"][loc] = {
                    "sum_vert": decimal.Decimal(0),
                    "sum_horiz": decimal.Decimal(0),
                }
        self.precomputed["data"][curr_locus]["sum_vert"] += val
        self.precomputed["data"][key]["sum_horiz"] += val

    def _add_locus(self, locus: int) -> None:
        self.precomputed["locus_list"].append(locus)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self) -> tuple[int | None, dict | None]:
        """Find the locally-optimal breakpoint position.

        Returns:
            ``(best_position, metric_details)`` where *metric_details* has keys
            ``sum`` and ``N_zero``.  Returns the initial breakpoint unchanged if
            no better position is found.
        """
        if not self.init_complete:
            log_msg("init_search() not called — running automatically")
            self.init_search()

        log_msg("Starting local search")
        locus_list = self.precomputed["locus_list"]

        try:
            snp_bottom_ind = find_ge_ind(locus_list, self.snp_bottom)
            snp_top_ind = find_le_ind(locus_list, self.snp_top)
        except ValueError as exc:
            log_msg(f"Error finding bounds in precomputed locus_list: {exc}")
            log_msg(f"snp_bottom={self.snp_bottom} snp_top={self.snp_top}")
            log_msg("Returning initial breakpoint unchanged")
            return self.breakpoints[self.initial_breakpoint_index], None

        bp_ind = find_le_ind(locus_list, self.breakpoints[self.initial_breakpoint_index])
        init_bp_locus = locus_list[bp_ind]

        curr_sum = self.total_sum
        curr_n = self.total_n
        min_metric = decimal.Decimal(self.total_sum) / decimal.Decimal(self.total_n)
        min_breakpoint: int | None = None
        min_metric_details: dict = {"sum": self.total_sum, "N_zero": self.total_n}
        min_distance_right = 0

        # Search RIGHT
        log_msg("Searching right...")
        if bp_ind + 1 < len(locus_list):
            curr_loc_ind = bp_ind + 1
            curr_loc = locus_list[curr_loc_ind]

            while curr_loc <= self.snp_last:
                data = self.precomputed["data"]
                curr_sum = (
                    curr_sum
                    - data[curr_loc]["sum_horiz"]
                    + data[curr_loc]["sum_vert"]
                )
                horiz_n = curr_loc_ind - snp_bottom_ind - 1
                vert_n = snp_top_ind - curr_loc_ind
                curr_n = curr_n - horiz_n + vert_n

                curr_metric = decimal.Decimal(curr_sum) / decimal.Decimal(curr_n)
                if curr_metric < min_metric:
                    min_metric = curr_metric
                    min_breakpoint = curr_loc
                    min_metric_details = {"sum": curr_sum, "N_zero": curr_n}
                    min_distance_right = curr_loc - init_bp_locus

                if curr_loc_ind + 1 < len(locus_list):
                    curr_loc_ind += 1
                    curr_loc = locus_list[curr_loc_ind]
                else:
                    break
        else:
            log_msg("Warning: no loci to the right of initial breakpoint")

        # Reset for left search
        curr_sum = self.total_sum
        curr_n = self.total_n

        # Search LEFT
        log_msg("Searching left...")
        if bp_ind - 1 >= 0:
            curr_loc_ind = bp_ind - 1
            curr_loc = locus_list[curr_loc_ind]

            while curr_loc > self.snp_first:
                data = self.precomputed["data"]
                curr_sum = (
                    curr_sum
                    + data[curr_loc]["sum_horiz"]
                    - data[curr_loc]["sum_vert"]
                )
                horiz_n = curr_loc_ind - snp_bottom_ind - 1
                vert_n = snp_top_ind - curr_loc_ind
                curr_n = curr_n + horiz_n - vert_n

                curr_metric = decimal.Decimal(curr_sum) / decimal.Decimal(curr_n)
                left_dist = init_bp_locus - curr_loc
                if curr_metric < min_metric or (
                    curr_metric == min_metric and left_dist < min_distance_right
                ):
                    min_metric = curr_metric
                    min_breakpoint = curr_loc
                    min_metric_details = {"sum": curr_sum, "N_zero": curr_n}

                if curr_loc_ind - 1 >= 0:
                    curr_loc_ind -= 1
                    curr_loc = locus_list[curr_loc_ind]
                else:
                    break
        else:
            log_msg("Warning: no loci to the left of initial breakpoint")

        self.search_complete = True
        log_msg("Search done")
        return min_breakpoint, min_metric_details
