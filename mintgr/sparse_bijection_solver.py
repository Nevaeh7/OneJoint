"""Budgeted sparse permutation projection for the hard item-leaf map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class BijectionUpdate:
    item_to_leaf: torch.Tensor
    changed_items: int
    previous_cost: float
    updated_cost: float
    accepted: bool


class SparseBijectionSolver:
    """Solve a closed sparse assignment with current mapping always feasible."""

    def __init__(self, infeasible_cost: float = 1e6, tolerance: float = 1e-8):
        self.infeasible_cost = float(infeasible_cost)
        self.tolerance = float(tolerance)

    def solve(
        self,
        current_item_to_leaf: torch.Tensor,
        candidate_leaf_ids: torch.Tensor,
        candidate_costs: torch.Tensor,
        max_changed_items: int,
    ) -> BijectionUpdate:
        current = current_item_to_leaf.detach().cpu().long().clone()
        candidates = candidate_leaf_ids.detach().cpu().long()
        costs = candidate_costs.detach().cpu().double()
        item_count = current.numel()
        if candidates.shape != costs.shape or candidates.shape[0] != item_count:
            raise ValueError("candidate leaves and costs must align with hard map")
        if torch.unique(current).numel() != item_count:
            raise ValueError("current hard map must be a bijection")
        if max_changed_items <= 1:
            return BijectionUpdate(current, 0, 0.0, 0.0, False)

        owner = torch.empty_like(current)
        owner[current] = torch.arange(item_count)
        current_cost = torch.full((item_count,), self.infeasible_cost, dtype=torch.double)
        best_gain = torch.full((item_count,), float("-inf"), dtype=torch.double)
        candidate_maps: List[Dict[int, float]] = []
        for item in range(item_count):
            mapping: Dict[int, float] = {}
            for leaf, value in zip(candidates[item].tolist(), costs[item].tolist()):
                mapping[int(leaf)] = min(mapping.get(int(leaf), float("inf")), float(value))
            mapping.setdefault(int(current[item]), 0.0)
            candidate_maps.append(mapping)
            current_cost[item] = float(mapping[int(current[item])])
            alternatives = [
                value for leaf, value in mapping.items() if leaf != int(current[item])
            ]
            if alternatives:
                best_gain[item] = current_cost[item] - min(alternatives)

        ranked = torch.argsort(best_gain, descending=True).tolist()
        selected: Set[int] = set()
        for seed in ranked:
            if len(selected) >= max_changed_items:
                break
            if not np.isfinite(float(best_gain[seed])) or best_gain[seed] <= 0:
                break
            closure = set(selected)
            queue = [int(seed)]
            while queue and len(closure) <= max_changed_items:
                item = queue.pop()
                if item in closure:
                    continue
                closure.add(item)
                for leaf in candidate_maps[item]:
                    owner_item = int(owner[int(leaf)])
                    if owner_item not in closure:
                        queue.append(owner_item)
            if len(closure) <= max_changed_items:
                selected = closure

        if len(selected) <= 1:
            return BijectionUpdate(current, 0, 0.0, 0.0, False)
        items = sorted(selected)
        leaves = [int(current[item]) for item in items]
        leaf_column = {leaf: column for column, leaf in enumerate(leaves)}
        matrix = np.full(
            (len(items), len(leaves)), self.infeasible_cost, dtype=np.float64
        )
        for row, item in enumerate(items):
            for leaf, value in candidate_maps[item].items():
                column = leaf_column.get(int(leaf))
                if column is not None:
                    matrix[row, column] = float(value)
            matrix[row, leaf_column[int(current[item])]] = float(current_cost[item])
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as error:
            raise RuntimeError("scipy is required for sparse bijection projection") from error
        row_index, column_index = linear_sum_assignment(matrix)
        previous = float(sum(current_cost[item].item() for item in items))
        updated = float(matrix[row_index, column_index].sum())
        if not np.isfinite(updated) or updated >= previous - self.tolerance:
            return BijectionUpdate(current, 0, previous, updated, False)
        proposal = current.clone()
        for row, column in zip(row_index.tolist(), column_index.tolist()):
            proposal[items[row]] = leaves[column]
        if torch.unique(proposal).numel() != item_count:
            raise AssertionError("bijection solver produced a collision")
        changed = int((proposal != current).sum().item())
        if changed > max_changed_items:
            raise AssertionError("bijection solver exceeded the move budget")
        return BijectionUpdate(proposal, changed, previous, updated, True)
