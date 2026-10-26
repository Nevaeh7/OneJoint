"""Exact projection of item distributions to active-tree conditionals."""

from __future__ import annotations

from typing import List

import torch

from .balanced_code_lattice import BalancedCodeTree


def project_item_distribution_to_target_path(
    item_probability: torch.Tensor,
    inverse_leaf_owner: torch.Tensor,
    target_paths: torch.Tensor,
    tree: BalancedCodeTree,
    epsilon: float = 1e-12,
) -> List[torch.Tensor]:
    """Project item mass to legal child distributions on target prefixes.

    Returns one ``[batch, branching]`` tensor per tree level. Illegal children
    have zero mass. ``inverse_leaf_owner[leaf]`` is the zero-based catalog item
    row that owns the active leaf.
    """
    if item_probability.ndim != 2:
        raise ValueError("item_probability must have shape [batch, items]")
    if inverse_leaf_owner.shape != (tree.num_leaves,):
        raise ValueError("inverse leaf owner must cover every active leaf")
    if target_paths.shape != (item_probability.shape[0], tree.depth):
        raise ValueError("target paths must match batch and tree depth")
    leaf_probability = item_probability[:, inverse_leaf_owner.long()]
    leaves = tree.active_leaves.to(item_probability.device)
    results: List[torch.Tensor] = []
    for level in range(tree.depth):
        conditional = torch.zeros(
            item_probability.shape[0],
            tree.branching,
            dtype=item_probability.dtype,
            device=item_probability.device,
        )
        for row in range(item_probability.shape[0]):
            if level == 0:
                parent_mask = torch.ones(
                    tree.num_leaves, dtype=torch.bool, device=item_probability.device
                )
            else:
                parent_mask = (
                    leaves[:, :level]
                    == target_paths[row, :level].unsqueeze(0)
                ).all(dim=1)
            selected_mass = leaf_probability[row, parent_mask]
            selected_children = leaves[parent_mask, level]
            conditional[row].scatter_add_(0, selected_children, selected_mass)
            denominator = conditional[row].sum().clamp_min(float(epsilon))
            conditional[row] = conditional[row] / denominator
        results.append(conditional)
    return results


def reconstruct_leaf_distribution(
    conditionals: List[torch.Tensor],
    tree: BalancedCodeTree,
) -> torch.Tensor:
    """Multiply projected conditionals along every active leaf."""
    if len(conditionals) != tree.depth:
        raise ValueError("one conditional tensor is required per tree level")
    batch_size = conditionals[0].shape[0]
    leaves = tree.active_leaves.to(conditionals[0].device)
    output = torch.ones(
        batch_size,
        tree.num_leaves,
        dtype=conditionals[0].dtype,
        device=conditionals[0].device,
    )
    # These conditionals may be target-prefix-specific. Exact reconstruction is
    # therefore only valid when each tensor stores conditionals for every path;
    # the helper is intentionally restricted to batch size == num leaves tests.
    if batch_size != tree.num_leaves:
        raise ValueError(
            "target-prefix reconstruction requires one row per active leaf"
        )
    for leaf_index, path in enumerate(leaves.tolist()):
        probability = torch.tensor(
            1.0, dtype=output.dtype, device=output.device
        )
        for level, token in enumerate(path):
            probability = probability * conditionals[level][leaf_index, token]
        output[leaf_index, leaf_index] = probability
    return output


def target_path_probability(
    conditionals: List[torch.Tensor], target_paths: torch.Tensor
) -> torch.Tensor:
    if not conditionals:
        raise ValueError("conditionals cannot be empty")
    if target_paths.shape != (conditionals[0].shape[0], len(conditionals)):
        raise ValueError("target paths must align with conditionals")
    probability = torch.ones(
        target_paths.shape[0],
        dtype=conditionals[0].dtype,
        device=conditionals[0].device,
    )
    rows = torch.arange(target_paths.shape[0], device=target_paths.device)
    for level, conditional in enumerate(conditionals):
        probability = probability * conditional[
            rows, target_paths[:, level].long()
        ]
    return probability
