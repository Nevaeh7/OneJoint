"""Occupancy-normalized latent item probability primitives."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def sparse_occupancy(
    candidate_leaf_ids: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    num_leaves: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if candidate_leaf_ids.shape != candidate_probabilities.shape:
        raise ValueError("candidate IDs and probabilities must have the same shape")
    if candidate_leaf_ids.numel() and (
        candidate_leaf_ids.min().item() < 0
        or candidate_leaf_ids.max().item() >= num_leaves
    ):
        raise ValueError("candidate leaf ID is outside the active leaf set")
    occupancy = torch.zeros(
        num_leaves,
        dtype=candidate_probabilities.dtype,
        device=candidate_probabilities.device,
    )
    occupancy.scatter_add_(
        0,
        candidate_leaf_ids.reshape(-1).long(),
        candidate_probabilities.reshape(-1),
    )
    return occupancy.clamp_min(float(epsilon))


def marginalized_item_log_probability(
    path_log_probability: torch.Tensor,
    router_probability: torch.Tensor,
    occupancy: torch.Tensor,
    candidate_leaf_ids: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if path_log_probability.shape != router_probability.shape:
        raise ValueError("path and router probability tensors must align")
    if candidate_leaf_ids.shape != router_probability.shape:
        raise ValueError("candidate leaf IDs must align with router probabilities")
    if occupancy.ndim != 1:
        raise ValueError("occupancy must be a one-dimensional active-leaf tensor")
    log_router = torch.log(router_probability.clamp_min(float(epsilon)))
    log_occupancy = torch.log(
        occupancy[candidate_leaf_ids.long()].clamp_min(float(epsilon))
    )
    return torch.logsumexp(
        path_log_probability + log_router - log_occupancy,
        dim=-1,
    )


def marginalized_item_nll(
    path_log_probability: torch.Tensor,
    router_probability: torch.Tensor,
    occupancy: torch.Tensor,
    candidate_leaf_ids: torch.Tensor,
    epsilon: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    losses = -marginalized_item_log_probability(
        path_log_probability=path_log_probability,
        router_probability=router_probability,
        occupancy=occupancy,
        candidate_leaf_ids=candidate_leaf_ids,
        epsilon=epsilon,
    )
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def enumerate_item_distribution(
    leaf_probability: torch.Tensor,
    router_probability: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Enumerate the proper item distribution for a small dense catalog.

    ``leaf_probability`` is ``[batch, leaves]`` and ``router_probability`` is
    ``[items, leaves]``. This function is intentionally dense and exists for
    exact tests and small-catalog audits, not for formal training.
    """
    if leaf_probability.ndim != 2 or router_probability.ndim != 2:
        raise ValueError("dense probabilities must be rank-two tensors")
    if leaf_probability.shape[1] != router_probability.shape[1]:
        raise ValueError("leaf dimensions must match")
    occupancy = router_probability.sum(dim=0).clamp_min(float(epsilon))
    item_given_leaf = router_probability / occupancy.unsqueeze(0)
    return torch.matmul(leaf_probability, item_given_leaf.transpose(0, 1))


def hard_limit_item_distribution(
    leaf_probability: torch.Tensor,
    item_to_leaf: torch.Tensor,
) -> torch.Tensor:
    """Return the item distribution induced by a hard item-leaf bijection."""
    if item_to_leaf.ndim != 1:
        raise ValueError("item_to_leaf must be one-dimensional")
    if torch.unique(item_to_leaf).numel() != item_to_leaf.numel():
        raise ValueError("hard-limit mapping must be injective")
    return leaf_probability[:, item_to_leaf.long()]
