"""Sparse item-to-active-leaf posterior with semantic-preserving residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroInitResidualAdapter(nn.Module):
    def __init__(self, dimension: int = 128, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.first = nn.Linear(dimension, hidden)
        self.second = nn.Linear(hidden, dimension)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.first.weight)
        nn.init.zeros_(self.first.bias)
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.second(self.dropout(F.gelu(self.first(inputs))))


@dataclass
class SparsePosterior:
    leaf_ids: torch.Tensor
    probabilities: torch.Tensor
    logits: torch.Tensor
    entropy: torch.Tensor
    top1_mass: torch.Tensor


class SparseLeafRouter(nn.Module):
    def __init__(
        self,
        semantic_base: torch.Tensor,
        initial_leaf_embedding: torch.Tensor,
        collaborative_embedding: nn.Embedding,
        adapter_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if semantic_base.ndim != 2:
            raise ValueError("semantic base must have shape [items, dimension]")
        if initial_leaf_embedding.shape != semantic_base.shape:
            raise ValueError("one initial leaf embedding is required per item")
        if collaborative_embedding.embedding_dim != semantic_base.shape[1]:
            raise ValueError("critic item embedding must match router dimension")
        self.register_buffer("semantic_base", semantic_base.float().clone())
        self.adapter = ZeroInitResidualAdapter(
            dimension=semantic_base.shape[1],
            hidden=int(adapter_hidden),
            dropout=float(dropout),
        )
        self.collaborative_embedding = collaborative_embedding
        self.leaf_embedding = nn.Parameter(initial_leaf_embedding.float().clone())

    @property
    def dimension(self) -> int:
        return int(self.semantic_base.shape[1])

    def item_representation(
        self,
        item_rows: torch.Tensor,
        adapter_gate: float,
        collaborative_gate: float,
    ) -> torch.Tensor:
        rows = item_rows.long()
        base = self.semantic_base[rows]
        # Catalog row zero corresponds to atomic item ID one. Padding is kept
        # outside the router; hence collaborative rows are shifted by one.
        collaborative = self.collaborative_embedding(rows + 1)
        return F.normalize(
            base
            + float(adapter_gate) * self.adapter(base)
            + float(collaborative_gate) * collaborative,
            dim=-1,
        )

    def forward(
        self,
        item_rows: torch.Tensor,
        candidate_leaf_ids: torch.Tensor,
        temperature: float,
        top_m: int,
        adapter_gate: float,
        collaborative_gate: float,
        required_leaf_ids: torch.Tensor | None = None,
    ) -> SparsePosterior:
        if candidate_leaf_ids.shape[0] != item_rows.shape[0]:
            raise ValueError("one candidate row is required per item")
        if temperature <= 0:
            raise ValueError("router temperature must be positive")
        item = self.item_representation(
            item_rows,
            adapter_gate=adapter_gate,
            collaborative_gate=collaborative_gate,
        )
        leaf = F.normalize(self.leaf_embedding[candidate_leaf_ids.long()], dim=-1)
        logits = torch.einsum("bd,bcd->bc", item, leaf) / float(temperature)
        keep = min(max(1, int(top_m)), logits.shape[1])
        if required_leaf_ids is None:
            top_logits, top_positions = torch.topk(logits, keep, dim=-1)
        else:
            if required_leaf_ids.shape != (item_rows.shape[0],):
                raise ValueError("required leaf IDs must have one value per item")
            required_match = candidate_leaf_ids.eq(required_leaf_ids[:, None])
            if not required_match.any(dim=1).all():
                raise ValueError("required hard leaf is missing from router candidates")
            required_position = required_match.float().argmax(dim=1).long()
            if keep == 1:
                top_positions = required_position[:, None]
            else:
                other_logits = logits.masked_fill(required_match, float("-inf"))
                _, other_positions = torch.topk(other_logits, keep - 1, dim=-1)
                top_positions = torch.cat(
                    [required_position[:, None], other_positions], dim=1
                )
            top_logits = logits.gather(1, top_positions)
        top_leaf_ids = candidate_leaf_ids.gather(1, top_positions)
        probability = F.softmax(top_logits.float(), dim=-1).to(logits.dtype)
        entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=-1)
        return SparsePosterior(
            leaf_ids=top_leaf_ids,
            probabilities=probability,
            logits=top_logits,
            entropy=entropy,
            top1_mass=probability.max(dim=-1).values,
        )

    @torch.no_grad()
    def ema_update_leaf_embeddings(self, item_to_leaf: torch.Tensor, momentum: float) -> None:
        if not 0 <= momentum < 1:
            raise ValueError("EMA momentum must be in [0, 1)")
        rows = torch.arange(item_to_leaf.numel(), device=item_to_leaf.device)
        item = self.item_representation(rows, adapter_gate=1.0, collaborative_gate=0.5)
        inverse = torch.empty_like(item_to_leaf)
        inverse[item_to_leaf.long()] = rows
        target = item[inverse]
        self.leaf_embedding.mul_(float(momentum)).add_(
            target, alpha=1.0 - float(momentum)
        )
