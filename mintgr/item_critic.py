"""Training-only full-softmax sequential item critic."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MintItemCritic(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 128,
        layers: int = 2,
        heads: int = 4,
        feedforward_size: int = 512,
        dropout: float = 0.2,
        max_history: int = 50,
    ):
        super().__init__()
        if num_items < 2:
            raise ValueError("critic requires padding plus at least one item")
        self.num_items = int(num_items)
        self.hidden_size = int(hidden_size)
        self.item_embedding = nn.Embedding(
            self.num_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_history, self.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(heads),
            dim_feedforward=int(feedforward_size),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(layers))
        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        self.output_bias = nn.Parameter(torch.zeros(self.num_items))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_bias)

    def forward(
        self, item_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if item_ids.shape != attention_mask.shape:
            raise ValueError("critic IDs and attention mask must align")
        batch_size, history_length = item_ids.shape
        if history_length > self.position_embedding.num_embeddings:
            raise ValueError(
                f"critic history length {history_length} exceeds configured "
                f"maximum {self.position_embedding.num_embeddings}"
            )
        if item_ids.numel() and (
            item_ids.min().item() < 0 or item_ids.max().item() >= self.num_items
        ):
            raise ValueError("critic input contains an out-of-range atomic item ID")
        positions = torch.arange(history_length, device=item_ids.device)
        hidden = self.item_embedding(item_ids.long())
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = self.dropout(hidden)
        causal_mask = torch.triu(
            torch.ones(
                history_length,
                history_length,
                dtype=torch.bool,
                device=item_ids.device,
            ),
            diagonal=1,
        )
        encoded = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask.bool(),
        )
        lengths = attention_mask.long().sum(dim=1).clamp_min(1)
        rows = torch.arange(batch_size, device=item_ids.device)
        final = encoded[rows, lengths - 1]
        final = self.norm(final)
        logits = F.linear(final, self.item_embedding.weight, self.output_bias)
        logits[:, 0] = torch.finfo(logits.dtype).min
        return logits
