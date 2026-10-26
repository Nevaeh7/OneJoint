"""Bounded candidate leaves for sparse MINT-GR routing."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .balanced_code_lattice import BalancedCodeTree


def semantic_neighbors(
    semantic_base: torch.Tensor,
    count: int = 32,
    seed: int = 0,
) -> torch.Tensor:
    base = semantic_base.detach().cpu().float().numpy()
    count = min(max(1, int(count)), base.shape[0])
    try:
        import faiss

        index = faiss.IndexHNSWFlat(base.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = max(64, count * 2)
        index.add(base.astype(np.float32, copy=False))
        _, neighbors = index.search(base.astype(np.float32, copy=False), count)
    except ImportError:
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=count, metric="cosine", n_jobs=-1)
        model.fit(base)
        neighbors = model.kneighbors(return_distance=False)
    return torch.from_numpy(neighbors.astype(np.int64, copy=False))


def build_candidate_leaf_cache(
    tree: BalancedCodeTree,
    item_to_leaf: torch.Tensor,
    semantic_neighbor_rows: torch.Tensor,
    candidate_count: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """Create deterministic candidate rows with the current leaf at column zero."""
    tree.validate_item_to_leaf(item_to_leaf)
    item_to_leaf = item_to_leaf.detach().cpu().long()
    neighbors = semantic_neighbor_rows.detach().cpu().long()
    if neighbors.shape[0] != tree.num_leaves:
        raise ValueError("semantic neighbor rows must cover every catalog item")
    candidate_count = min(max(1, int(candidate_count)), tree.num_leaves)
    inverse = torch.empty_like(item_to_leaf)
    inverse[item_to_leaf] = torch.arange(tree.num_leaves)
    paths = tree.active_leaves
    rng = np.random.default_rng(seed)
    result = torch.empty(tree.num_leaves, candidate_count, dtype=torch.long)

    prefix_to_leaves: dict[tuple[int, ...], list[int]] = {}
    prefix_level = max(1, tree.depth - 2)
    for leaf_index, path in enumerate(paths.tolist()):
        prefix = tuple(int(value) for value in path[:prefix_level])
        prefix_to_leaves.setdefault(prefix, []).append(leaf_index)

    for item in range(tree.num_leaves):
        current_leaf = int(item_to_leaf[item])
        candidates = [current_leaf]
        seen = {current_leaf}
        for neighbor in neighbors[item].tolist():
            leaf = int(item_to_leaf[int(neighbor)])
            if leaf not in seen:
                candidates.append(leaf)
                seen.add(leaf)
            if len(candidates) >= candidate_count:
                break
        prefix = tuple(
            int(value) for value in paths[current_leaf, :prefix_level].tolist()
        )
        local = prefix_to_leaves[prefix]
        for leaf in rng.permutation(local).tolist():
            leaf = int(leaf)
            if leaf not in seen:
                candidates.append(leaf)
                seen.add(leaf)
            if len(candidates) >= candidate_count:
                break
        if len(candidates) < candidate_count:
            for leaf in rng.permutation(tree.num_leaves).tolist():
                leaf = int(leaf)
                if leaf not in seen:
                    candidates.append(leaf)
                    seen.add(leaf)
                if len(candidates) >= candidate_count:
                    break
        result[item] = torch.tensor(candidates[:candidate_count], dtype=torch.long)
    return result
