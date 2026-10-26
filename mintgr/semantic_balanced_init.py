"""Deterministic, non-neural semantic initialization for MINT-GR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .balanced_code_lattice import BalancedCodeTree


def _l2_normalize(array: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, epsilon)


def pca_project_and_normalize(
    embeddings: np.ndarray,
    output_dim: int = 128,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if embeddings.ndim != 2:
        raise ValueError("semantic embeddings must have shape [items, dimension]")
    if embeddings.shape[0] < 2:
        raise ValueError("semantic initialization needs at least two items")
    if not np.isfinite(embeddings).all():
        raise ValueError("semantic embeddings contain NaN or infinity")
    effective_dim = min(
        int(output_dim), embeddings.shape[0] - 1, embeddings.shape[1]
    )
    if effective_dim < 1:
        raise ValueError("PCA output dimension is empty")
    try:
        from sklearn.decomposition import PCA

        pca = PCA(
            n_components=effective_dim,
            svd_solver="randomized" if effective_dim < embeddings.shape[1] else "full",
            random_state=int(seed),
        )
        projected = pca.fit_transform(embeddings.astype(np.float32, copy=False))
        components = pca.components_.astype(np.float32, copy=False)
        mean = pca.mean_.astype(np.float32, copy=False)
        explained = pca.explained_variance_ratio_.astype(np.float32, copy=False)
    except ImportError:
        centered = embeddings.astype(np.float64) - embeddings.mean(axis=0)
        _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
        components = right[:effective_dim].astype(np.float32)
        mean = embeddings.mean(axis=0).astype(np.float32)
        projected = centered @ components.T
        variance = singular_values**2
        explained = (variance[:effective_dim] / variance.sum()).astype(np.float32)
    if effective_dim < output_dim:
        projected = np.pad(
            projected,
            ((0, 0), (0, int(output_dim) - effective_dim)),
            mode="constant",
        )
        components = np.pad(
            components,
            ((0, int(output_dim) - effective_dim), (0, 0)),
            mode="constant",
        )
        explained = np.pad(
            explained,
            (0, int(output_dim) - effective_dim),
            mode="constant",
        )
    projected = _l2_normalize(projected.astype(np.float32, copy=False))
    return projected, {
        "components": components,
        "mean": mean,
        "explained_variance_ratio": explained,
    }


def _kmeans_plus_plus(
    features: np.ndarray,
    groups: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = [int(rng.integers(features.shape[0]))]
    min_distance = np.full(features.shape[0], np.inf, dtype=np.float64)
    for _ in range(1, groups):
        similarity = features @ features[centers[-1]]
        distance = np.maximum(0.0, 2.0 - 2.0 * similarity)
        min_distance = np.minimum(min_distance, distance)
        total = float(min_distance.sum())
        if total <= 1e-12:
            remaining = [index for index in range(features.shape[0]) if index not in centers]
            centers.append(int(remaining[0]))
        else:
            centers.append(int(rng.choice(features.shape[0], p=min_distance / total)))
    return features[np.asarray(centers, dtype=np.int64)].copy()


def _capacity_greedy_assignment(
    scores: np.ndarray,
    capacities: Sequence[int],
) -> np.ndarray:
    item_count, group_count = scores.shape
    if sum(int(value) for value in capacities) != item_count:
        raise ValueError("balanced assignment capacities must sum to item count")
    remaining = np.asarray(capacities, dtype=np.int64).copy()
    assignment = np.full(item_count, -1, dtype=np.int64)
    # Assign the most decisive item/group pairs first. Stable mergesort and the
    # flattened row-major index make ties deterministic.
    order = np.argsort(-scores.reshape(-1), kind="mergesort")
    for pair in order.tolist():
        item = pair // group_count
        group = pair % group_count
        if assignment[item] < 0 and remaining[group] > 0:
            assignment[item] = group
            remaining[group] -= 1
        if (assignment >= 0).all():
            break
    if (assignment < 0).any() or remaining.sum() != 0:
        raise RuntimeError("capacity-constrained semantic assignment is incomplete")
    return assignment


def _balanced_partition(
    features: np.ndarray,
    capacities: Sequence[int],
    rng: np.random.Generator,
    iterations: int,
) -> List[np.ndarray]:
    groups = len(capacities)
    if groups == 1:
        return [np.arange(features.shape[0], dtype=np.int64)]
    centers = _kmeans_plus_plus(features, groups, rng)
    assignment = np.zeros(features.shape[0], dtype=np.int64)
    for _ in range(max(1, int(iterations))):
        scores = features @ centers.T
        updated = _capacity_greedy_assignment(scores, capacities)
        if np.array_equal(updated, assignment):
            assignment = updated
            break
        assignment = updated
        next_centers = []
        for group in range(groups):
            members = features[assignment == group]
            center = members.mean(axis=0)
            center /= max(float(np.linalg.norm(center)), 1e-12)
            next_centers.append(center)
        centers = np.asarray(next_centers, dtype=np.float32)
    return [
        np.flatnonzero(assignment == group).astype(np.int64)
        for group in range(groups)
    ]


def recursive_balanced_semantic_assignment(
    features: np.ndarray,
    tree: BalancedCodeTree,
    seed: int = 0,
    lloyd_iterations: int = 10,
) -> np.ndarray:
    """Assign every item to an active leaf while preserving semantic locality."""
    if features.shape != (tree.num_leaves, features.shape[1]):
        if features.ndim != 2 or features.shape[0] != tree.num_leaves:
            raise ValueError("one semantic feature row is required per active leaf")
    features = _l2_normalize(features.astype(np.float32, copy=False))
    rng = np.random.default_rng(seed)
    leaves = tree.active_leaves.cpu().numpy()
    item_to_leaf = np.full(tree.num_leaves, -1, dtype=np.int64)

    def recurse(item_rows: np.ndarray, leaf_rows: np.ndarray, level: int) -> None:
        if item_rows.size != leaf_rows.size:
            raise AssertionError("semantic partition lost item/leaf cardinality")
        if level == tree.depth:
            if item_rows.size != 1:
                raise AssertionError("complete tree path must own exactly one item")
            item_to_leaf[item_rows[0]] = leaf_rows[0]
            return
        tokens = sorted(set(int(value) for value in leaves[leaf_rows, level]))
        child_leaf_rows = [
            leaf_rows[leaves[leaf_rows, level] == token] for token in tokens
        ]
        capacities = [int(rows.size) for rows in child_leaf_rows]
        partitions = _balanced_partition(
            features[item_rows], capacities, rng, lloyd_iterations
        )
        for local_rows, next_leaf_rows in zip(partitions, child_leaf_rows):
            recurse(item_rows[local_rows], next_leaf_rows, level + 1)

    recurse(
        np.arange(tree.num_leaves, dtype=np.int64),
        np.arange(tree.num_leaves, dtype=np.int64),
        0,
    )
    if (item_to_leaf < 0).any() or np.unique(item_to_leaf).size != tree.num_leaves:
        raise RuntimeError("semantic initialization did not produce a bijection")
    tree.validate_item_to_leaf(torch.from_numpy(item_to_leaf))
    return item_to_leaf


@dataclass
class SemanticInitialization:
    base: torch.Tensor
    item_to_leaf: torch.Tensor
    pca_state: Dict[str, np.ndarray]


def initialize_semantic_tree(
    semantic_embeddings: np.ndarray,
    tree: BalancedCodeTree,
    output_dim: int = 128,
    seed: int = 0,
    lloyd_iterations: int = 10,
) -> SemanticInitialization:
    base, pca_state = pca_project_and_normalize(
        semantic_embeddings, output_dim=output_dim, seed=seed
    )
    item_to_leaf = recursive_balanced_semantic_assignment(
        base,
        tree=tree,
        seed=seed,
        lloyd_iterations=lloyd_iterations,
    )
    return SemanticInitialization(
        base=torch.from_numpy(base),
        item_to_leaf=torch.from_numpy(item_to_leaf).long(),
        pca_state=pca_state,
    )
