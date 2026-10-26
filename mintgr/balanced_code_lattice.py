"""Fixed active code tree with exact capacity and injectivity invariants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


Prefix = Tuple[int, ...]


def _balanced_parts(total: int, parts: int) -> List[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if parts < 1:
        raise ValueError("parts must be positive")
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]


@dataclass(frozen=True)
class BalancedCodeTree:
    """A fixed set of active leaves behind one compact tree interface.

    The active leaf universe is independent of the item-to-leaf permutation.
    Consequently every valid hard map has zero collision and identical prefix
    capacity. Token IDs are level-local integers in ``[0, branching)``.
    """

    branching: int
    depth: int
    active_leaves: torch.Tensor

    @classmethod
    def build(
        cls,
        num_items: int,
        branching: int = 16,
        depth: int = 4,
        seed: int = 0,
    ) -> "BalancedCodeTree":
        if num_items < 1:
            raise ValueError("num_items must be positive")
        if branching < 2:
            raise ValueError("branching must be at least two")
        if depth < 1:
            raise ValueError("depth must be positive")
        if num_items > branching**depth:
            raise ValueError(
                f"{num_items} items exceed tree capacity {branching ** depth}"
            )

        rng = np.random.default_rng(seed)
        level_permutations: Dict[Prefix, np.ndarray] = {}

        def child_tokens(prefix: Prefix) -> np.ndarray:
            if prefix not in level_permutations:
                level_permutations[prefix] = rng.permutation(branching)
            return level_permutations[prefix]

        leaves: List[Prefix] = []

        def allocate(count: int, prefix: Prefix) -> None:
            remaining_depth = depth - len(prefix)
            if remaining_depth == 0:
                if count != 1:
                    raise AssertionError("a complete active path must hold one leaf")
                leaves.append(prefix)
                return

            maximum_child_capacity = branching ** (remaining_depth - 1)
            child_count = min(branching, count)
            while child_count > 1 and (
                (count + child_count - 1) // child_count
                > maximum_child_capacity
            ):
                child_count -= 1
            if (count + child_count - 1) // child_count > maximum_child_capacity:
                raise AssertionError("recursive active-leaf allocation exceeded capacity")
            counts = _balanced_parts(count, child_count)
            tokens = child_tokens(prefix)[:child_count]
            for token, child_size in zip(tokens.tolist(), counts):
                allocate(int(child_size), prefix + (int(token),))

        allocate(num_items, ())
        leaf_tensor = torch.tensor(sorted(leaves), dtype=torch.long)
        tree = cls(
            branching=int(branching),
            depth=int(depth),
            active_leaves=leaf_tensor,
        )
        tree.validate()
        return tree

    @property
    def num_leaves(self) -> int:
        return int(self.active_leaves.shape[0])

    def validate(self) -> None:
        leaves = self.active_leaves
        if leaves.ndim != 2 or leaves.shape[1] != self.depth:
            raise ValueError("active leaves must have shape [items, depth]")
        if leaves.numel() and (
            leaves.min().item() < 0
            or leaves.max().item() >= self.branching
        ):
            raise ValueError("active leaf token is outside the configured vocabulary")
        if torch.unique(leaves, dim=0).shape[0] != leaves.shape[0]:
            raise ValueError("active leaves must be unique")
        for level in range(1, self.depth + 1):
            _, counts = torch.unique(
                leaves[:, :level], dim=0, return_counts=True
            )
            if counts.numel() and int(counts.max() - counts.min()) > 1:
                raise ValueError(
                    f"active prefix capacities are not balanced at level {level}"
                )

    def path_to_id(self, paths: torch.Tensor) -> torch.Tensor:
        if paths.shape[-1] != self.depth:
            raise ValueError("path depth does not match the tree")
        powers = torch.tensor(
            [self.branching ** power for power in range(self.depth - 1, -1, -1)],
            dtype=torch.long,
            device=paths.device,
        )
        return (paths.long() * powers).sum(dim=-1)

    def compact_index(self) -> Dict[int, int]:
        path_ids = self.path_to_id(self.active_leaves).tolist()
        return {int(path_id): index for index, path_id in enumerate(path_ids)}

    def prefix_tables(self) -> List[Dict[Prefix, Tuple[int, ...]]]:
        """Return legal children for every prefix at every decoder level."""
        tables: List[Dict[Prefix, Tuple[int, ...]]] = []
        leaves = self.active_leaves.tolist()
        for level in range(self.depth):
            table: Dict[Prefix, set[int]] = {}
            for leaf in leaves:
                prefix = tuple(int(token) for token in leaf[:level])
                table.setdefault(prefix, set()).add(int(leaf[level]))
            tables.append(
                {
                    prefix: tuple(sorted(children))
                    for prefix, children in table.items()
                }
            )
        return tables

    def allowed_children(self, prefix: Sequence[int]) -> Tuple[int, ...]:
        prefix_tuple = tuple(int(token) for token in prefix)
        if len(prefix_tuple) >= self.depth:
            return ()
        return self.prefix_tables()[len(prefix_tuple)].get(prefix_tuple, ())

    def legal_token_mask(self, paths: torch.Tensor) -> torch.Tensor:
        """Return legal-child masks for each teacher-forced path position."""
        if paths.shape[-1] != self.depth:
            raise ValueError("paths must end in the configured tree depth")
        flat = paths.detach().long().reshape(-1, self.depth)
        mask = torch.zeros(
            flat.shape[0],
            self.depth,
            self.branching,
            dtype=torch.bool,
            device=paths.device,
        )
        tables = self.prefix_tables()
        flat_cpu = flat.cpu().tolist()
        for row_index, path in enumerate(flat_cpu):
            for level in range(self.depth):
                prefix = tuple(int(token) for token in path[:level])
                children = tables[level].get(prefix)
                if not children:
                    raise ValueError(f"illegal active-tree prefix: {prefix}")
                mask[row_index, level, list(children)] = True
        return mask.reshape(*paths.shape[:-1], self.depth, self.branching)

    def is_active(self, paths: torch.Tensor) -> torch.Tensor:
        if paths.shape[-1] != self.depth:
            raise ValueError("paths must end in the configured tree depth")
        active_ids = set(self.path_to_id(self.active_leaves).tolist())
        values = [
            int(path_id) in active_ids
            for path_id in self.path_to_id(paths).reshape(-1).tolist()
        ]
        return torch.tensor(values, dtype=torch.bool, device=paths.device).reshape(
            paths.shape[:-1]
        )

    def validate_item_to_leaf(self, item_to_leaf: torch.Tensor) -> None:
        mapping = item_to_leaf.detach().long().reshape(-1)
        if mapping.numel() != self.num_leaves:
            raise ValueError("hard map must assign every catalog item")
        if mapping.min().item() < 0 or mapping.max().item() >= self.num_leaves:
            raise ValueError("hard map contains an inactive leaf")
        if torch.unique(mapping).numel() != self.num_leaves:
            raise ValueError("hard map must be a bijection")

    def hard_paths(self, item_to_leaf: torch.Tensor) -> torch.Tensor:
        self.validate_item_to_leaf(item_to_leaf)
        return self.active_leaves.to(item_to_leaf.device)[item_to_leaf.long()]

    def prefix_capacity(self, level: int) -> Dict[Prefix, int]:
        if not 1 <= level <= self.depth:
            raise ValueError("level must be in [1, depth]")
        result: Dict[Prefix, int] = {}
        for leaf in self.active_leaves[:, :level].tolist():
            prefix = tuple(int(value) for value in leaf)
            result[prefix] = result.get(prefix, 0) + 1
        return result


class ActiveLeafConstraint:
    """Adapter from ``BalancedCodeTree`` to the generator beam interface."""

    def __init__(self, tree: BalancedCodeTree):
        self.depth = tree.depth
        self.tables = tree.prefix_tables()

    def __call__(self, batch_id: int, decoder_input_ids: torch.Tensor) -> List[int]:
        del batch_id
        prefix = tuple(int(value) for value in decoder_input_ids.tolist()[1:])
        if len(prefix) >= self.depth:
            return []
        return list(self.tables[len(prefix)].get(prefix, ()))
