"""Deep MINT-GR training module.

Callers provide an atomic-ID batch and a scalar training progress. The module
hides the active tree, hard bijection, sparse router cache, training-only item
critic, exact item-to-tree projection, and all inference-aligned generator
losses behind ``compute_loss``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

from .balanced_code_lattice import ActiveLeafConstraint, BalancedCodeTree
from .candidate_cache import build_candidate_leaf_cache, semantic_neighbors
from .item_critic import MintItemCritic
from .latent_item_likelihood import marginalized_item_nll, sparse_occupancy
from .mint_schedule import MintSchedule
from .semantic_balanced_init import initialize_semantic_tree
from .sparse_bijection_solver import BijectionUpdate, SparseBijectionSolver
from .sparse_leaf_router import SparseLeafRouter, SparsePosterior


@dataclass
class MintLossOutput:
    loss: torch.Tensor
    losses: Dict[str, torch.Tensor]
    diagnostics: Dict[str, float]


def _prefix_id(paths: torch.Tensor, level: int, branching: int) -> torch.Tensor:
    if level == 0:
        return torch.zeros(paths.shape[:-1], dtype=torch.long, device=paths.device)
    value = torch.zeros(paths.shape[:-1], dtype=torch.long, device=paths.device)
    for position in range(level):
        value = value * int(branching) + paths[..., position].long()
    return value


class MintGRSystem(nn.Module):
    """Joint generator/router/critic with a fixed active-leaf universe."""

    def __init__(
        self,
        generator: nn.Module,
        semantic_embeddings: np.ndarray,
        num_items_with_padding: int,
        config: Dict,
    ):
        super().__init__()
        self.config_dict = dict(config)
        self.generator = generator
        self.num_items = int(num_items_with_padding)
        self.num_catalog_items = self.num_items - 1
        self.branching = int(config.get("mint_branching", 16))
        self.depth = int(config.get("mint_depth", 4))
        self.candidate_count = int(config.get("mint_candidate_leaves", 64))
        self.seed = int(config.get("seed", 2020))
        if semantic_embeddings.shape[0] != self.num_catalog_items:
            raise ValueError(
                "semantic embedding rows must equal non-padding catalog items"
            )
        if generator.code_length != self.depth:
            raise ValueError("generator SID length must equal MINT tree depth")
        if generator.code_number != self.branching:
            raise ValueError("generator vocabulary must equal MINT branching")

        tree = BalancedCodeTree.build(
            num_items=self.num_catalog_items,
            branching=self.branching,
            depth=self.depth,
            seed=self.seed,
        )
        initialization = initialize_semantic_tree(
            semantic_embeddings=semantic_embeddings,
            tree=tree,
            output_dim=int(config.get("mint_router_dim", 128)),
            seed=self.seed,
            lloyd_iterations=int(config.get("mint_init_lloyd_iterations", 10)),
        )
        self.tree = tree
        self.prefix_constraint = ActiveLeafConstraint(tree)
        self.register_buffer("active_leaves", tree.active_leaves.clone())
        self.register_buffer(
            "item_to_leaf", initialization.item_to_leaf.long().clone()
        )
        inverse = torch.empty_like(self.item_to_leaf)
        inverse[self.item_to_leaf] = torch.arange(self.num_catalog_items)
        self.register_buffer("inverse_leaf_owner", inverse)

        critic_hidden = int(config.get("mint_critic_hidden", 128))
        self.critic = MintItemCritic(
            num_items=self.num_items,
            hidden_size=critic_hidden,
            layers=int(config.get("mint_critic_layers", 2)),
            heads=int(config.get("mint_critic_heads", 4)),
            feedforward_size=int(config.get("mint_critic_ffn", 512)),
            dropout=float(config.get("mint_critic_dropout", 0.2)),
            max_history=int(config.get("max_his_len", 50)),
        )
        if critic_hidden != initialization.base.shape[1]:
            raise ValueError("critic hidden size and router dimension must match")
        leaf_owner = torch.empty_like(self.item_to_leaf)
        leaf_owner[self.item_to_leaf] = torch.arange(self.num_catalog_items)
        initial_leaf_embedding = initialization.base[leaf_owner]
        self.router = SparseLeafRouter(
            semantic_base=initialization.base,
            initial_leaf_embedding=initial_leaf_embedding,
            collaborative_embedding=self.critic.item_embedding,
            adapter_hidden=int(config.get("mint_adapter_hidden", 256)),
            dropout=float(config.get("mint_router_dropout", 0.1)),
        )

        neighbor_rows = semantic_neighbors(
            initialization.base,
            count=int(config.get("mint_semantic_neighbors", 32)),
            seed=self.seed,
        )
        self.register_buffer("semantic_neighbor_rows", neighbor_rows.long())
        candidates = build_candidate_leaf_cache(
            tree,
            self.item_to_leaf,
            self.semantic_neighbor_rows,
            candidate_count=self.candidate_count,
            seed=self.seed,
        )
        self.register_buffer("candidate_leaf_ids", candidates)
        initial_q = torch.zeros(candidates.shape, dtype=torch.float32)
        initial_q[:, 0] = 1.0
        self.register_buffer("cached_q", initial_q)
        self.register_buffer("occupancy", torch.ones(self.num_catalog_items))

        self._register_tree_tables()
        self.bijection_solver = SparseBijectionSolver()
        self.map_update_count = 0
        self.cumulative_changed_items = 0
        self.last_map_update: Optional[BijectionUpdate] = None
        self.last_refresh_step = -1
        self.pca_explained_variance = float(
            np.asarray(initialization.pca_state["explained_variance_ratio"]).sum()
        )
        self.validate_invariants()

    def _register_tree_tables(self) -> None:
        leaves = self.active_leaves.cpu()
        for level in range(self.depth):
            group_count = self.branching**level
            legal = torch.zeros(group_count, self.branching, dtype=torch.bool)
            parent = _prefix_id(leaves, level, self.branching).cpu()
            legal[parent, leaves[:, level]] = True
            self.register_buffer(f"legal_children_{level}", legal)
            parent_child = parent * self.branching + leaves[:, level]
            self.register_buffer(
                f"leaf_parent_child_{level}", parent_child.long()
            )

    def validate_invariants(self) -> None:
        self.tree.validate_item_to_leaf(self.item_to_leaf.detach().cpu())
        if torch.unique(self.hard_sid_table()[1:], dim=0).shape[0] != self.num_catalog_items:
            raise AssertionError("MINT hard Item2SID table is not injective")
        if not self.tree.is_active(self.hard_sid_table()[1:].cpu()).all():
            raise AssertionError("MINT hard map contains an inactive SID")

    def hard_sid_table(self) -> torch.Tensor:
        table = torch.zeros(
            self.num_items,
            self.depth,
            dtype=torch.long,
            device=self.item_to_leaf.device,
        )
        table[1:] = self.active_leaves[self.item_to_leaf.long()]
        return table

    def critic_teacher_is_frozen(self, schedule: MintSchedule) -> bool:
        """Whether stage 2 should keep the current critic as a fixed teacher."""
        threshold = float(
            self.config_dict.get("mint_critic_freeze_progress", -1.0)
        )
        return (
            int(self.config_dict.get("mint_stage", 5)) == 2
            and threshold >= 0.0
            and float(schedule.progress) >= threshold
        )

    def _legal_mask(self, paths: torch.Tensor) -> torch.Tensor:
        if paths.shape[-1] != self.depth:
            raise ValueError("path tensor has the wrong SID length")
        masks = []
        for level in range(self.depth):
            parent = _prefix_id(paths, level, self.branching)
            table = getattr(self, f"legal_children_{level}")
            masks.append(table[parent.long()])
        return torch.stack(masks, dim=-2)

    def _masked_logits(self, logits: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        if logits.shape != paths.shape + (self.branching,):
            raise ValueError("generator logits and SID paths do not align")
        legal = self._legal_mask(paths)
        if not legal.gather(-1, paths.long().unsqueeze(-1)).all():
            raise AssertionError("target path is illegal in the active tree")
        return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)

    def _path_log_probability(
        self, logits: torch.Tensor, paths: torch.Tensor
    ) -> torch.Tensor:
        masked = self._masked_logits(logits, paths)
        token_log_probability = F.log_softmax(masked.float(), dim=-1).gather(
            -1, paths.long().unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probability.sum(dim=-1)

    def _hard_history(
        self, raw_item_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sid = self.hard_sid_table()[raw_item_ids.long()]
        expanded_mask = attention_mask.bool().unsqueeze(-1).expand(
            *attention_mask.shape, self.depth
        )
        return sid.reshape(raw_item_ids.shape[0], -1), expanded_mask.reshape(
            raw_item_ids.shape[0], -1
        )

    def _soft_history_embeddings(
        self, raw_item_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        item_rows = (raw_item_ids.long() - 1).clamp_min(0)
        candidate_ids = self.candidate_leaf_ids[item_rows]
        q = self.cached_q[item_rows]
        paths = self.active_leaves[candidate_ids]
        level_embeddings = []
        for level in range(self.depth):
            token_probability = torch.zeros(
                *raw_item_ids.shape,
                self.branching,
                dtype=q.dtype,
                device=q.device,
            )
            token_probability.scatter_add_(
                -1, paths[..., level].long(), q
            )
            level_embeddings.append(
                torch.matmul(
                    token_probability,
                    self.generator.token_embeddings[level].weight,
                )
            )
        item_major = torch.stack(level_embeddings, dim=2)
        embeddings = item_major.reshape(
            raw_item_ids.shape[0], -1, self.generator.hidden_size
        )
        expanded_mask = attention_mask.bool().unsqueeze(-1).expand(
            *attention_mask.shape, self.depth
        ).reshape(raw_item_ids.shape[0], -1)
        pad = self.generator.model.shared.weight[0].reshape(1, 1, -1)
        embeddings = torch.where(expanded_mask.unsqueeze(-1), embeddings, pad)
        return embeddings, expanded_mask

    def _score_paths_from_embeddings(
        self,
        history_embeddings: torch.Tensor,
        history_attention_mask: torch.Tensor,
        paths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if paths.ndim == 2:
            paths = paths.unsqueeze(1)
        batch_size, path_count, depth = paths.shape
        if depth != self.depth:
            raise ValueError("candidate path depth does not match generator")
        encoder = self.generator.get_encoder()(
            inputs_embeds=history_embeddings,
            attention_mask=history_attention_mask,
            return_dict=True,
        )
        repeated_hidden = encoder.last_hidden_state.repeat_interleave(
            path_count, dim=0
        )
        repeated_mask = history_attention_mask.repeat_interleave(path_count, dim=0)
        flat_paths = paths.reshape(batch_size * path_count, depth)
        output = self.generator(
            encoder_outputs=BaseModelOutput(last_hidden_state=repeated_hidden),
            attention_mask=repeated_mask,
            labels=flat_paths,
        )
        logits = output.logits.reshape(
            batch_size, path_count, depth, self.branching
        )
        log_probability = self._path_log_probability(logits, paths)
        return log_probability, logits

    def _project_teacher(
        self, item_probability: torch.Tensor, target_paths: torch.Tensor
    ) -> List[torch.Tensor]:
        leaf_probability = item_probability[:, self.inverse_leaf_owner.long()]
        conditionals = []
        for level in range(self.depth):
            group_count = self.branching**level
            destination = getattr(self, f"leaf_parent_child_{level}")
            mass = torch.zeros(
                item_probability.shape[0],
                group_count * self.branching,
                dtype=item_probability.dtype,
                device=item_probability.device,
            )
            mass.scatter_add_(
                1,
                destination.unsqueeze(0).expand(item_probability.shape[0], -1),
                leaf_probability,
            )
            parent = _prefix_id(target_paths, level, self.branching)
            child_offsets = torch.arange(
                self.branching, device=item_probability.device
            ).unsqueeze(0)
            selected = mass.gather(
                1, parent.unsqueeze(1) * self.branching + child_offsets
            )
            conditionals.append(selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-12))
        return conditionals

    def compute_loss(
        self,
        raw_item_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        targets: torch.Tensor,
        schedule: MintSchedule,
        step: int = 0,
    ) -> MintLossOutput:
        stage = int(self.config_dict.get("mint_stage", 5))
        targets = targets.reshape(-1).long()
        target_rows = targets - 1
        target_leaf = self.item_to_leaf[target_rows]
        target_paths = self.active_leaves[target_leaf]

        critic_frozen = self.critic_teacher_is_frozen(schedule)
        if critic_frozen:
            # A fixed teacher must also disable dropout; no_grad alone would
            # freeze the weights while leaving a different stochastic target
            # distribution on every batch.
            self.critic.eval()
            with torch.no_grad():
                critic_logits = self.critic(raw_item_ids, attention_mask)
        else:
            critic_logits = self.critic(raw_item_ids, attention_mask)
        critic_loss = F.cross_entropy(critic_logits.float(), targets)
        teacher_item_probability = F.softmax(
            critic_logits.detach().float(), dim=-1
        )[:, 1:]

        hard_history_ids, hard_attention = self._hard_history(
            raw_item_ids, attention_mask
        )
        hard_output = self.generator(
            input_ids=hard_history_ids,
            attention_mask=hard_attention,
            labels=target_paths,
        )
        hard_logits = hard_output.logits
        hard_path_log_probability = self._path_log_probability(
            hard_logits, target_paths
        )
        hard_loss = -hard_path_log_probability.mean()

        posterior = self.router(
            target_rows,
            self.candidate_leaf_ids[target_rows],
            temperature=schedule.router_temperature,
            top_m=schedule.top_m,
            adapter_gate=schedule.adapter_gate,
            collaborative_gate=schedule.collaborative_gate,
            required_leaf_ids=target_leaf,
        )
        candidate_paths = self.active_leaves[posterior.leaf_ids]
        soft_history, soft_attention = self._soft_history_embeddings(
            raw_item_ids, attention_mask
        )
        soft_path_log_probability, soft_path_logits = self._score_paths_from_embeddings(
            soft_history, soft_attention, candidate_paths
        )
        marginal_loss = marginalized_item_nll(
            path_log_probability=soft_path_log_probability,
            router_probability=posterior.probabilities,
            occupancy=self.occupancy.detach(),
            candidate_leaf_ids=posterior.leaf_ids,
        )

        teacher_conditionals = self._project_teacher(
            teacher_item_probability, target_paths
        )
        masked_hard = self._masked_logits(hard_logits, target_paths)
        hard_log_distribution = F.log_softmax(masked_hard.float(), dim=-1)
        projection_terms = []
        for level, teacher in enumerate(teacher_conditionals):
            projection_terms.append(
                (
                    teacher
                    * (
                        torch.log(teacher.clamp_min(1e-12))
                        - hard_log_distribution[:, level]
                    )
                ).sum(dim=-1)
            )
        projection_loss = torch.stack(projection_terms, dim=-1).sum(dim=-1).mean()

        soft_current_logits = soft_path_logits[:, 0]
        soft_current_masked = self._masked_logits(
            soft_current_logits, target_paths
        )
        soft_current_probability = F.softmax(
            soft_current_masked.detach().float(), dim=-1
        )
        consistency_loss = (
            soft_current_probability
            * (
                torch.log(soft_current_probability.clamp_min(1e-12))
                - hard_log_distribution
            )
        ).sum(dim=-1).mean()

        entropy_loss = posterior.entropy.mean()
        zero = hard_loss.new_zeros(())
        ranking_loss = self._ranking_loss(
            raw_item_ids,
            attention_mask,
            targets,
            critic_logits,
            step=step,
            enabled=stage >= 5 and schedule.ranking_weight > 0,
        )
        capacity_loss, locality_loss = self._catalog_regularizers(
            schedule=schedule,
            step=step,
            enabled=stage >= 3 and (
                schedule.capacity_weight > 0 or schedule.locality_weight > 0
            ),
        )
        losses = {
            "hard": hard_loss,
            "marginal": marginal_loss,
            "critic": critic_loss,
            "projection": projection_loss,
            "ranking": ranking_loss,
            "capacity": capacity_loss,
            "locality": locality_loss,
            "consistency": consistency_loss,
            "entropy": entropy_loss,
        }
        marginal_scale = float(stage >= 3) * float(
            self.config_dict.get("mint_marginal_scale", 1.0)
        )
        critic_scale = float(stage >= 2 and not critic_frozen) * float(
            self.config_dict.get("mint_critic_scale", 1.0)
        )
        projection_scale = float(stage >= 2) * float(
            self.config_dict.get("mint_projection_scale", 1.0)
        )
        ranking_scale = float(stage >= 5) * float(
            self.config_dict.get("mint_ranking_scale", 1.0)
        )
        capacity_scale = float(stage >= 3) * float(
            self.config_dict.get("mint_capacity_scale", 1.0)
        )
        locality_scale = float(stage >= 5) * float(
            self.config_dict.get("mint_locality_scale", 1.0)
        )
        consistency_scale = float(stage >= 3) * float(
            self.config_dict.get("mint_consistency_scale", 1.0)
        )
        entropy_scale = float(stage >= 3) * float(
            self.config_dict.get("mint_entropy_scale", 1.0)
        )
        total = (
            schedule.hard_weight * hard_loss
            + marginal_scale * schedule.marginal_weight * marginal_loss
            + critic_scale * schedule.critic_weight * critic_loss
            + projection_scale * schedule.projection_weight * projection_loss
            + ranking_scale * schedule.ranking_weight * ranking_loss
            + capacity_scale * schedule.capacity_weight * capacity_loss
            + locality_scale * schedule.locality_weight * locality_loss
            + consistency_scale * schedule.consistency_weight * consistency_loss
            + entropy_scale * schedule.entropy_weight * entropy_loss
        )
        diagnostics = {
            "router_entropy": float(posterior.entropy.detach().mean()),
            "router_top1_mass": float(posterior.top1_mass.detach().mean()),
            "hard_path_nll": float(hard_loss.detach()),
            "marginal_item_nll": float(marginal_loss.detach()),
            "critic_nll": float(critic_loss.detach()),
            "critic_frozen": float(critic_frozen),
            "projection_kl": float(projection_loss.detach()),
            "soft_hard_kl": float(consistency_loss.detach()),
            "collision_rate": 0.0,
        }
        return MintLossOutput(total, losses, diagnostics)

    def _ranking_loss(
        self,
        raw_item_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        targets: torch.Tensor,
        critic_logits: torch.Tensor,
        step: int,
        enabled: bool,
    ) -> torch.Tensor:
        zero = critic_logits.new_zeros(())
        interval = int(self.config_dict.get("mint_rank_interval", 2))
        if not enabled or interval < 1 or int(step) % interval:
            return zero
        fraction = float(self.config_dict.get("mint_rank_batch_fraction", 0.25))
        subset_size = max(1, int(raw_item_ids.shape[0] * fraction))
        subset_size = min(subset_size, raw_item_ids.shape[0])
        negative_count = min(
            int(self.config_dict.get("mint_rank_negatives", 32)),
            self.num_catalog_items - 1,
        )
        rows = torch.arange(subset_size, device=raw_item_ids.device)
        selected_targets = targets[:subset_size]
        ranking_logits = critic_logits[:subset_size].detach().clone()
        ranking_logits[:, 0] = torch.finfo(ranking_logits.dtype).min
        ranking_logits[rows, selected_targets] = torch.finfo(ranking_logits.dtype).min
        negatives = torch.topk(
            ranking_logits, k=negative_count, dim=-1
        ).indices
        item_candidates = torch.cat(
            [selected_targets.unsqueeze(1), negatives], dim=1
        )
        catalog_rows = item_candidates - 1
        leaves = self.item_to_leaf[catalog_rows]
        paths = self.active_leaves[leaves]
        hard_ids, hard_mask = self._hard_history(
            raw_item_ids[:subset_size], attention_mask[:subset_size]
        )
        history_embeddings = self.generator.get_input_embeddings(
            hard_ids.clone(), hard_mask
        )
        path_scores, _ = self._score_paths_from_embeddings(
            history_embeddings, hard_mask, paths
        )
        return F.cross_entropy(
            path_scores.float(),
            torch.zeros(subset_size, dtype=torch.long, device=path_scores.device),
        )

    def _catalog_regularizers(
        self,
        schedule: MintSchedule,
        step: int,
        enabled: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = self.router.leaf_embedding.new_zeros(())
        interval = int(self.config_dict.get("mint_catalog_regularizer_interval", 4))
        if not enabled or interval < 1 or int(step) % interval:
            return zero, zero
        sample_size = min(
            int(self.config_dict.get("mint_catalog_regularizer_items", 512)),
            self.num_catalog_items,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(step))
        sampled_cpu = torch.randperm(
            self.num_catalog_items, generator=generator
        )[:sample_size]
        sampled = sampled_cpu.to(self.item_to_leaf.device)
        sampled_current = self.item_to_leaf[sampled]
        posterior = self.router(
            sampled,
            self.candidate_leaf_ids[sampled],
            temperature=schedule.router_temperature,
            top_m=schedule.top_m,
            adapter_gate=schedule.adapter_gate,
            collaborative_gate=schedule.collaborative_gate,
            required_leaf_ids=sampled_current,
        )
        sampled_paths = self.active_leaves[posterior.leaf_ids]
        capacity_terms = []
        for level in range(1, self.depth):
            group_count = self.branching**level
            prefix = _prefix_id(sampled_paths, level, self.branching)
            soft_mass = torch.zeros(
                group_count,
                dtype=posterior.probabilities.dtype,
                device=posterior.probabilities.device,
            )
            soft_mass.scatter_add_(
                0, prefix.reshape(-1), posterior.probabilities.reshape(-1)
            )
            soft_mass = soft_mass / float(sample_size)
            hard_prefix = _prefix_id(
                self.active_leaves, level, self.branching
            )
            target_mass = torch.bincount(
                hard_prefix,
                minlength=group_count,
            ).to(soft_mass.dtype) / float(self.num_catalog_items)
            active = target_mass > 0
            capacity_terms.append(
                (((soft_mass[active] - target_mass[active]) / target_mass[active]) ** 2).mean()
            )
        capacity_loss = torch.stack(capacity_terms).mean() if capacity_terms else zero

        neighbor = self.semantic_neighbor_rows[sampled, 1].to(sampled.device)
        neighbor_posterior = self.router(
            neighbor,
            self.candidate_leaf_ids[neighbor],
            temperature=schedule.router_temperature,
            top_m=schedule.top_m,
            adapter_gate=schedule.adapter_gate,
            collaborative_gate=schedule.collaborative_gate,
            required_leaf_ids=self.item_to_leaf[neighbor],
        )
        neighbor_paths = self.active_leaves[neighbor_posterior.leaf_ids]
        locality_terms = []
        for level, weight in ((1, 1.0), (2, 0.5)):
            if level >= self.depth:
                continue
            left_prefix = _prefix_id(sampled_paths, level, self.branching)
            right_prefix = _prefix_id(neighbor_paths, level, self.branching)
            shared = left_prefix.unsqueeze(-1).eq(right_prefix.unsqueeze(-2))
            probability = (
                posterior.probabilities.unsqueeze(-1)
                * neighbor_posterior.probabilities.unsqueeze(-2)
                * shared.to(posterior.probabilities.dtype)
            ).sum(dim=(-1, -2))
            locality_terms.append(-float(weight) * torch.log(probability.clamp_min(1e-8)))
        locality_loss = (
            torch.stack(locality_terms, dim=-1).sum(dim=-1).mean()
            if locality_terms else zero
        )
        return capacity_loss, locality_loss

    @torch.no_grad()
    def refresh_catalog(
        self,
        schedule: MintSchedule,
        step: int,
        chunk_size: int = 256,
    ) -> Dict[str, float]:
        q = torch.zeros_like(self.cached_q)
        for start in range(0, self.num_catalog_items, int(chunk_size)):
            end = min(self.num_catalog_items, start + int(chunk_size))
            rows = torch.arange(start, end, device=self.item_to_leaf.device)
            candidates = self.candidate_leaf_ids[rows]
            posterior = self.router(
                rows,
                candidates,
                temperature=schedule.router_temperature,
                top_m=schedule.top_m,
                adapter_gate=schedule.adapter_gate,
                collaborative_gate=schedule.collaborative_gate,
                required_leaf_ids=self.item_to_leaf[rows],
            )
            matches = candidates.unsqueeze(-1).eq(posterior.leaf_ids.unsqueeze(1))
            q[start:end] = (
                matches.to(posterior.probabilities.dtype)
                * posterior.probabilities.unsqueeze(1)
            ).sum(dim=-1)
        self.cached_q.copy_(q)
        self.occupancy.copy_(
            sparse_occupancy(
                self.candidate_leaf_ids,
                self.cached_q,
                self.num_catalog_items,
            )
        )
        self.last_refresh_step = int(step)
        entropy = -(
            q * torch.log(q.clamp_min(1e-8))
        ).sum(dim=-1)
        current_probability = q[:, 0]
        return {
            "mean_entropy": float(entropy.mean()),
            "top1_mass": float(q.max(dim=-1).values.mean()),
            "current_leaf_mass": float(current_probability.mean()),
            "occupancy_min": float(self.occupancy.min()),
            "occupancy_max": float(self.occupancy.max()),
            "occupancy_cv": float(
                self.occupancy.std() / self.occupancy.mean().clamp_min(1e-8)
            ),
        }

    @torch.no_grad()
    def project_hard_map(
        self,
        move_budget: float,
        move_penalty: float,
    ) -> BijectionUpdate:
        maximum = int(self.num_catalog_items * float(move_budget))
        costs = -torch.log(self.cached_q.clamp_min(1e-8))
        costs = costs + float(move_penalty) * self.candidate_leaf_ids.ne(
            self.item_to_leaf[:, None]
        )
        update = self.bijection_solver.solve(
            self.item_to_leaf,
            self.candidate_leaf_ids,
            costs,
            max_changed_items=maximum,
        )
        if update.accepted:
            self.item_to_leaf.copy_(update.item_to_leaf.to(self.item_to_leaf.device))
            self.inverse_leaf_owner[self.item_to_leaf] = torch.arange(
                self.num_catalog_items, device=self.item_to_leaf.device
            )
            candidates = build_candidate_leaf_cache(
                self.tree,
                self.item_to_leaf.detach().cpu(),
                self.semantic_neighbor_rows.detach().cpu(),
                candidate_count=self.candidate_count,
                seed=self.seed + self.map_update_count + 1,
            ).to(self.item_to_leaf.device)
            self.candidate_leaf_ids.copy_(candidates)
            self.cached_q.zero_()
            self.cached_q[:, 0] = 1.0
            self.occupancy.fill_(1.0)
            self.map_update_count += 1
            self.cumulative_changed_items += update.changed_items
            self.validate_invariants()
        self.last_map_update = update
        return update

    def audit_state(self) -> Dict[str, float]:
        self.validate_invariants()
        occupancy_mean = self.occupancy.mean().clamp_min(1e-8)
        return {
            "items": float(self.num_catalog_items),
            "active_leaves": float(self.active_leaves.shape[0]),
            "unique_hard_sids": float(
                torch.unique(self.hard_sid_table()[1:], dim=0).shape[0]
            ),
            "collision_rate": 0.0,
            "cumulative_changed_rate": float(
                self.cumulative_changed_items / self.num_catalog_items
            ),
            "occupancy_min": float(self.occupancy.min()),
            "occupancy_max": float(self.occupancy.max()),
            "occupancy_cv": float(self.occupancy.std() / occupancy_mean),
            "pca_explained_variance": self.pca_explained_variance,
        }
