"""Validation-selected single-run trainer for MINT-GR."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from logging import getLogger
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import get_scheduler

from utils import ensure_dir, init_device_seed, log

from .mint_schedule import schedule_at
from .system import MintGRSystem


def _atomic_torch_save(payload: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(path)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _atomic_json_save(payload: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(path)
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class MintGRTrainer:
    """Own optimization, validation selection, checkpoints, and audit logs."""

    def __init__(
        self,
        config: Dict,
        system: MintGRSystem,
        accelerator,
        train_data,
        valid_data,
        test_data,
    ):
        self.config = config
        self.accelerator = accelerator
        self.logger = getLogger()
        self.system = system
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.save_path = config["save_path"]
        ensure_dir(self.save_path)
        self.manifest_path = os.path.join(self.save_path, "manifest.json")
        self.epochs = int(config["epochs"])
        self.early_stop = int(config["early_stop"])
        self.valid_metric = str(config.get("valid_metric", "ndcg@10"))
        self.max_topk = max(
            int(metric.split("@")[1])
            for metric in str(config["metrics"]).split(",")
        )
        if int(config["num_beams"]) < self.max_topk:
            raise ValueError("MintGR beam size must cover the largest metric cutoff")
        if accelerator.num_processes != 1:
            raise ValueError("MintGR formal v1 is intentionally single-process")
        if config.get("rqvae_path"):
            raise ValueError("MintGR forbids every RQ-VAE checkpoint")
        if bool(config.get("evaluate_test_each_epoch", False)):
            raise ValueError("MintGR forbids per-epoch test tuning")

        self.max_steps = max(1, len(train_data) * self.epochs)
        self.optimizer = self._build_optimizer(system)
        self.scheduler = get_scheduler(
            name=str(config.get("lr_scheduler_type", "cosine")),
            optimizer=self.optimizer,
            num_warmup_steps=int(config.get("warmup_steps", 500)),
            num_training_steps=self.max_steps,
        )
        (
            self.system,
            self.optimizer,
            self.scheduler,
            self.train_data,
            self.valid_data,
            self.test_data,
        ) = accelerator.prepare(
            self.system,
            self.optimizer,
            self.scheduler,
            self.train_data,
            self.valid_data,
            self.test_data,
        )
        unwrapped = accelerator.unwrap_model(self.system)
        unwrapped.generator.device = accelerator.device
        self.process_seed = init_device_seed(
            int(config["seed"]), accelerator.process_index
        )

        self.global_step = 0
        self.start_epoch = 0
        self.best_score = float("-inf")
        self.best_result: Dict[str, float] = {}
        self.best_epoch: Optional[int] = None
        self.best_ckpt: Optional[str] = None
        self.final_test_results: Optional[Dict[str, float]] = None
        self.epoch_history = []
        self.bad_epochs = 0
        resume_path = str(config.get("resume_from", "") or "")
        if resume_path:
            self.load_checkpoint(resume_path, load_optimizer=True)
        self._write_manifest(status="initialized")

    def _build_optimizer(self, system: MintGRSystem) -> torch.optim.Optimizer:
        seen = set()
        groups = []

        def add(parameters: Iterable[torch.nn.Parameter], lr: float, name: str) -> None:
            values = []
            for parameter in parameters:
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    values.append(parameter)
            if values:
                groups.append({"params": values, "lr": float(lr), "name": name})

        add(
            system.generator.parameters(),
            self.config.get("mint_generator_lr", self.config.get("lr_rec", 1e-3)),
            "generator",
        )
        add(
            (
                parameter
                for name, parameter in system.router.named_parameters()
                if not name.startswith("collaborative_embedding.")
            ),
            self.config.get("mint_router_lr", 1e-4),
            "router",
        )
        add(
            system.critic.item_embedding.parameters(),
            self.config.get("mint_collaborative_lr", 5e-4),
            "collaborative_item_embedding",
        )
        add(
            (
                parameter
                for name, parameter in system.critic.named_parameters()
                if not name.startswith("item_embedding.")
            ),
            self.config.get("mint_critic_lr", 1e-3),
            "critic",
        )
        return torch.optim.AdamW(
            groups,
            weight_decay=float(self.config.get("weight_decay", 0.05)),
        )

    def log(self, message: str, level: str = "info") -> None:
        log(message, self.accelerator, self.logger, level=level)

    def _system(self) -> MintGRSystem:
        return self.accelerator.unwrap_model(self.system)

    def _checkpoint_payload(self, epoch: int) -> Dict:
        return {
            "format": "mintgr_resume_v1",
            "epoch": int(epoch),
            "next_epoch": int(epoch) + 1,
            "global_step": int(self.global_step),
            "system": self._system().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_score": float(self.best_score),
            "best_result": _json_safe(self.best_result),
            "best_epoch": self.best_epoch,
            "best_ckpt": self.best_ckpt,
            "bad_epochs": int(self.bad_epochs),
            "epoch_history": _json_safe(self.epoch_history),
            "config": {
                key: _json_safe(value)
                for key, value in self.config.items()
                if key not in {"accelerator", "device"}
            },
        }

    def save_checkpoint(self, epoch: int, name: str) -> str:
        path = os.path.join(self.save_path, name)
        if self.accelerator.is_main_process:
            _atomic_torch_save(self._checkpoint_payload(epoch), path)
        self.accelerator.wait_for_everyone()
        return path

    def load_checkpoint(self, path: str, load_optimizer: bool) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "mintgr_resume_v1":
            raise ValueError("checkpoint is not a MINT-GR resumable bundle")
        self._system().load_state_dict(checkpoint["system"], strict=True)
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = int(checkpoint["next_epoch"])
            self.global_step = int(checkpoint["global_step"])
            self.best_score = float(checkpoint.get("best_score", float("-inf")))
            self.best_result = checkpoint.get("best_result", {})
            self.best_epoch = checkpoint.get("best_epoch")
            self.best_ckpt = checkpoint.get("best_ckpt")
            self.bad_epochs = int(checkpoint.get("bad_epochs", 0))
            self.epoch_history = checkpoint.get("epoch_history", [])
        self._system().validate_invariants()
        self.log(
            f"[MINT Resume] path={path}, next_epoch={self.start_epoch}, "
            f"global_step={self.global_step}, best_{self.valid_metric}={self.best_score:.6f}"
        )

    def _write_manifest(self, **updates) -> None:
        if not self.accelerator.is_main_process:
            return
        manifest = {
            "method": "OneJoint",
            "status": "running",
            "dataset": self.config["dataset"],
            "seed": int(self.config["seed"]),
            "rqvae_checkpoint": self.config.get("rqvae_path"),
            "tokenizer_pretraining": False,
            "critic_used_at_inference": False,
            "sid": {
                "branching": int(self.config.get("mint_branching", 16)),
                "depth": int(self.config.get("mint_depth", 4)),
                "suffix": False,
                "level_specific_tokens": True,
            },
            "selection_metric": self.valid_metric,
            "test_during_tuning": False,
            "global_step": int(self.global_step),
            "best_score": float(self.best_score),
            "best_result": _json_safe(self.best_result),
            "best_epoch": self.best_epoch,
            "best_checkpoint": self.best_ckpt,
            "epoch_history": _json_safe(self.epoch_history),
            "audit": _json_safe(self._system().audit_state()),
        }
        manifest.update(_json_safe(updates))
        _atomic_json_save(manifest, self.manifest_path)

    @staticmethod
    def _module_gradient_norm(module: torch.nn.Module) -> float:
        squared = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                value = parameter.grad.detach().float()
                squared += float((value * value).sum())
        return math.sqrt(squared)

    def _gradient_audit(self, epoch: int) -> Dict[str, float]:
        system = self._system()
        report = {
            "generator": self._module_gradient_norm(system.generator),
            "router": self._module_gradient_norm(system.router),
            "critic": self._module_gradient_norm(system.critic),
        }
        for name, value in report.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"MINT {name} gradient is not finite")
        progress = self.global_step / self.max_steps
        stage = int(self.config.get("mint_stage", 5))
        critic_freeze_progress = float(
            self.config.get("mint_critic_freeze_progress", -1.0)
        )
        critic_frozen = (
            stage == 2
            and critic_freeze_progress >= 0.0
            and progress >= critic_freeze_progress
        )
        required = ["generator"]
        if stage >= 2 and not critic_frozen:
            required.append("critic")
        if stage >= 3 and progress < 0.85:
            required.append("router")
        if progress < 0.85:
            for name in required:
                value = report[name]
                if value <= 0:
                    raise RuntimeError(
                        f"MINT joint-gradient gate failed for {name} at epoch {epoch}"
                    )
        return report

    def _train_epoch(self, epoch: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        self.system.train()
        totals: Dict[str, float] = {}
        samples = 0
        first_gradient_report = None
        maximum_batches = int(self.config.get("max_train_batches", 0))
        log_every = int(self.config.get("mint_log_every", 50))
        loss_limit = float(self.config.get("mint_fail_loss_abs_max", 100.0))
        refresh_interval = int(self.config.get("mint_catalog_refresh_steps", 200))
        start = time.time()
        for batch_index, batch in enumerate(self.train_data):
            if maximum_batches > 0 and batch_index >= maximum_batches:
                break
            progress = min(1.0, self.global_step / self.max_steps)
            schedule = schedule_at(
                progress,
                move_budget_peak=float(self.config.get("mint_move_budget_peak", 0.05)),
                hard_alignment_start=float(
                    self.config.get("mint_hard_alignment_start", 0.85)
                ),
            )
            with self.accelerator.accumulate(self.system):
                output = self.system.compute_loss(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["targets"],
                    schedule=schedule,
                    step=self.global_step,
                )
                if not torch.isfinite(output.loss):
                    raise FloatingPointError("MINT total loss became NaN or infinity")
                if abs(float(output.loss.detach())) > loss_limit:
                    raise RuntimeError(
                        f"MINT total loss {float(output.loss.detach()):.6f} "
                        f"exceeded fail-closed limit {loss_limit}"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                self.accelerator.backward(output.loss)
                if first_gradient_report is None:
                    first_gradient_report = self._gradient_audit(epoch)
                gradient_norm = self.accelerator.clip_grad_norm_(
                    self.system.parameters(),
                    float(self.config.get("mint_grad_clip", 1.0)),
                )
                if not torch.isfinite(torch.as_tensor(gradient_norm)):
                    raise FloatingPointError("MINT clipped gradient norm is not finite")
                self.optimizer.step()
                self.scheduler.step()

            batch_size = int(batch["targets"].shape[0])
            samples += batch_size
            totals["loss"] = totals.get("loss", 0.0) + float(output.loss.detach()) * batch_size
            for name, value in output.losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            for name, value in output.diagnostics.items():
                totals[name] = totals.get(name, 0.0) + float(value) * batch_size
            self.global_step += 1
            if refresh_interval > 0 and self.global_step % refresh_interval == 0:
                refresh = self._system().refresh_catalog(
                    schedule=schedule,
                    step=self.global_step,
                    chunk_size=int(self.config.get("mint_catalog_refresh_chunk", 256)),
                )
                self.log(f"[MINT Catalog Refresh] step={self.global_step}, {refresh}")
            if batch_index == 0 or (batch_index + 1) % log_every == 0:
                self.log(
                    f"[MINT Train] epoch={epoch}, batch={batch_index + 1}, "
                    f"phase={schedule.phase}, loss={float(output.loss.detach()):.6f}, "
                    f"hard={float(output.losses['hard'].detach()):.6f}, "
                    f"marg={float(output.losses['marginal'].detach()):.6f}, "
                    f"critic={float(output.losses['critic'].detach()):.6f}, "
                    f"proj={float(output.losses['projection'].detach()):.6f}"
                )
        if samples == 0:
            raise RuntimeError("MINT training epoch consumed no samples")
        means = {name: value / samples for name, value in totals.items()}
        means["seconds"] = time.time() - start
        return means, first_gradient_report or {}

    @staticmethod
    def _metric_sums(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        matches = predictions.eq(labels[:, None, :]).all(dim=-1)
        ranks = torch.arange(
            predictions.shape[1], device=predictions.device
        ).unsqueeze(0)
        found = matches.any(dim=1)
        first_rank = torch.where(
            found,
            torch.where(matches, ranks, predictions.shape[1]).min(dim=1).values,
            torch.full((predictions.shape[0],), predictions.shape[1], device=predictions.device),
        )
        result = {}
        for cutoff in (1, 5, 10):
            hit = first_rank < cutoff
            result[f"recall@{cutoff}"] = float(hit.float().sum())
            gain = torch.where(
                hit,
                1.0 / torch.log2(first_rank.float() + 2.0),
                torch.zeros_like(first_rank, dtype=torch.float),
            )
            result[f"ndcg@{cutoff}"] = float(gain.sum())
        return result

    @torch.no_grad()
    def evaluate(self, data_loader, split: str) -> Dict[str, float]:
        self.system.eval()
        system = self._system()
        totals = {
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "ndcg@1": 0.0,
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "critic_recall@10": 0.0,
            "critic_ndcg@10": 0.0,
            "prefix_survival@1": 0.0,
            "prefix_survival@2": 0.0,
            "prefix_survival@3": 0.0,
            "prefix_survival@4": 0.0,
        }
        total_examples = 0
        valid_predictions = 0
        generated_predictions = 0
        hard_sid = system.hard_sid_table()
        maximum_batches = int(self.config.get("mint_max_eval_batches", 0))
        for batch_index, batch in enumerate(data_loader):
            if maximum_batches > 0 and batch_index >= maximum_batches:
                break
            raw_ids = batch["input_ids"]
            raw_mask = batch["attention_mask"]
            targets = batch["targets"].reshape(-1).long()
            sid_history = hard_sid[raw_ids.long()]
            input_ids = sid_history.reshape(raw_ids.shape[0], -1)
            attention = raw_mask.bool().unsqueeze(-1).expand(
                *raw_mask.shape, system.depth
            ).reshape(raw_ids.shape[0], -1)
            predictions = system.generator.generate(
                input_ids=input_ids,
                attention_mask=attention,
                n_return_sequences=self.max_topk,
                prefix_allowed_tokens_fn=system.prefix_constraint,
            )
            labels = hard_sid[targets]
            gathered_predictions, gathered_labels = self.accelerator.gather_for_metrics(
                (predictions, labels)
            )
            batch_metrics = self._metric_sums(gathered_predictions, gathered_labels)
            for name in ("recall@1", "recall@5", "recall@10", "ndcg@1", "ndcg@5", "ndcg@10"):
                totals[name] += batch_metrics[name]
            for level in range(1, system.depth + 1):
                survived = gathered_predictions[:, :, :level].eq(
                    gathered_labels[:, None, :level]
                ).all(dim=-1).any(dim=-1)
                totals[f"prefix_survival@{level}"] += float(survived.float().sum())
            active = system.tree.is_active(gathered_predictions)
            valid_predictions += int(active.sum())
            generated_predictions += int(active.numel())

            critic_logits = system.critic(raw_ids, raw_mask)
            critic_top = torch.topk(critic_logits, k=self.max_topk, dim=-1).indices
            gathered_critic, gathered_target = self.accelerator.gather_for_metrics(
                (critic_top, targets)
            )
            critic_match = gathered_critic.eq(gathered_target[:, None])
            ranks = torch.arange(self.max_topk, device=gathered_critic.device).unsqueeze(0)
            found = critic_match.any(dim=1)
            first = torch.where(
                found,
                torch.where(critic_match, ranks, self.max_topk).min(dim=1).values,
                torch.full_like(gathered_target, self.max_topk),
            )
            hit = first < 10
            totals["critic_recall@10"] += float(hit.float().sum())
            totals["critic_ndcg@10"] += float(
                torch.where(
                    hit,
                    1.0 / torch.log2(first.float() + 2.0),
                    torch.zeros_like(first, dtype=torch.float),
                ).sum()
            )
            total_examples += int(gathered_target.numel())
        if total_examples == 0:
            raise RuntimeError(f"MINT {split} evaluation consumed no examples")
        metrics = {
            name: round(value / total_examples, 6)
            for name, value in totals.items()
        }
        metrics["valid_prediction_rate"] = round(
            valid_predictions / max(1, generated_predictions), 6
        )
        if metrics["valid_prediction_rate"] != 1.0:
            raise RuntimeError(
                f"MINT {split} generated an inactive SID: "
                f"validity={metrics['valid_prediction_rate']}"
            )
        self.log(f"[MINT {split}] {metrics}")
        return metrics

    def train(self, verbose: bool = True) -> float:
        del verbose
        self._write_manifest(status="running")
        initial_schedule = schedule_at(
            self.global_step / self.max_steps,
            move_budget_peak=float(self.config.get("mint_move_budget_peak", 0.05)),
            hard_alignment_start=float(
                self.config.get("mint_hard_alignment_start", 0.85)
            ),
        )
        refresh = self._system().refresh_catalog(
            initial_schedule,
            step=self.global_step,
            chunk_size=int(self.config.get("mint_catalog_refresh_chunk", 256)),
        )
        self.log(f"[MINT Initial Catalog] {self._system().audit_state()}, refresh={refresh}")
        stop_after_epoch = int(self.config.get("stop_after_epoch", -1))
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics, gradient_report = self._train_epoch(epoch)
            progress = min(1.0, self.global_step / self.max_steps)
            schedule = schedule_at(
                progress,
                move_budget_peak=float(self.config.get("mint_move_budget_peak", 0.05)),
                hard_alignment_start=float(
                    self.config.get("mint_hard_alignment_start", 0.85)
                ),
            )
            map_report = None
            if (
                int(self.config.get("mint_stage", 5)) >= 4
                and bool(self.config.get("mint_enable_map_evolution", True))
                and schedule.move_budget > 0
            ):
                update = self._system().project_hard_map(
                    move_budget=schedule.move_budget,
                    move_penalty=float(self.config.get("mint_map_move_penalty", 0.25)),
                )
                map_report = {
                    "accepted": update.accepted,
                    "changed_items": update.changed_items,
                    "previous_cost": update.previous_cost,
                    "updated_cost": update.updated_cost,
                }
                self.log(f"[MINT Map Update] epoch={epoch}, {map_report}")
                refresh = self._system().refresh_catalog(
                    schedule,
                    step=self.global_step,
                    chunk_size=int(self.config.get("mint_catalog_refresh_chunk", 256)),
                )
                self.log(f"[MINT Post-map Catalog] epoch={epoch}, {refresh}")

            validation = self.evaluate(self.valid_data, split="Validation")
            score = float(validation[self.valid_metric])
            is_best = self.best_epoch is None or score > self.best_score
            if is_best:
                self.best_score = score
                self.best_result = validation
                self.best_epoch = epoch
                self.bad_epochs = 0
                self.best_ckpt = os.path.join(self.save_path, f"{epoch}.pt.resume")
                self.save_checkpoint(epoch, f"{epoch}.pt.resume")
            else:
                self.bad_epochs += 1
            latest = self.save_checkpoint(epoch, "latest.pt.resume")
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "gradient_report": gradient_report,
                "map_update": map_report,
                "validation": validation,
                "is_best": is_best,
                "audit": self._system().audit_state(),
            }
            self.epoch_history.append(_json_safe(record))
            self._write_manifest(
                status="running",
                latest_epoch=epoch,
                latest_checkpoint=latest,
                latest_train=train_metrics,
                latest_validation=validation,
                latest_gradient_report=gradient_report,
                latest_map_update=map_report,
            )
            self.log(
                f"[MINT Epoch {epoch}] train={train_metrics}, val={validation}, "
                f"best_{self.valid_metric}={self.best_score:.6f}@{self.best_epoch}"
            )
            if stop_after_epoch >= 0 and epoch >= stop_after_epoch:
                self.log(f"[MINT Stop] reached stop_after_epoch={stop_after_epoch}")
                break
            if self.bad_epochs >= self.early_stop:
                self.log(
                    f"[MINT Early Stop] no {self.valid_metric} improvement for "
                    f"{self.bad_epochs} epochs"
                )
                break
        if self.best_ckpt is None:
            raise RuntimeError("MINT training ended without a validation checkpoint")
        self.load_checkpoint(self.best_ckpt, load_optimizer=False)
        if bool(self.config.get("evaluate_test_at_end", True)):
            self.final_test_results = self.evaluate(self.test_data, split="Official Test")
        self._write_manifest(
            status="succeeded",
            final_test=self.final_test_results,
            final_generator_only=True,
        )
        return self.best_score
