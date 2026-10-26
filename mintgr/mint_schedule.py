"""Single-run MINT-GR annealing schedule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MintSchedule:
    progress: float
    phase: str
    top_m: int
    router_temperature: float
    move_budget: float
    adapter_gate: float
    collaborative_gate: float
    hard_weight: float
    marginal_weight: float
    critic_weight: float
    projection_weight: float
    ranking_weight: float
    capacity_weight: float
    locality_weight: float
    consistency_weight: float
    entropy_weight: float


def _linear(value: float, start: float, end: float, left: float, right: float) -> float:
    if right <= left:
        return float(end if value >= right else start)
    if value <= left:
        return float(start)
    if value >= right:
        return float(end)
    fraction = (value - left) / (right - left)
    return float(start + fraction * (end - start))


def schedule_at(
    progress: float,
    move_budget_peak: float = 0.05,
    hard_alignment_start: float = 0.85,
) -> MintSchedule:
    p = min(1.0, max(0.0, float(progress)))
    hard_alignment_start = float(hard_alignment_start)
    if not 0.60 < hard_alignment_start <= 1.0:
        raise ValueError("hard_alignment_start must be in (0.60, 1.0]")
    adapter_gate = _linear(p, 0.0, 1.0, 0.0, 0.60)
    collaborative_gate = _linear(p, 0.0, 0.5, 0.05, 0.70)
    if p < 0.10:
        return MintSchedule(
            p, "bootstrap", 8, 1.5, 0.0, adapter_gate,
            collaborative_gate, 1.0, 0.25, 1.0, 0.5, 0.0, 0.1, 0.05,
            0.1, -0.01,
        )
    if p < 0.60:
        return MintSchedule(
            p,
            "adaptation",
            4,
            _linear(p, 1.0, 0.4, 0.10, 0.60),
            _linear(p, 0.02, move_budget_peak, 0.10, 0.40),
            adapter_gate,
            collaborative_gate,
            1.0,
            1.0,
            0.5,
            0.5,
            0.2,
            0.1,
            0.05,
            0.1,
            _linear(p, 0.0, 0.01, 0.40, 0.60),
        )
    if p < hard_alignment_start:
        return MintSchedule(
            p,
            "consolidation",
            2,
            _linear(p, 0.4, 0.2, 0.60, hard_alignment_start),
            _linear(p, move_budget_peak, 0.02, 0.60, hard_alignment_start),
            adapter_gate,
            collaborative_gate,
            1.0,
            1.0,
            0.5,
            0.5,
            0.2,
            0.2,
            0.02,
            0.2,
            0.01,
        )
    return MintSchedule(
        p,
        "hard_alignment",
        1,
        _linear(p, 0.15, 0.08, hard_alignment_start, 1.0),
        0.0,
        1.0,
        0.5,
        1.0,
        1.0,
        _linear(
            p,
            0.1,
            0.0,
            hard_alignment_start,
            min(1.0, hard_alignment_start + 0.10),
        ),
        _linear(
            p,
            0.2,
            0.0,
            hard_alignment_start,
            min(1.0, hard_alignment_start + 0.10),
        ),
        0.2,
        0.2,
        0.02,
        0.2,
        0.05,
    )
