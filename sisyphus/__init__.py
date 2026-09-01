"""
Sisyphus — an inference operating system for underutilized consumer hardware.

Runs a 2.8T-param MoE (Kimi-K3) as an overnight *throughput* workload on ordinary
metal by owning every scarce resource a disk-offloaded MoE fights over: expert
residency, fetch order, phase alignment and — V3 — the gate mass a token is willing
to trade for bytes. This package is the scheduler core: pure algorithm, stdlib only,
no model present needed to build, test or measure.

Public surface:
    geometry            — Kimi-K3 physical constants and the machine envelope (RAM budget,
                          KV, prefill, stable seeding).
    coordinator         — ResidencyCache (decayed demand), DemandHistogram,
                          ExpertMajorScheduler (cold-pick policy), Coordinator.
    routing             — bracketed routing model with gate weights (the unknown, explicit).
    tape                — the full-sweep floor and the KV-vs-residency RAM split.
    engine_sim          — drives the real scheduler; the throughput tables.
    schedule            — jobs -> convoys -> night plan.
    profiling           — rent-to-profile: capture, analyze, replay real routes.
"""
from .geometry import (
    LAYERS, EXPERTS, TOPK, EXPERT_MB, TRUNK_GB, MODEL_GB, KV_MB_PER_TOKEN,
    experts_per_gb, hot_set_gb, kv_gb, cache_budget_gb, night_tokens, stable_seed,
)
from .coordinator import (
    ResidencyCache, ResidencyError, DemandHistogram, Call,
    ExpertMajorScheduler, Coordinator, LayerResult, RunResult,
)
from .routing import RoutingModel, BRACKETS, Route, as_route
from .tape import TapePlan, plan_tape, best_tape, max_batch

__all__ = [
    "LAYERS", "EXPERTS", "TOPK", "EXPERT_MB", "TRUNK_GB", "MODEL_GB", "KV_MB_PER_TOKEN",
    "experts_per_gb", "hot_set_gb", "kv_gb", "cache_budget_gb", "night_tokens", "stable_seed",
    "ResidencyCache", "ResidencyError", "DemandHistogram", "Call",
    "ExpertMajorScheduler", "Coordinator", "LayerResult", "RunResult",
    "RoutingModel", "BRACKETS", "Route", "as_route",
    "TapePlan", "plan_tape", "best_tape", "max_batch",
]
