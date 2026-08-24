"""
Sisyphus — an inference operating system for underutilized consumer hardware.

Runs a 2.8T-param MoE (Kimi-K3) as an overnight *throughput* workload on ordinary
metal by owning every scarce resource a disk-offloaded MoE fights over: expert
residency, fetch order, and phase alignment. This package is the scheduler core —
the part that is pure algorithm and needs no model present to build or test.

Public surface:
    geometry            — Kimi-K3 physical constants (real, off the download).
    coordinator         — ResidencyCache, DemandHistogram, ExpertMajorScheduler,
                          Coordinator (V2.18 + V2.19).
    routing             — bracketed routing models (the one real unknown, made explicit).
    engine_sim          — drives the real scheduler to measure bytes/token per policy.
"""
from .geometry import (
    LAYERS, EXPERTS, TOPK, EXPERT_MB, experts_per_gb, night_tokens,
)
from .coordinator import (
    ResidencyCache, ResidencyError, DemandHistogram, Call,
    ExpertMajorScheduler, Coordinator, LayerResult, RunResult,
)
from .routing import RoutingModel, BRACKETS
from .schedule import run_night, NightPlan

__all__ = [
    "LAYERS", "EXPERTS", "TOPK", "EXPERT_MB", "experts_per_gb", "night_tokens",
    "ResidencyCache", "ResidencyError", "DemandHistogram", "Call",
    "ExpertMajorScheduler", "Coordinator", "LayerResult", "RunResult",
    "RoutingModel", "BRACKETS", "run_night", "NightPlan",
]
