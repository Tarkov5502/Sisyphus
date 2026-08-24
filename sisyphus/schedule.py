"""
sisyphus.schedule — the night pipeline: jobs -> convoys -> expert-major throughput.

Ties the two halves together. The convoy composer (route-aware batching + route-chaining,
V2.11/V2.16) decides WHICH jobs ride together and in WHAT order so each convoy inherits
the previous one's warm roads; the Coordinator + ExpertMajorScheduler then runs each
convoy as a phase-aligned batch, demand-ordered, over the residency cache. The result is
a night plan with an estimated aggregate token budget.

A job is the convoy_composer dict: {"id", "domain", "prompt", optional "route"}.
"""
from __future__ import annotations

from dataclasses import dataclass

from .convoy_composer import plan_night
from .coordinator import Call, Coordinator, ExpertMajorScheduler, ResidencyCache
from .geometry import LAYERS, MB_PER_GB, COMPUTE_S_PER_TOKEN, experts_per_gb, night_tokens
from .routing import RoutingModel


@dataclass
class NightPlan:
    convoys: int
    total_calls: int
    bytes_mb: float
    mb_per_token: float
    tok_per_s: float
    tokens_per_night: float
    convoy_order: list[list[str]]        # job ids per convoy, in run order


def run_night(jobs: list[dict], *, convoy_size: int = 16, cache_gb: float = 100.0,
              drive_gbps: float = 10.0, tokens_per_job: int = 12, spec_mult: float = 1.0,
              hot_mass: float = 0.75, seed: int = 42) -> NightPlan:
    """Compose jobs into chained convoys and run each through the scheduler on ONE shared
    residency cache (so warm roads carry across convoys — route-chaining's whole point).
    Returns an aggregate NightPlan."""
    _, convoys = plan_night(jobs, convoy_size=convoy_size)
    rm = RoutingModel(hot_mass=hot_mass, coherent=True, seed=seed)

    cache = ResidencyCache(max(1, experts_per_gb(cache_gb)), policy="demand")
    sched = ExpertMajorScheduler(cache, hot_min=2, max_deferrals=4)
    coord = Coordinator(cache, sched, LAYERS)

    total_tokens = 0
    order: list[list[str]] = []
    for convoy in convoys:                       # already chained for warm-road inheritance
        calls = [Call(id=j["id"], route=[], domain=j.get("domain", "")) for j in convoy]
        cube = rm.build_cube(calls, tokens_per_job)
        coord.run(calls, tokens_per_job, routing=lambda c, t: cube[c.id][t])
        total_tokens += tokens_per_job * len(calls)
        order.append([j["id"] for j in convoy])

    bytes_mb = cache.bytes_fetched
    io_s = (bytes_mb / MB_PER_GB) / drive_gbps
    compute_s = total_tokens * COMPUTE_S_PER_TOKEN
    wall = max(io_s, compute_s)
    tps = total_tokens / wall if wall else 0.0
    return NightPlan(
        convoys=len(convoys),
        total_calls=sum(len(c) for c in convoys),
        bytes_mb=bytes_mb,
        mb_per_token=bytes_mb / total_tokens if total_tokens else 0.0,
        tok_per_s=tps,
        tokens_per_night=night_tokens(tps, spec_mult),
        convoy_order=order,
    )


if __name__ == "__main__":
    # A representative night: mixed code-audit and diligence jobs. The composer clusters
    # them into route-coherent convoys; the scheduler runs them warm.
    jobs = (
        [{"id": f"audit{i}", "domain": "code",
          "prompt": f"audit the merge gate worktree scheduler module {i}"} for i in range(10)]
        + [{"id": f"dili{i}", "domain": "diligence",
            "prompt": f"score hvac acquisition target revenue multiple {i}"} for i in range(10)]
    )
    plan = run_night(jobs, convoy_size=8, cache_gb=100, drive_gbps=10.0, spec_mult=1.4)
    print(f"convoys={plan.convoys}  calls={plan.total_calls}  "
          f"MB/tok={plan.mb_per_token:.1f}  tok/s={plan.tok_per_s:.2f}  "
          f"tokens/night={plan.tokens_per_night:,.0f}")
    for i, ids in enumerate(plan.convoy_order):
        print(f"  convoy {i}: {ids}")
