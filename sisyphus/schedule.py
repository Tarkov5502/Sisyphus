"""
sisyphus.schedule — the night pipeline: jobs -> convoys -> expert-major throughput.

Ties the two halves together. The convoy composer (route-aware batching + route-chaining,
V2.11/V2.16) decides WHICH jobs ride together and in WHAT order so each convoy inherits
the previous one's warm roads; the Coordinator + ExpertMajorScheduler then runs each
convoy as a phase-aligned batch, demand-ordered, over ONE shared residency cache.

This is also where the mixed-domain night exposes the decay fix: with cumulative demand
the first domain's experts squat in the cache while the second domain thrashes; with
decay the cache follows the convoys. `night_comparison()` shows both.

A job is the convoy_composer dict: {"id", "domain", "prompt", optional "route"}.
"""
from __future__ import annotations

from dataclasses import dataclass

from .convoy_composer import plan_night
from .coordinator import Call, Coordinator
from .engine_sim import MACHINES, Machine, build, prefill_seconds, throughput
from .geometry import (
    DEFAULT_DECODE_TOKENS, DEFAULT_PROMPT_TOKENS, LAYERS, experts_per_gb, night_tokens,
)
from .routing import BRACKETS, RoutingModel


@dataclass
class NightPlan:
    policy: str
    convoys: int
    total_calls: int
    bytes_mb: float
    mb_per_token: float
    hit_rate: float
    skipped_mass: float
    tok_per_s: float                     # steady-state decode, mean over convoys
    tokens_per_night: float              # net of one prefill per convoy
    cache_gb: float
    convoy_order: list[list[str]]        # job ids per convoy, in run order


def run_night(jobs: list[dict], *, policy: str = "skip", machine: Machine = MACHINES["192"],
              convoy_size: int = 32, tokens_per_job: int = 12,
              prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
              decode_tokens: int = DEFAULT_DECODE_TOKENS, spec_mult: float = 1.0,
              rm: RoutingModel | None = None, seed: int = 42) -> NightPlan:
    """Compose jobs into chained convoys and run each through the scheduler on ONE shared
    residency cache (so warm roads carry across convoys — route-chaining's whole point).
    `tokens_per_job` decode steps are simulated per convoy to measure bytes/token; the
    night total then charges `decode_tokens` steps plus one prefill per convoy."""
    _, convoys = plan_night(jobs, convoy_size=convoy_size)
    rm = rm or RoutingModel(**BRACKETS["expected"], coherent=True, seed=seed)
    largest = max(len(c) for c in convoys)
    cache_gb, trunk_streams = machine.budget(largest, prompt_tokens + decode_tokens)
    cache, sched = build(policy, experts_per_gb(cache_gb))
    coord = Coordinator(cache, sched, LAYERS)

    total_tokens = 0
    skipped_mass = 0.0
    token_layers = 0
    night_s = 0.0
    order: list[list[str]] = []
    for convoy in convoys:                       # already chained for warm-road inheritance
        calls = [Call(id=j["id"], route=[], domain=j.get("domain", "")) for j in convoy]
        cube = rm.build_cube(calls, tokens_per_job)
        before = cache.bytes_fetched
        r = coord.run(calls, tokens_per_job, routing=lambda c, t: cube[c.id][t])
        total_tokens += r.tokens
        skipped_mass += r.skipped_mass
        token_layers += r.token_layers
        tps, _ = throughput(r.tokens, cache.bytes_fetched - before, len(calls), machine,
                            trunk_streams, r.applied_experts_per_step, r.expert_flop_fraction,
                            prompt_tokens + decode_tokens)
        pre, _ = prefill_seconds(rm, len(calls), prompt_tokens, cache_gb, machine,
                                 trunk_streams, r.expert_flop_fraction)
        night_s += pre + (decode_tokens * len(calls) / tps if tps else float("inf"))
        order.append([j["id"] for j in convoy])

    produced = sum(len(c) for c in convoys) * decode_tokens
    per_night = night_tokens(1.0) / night_s * produced * spec_mult if night_s else 0.0
    return NightPlan(
        policy=policy, convoys=len(convoys), total_calls=sum(len(c) for c in convoys),
        bytes_mb=cache.bytes_fetched,
        mb_per_token=cache.bytes_fetched / total_tokens if total_tokens else 0.0,
        hit_rate=cache.hit_rate,
        skipped_mass=skipped_mass / token_layers if token_layers else 0.0,
        tok_per_s=produced / (night_s - 0.0) if night_s else 0.0,
        tokens_per_night=per_night, cache_gb=cache_gb, convoy_order=order,
    )


def sample_jobs(per_domain: int = 32) -> list[dict]:
    """A representative night: code audits and acquisition diligence, interleaved as
    they would arrive, for the composer to sort out."""
    jobs = []
    for i in range(per_domain):
        jobs.append({"id": f"audit{i}", "domain": "code",
                     "prompt": f"audit the merge gate worktree scheduler module {i}"})
        jobs.append({"id": f"dili{i}", "domain": "diligence",
                     "prompt": f"score hvac acquisition target revenue multiple {i}"})
    return jobs


def night_comparison(machine_key: str = "192") -> str:
    mach = MACHINES[machine_key]
    jobs = sample_jobs()
    out = [f"NIGHT PIPELINE on {mach.label} (cpu and GPU x16) — {len(jobs)} mixed jobs -> convoys of 32, one shared cache",
           f"  {'compute':8} {'policy':10} {'convoys':>7} {'cache':>6} {'MB/tok':>7} {'hit':>5} {'skip':>5} {'tok/night':>12}"]
    for key in (machine_key, machine_key + "+GPU"):
        mach = MACHINES[key]
        for pol in ("union_lru", "sisyphus", "decay", "skip"):
            p = run_night(jobs, policy=pol, machine=mach)
            out.append(f"  {mach.compute.name:8} {pol:10} {p.convoys:7d} {p.cache_gb:6.0f} {p.mb_per_token:7.0f} "
                       f"{p.hit_rate:5.2f} {p.skipped_mass:5.2f} {p.tokens_per_night:12,.0f}")
    out.append("  convoy order: " + " -> ".join(
        f"{ids[0].rstrip('0123456789')}x{len(ids)}" for ids in p.convoy_order))
    return "\n".join(out)


if __name__ == "__main__":
    print(night_comparison())
