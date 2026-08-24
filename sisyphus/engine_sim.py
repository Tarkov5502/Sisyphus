"""
sisyphus.engine_sim — drive the REAL scheduler and measure bytes fetched per token.

Unlike the first-pass Monte-Carlo (atlas_sim/), this runs actual Call objects through
the actual Coordinator + ExpertMajorScheduler + ResidencyCache, so the numbers come out
of the code that would ship. It compares three policies on IDENTICAL routing (a route
cube built once per config, replayed by every policy):

  naive_single  — token-major: every call streams alone against a tiny per-call cache.
  union_lru     — batch: load each layer's union once, LRU eviction.
  sisyphus      — batch + demand-frequency eviction + straggler consolidation + aging.

Reported: MB/token, disk- vs compute-bound, resident hit rate, tok/s, tokens/night.
Deterministic given the seed.
"""
from __future__ import annotations

from dataclasses import dataclass

from .coordinator import Call, Coordinator, ExpertMajorScheduler, ResidencyCache
from .geometry import (
    LAYERS, COMPUTE_S_PER_TOKEN, MB_PER_GB, experts_per_gb, night_tokens,
)
from .routing import RoutingModel


@dataclass
class Metrics:
    policy: str
    tokens: int
    bytes_mb: float
    mb_per_token: float
    hit_rate: float
    tok_per_s: float
    io_bound: bool
    tokens_per_night: float


def _throughput(tokens: int, bytes_mb: float, drive_gbps: float,
                spec_mult: float) -> tuple[float, bool]:
    io_s = (bytes_mb / MB_PER_GB) / drive_gbps
    compute_s = tokens * COMPUTE_S_PER_TOKEN
    wall = max(io_s, compute_s)              # Spotlight overlaps I/O behind compute
    return (tokens / wall if wall else 0.0), (io_s > compute_s)


def _cube_routing(cube):
    """Adapt a prebuilt route cube to the Coordinator.run(routing=...) contract."""
    return lambda call, tok: cube[call.id][tok]


def measure(policy: str, calls: list[Call], cube: dict, drive_gbps: float,
            cache_gb: float, tokens: int, spec_mult: float = 1.0) -> Metrics:
    """Run one policy over a prebuilt route cube and return its metrics."""
    capacity = max(1, experts_per_gb(cache_gb))
    batch = len(calls)
    routing = _cube_routing(cube)

    if policy == "naive_single":
        per_call_cap = max(1, capacity // batch)
        total_bytes = 0.0
        hits = fetches = 0
        for c in calls:
            cache = ResidencyCache(per_call_cap, policy="lru")
            sched = ExpertMajorScheduler(cache, hot_min=1)
            solo = Call(id=c.id, route=[], domain=c.domain)
            r = Coordinator(cache, sched, LAYERS).run([solo], tokens, routing=routing)
            total_bytes += r.bytes_fetched
            hits += cache.hits
            fetches += cache.fetches
        tok = tokens * batch
        hit_rate = hits / (hits + fetches) if (hits + fetches) else 0.0
        tps, io_bound = _throughput(tok, total_bytes, drive_gbps, spec_mult)
        return Metrics(policy, tok, total_bytes, total_bytes / tok, hit_rate,
                       tps, io_bound, night_tokens(tps, spec_mult))

    cache_policy = "lru" if policy == "union_lru" else "demand"
    hot_min = 1 if policy == "union_lru" else 2
    cache = ResidencyCache(capacity, policy=cache_policy)
    sched = ExpertMajorScheduler(cache, hot_min=hot_min, max_deferrals=4)
    # fresh Call objects so aging state doesn't leak between policy runs
    fresh = [Call(id=c.id, route=[], domain=c.domain) for c in calls]
    r = Coordinator(cache, sched, LAYERS).run(fresh, tokens, routing=routing)
    tps, io_bound = _throughput(r.tokens, r.bytes_fetched, drive_gbps, spec_mult)
    return Metrics(policy, r.tokens, r.bytes_fetched, r.mb_per_token, r.hit_rate,
                   tps, io_bound, night_tokens(tps, spec_mult))


def _calls(batch: int, domains: tuple[str, ...]) -> list[Call]:
    return [Call(id=f"c{i}", route=[], domain=domains[i % len(domains)])
            for i in range(batch)]


# --------------------------------------------------------------------------- #
#  Reports
# --------------------------------------------------------------------------- #
def scenario_table(tokens: int = 12) -> str:
    out = [f"{'config':50} {'policy':13} {'MB/tok':>7} {'bnd':>4} "
           f"{'hit':>5} {'tok/s':>7} {'tok/night':>12}", "-" * 104]
    configs = [
        # (label, batch, drive_gbps, cache_gb, spec, policies)
        ("32GB RAM · scattered 1.5 · batch1",  1, 1.5, 20, 1.0, ["naive_single"]),
        ("32GB RAM · scattered 1.5 · batch16", 16, 1.5, 20, 1.0, ["union_lru", "sisyphus"]),
        ("32GB RAM · layout 3.2 · batch16",    16, 3.2, 20, 1.0, ["union_lru", "sisyphus"]),
        ("128GB RAM · striped 10 · batch16",   16, 10.0, 100, 1.0, ["union_lru", "sisyphus"]),
        ("128GB RAM · striped 10 · batch32 · spec1.4", 32, 10.0, 100, 1.4,
         ["union_lru", "sisyphus"]),
    ]
    rm = RoutingModel(hot_mass=0.75, coherent=True)
    for label, B, drive, cache_gb, spec, policies in configs:
        calls = _calls(B, ("code",))
        cube = rm.build_cube(calls, tokens)
        for pol in policies:
            m = measure(pol, calls, cube, drive, cache_gb, tokens, spec)
            out.append(f"{label:50} {m.policy:13} {m.mb_per_token:7.1f} "
                       f"{'I/O' if m.io_bound else 'CPU':>4} {m.hit_rate:5.2f} "
                       f"{m.tok_per_s:7.2f} {m.tokens_per_night:12,.0f}")
        out.append("")
    return "\n".join(out)


def sensitivity_table(tokens: int = 12) -> str:
    out = ["SENSITIVITY — Sisyphus policy, 128GB/striped/batch32/spec1.4, "
           "vary the one unknown (domain coherence):"]
    for name, hm in (("weak 0.60", 0.60), ("expected 0.75", 0.75), ("strong 0.85", 0.85)):
        rm = RoutingModel(hot_mass=hm, coherent=True)
        calls = _calls(32, ("code",))
        cube = rm.build_cube(calls, tokens)
        m = measure("sisyphus", calls, cube, 10.0, 100, tokens, 1.4)
        out.append(f"  {name:16} -> MB/tok {m.mb_per_token:6.1f}  hit {m.hit_rate:4.2f}  "
                   f"{m.tok_per_s:6.2f} tok/s = {m.tokens_per_night:11,.0f}/night")
    return "\n".join(out)


if __name__ == "__main__":
    print(scenario_table())
    print(sensitivity_table())
    print("\nNOTE: bytes come from the real ResidencyCache/scheduler, not a formula. "
          "tok/s = tokens / max(I/O, compute) with I/O overlapped behind compute "
          "(Spotlight); compute 0.12 core-s/token. Routing skew is the bracketed unknown "
          "the profiling harness replaces with measured numbers.")
