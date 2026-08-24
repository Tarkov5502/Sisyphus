"""
sisyphus.routing — the one real unknown, made explicit.

Everything in the doctrine turns on ONE empirical question we cannot answer until the
profiling harness runs K3 on real workloads: how concentrated is expert activation, and
how much do same-domain jobs share experts? Rather than bury that guess, this module
makes it a first-class, seedable, bracketed model so every result carries its assumption.

  hot_frac   fraction of a layer's experts that form the "hot set"       (skew)
  hot_mass   probability a routed pick lands in the hot set              (concentration)
  coherent   do same-domain calls draw from a SHARED hot pool?           (convoy premise)

WASTE measured a 17% LRU hit at 64 GB with NO domain coherence; published MoE routing
skew is strong. We bracket weak / expected / strong and report sensitivity, never a
single number dressed up as fact.
"""
from __future__ import annotations

import random
from typing import Optional

from .geometry import LAYERS, EXPERTS, TOPK


class RoutingModel:
    """Deterministic per-(call, token, layer) expert routing under a skew model.

    A call carries a `domain`. When `coherent` is set, all calls of the same domain
    share that domain's per-layer hot pool, so their hot picks overlap heavily — the
    convoy premise. Cold picks are uniform over all experts (the long tail that becomes
    straggler work).
    """

    def __init__(self, hot_frac: float = 0.10, hot_mass: float = 0.75,
                 coherent: bool = True, seed: int = 42,
                 experts: int = EXPERTS, topk: int = TOPK, layers: int = LAYERS):
        self.hot_frac = hot_frac
        self.hot_mass = hot_mass
        self.coherent = coherent
        self.experts = experts
        self.topk = topk
        self.layers = layers
        self.hot_n = max(1, int(experts * hot_frac))
        self._seed = seed
        # Per-domain, per-layer hot pool (a sorted list of expert ids). Built lazily and
        # cached so it is stable across tokens — the "domain has a home turf" property.
        self._domain_pools: dict[tuple[str, int], list[int]] = {}

    def _pool(self, domain: str, layer: int) -> list[int]:
        key = (domain if self.coherent else f"{domain}#solo", layer)
        pool = self._domain_pools.get(key)
        if pool is None:
            rng = random.Random(hash((self._seed, key)) & 0xFFFFFFFF)
            pool = rng.sample(range(self.experts), self.hot_n)
            self._domain_pools[key] = pool
        return pool

    def route_layer(self, call_id: str, domain: str, token: int, layer: int) -> set[int]:
        # Deterministic per (call, token, layer): same inputs → same route, always.
        # Experts are returned as GLOBALLY-UNIQUE ids (layer*EXPERTS + local): expert 5
        # at layer 0 is a different tensor from expert 5 at layer 1, so the residency
        # cache must not conflate them. The offset makes the cache model a real
        # cross-layer working set.
        rng = random.Random(hash((self._seed, call_id, token, layer)) & 0xFFFFFFFF)
        pool = self._pool(domain, layer)
        off = layer * self.experts
        picks: set[int] = set()
        while len(picks) < self.topk:
            if rng.random() < self.hot_mass:
                picks.add(off + pool[rng.randrange(self.hot_n)])
            else:
                picks.add(off + rng.randrange(self.experts))
        return picks

    def route(self, call, token: int) -> list[set]:
        """Full per-layer route for a Call this token — the shape Coordinator.run wants."""
        return [self.route_layer(call.id, call.domain, token, L)
                for L in range(self.layers)]

    def build_cube(self, calls, tokens: int) -> dict:
        """Precompute every call's per-token, per-layer route ONCE, so many policies can
        replay identical routing without recomputation. One RNG per (call, token) keeps
        it fast and still deterministic. Returns {call_id: [ [set]*layers ]*tokens }."""
        cube: dict[str, list] = {}
        for c in calls:
            pools = [self._pool(c.domain, L) for L in range(self.layers)]
            offs = [L * self.experts for L in range(self.layers)]
            per_tok = []
            for t in range(tokens):
                rng = random.Random(hash((self._seed, c.id, t)) & 0xFFFFFFFF)
                rand = rng.random
                randrange = rng.randrange
                layers = []
                hn, hm, ex, k = self.hot_n, self.hot_mass, self.experts, self.topk
                for L in range(self.layers):
                    pool, off = pools[L], offs[L]
                    picks: set[int] = set()
                    add = picks.add
                    while len(picks) < k:
                        if rand() < hm:
                            add(off + pool[randrange(hn)])
                        else:
                            add(off + randrange(ex))
                    layers.append(picks)
                per_tok.append(layers)
            cube[c.id] = per_tok
        return cube


# Named brackets for the sensitivity sweep.
BRACKETS = {
    "weak":     dict(hot_frac=0.10, hot_mass=0.60),
    "expected": dict(hot_frac=0.10, hot_mass=0.75),
    "strong":   dict(hot_frac=0.10, hot_mass=0.85),
}
