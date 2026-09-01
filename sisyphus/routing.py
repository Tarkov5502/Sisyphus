"""
sisyphus.routing — the one real unknown, made explicit.

Everything in the doctrine turns on ONE empirical question we cannot answer until the
profiling harness runs K3 on real workloads: how concentrated is expert activation, how
much do same-domain jobs share experts, and how much gate mass rides on the cold tail?
Rather than bury that guess, this module makes it a first-class, seedable, bracketed
model so every result carries its assumption.

  hot_frac          fraction of a layer's experts that form the "hot set"     (skew)
  hot_mass          probability a routed pick lands in the hot set            (concentration)
  coherent          do same-domain calls draw from a SHARED hot pool?         (convoy premise)
  cold_weight_ratio mean gate weight of a cold pick relative to a hot pick    (skippability)

Each route is a dict {expert_id: gate_weight} with weights normalised to sum to 1 across
the top-k, so a scheduler can reason about *how much* of a token's output a given expert
carries — the quantity residency-aware routing trades against bytes.

WASTE measured a 17% LRU hit at 64 GB with NO domain coherence; published MoE routing
skew is strong, but K3-family models train against skew with aux-loss-free balancing,
so we bracket weak / expected / strong and report sensitivity, never a single number
dressed up as fact. `sisyphus.profiling` replaces this model with measured routes.
"""
from __future__ import annotations

import random
from typing import Iterable

from .geometry import LAYERS, EXPERTS, TOPK, stable_seed

Route = dict[int, float]            # expert id -> gate weight (sums to 1 over the top-k)


class RoutingModel:
    """Deterministic per-(call, token, layer) expert routing under a skew model.

    A call carries a `domain`. When `coherent` is set, all calls of the same domain
    share that domain's per-layer hot pool, so their hot picks overlap heavily — the
    convoy premise. Cold picks are uniform over all experts (the long tail that becomes
    straggler work). Gate weights: a hot pick draws raw mass U(0,1)+1, a cold pick
    draws `cold_weight_ratio` * (U(0,1)+1); the top-k raw masses are normalised.
    """

    def __init__(self, hot_frac: float = 0.10, hot_mass: float = 0.75,
                 coherent: bool = True, seed: int = 42, cold_weight_ratio: float = 0.5,
                 experts: int = EXPERTS, topk: int = TOPK, layers: int = LAYERS):
        if not 0.0 < hot_frac <= 1.0:
            raise ValueError("hot_frac must be in (0, 1]")
        if not 0.0 <= hot_mass <= 1.0:
            raise ValueError("hot_mass must be in [0, 1]")
        self.hot_frac = hot_frac
        self.hot_mass = hot_mass
        self.coherent = coherent
        self.cold_weight_ratio = cold_weight_ratio
        self.experts = experts
        self.topk = topk
        self.layers = layers
        self.hot_n = max(1, int(experts * hot_frac))
        self._seed = seed
        # Per-domain, per-layer hot pool (a list of expert ids). Built lazily and cached
        # so it is stable across tokens — the "domain has a home turf" property.
        self._domain_pools: dict[tuple[str, int], list[int]] = {}

    # -- pools -----------------------------------------------------------------
    def _pool_key(self, domain: str, call_id: str | None, layer: int) -> tuple[str, int]:
        # Incoherent routing gives every CALL its own pool: same-domain jobs share nothing.
        name = domain if self.coherent else f"{domain}#{call_id}"
        return (name, layer)

    def _pool(self, domain: str, layer: int, call_id: str | None = None) -> list[int]:
        key = self._pool_key(domain, call_id, layer)
        pool = self._domain_pools.get(key)
        if pool is None:
            rng = random.Random(stable_seed("pool", self._seed, key))
            pool = rng.sample(range(self.experts), self.hot_n)
            self._domain_pools[key] = pool
        return pool

    def hot_pool(self, domain: str, layer: int) -> set[int]:
        """Globally-unique ids of the domain's hot experts at `layer` (for analysis)."""
        off = layer * self.experts
        return {off + e for e in self._pool(domain, layer)}

    # -- routing ---------------------------------------------------------------
    def _route_with(self, rng: random.Random, pool: list[int], off: int) -> Route:
        """One layer's route: top-k globally-unique expert ids with normalised gates.
        Experts are GLOBALLY-UNIQUE ids (layer*EXPERTS + local): expert 5 at layer 0 is a
        different tensor from expert 5 at layer 1, so the residency cache must not
        conflate them."""
        rand, randrange = rng.random, rng.randrange
        hn, hm, ex, k, cwr = self.hot_n, self.hot_mass, self.experts, self.topk, self.cold_weight_ratio
        raw: dict[int, float] = {}
        while len(raw) < k:
            if rand() < hm:
                e = off + pool[randrange(hn)]
                if e not in raw:
                    raw[e] = 1.0 + rand()
            else:
                e = off + randrange(ex)
                if e not in raw:
                    raw[e] = cwr * (1.0 + rand())
        total = sum(raw.values())
        return {e: w / total for e, w in raw.items()}

    def route_layer(self, call_id: str, domain: str, token: int, layer: int) -> Route:
        """Deterministic per (call, token, layer): same inputs → same route, always."""
        rng = random.Random(stable_seed("route", self._seed, call_id, token, layer))
        pool = self._pool(domain, layer, call_id)
        return self._route_with(rng, pool, layer * self.experts)

    def route(self, call, token: int) -> list[Route]:
        """Full per-layer route for a Call this token — the shape Coordinator.run wants."""
        return [self.route_layer(call.id, call.domain, token, L) for L in range(self.layers)]

    def build_cube(self, calls, tokens: int) -> dict[str, list[list[Route]]]:
        """Precompute every call's per-token, per-layer route ONCE, so many policies can
        replay identical routing without recomputation. One RNG per (call, token) keeps
        it fast and still deterministic. Returns {call_id: [ [Route]*layers ]*tokens }."""
        cube: dict[str, list] = {}
        for c in calls:
            pools = [self._pool(c.domain, L, c.id) for L in range(self.layers)]
            per_tok = []
            for t in range(tokens):
                rng = random.Random(stable_seed("cube", self._seed, c.id, t))
                per_tok.append([self._route_with(rng, pools[L], L * self.experts)
                                for L in range(self.layers)])
            cube[c.id] = per_tok
        return cube

    # -- analytic helpers ------------------------------------------------------
    def pick_distribution(self) -> tuple[float, float]:
        """Per-pick probability of (a specific hot expert, a specific cold expert)."""
        p_hot = self.hot_mass / self.hot_n + (1.0 - self.hot_mass) / self.experts
        p_cold = (1.0 - self.hot_mass) / self.experts
        return p_hot, p_cold

    def expected_unique_experts(self, n_picks: int) -> float:
        """Expected number of DISTINCT experts touched in one layer by `n_picks`
        independent routed picks from one coherent pool — the prefill working set.
        E[unique] = sum_i 1 - (1 - p_i)^n, exact under independence (top-k without
        replacement makes it very slightly conservative)."""
        p_hot, p_cold = self.pick_distribution()
        hot = self.hot_n * (1.0 - (1.0 - p_hot) ** n_picks)
        cold = (self.experts - self.hot_n) * (1.0 - (1.0 - p_cold) ** n_picks)
        return hot + cold

    def expected_cold_gate_mass(self) -> float:
        """Expected fraction of a token's gate mass that rides on cold picks — the
        ceiling on what residency-aware skipping could ever remove at one layer."""
        k = self.topk
        n_cold = k * (1.0 - self.hot_mass)
        n_hot = k - n_cold
        cold = n_cold * 1.5 * self.cold_weight_ratio
        hot = n_hot * 1.5
        return cold / (cold + hot) if (cold + hot) else 0.0


# Named brackets for the sensitivity sweep. `hot_frac` decides whether the hot set fits
# in RAM at all, so it is swept too — the original sweep only varied hot_mass.
BRACKETS = {
    "weak":     dict(hot_frac=0.15, hot_mass=0.60, cold_weight_ratio=0.8),
    "expected": dict(hot_frac=0.10, hot_mass=0.75, cold_weight_ratio=0.5),
    "strong":   dict(hot_frac=0.10, hot_mass=0.85, cold_weight_ratio=0.3),
}


def as_route(picks: Iterable[int] | Route) -> Route:
    """Accept a bare set of expert ids (uniform gates) or a weighted route."""
    if isinstance(picks, dict):
        return picks
    picks = list(picks)
    if not picks:
        return {}
    w = 1.0 / len(picks)
    return {e: w for e in picks}
