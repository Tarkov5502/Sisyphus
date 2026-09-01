"""
sisyphus.coordinator — the concurrency-safe residency owner and the demand-ordered
expert-major scheduler (design doc V2.18 + V2.19, revised V3).

THE KEY INSIGHT (V2.18). Weights are READ-ONLY during inference. Many concurrent
convoys reading the same expert can never corrupt each other's outputs — the scary
race needs a writer, and there are none. So correctness is free and needs no locks.
The only genuinely contended resource is scarce RAM *residency*: which experts occupy
the cache. Uncoordinated convoys don't corrupt each other, they THRASH — convoy A
loads expert 47, convoy B evicts it for 88, A refetches 47. Wasted bytes, never wrong
answers. This module owns that mutable residency centrally so workers never manage it
ad hoc.

  ResidencyCache        — bounded, reference-counted expert residency. An expert a
                          live call still stands on (refcount > 0) cannot be evicted.
                          Eviction policy is pluggable; the Sisyphus policy is
                          DEMAND-FREQUENCY with EXPONENTIAL DECAY (V3), not LRU: a
                          layer-sequential MoE sweep is nearly the worst case for LRU
                          (every union expert was "just used"), and un-decayed demand
                          is anti-adaptive (a stale domain squats in the cache forever).

  DemandHistogram       — per-layer cross-convoy expert demand: who wants what, how
                          many, and with how much gate weight.

  ExpertMajorScheduler  — per layer, across every phase-aligned call, service experts
                          HOTTEST-FIRST (load once, serve the whole crowd), then sweep
                          the cold single-use experts in one consolidated late pass.
                          V3 adds the COLD-PICK POLICY: a cold expert that is not
                          resident may be SKIPPED (its gate mass renormalised over the
                          experts that are) under a per-token budget — the only lever
                          that touches the cold tail, which is where the bytes are.

What happened to aging / deferral (V2.19). Deferring a call's cold expert to a later
straggler batch cannot save bytes in a lockstep decode: the call's next layer depends on
this layer's output, so it cannot proceed without the expert, and consolidating its load
with a LATER convoy is identical, byte for byte, to having run a larger batch — at the
cost of latency. The batch-size curve in engine_sim.py IS the deferral curve. The
mechanism was therefore removed rather than kept as bookkeeping that could never change
a measurement.

Stdlib only. Deterministic given a seeded routing model. The real inference engine is
meant to call this; engine_sim.py drives it with a routing model to measure the one
number that matters on disk-offload hardware — bytes fetched per generated token — and
the one number that bounds residency-aware routing — gate mass skipped per token.
"""
from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .geometry import EXPERT_MB
from .routing import Route, as_route


# --------------------------------------------------------------------------- #
#  Residency: the one piece of scarce mutable state, owned centrally.
# --------------------------------------------------------------------------- #
class ResidencyError(RuntimeError):
    """Raised when the cache cannot honor a required residency (capacity too small to
    hold even the simultaneously-pinned working set). Loud beats silent — V.4."""


class ResidencyCache:
    """Reference-counted, capacity-bounded expert residency.

    An expert is one of three states:
      * pinned-permanent (the trunk / profiled hot set) — never evicted, does not
        consume the dynamic capacity budget;
      * resident with refcount > 0 — a live call is standing on it, eviction forbidden;
      * resident with refcount == 0 — cached but idle, eligible for eviction by policy.

    `acquire` returns True when it had to fetch from disk (a miss → bytes moved), False
    on a hit. The caller pairs every acquire with a release once the expert's matmul for
    this layer is done, which drops the refcount but LEAVES the expert cached — so the
    next layer/token can hit it for free if it survives eviction.

    Demand decay. `decay` is the per-decode-step retention factor: 1.0 reproduces the
    original cumulative counter; 0.8 means a demand observation is worth 80% of itself
    one step later. Decay is implemented by INFLATING new observations by 1/decay^t
    instead of deflating old ones — every stored value scales uniformly, so heap order
    is preserved and `decay_step()` is O(1) except for a rare renormalisation.
    """

    _RENORM_AT = 1e100

    def __init__(self, capacity: int, policy: str = "demand",
                 pinned: Optional[Iterable[int]] = None, decay: float = 1.0):
        if capacity < 1:
            raise ValueError("capacity must be >= 1 expert")
        if policy not in ("demand", "lru"):
            raise ValueError("policy must be 'demand' or 'lru'")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.capacity = capacity
        self.policy = policy
        self.decay = decay
        self.pinned: set[int] = set(pinned or ())
        # resident (non-pinned) experts -> refcount. OrderedDict gives LRU order:
        # least-recently-acquired is left-most.
        self._resident: "OrderedDict[int, int]" = OrderedDict()
        # decayed demand per expert in INFLATED units (divide by _scale for real units).
        self._demand: dict[int, float] = {}
        self._scale = 1.0
        # lazy min-heap of (demand_at_push, expert) over idle (refcount-0) experts, the
        # demand-policy eviction structure. Entries go stale as demand rises or refcount
        # changes; _pick_victim validates on pop and re-pushes the stale ones.
        self._heap: list[tuple[float, int]] = []
        # ---- instrumentation ----
        self.fetches = 0          # disk loads (misses)
        self.hits = 0             # resident/pinned re-uses
        self.evictions = 0
        self.bytes_fetched = 0.0  # MB
        self.steps = 0            # decay steps seen

    # -- demand signal (fed by the scheduler before it services a layer) -------
    def note_demand(self, expert: int, count: float = 1.0) -> None:
        """Record that `expert` is wanted `count` times this step. Drives demand-eviction
        and lets the cache keep currently-popular experts warm across tokens."""
        self._demand[expert] = self._demand.get(expert, 0.0) + count * self._scale

    def demand(self, expert: int) -> float:
        """Current decayed demand for `expert`, in real (un-inflated) units."""
        return self._demand.get(expert, 0.0) / self._scale

    def decay_step(self) -> None:
        """Advance one decode step: everything observed so far is worth `decay` of what
        it was. O(1) amortised — see class docstring."""
        self.steps += 1
        if self.decay >= 1.0:
            return
        self._scale /= self.decay
        if self._scale > self._RENORM_AT:
            s = self._scale
            self._demand = {e: d / s for e, d in self._demand.items()}
            self._heap = [(d / s, e) for d, e in self._heap]
            heapq.heapify(self._heap)
            self._scale = 1.0

    # -- the hot path ----------------------------------------------------------
    def is_resident(self, expert: int) -> bool:
        return expert in self.pinned or expert in self._resident

    def acquire(self, expert: int) -> bool:
        """Pin `expert` for use. Returns True iff it had to be fetched from disk."""
        if expert in self.pinned:
            self.hits += 1
            return False
        if expert in self._resident:
            self._resident[expert] += 1
            self._resident.move_to_end(expert)      # mark most-recently-used
            self.hits += 1
            return False
        # miss → make room, fetch, pin
        self._ensure_room_for_one()
        self._resident[expert] = 1
        self._resident.move_to_end(expert)
        self.fetches += 1
        self.bytes_fetched += EXPERT_MB
        return True

    def release(self, expert: int) -> None:
        """Drop one pin. The expert stays cached (refcount 0) until evicted."""
        if expert in self.pinned:
            return
        rc = self._resident.get(expert)
        if rc is None:
            raise ResidencyError(f"release of non-resident expert {expert}")
        if rc <= 1:
            self._resident[expert] = 0
            if self.policy == "demand":
                # now idle → an eviction candidate at its current demand
                heapq.heappush(self._heap, (self._demand.get(expert, 0.0), expert))
        else:
            self._resident[expert] = rc - 1

    def refcount(self, expert: int) -> int:
        return self._resident.get(expert, 0)

    # -- eviction --------------------------------------------------------------
    def _ensure_room_for_one(self) -> None:
        if len(self._resident) < self.capacity:
            return
        victim = self._pick_victim()
        if victim is None:
            raise ResidencyError(
                f"cache full ({self.capacity}) and every resident expert is pinned "
                f"(refcount>0); the simultaneously-needed working set exceeds capacity")
        del self._resident[victim]
        self.evictions += 1

    def _pick_victim(self) -> Optional[int]:
        """Choose an evictable (refcount 0) resident expert by policy. LRU picks the
        left-most (oldest) idle entry. Demand pops the global lowest-current-demand idle
        expert from a lazy min-heap — keeping popular experts warm regardless of age."""
        if self.policy == "lru":
            for e, rc in self._resident.items():        # OrderedDict: oldest first
                if rc == 0:
                    return e
            return None
        heap = self._heap
        while heap:
            dmd, e = heapq.heappop(heap)
            rc = self._resident.get(e)
            if rc is None or rc != 0:
                continue                                # gone or re-pinned → stale
            cur = self._demand.get(e, 0.0)
            if dmd != cur:                              # demand rose → re-file, keep going
                heapq.heappush(heap, (cur, e))
                continue
            return e
        return None

    def resident_set(self) -> set[int]:
        return set(self.pinned) | set(self._resident.keys())

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.fetches
        return self.hits / total if total else 0.0


# --------------------------------------------------------------------------- #
#  Demand histogram: who wants what, this layer, and how badly.
# --------------------------------------------------------------------------- #
class DemandHistogram:
    """Cross-convoy expert demand for a single layer: expert -> [(call id, gate)]."""

    def __init__(self) -> None:
        self._callers: dict[int, list[tuple[str, float]]] = {}

    def add(self, call_id: str, route: Route | Iterable[int]) -> None:
        for e, w in as_route(route).items():
            self._callers.setdefault(e, []).append((call_id, w))

    def count(self, expert: int) -> int:
        return len(self._callers.get(expert, ()))

    def callers(self, expert: int) -> list[str]:
        return [c for c, _ in self._callers.get(expert, [])]

    def weighted(self, expert: int) -> list[tuple[str, float]]:
        return self._callers.get(expert, [])

    def max_gate(self, expert: int) -> float:
        return max((w for _, w in self._callers.get(expert, [])), default=0.0)

    def by_popularity(self) -> list[tuple[int, int]]:
        """(expert, demand) descending by demand, expert id breaking ties for
        determinism."""
        return sorted(((e, len(c)) for e, c in self._callers.items()),
                      key=lambda kv: (-kv[1], kv[0]))

    def experts(self) -> set[int]:
        return set(self._callers.keys())


# --------------------------------------------------------------------------- #
#  A call in flight.
# --------------------------------------------------------------------------- #
@dataclass
class Call:
    """One inference stream. `route[L]` is the routed experts it needs at layer L —
    either a weighted Route {expert: gate} (from a routing model or a profiled log) or
    a bare set of ids (uniform gates; test mode)."""
    id: str
    route: list
    domain: str = ""

    def experts_at(self, layer: int):
        r = self.route[layer]
        return r.keys() if isinstance(r, dict) else r

    def gate(self, layer: int, expert: int) -> float:
        r = self.route[layer]
        if isinstance(r, dict):
            return r.get(expert, 0.0)
        return 1.0 / len(r) if expert in r and r else 0.0


@dataclass
class LayerResult:
    layer: int
    bytes_fetched: float
    load_order: list[int]                 # experts, in the order the layer loaded them
    hot_loads: int = 0                    # experts served in the hot pass
    straggler_loads: int = 0              # cold experts actually loaded in the sweep
    skipped_loads: int = 0                # cold experts NOT loaded (residency-aware skip)
    skipped_applications: int = 0         # (call, expert) pairs whose output was dropped
    skipped_mass: float = 0.0             # gate mass dropped, summed over calls
    calls: int = 0


# --------------------------------------------------------------------------- #
#  Demand-ordered, straggler-consolidated, expert-major scheduler (V3).
# --------------------------------------------------------------------------- #
class ExpertMajorScheduler:
    """Services one MoE layer for a set of phase-aligned calls.

    Ordering per layer:
      1. Tally cross-convoy demand (the histogram) and feed the cache's decayed demand.
      2. HOT experts (demand >= hot_min) are serviced first, hottest-first — each
         loaded exactly once and applied to the whole crowd that needs it. Hot-first
         also lets a real engine start compute while the cold I/O is still in flight.
      3. COLD experts (demand < hot_min) are swept at the layer's end under the
         cold-pick policy:
           "load" — fetch each cold expert once (the full union; V2.19 behaviour);
           "skip" — a cold expert that is NOT resident and whose gate weight is small
                    is dropped: its caller renormalises the gate over the experts it
                    did get. Skips are bounded per call per layer by `skip_budget`
                    (total gate mass droppable) and per pick by `skip_max_gate`.
                    Resident cold experts are always applied — a hit is free.

    Because weights are read-only, applying one loaded expert to many calls is always
    correct; the scheduler reorders *loads* and, under "skip", trades a bounded slice
    of gate mass for the bytes it would have cost. The skipped mass is reported so the
    trade is visible, never silent.
    """

    def __init__(self, cache: ResidencyCache, hot_min: int = 2,
                 cold_policy: str = "load", skip_budget: float = 0.10,
                 skip_max_gate: float = 0.06):
        if hot_min < 1:
            raise ValueError("hot_min must be >= 1")
        if cold_policy not in ("load", "skip"):
            raise ValueError("cold_policy must be 'load' or 'skip'")
        if not 0.0 <= skip_budget <= 1.0:
            raise ValueError("skip_budget is a fraction of a token-layer's gate mass")
        self.cache = cache
        self.hot_min = hot_min
        self.cold_policy = cold_policy
        self.skip_budget = skip_budget
        self.skip_max_gate = skip_max_gate

    def run_layer(self, layer: int, calls: list[Call]) -> LayerResult:
        cache = self.cache
        hist = DemandHistogram()
        for c in calls:
            route = c.route[layer]
            hist.add(c.id, route)
            for e in c.experts_at(layer):
                cache.note_demand(e)

        hot: list[int] = []
        cold: list[int] = []
        for expert, demand in hist.by_popularity():
            (hot if demand >= self.hot_min else cold).append(expert)

        load_order: list[int] = []
        bytes_before = cache.bytes_fetched

        # --- hot pass: hottest-first, load once, serve the crowd ---
        for expert in hot:
            cache.acquire(expert)          # fetch iff not resident
            load_order.append(expert)
            cache.release(expert)          # matmul done for all its callers this layer

        # --- straggler sweep: consolidated cold batch at the layer's end ---
        skipped_loads = skipped_apps = 0
        skipped_mass = 0.0
        if self.cold_policy == "skip" and cold:
            budget: dict[str, float] = {c.id: self.skip_budget for c in calls}
            # Cheapest skips first so a fixed budget removes the most loads.
            for expert in sorted(cold, key=lambda e: (hist.max_gate(e), e)):
                if cache.is_resident(expert):
                    cache.acquire(expert); load_order.append(expert); cache.release(expert)
                    continue
                callers = hist.weighted(expert)
                skippable = all(w <= self.skip_max_gate and budget[cid] >= w
                                for cid, w in callers)
                if skippable:
                    for cid, w in callers:
                        budget[cid] -= w
                        skipped_mass += w
                    skipped_apps += len(callers)
                    skipped_loads += 1
                else:
                    cache.acquire(expert); load_order.append(expert); cache.release(expert)
        else:
            for expert in cold:            # each rare expert loaded exactly once
                cache.acquire(expert)
                load_order.append(expert)
                cache.release(expert)

        return LayerResult(
            layer=layer,
            bytes_fetched=cache.bytes_fetched - bytes_before,
            load_order=load_order,
            hot_loads=len(hot),
            straggler_loads=len(cold) - skipped_loads,
            skipped_loads=skipped_loads,
            skipped_applications=skipped_apps,
            skipped_mass=skipped_mass,
            calls=len(calls),
        )


@dataclass
class RunResult:
    tokens: int                           # generated tokens (steps x calls)
    bytes_fetched: float                  # MB
    layer_results: list[LayerResult]
    cache: ResidencyCache
    skipped_mass: float = 0.0
    skipped_applications: int = 0
    token_layers: int = 0                 # tokens x layers — the unit of skipped mass

    @property
    def mb_per_token(self) -> float:
        return self.bytes_fetched / self.tokens if self.tokens else 0.0

    @property
    def hit_rate(self) -> float:
        return self.cache.hit_rate

    @property
    def skipped_mass_fraction(self) -> float:
        """Mean fraction of a token-layer's gate mass that was dropped. 0 under
        cold_policy="load"; bounded by skip_budget under "skip". The quality proxy."""
        return self.skipped_mass / self.token_layers if self.token_layers else 0.0


class Coordinator:
    """Drives a phase-aligned batch of calls through all layers for `tokens` decode
    steps, in lockstep, accumulating bytes fetched and advancing demand decay once per
    step. This is the throughput engine the simulator measures and the real server would
    wrap around llama.cpp / WASTE."""

    def __init__(self, cache: ResidencyCache, scheduler: ExpertMajorScheduler,
                 layers: int):
        self.cache = cache
        self.scheduler = scheduler
        self.layers = layers

    def run(self, calls: list[Call], tokens: int,
            routing: "Optional[Callable[[Call, int], list]]" = None,
            keep_layer_results: bool = False) -> RunResult:
        """Run `tokens` decode steps for the lockstep batch `calls`.

        If `routing` is given it is called once per call per token to refresh that
        token's per-layer route (real routing varies token to token). If omitted, each
        call's fixed `route` is reused every token (single-token and test mode).
        """
        results: list[LayerResult] = []
        skipped_mass = 0.0
        skipped_apps = 0
        total_bytes_before = self.cache.bytes_fetched
        for tok in range(tokens):
            if routing is not None:
                for c in calls:
                    c.route = routing(c, tok)
            for L in range(self.layers):
                r = self.scheduler.run_layer(L, calls)
                skipped_mass += r.skipped_mass
                skipped_apps += r.skipped_applications
                if keep_layer_results:
                    results.append(r)
            self.cache.decay_step()
        n_tokens = tokens * len(calls)
        return RunResult(
            tokens=n_tokens,
            bytes_fetched=self.cache.bytes_fetched - total_bytes_before,
            layer_results=results,
            cache=self.cache,
            skipped_mass=skipped_mass,
            skipped_applications=skipped_apps,
            token_layers=n_tokens * self.layers,
        )
