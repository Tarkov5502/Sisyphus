"""
sisyphus.coordinator — the concurrency-safe residency owner and the demand-ordered
expert-major scheduler (design doc V2.18 + V2.19).

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
                          DEMAND-FREQUENCY, not LRU, because a layer-sequential MoE
                          sweep is nearly the worst case for LRU (every union expert
                          was "just used", so LRU keeps precisely the experts about to
                          become useless next layer).

  DemandHistogram       — per-layer cross-convoy expert demand: who wants what, and
                          how many.

  ExpertMajorScheduler  — V2.19, the flagship. Per layer, across every phase-aligned
                          call, service experts HOTTEST-FIRST (load once, serve the
                          whole crowd), sweep cold single-use experts into one
                          consolidated late batch (rare work done once, late, cheap),
                          and AGE deferred calls so a rare-expert-only job can never
                          starve.

Stdlib only. Deterministic given a seeded routing model. The real inference engine is
meant to call this; engine_sim.py drives it with a routing model to measure the one
number that matters on disk-offload hardware — bytes fetched per generated token.
"""
from __future__ import annotations

import heapq
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .geometry import EXPERT_MB


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
    """

    def __init__(self, capacity: int, policy: str = "demand",
                 pinned: Optional[Iterable[int]] = None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1 expert")
        if policy not in ("demand", "lru"):
            raise ValueError("policy must be 'demand' or 'lru'")
        self.capacity = capacity
        self.policy = policy
        self.pinned: set[int] = set(pinned or ())
        # resident (non-pinned) experts -> refcount. OrderedDict gives LRU order:
        # least-recently-acquired is left-most.
        self._resident: "OrderedDict[int, int]" = OrderedDict()
        # cumulative demand seen per expert, the demand-policy eviction key.
        self._demand: Counter = Counter()
        # lazy min-heap of (demand_at_push, expert) over idle (refcount-0) experts, the
        # demand-policy eviction structure. Entries go stale as demand rises or refcount
        # changes; _pick_victim validates on pop and re-pushes the stale ones.
        self._heap: list[tuple[int, int]] = []
        # ---- instrumentation ----
        self.fetches = 0          # disk loads (misses)
        self.hits = 0             # resident/pinned re-uses
        self.evictions = 0
        self.bytes_fetched = 0.0  # MB

    # -- demand signal (fed by the scheduler before it services a layer) -------
    def note_demand(self, expert: int, count: int = 1) -> None:
        """Record that `expert` is wanted `count` times. Drives demand-eviction and
        lets the cache keep globally-popular experts warm across tokens."""
        self._demand[expert] += count

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
                heapq.heappush(self._heap, (self._demand[expert], expert))
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
        expert from a lazy min-heap — keeping popular experts warm regardless of age,
        which a bounded/oldest sample cannot do (a hot expert loaded early in the sweep
        must outlive cold experts loaded later)."""
        if self.policy == "lru":
            for e, rc in self._resident.items():        # OrderedDict: oldest first
                if rc == 0:
                    return e
            return None
        # demand: pop the lazy heap until a valid, current-priority victim surfaces.
        heap = self._heap
        while heap:
            dmd, e = heapq.heappop(heap)
            rc = self._resident.get(e)
            if rc is None or rc != 0:
                continue                                # gone or re-pinned → stale
            cur = self._demand[e]
            if dmd != cur:                              # demand rose → re-file, keep going
                heapq.heappush(heap, (cur, e))
                continue
            return e
        return None

    def resident_set(self) -> set[int]:
        return set(self.pinned) | set(self._resident.keys())


# --------------------------------------------------------------------------- #
#  Demand histogram: who wants what, this layer.
# --------------------------------------------------------------------------- #
class DemandHistogram:
    """Cross-convoy expert demand for a single layer: expert -> list of call ids."""

    def __init__(self) -> None:
        self._callers: dict[int, list[str]] = {}

    def add(self, call_id: str, experts: Iterable[int]) -> None:
        for e in experts:
            self._callers.setdefault(e, []).append(call_id)

    def count(self, expert: int) -> int:
        return len(self._callers.get(expert, ()))

    def callers(self, expert: int) -> list[str]:
        return self._callers.get(expert, [])

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
    """One inference stream. `route[L]` is the set of routed experts it needs at
    layer L (precomputed by a routing model, or logged from a real run)."""
    id: str
    route: list[set[int]]
    domain: str = ""
    deferrals: int = 0            # times parked into a straggler pool (aging counter)
    layer: int = 0               # current layer (lockstep convoys advance together)

    def experts_at(self, layer: int) -> set[int]:
        return self.route[layer]


@dataclass
class LayerResult:
    layer: int
    bytes_fetched: float
    load_order: list[int]                 # experts, in the order the layer loaded them
    hot_loads: int = 0
    straggler_loads: int = 0
    forced_by_aging: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  V2.19 — demand-ordered, straggler-consolidated, expert-major scheduler.
# --------------------------------------------------------------------------- #
class ExpertMajorScheduler:
    """Services one MoE layer for a set of phase-aligned calls.

    Ordering per layer:
      1. Tally cross-convoy demand (the histogram).
      2. HOT experts (demand >= hot_min) are serviced first, hottest-first — each
         loaded exactly once and applied to the whole crowd that needs it.
      3. COLD experts (demand < hot_min, i.e. single-use rare experts) are deferred
         into a consolidated straggler batch swept at the LAYER'S END — one load of a
         rare expert serves every straggler from every convoy that wanted it.
      4. AGING: any call parked more than `max_deferrals` times has ALL its experts
         promoted to the hot pass this layer, so a rare-expert-only job cannot starve.

    Because weights are read-only, applying one loaded expert to many calls is always
    correct; the scheduler only reorders *loads*, never results.
    """

    def __init__(self, cache: ResidencyCache, hot_min: int = 2, max_deferrals: int = 4):
        if hot_min < 1:
            raise ValueError("hot_min must be >= 1")
        self.cache = cache
        self.hot_min = hot_min
        self.max_deferrals = max_deferrals

    def run_layer(self, layer: int, calls: list[Call]) -> LayerResult:
        hist = DemandHistogram()
        for c in calls:
            hist.add(c.id, c.experts_at(layer))
            for e in c.experts_at(layer):
                self.cache.note_demand(e)

        # Aging: which calls must be fully served this layer regardless of coldness.
        forced = {c.id for c in calls if c.deferrals >= self.max_deferrals}

        # Partition experts into hot (serve now) and cold (defer), with aged calls'
        # experts force-promoted to hot.
        hot: list[int] = []
        cold: list[int] = []
        forced_experts: set[int] = set()
        for c in calls:
            if c.id in forced:
                forced_experts |= c.experts_at(layer)
        for expert, demand in hist.by_popularity():
            if demand >= self.hot_min or expert in forced_experts:
                hot.append(expert)
            else:
                cold.append(expert)

        load_order: list[int] = []
        bytes_before = self.cache.bytes_fetched

        # --- hot pass: hottest-first, load once, serve the crowd ---
        for expert in hot:
            self.cache.acquire(expert)     # fetch iff not resident
            load_order.append(expert)
            self.cache.release(expert)     # matmul done for all its callers this layer
        hot_bytes = self.cache.bytes_fetched - bytes_before
        hot_loads = len(hot)

        # --- straggler sweep: consolidated cold batch at the layer's end ---
        # A call is a straggler if any of its experts fell into `cold`. It is served
        # here (this layer still completes — lockstep), but its coldness is charged to
        # a deferral for aging, unless it was force-served.
        cold_set = set(cold)
        straggler_before = self.cache.bytes_fetched
        for expert in cold:                # each rare expert loaded exactly once
            self.cache.acquire(expert)
            load_order.append(expert)
            self.cache.release(expert)
        straggler_bytes = self.cache.bytes_fetched - straggler_before
        straggler_loads = len(cold)

        for c in calls:
            if c.id in forced:
                c.deferrals = 0            # aged out and served; reset the clock
            elif c.experts_at(layer) & cold_set:
                c.deferrals += 1           # rode in a straggler batch this layer
            else:
                c.deferrals = 0            # fully served by the hot crowd

        return LayerResult(
            layer=layer,
            bytes_fetched=hot_bytes + straggler_bytes,
            load_order=load_order,
            hot_loads=hot_loads,
            straggler_loads=straggler_loads,
            forced_by_aging=sorted(forced),
        )


@dataclass
class RunResult:
    tokens: int
    bytes_fetched: float
    layer_results: list[LayerResult]
    cache: ResidencyCache

    @property
    def mb_per_token(self) -> float:
        return self.bytes_fetched / self.tokens if self.tokens else 0.0

    @property
    def hit_rate(self) -> float:
        total = self.cache.hits + self.cache.fetches
        return self.cache.hits / total if total else 0.0


class Coordinator:
    """Drives a phase-aligned batch of calls through all layers for `tokens` decode
    steps, in lockstep, accumulating bytes fetched. This is the throughput engine the
    simulator measures and the real server would wrap around llama.cpp / WASTE."""

    def __init__(self, cache: ResidencyCache, scheduler: ExpertMajorScheduler,
                 layers: int):
        self.cache = cache
        self.scheduler = scheduler
        self.layers = layers

    def run(self, calls: list[Call], tokens: int,
            routing: "Optional[Callable[[Call, int], list[set]]]" = None) -> RunResult:
        """Run `tokens` decode steps for the lockstep batch `calls`.

        If `routing` is given it is called once per call per token to refresh that
        token's per-layer route (real routing varies token to token); the persistent
        Call objects carry aging state across the whole run. If omitted, each call's
        fixed `route` is reused every token (single-token and test mode).
        """
        results: list[LayerResult] = []
        total_bytes_before = self.cache.bytes_fetched
        for tok in range(tokens):
            if routing is not None:
                for c in calls:
                    c.route = routing(c, tok)
            for L in range(self.layers):
                results.append(self.scheduler.run_layer(L, calls))
        return RunResult(
            tokens=tokens * len(calls),
            bytes_fetched=self.cache.bytes_fetched - total_bytes_before,
            layer_results=results,
            cache=self.cache,
        )
