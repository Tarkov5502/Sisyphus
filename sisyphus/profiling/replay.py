"""
sisyphus.profiling.replay — run the real scheduler on REAL routes.

Once a log exists, the simulator's routing model is no longer needed: `LoggedRouting`
implements the `routing(call, token) -> list[Route]` contract Coordinator.run expects,
straight from the log, with local expert ids lifted to Sisyphus's globally-unique ids.
`measure_logged` is engine_sim.measure with the routing model swapped for the log, so
every table in RESULTS.md can be regenerated from measured routes with one call.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..coordinator import Call
from ..geometry import EXPERTS
from ..routing import Route
from .schema import RouteRecord, read_log


class LoggedRouting:
    """Decode routes from a log, indexed for replay.

    Calls with fewer decode tokens than requested wrap around (the log is treated as a
    stationary sample of that call's routing), which keeps a lockstep batch aligned.
    """

    def __init__(self, records: Iterable[RouteRecord], n_experts: int = EXPERTS,
                 phase: str = "decode"):
        self.n_experts = n_experts
        routes: dict[str, dict[int, dict[int, Route]]] = defaultdict(lambda: defaultdict(dict))
        domains: dict[str, str] = {}
        layers: set[int] = set()
        for r in records:
            if r.phase != phase:
                continue
            off = r.layer * n_experts
            routes[r.call][r.token][r.layer] = {off + e: g for e, g in zip(r.experts, r.gates)}
            domains[r.call] = r.domain
            layers.add(r.layer)
        if not routes:
            raise ValueError(f"log has no {phase} records")
        self.layers = max(layers) + 1
        self._domains = domains
        # dense per-call token list, only tokens that have every layer
        self._routes: dict[str, list[list[Route]]] = {}
        for call, toks in routes.items():
            full = []
            for t in sorted(toks):
                if len(toks[t]) == self.layers:
                    full.append([toks[t][L] for L in range(self.layers)])
            if full:
                self._routes[call] = full

    @property
    def call_ids(self) -> list[str]:
        return sorted(self._routes)

    def tokens_for(self, call_id: str) -> int:
        return len(self._routes[call_id])

    def calls(self, ids: Iterable[str] | None = None) -> list[Call]:
        ids = list(ids) if ids is not None else self.call_ids
        return [Call(id=c, route=[], domain=self._domains.get(c, "")) for c in ids]

    def __call__(self, call: Call, token: int) -> list[Route]:
        seq = self._routes[call.id]
        return seq[token % len(seq)]

    @staticmethod
    def from_file(path: str | Path, **kw) -> "LoggedRouting":
        return LoggedRouting(read_log(path), **kw)


def measure_logged(policy: str, log: LoggedRouting, machine, tokens: int,
                   call_ids: Iterable[str] | None = None, **kw):
    """engine_sim.measure on logged routes. `kw` passes through (spec_mult, cache_gb,
    skip_budget, ...). The prefill term uses the conservative full-union bound because
    the log's prefill working set is measured separately by the analyzer."""
    from ..engine_sim import measure
    calls = log.calls(call_ids)
    cube = {c.id: [log(c, t) for t in range(tokens)] for c in calls}
    return measure(policy, calls, cube, machine, tokens, rm=None, layers=log.layers, **kw)
