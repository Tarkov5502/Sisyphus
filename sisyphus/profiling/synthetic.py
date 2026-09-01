"""
sisyphus.profiling.synthetic — a route log generated from a RoutingModel.

Two uses:
  * tests: the analyzer must recover the parameters the model was built with (a
    closed loop that proves the pipeline before a single dollar is spent on a GPU);
  * dry runs: exercise the whole rent-to-profile flow — write log, analyze, replay,
    measure — on a laptop, so the only thing left to do on the rented box is run
    `capture` for real.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..coordinator import Call
from ..routing import RoutingModel
from .schema import RouteLogWriter, RouteRecord


def synthesize(rm: RoutingModel, calls: Sequence[Call], decode_tokens: int,
               path: str | Path, prefill_tokens: int = 0) -> int:
    """Write `decode_tokens` (and optionally `prefill_tokens`) of routes per call.
    Returns the record count. Expert ids are written LOCAL per layer, as a model emits
    them, so the log is indistinguishable in shape from a real capture."""
    n = 0
    with RouteLogWriter(path) as w:
        for c in calls:
            for phase, count in (("prefill", prefill_tokens), ("decode", decode_tokens)):
                for t in range(count):
                    # separate token streams for the two phases so they do not collide
                    tok = t if phase == "decode" else -1 - t
                    for L in range(rm.layers):
                        route = rm.route_layer(c.id, c.domain, tok, L)
                        off = L * rm.experts
                        experts = tuple(e - off for e in route)
                        gates = tuple(route[e] for e in route)
                        w.write(RouteRecord(c.id, c.domain, phase, t, L, experts, gates))
                        n += 1
    return n
