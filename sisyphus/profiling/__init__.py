"""
sisyphus.profiling — rent-to-profile: replace the routing guess with measured routes.

The scheduler's every number rests on one unknown — how Kimi-K3 actually routes on the
operator's workload — and the laptop cannot run K3 to find out. A rented GPU node can,
for about the price of lunch. This package is the round trip:

    capture   (rented box)  hook the routers, log every (call, token, layer) route
    schema    (both)        the JSONL record both sides agree on
    analyze   (laptop)      log -> RoutingProfile: hot_frac, hot_mass, coherence,
                            cold gate weight, skippable mass, prefill working set
    replay    (laptop)      run the real scheduler on the real routes
    synthetic (laptop)      a log from RoutingModel, to prove the loop before paying

See RENT_TO_PROFILE.md for the runbook.
"""
from .schema import RouteRecord, RouteLogWriter, read_log
from .analyze import RoutingProfile, PhaseProfile, analyze, analyze_file
from .replay import LoggedRouting, measure_logged
from .synthetic import synthesize

__all__ = ["RouteRecord", "RouteLogWriter", "read_log", "RoutingProfile", "PhaseProfile",
           "analyze", "analyze_file", "LoggedRouting", "measure_logged", "synthesize"]
