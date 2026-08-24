"""
Invariant tests for the Sisyphus scheduler core. These prove the properties the design
doc claims — not just that it runs. Run: python -m pytest sisyphus/test_coordinator.py -q
"""
import pytest

from sisyphus.coordinator import (
    Call, Coordinator, DemandHistogram, ExpertMajorScheduler,
    ResidencyCache, ResidencyError,
)
from sisyphus.geometry import EXPERT_MB, LAYERS
from sisyphus.routing import RoutingModel
from sisyphus.engine_sim import measure


# --------------------------------------------------------------------------- #
#  ResidencyCache: reference-counted eviction (V2.18 correctness)
# --------------------------------------------------------------------------- #
def test_hit_on_reacquire_is_free():
    c = ResidencyCache(4)
    assert c.acquire(7) is True          # miss → fetched
    c.release(7)
    assert c.acquire(7) is False         # resident → hit, no fetch
    assert c.fetches == 1 and c.hits == 1
    assert c.bytes_fetched == EXPERT_MB


def test_pinned_expert_never_evicted():
    c = ResidencyCache(capacity=2, policy="lru")
    c.acquire(1)                          # refcount 1
    c.acquire(2)                          # refcount 1 — cache full, both pinned
    with pytest.raises(ResidencyError):
        c.acquire(3)                      # nothing evictable → loud failure, not silent
    c.release(1)                          # 1 now idle
    assert c.acquire(3) is True           # evicts idle 1, keeps pinned 2
    assert not c.is_resident(1)
    assert c.is_resident(2)               # a live call's expert survived


def test_refcount_survives_concurrent_holders():
    c = ResidencyCache(capacity=1, policy="lru")
    c.acquire(5)                          # convoy A pins 5
    c.acquire(5)                          # convoy B also needs 5 → refcount 2, one fetch
    assert c.refcount(5) == 2 and c.fetches == 1
    c.release(5)                          # A done; B still holds it
    assert c.refcount(5) == 1 and c.is_resident(5)


def test_demand_eviction_keeps_popular_expert():
    c = ResidencyCache(capacity=2, policy="demand")
    c.note_demand(1, 10)                  # expert 1 is hot
    c.note_demand(2, 1)                   # expert 2 is cold
    c.acquire(1); c.release(1)
    c.acquire(2); c.release(2)
    c.acquire(3)                          # must evict one idle → the cold one
    assert c.is_resident(1)               # hot expert kept warm
    assert not c.is_resident(2)           # cold expert evicted


def test_lru_eviction_drops_oldest():
    c = ResidencyCache(capacity=2, policy="lru")
    c.acquire(1); c.release(1)
    c.acquire(2); c.release(2)
    c.acquire(3)                          # evicts oldest idle = 1
    assert not c.is_resident(1) and c.is_resident(2) and c.is_resident(3)


# --------------------------------------------------------------------------- #
#  DemandHistogram
# --------------------------------------------------------------------------- #
def test_histogram_orders_by_demand_then_id():
    h = DemandHistogram()
    h.add("a", {1, 2}); h.add("b", {2, 3}); h.add("c", {2})
    order = h.by_popularity()
    assert order[0] == (2, 3)             # expert 2 wanted by all three
    assert h.count(2) == 3 and set(h.callers(2)) == {"a", "b", "c"}
    # ties (experts 1 and 3 each demand 1) break by ascending id
    assert order[1][0] == 1 and order[2][0] == 3


# --------------------------------------------------------------------------- #
#  ExpertMajorScheduler (V2.19)
# --------------------------------------------------------------------------- #
def test_never_double_loads_within_a_layer():
    cache = ResidencyCache(1000, policy="demand")
    sched = ExpertMajorScheduler(cache, hot_min=2)
    calls = [Call("a", [{1, 2, 3}]), Call("b", [{1, 2, 9}]), Call("c", [{1, 5}])]
    res = sched.run_layer(0, calls)
    assert len(res.load_order) == len(set(res.load_order))   # each expert loaded once
    assert set(res.load_order) == {1, 2, 3, 9, 5}            # the whole union, no more


def test_hot_served_before_cold():
    cache = ResidencyCache(1000, policy="demand")
    sched = ExpertMajorScheduler(cache, hot_min=2)
    # expert 1 wanted by all (hot); 2 by two (hot); 7,8,9 single-use (cold)
    calls = [Call("a", [{1, 2, 7}]), Call("b", [{1, 2, 8}]), Call("c", [{1, 9}])]
    res = sched.run_layer(0, calls)
    hot_positions = [res.load_order.index(e) for e in (1, 2)]
    cold_positions = [res.load_order.index(e) for e in (7, 8, 9)]
    assert max(hot_positions) < min(cold_positions)          # all hot before any cold
    assert res.hot_loads == 2 and res.straggler_loads == 3


def test_aging_bounds_deferrals_so_rare_jobs_dont_starve():
    cache = ResidencyCache(100000, policy="demand")
    max_def = 3
    sched = ExpertMajorScheduler(cache, hot_min=2, max_deferrals=max_def)
    # 'lonely' always routes to a unique cold expert; the crowd shares a hot one.
    layers = 20
    crowd = [Call(f"h{i}", [{0} for _ in range(layers)]) for i in range(3)]
    lonely = Call("lonely", [{1000 + L} for L in range(layers)])   # all cold, all unique
    calls = crowd + [lonely]
    forced_ever = False
    for L in range(layers):
        r = sched.run_layer(L, calls)
        assert lonely.deferrals <= max_def          # never allowed to exceed the cap
        if "lonely" in r.forced_by_aging:
            forced_ever = True
    assert forced_ever                              # aging did fire and rescue it


# --------------------------------------------------------------------------- #
#  End-to-end policy ordering + determinism
# --------------------------------------------------------------------------- #
def _cube_and_calls(batch, tokens, hot_mass=0.75):
    rm = RoutingModel(hot_mass=hot_mass, coherent=True)
    calls = [Call(f"c{i}", [], domain="code") for i in range(batch)]
    return rm, calls, rm.build_cube(calls, tokens)


def test_batching_beats_naive_and_sisyphus_beats_lru():
    rm, calls, cube = _cube_and_calls(batch=16, tokens=6)
    naive = measure("naive_single", calls, cube, 10.0, 100, tokens=6)
    union = measure("union_lru",   calls, cube, 10.0, 100, tokens=6)
    sisy  = measure("sisyphus",    calls, cube, 10.0, 100, tokens=6)
    # batching dedups shared experts → fewer bytes than streaming each call alone
    assert union.mb_per_token < naive.mb_per_token
    # demand eviction keeps the hot set warm at 128GB → strictly fewer bytes than LRU
    assert sisy.mb_per_token < union.mb_per_token
    assert sisy.hit_rate > union.hit_rate


def test_determinism_same_seed_same_bytes():
    rm1, calls1, cube1 = _cube_and_calls(batch=8, tokens=5)
    rm2, calls2, cube2 = _cube_and_calls(batch=8, tokens=5)
    a = measure("sisyphus", calls1, cube1, 10.0, 100, tokens=5)
    b = measure("sisyphus", calls2, cube2, 10.0, 100, tokens=5)
    assert a.bytes_mb == b.bytes_mb


def test_full_geometry_sanity():
    # a real-geometry run completes and reports sane numbers
    rm, calls, cube = _cube_and_calls(batch=8, tokens=4)
    m = measure("sisyphus", calls, cube, 10.0, 100, tokens=4)
    assert m.tokens == 8 * 4
    assert 0.0 <= m.hit_rate <= 1.0
    assert m.mb_per_token > 0
    assert len(cube["c0"]) == 4 and len(cube["c0"][0]) == LAYERS
