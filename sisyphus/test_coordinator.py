"""
Invariant tests for the Sisyphus scheduler core. These prove the properties the design
doc claims — not just that it runs. Run: python -m pytest sisyphus/test_coordinator.py -q
"""
import os
import subprocess
import sys

import pytest

from sisyphus.coordinator import (
    Call, Coordinator, DemandHistogram, ExpertMajorScheduler,
    ResidencyCache, ResidencyError,
)
from sisyphus.engine_sim import MACHINES, Machine, build, measure
from sisyphus.geometry import (
    EXPERT_MB, LAYERS, TRUNK_GB, cache_budget_gb, hot_set_gb, kv_gb, stable_seed,
)
from sisyphus.routing import BRACKETS, RoutingModel
from sisyphus.tape import max_batch, plan_tape


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
#  Demand decay (V3): the cache must follow the workload, not its history
# --------------------------------------------------------------------------- #
def test_decay_lets_new_domain_displace_stale_one():
    # Old regime: expert 1 was wanted 100 times long ago. New regime: expert 2 is wanted
    # once per step. Without decay, 1 squats forever; with decay, 2 wins within steps.
    for decay, expect_new_wins in ((1.0, False), (0.5, True)):
        c = ResidencyCache(capacity=2, policy="demand", decay=decay)
        c.note_demand(1, 100)
        c.acquire(1); c.release(1)
        for _ in range(12):
            c.decay_step()
            c.note_demand(2, 1)
        c.acquire(2); c.release(2)
        c.note_demand(3, 1)
        c.acquire(3)                      # must evict 1 or 2
        assert c.is_resident(2) is expect_new_wins, f"decay={decay}"


def test_decay_is_uniform_scaling_so_relative_order_holds():
    c = ResidencyCache(capacity=8, policy="demand", decay=0.8)
    c.note_demand(1, 5); c.note_demand(2, 3)
    c.decay_step(); c.decay_step()
    assert c.demand(1) == pytest.approx(5 * 0.8 ** 2)
    assert c.demand(2) == pytest.approx(3 * 0.8 ** 2)
    assert c.demand(1) > c.demand(2)


def test_decay_renormalisation_keeps_heap_valid(monkeypatch):
    monkeypatch.setattr(ResidencyCache, "_RENORM_AT", 10.0)
    c = ResidencyCache(capacity=2, policy="demand", decay=0.5)
    c.note_demand(1, 4); c.acquire(1); c.release(1)
    c.note_demand(2, 1); c.acquire(2); c.release(2)
    for _ in range(10):                   # scale crosses 10.0 repeatedly → renormalises
        c.decay_step()
    assert c._scale <= 10.0 and c.steps == 10
    assert c.demand(1) == pytest.approx(4 * 0.5 ** 10) and c.demand(2) == pytest.approx(0.5 ** 10)
    c.acquire(3)                          # evicts the lower-demand idle expert: 2
    assert c.is_resident(1) and not c.is_resident(2)


def test_cache_rejects_bad_decay():
    with pytest.raises(ValueError):
        ResidencyCache(2, decay=0.0)


# --------------------------------------------------------------------------- #
#  DemandHistogram
# --------------------------------------------------------------------------- #
def test_histogram_orders_by_demand_then_id():
    h = DemandHistogram()
    h.add("a", {1, 2}); h.add("b", {2, 3}); h.add("c", {2})
    order = h.by_popularity()
    assert order[0] == (2, 3)             # expert 2 wanted by all three
    assert h.count(2) == 3 and set(h.callers(2)) == {"a", "b", "c"}
    assert order[1][0] == 1 and order[2][0] == 3   # ties break by ascending id


def test_histogram_tracks_gate_weights():
    h = DemandHistogram()
    h.add("a", {1: 0.7, 2: 0.3}); h.add("b", {2: 0.9, 3: 0.1})
    assert h.max_gate(2) == pytest.approx(0.9)
    assert h.weighted(2) == [("a", 0.3), ("b", 0.9)]
    assert h.max_gate(3) == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
#  ExpertMajorScheduler: ordering invariants
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
    calls = [Call("a", [{1, 2, 7}]), Call("b", [{1, 2, 8}]), Call("c", [{1, 9}])]
    res = sched.run_layer(0, calls)
    hot_positions = [res.load_order.index(e) for e in (1, 2)]
    cold_positions = [res.load_order.index(e) for e in (7, 8, 9)]
    assert max(hot_positions) < min(cold_positions)          # all hot before any cold
    assert res.hot_loads == 2 and res.straggler_loads == 3 and res.skipped_loads == 0


# --------------------------------------------------------------------------- #
#  Residency-aware routing (V3): skips are bounded, visible, and never touch hits
# --------------------------------------------------------------------------- #
def _weighted_calls():
    # a: hot 1 (0.5), cold 7 (0.05), cold 8 (0.05), cold 9 (0.40)   -> 9 too heavy to skip
    # b: hot 1 (0.9), cold 10 (0.10)
    return [Call("a", [{1: 0.5, 7: 0.05, 8: 0.05, 9: 0.40}]),
            Call("b", [{1: 0.9, 10: 0.10}])]


def test_skip_respects_budget_and_max_gate():
    cache = ResidencyCache(1000, policy="demand")
    sched = ExpertMajorScheduler(cache, hot_min=2, cold_policy="skip",
                                 skip_budget=0.08, skip_max_gate=0.06)
    res = sched.run_layer(0, _weighted_calls())
    # a may skip 7 (0.05); 8 would push it to 0.10 > budget 0.08; 9 exceeds max_gate.
    # b's 10 (0.10) exceeds max_gate → loaded.
    assert 7 not in res.load_order
    assert {1, 8, 9, 10} <= set(res.load_order)
    assert res.skipped_loads == 1 and res.skipped_applications == 1
    assert res.skipped_mass == pytest.approx(0.05)


def test_skip_never_drops_a_resident_expert():
    cache = ResidencyCache(1000, policy="demand")
    cache.acquire(7); cache.release(7)                     # 7 is already resident
    sched = ExpertMajorScheduler(cache, hot_min=2, cold_policy="skip",
                                 skip_budget=1.0, skip_max_gate=1.0)
    res = sched.run_layer(0, _weighted_calls())
    assert 7 in res.load_order                             # a hit is free: applied, not skipped
    assert res.skipped_loads == 3                          # 8, 9, 10: cold, absent, in budget
    assert cache.fetches == 2                              # 7 (setup) + hot expert 1


def test_zero_budget_skip_equals_load():
    calls_a, calls_b = _weighted_calls(), _weighted_calls()
    ca = ResidencyCache(1000, policy="demand")
    cb = ResidencyCache(1000, policy="demand")
    la = ExpertMajorScheduler(ca, hot_min=2, cold_policy="load").run_layer(0, calls_a)
    lb = ExpertMajorScheduler(cb, hot_min=2, cold_policy="skip", skip_budget=0.0).run_layer(0, calls_b)
    assert set(la.load_order) == set(lb.load_order) and lb.skipped_mass == 0.0


def test_skipped_mass_fraction_is_bounded_by_budget_end_to_end():
    rm = RoutingModel(**BRACKETS["expected"], layers=6)
    calls = [Call(f"c{i}", [], domain="code") for i in range(8)]
    cube = rm.build_cube(calls, 4)
    cache, sched = build("skip", 2000, skip_budget=0.10)
    r = Coordinator(cache, sched, 6).run(calls, 4, routing=lambda c, t: cube[c.id][t])
    assert 0.0 < r.skipped_mass_fraction <= 0.10 + 1e-9
    per_layer_max = max(lr.skipped_mass / lr.calls for lr in
                        Coordinator(cache, sched, 6).run(calls, 1, routing=lambda c, t: cube[c.id][t],
                                                          keep_layer_results=True).layer_results)
    assert per_layer_max <= 0.10 + 1e-9


def test_scheduler_rejects_bad_config():
    cache = ResidencyCache(4)
    with pytest.raises(ValueError):
        ExpertMajorScheduler(cache, cold_policy="defer")
    with pytest.raises(ValueError):
        ExpertMajorScheduler(cache, hot_min=0)


# --------------------------------------------------------------------------- #
#  Policy ordering end to end (real geometry, small token counts)
# --------------------------------------------------------------------------- #
def _cube_and_calls(batch, tokens, **kw):
    rm = RoutingModel(**{**BRACKETS["expected"], **kw}, coherent=True)
    calls = [Call(f"c{i}", [], domain="code") for i in range(batch)]
    return rm, calls, rm.build_cube(calls, tokens)


def test_policy_ladder_on_192gb():
    rm, calls, cube = _cube_and_calls(batch=16, tokens=6)
    mach = MACHINES["192"]
    naive = measure("naive_single", calls, cube, mach, 6, rm=rm)
    union = measure("union_lru", calls, cube, mach, 6, rm=rm)
    sisy = measure("sisyphus", calls, cube, mach, 6, rm=rm)
    skip = measure("skip", calls, cube, mach, 6, rm=rm)
    assert union.mb_per_token < naive.mb_per_token     # batching dedups shared experts
    assert sisy.mb_per_token < union.mb_per_token      # demand eviction keeps the hot set
    assert skip.mb_per_token < sisy.mb_per_token       # skipping removes tail bytes
    assert skip.skipped_mass <= 0.10 and sisy.skipped_mass == 0.0
    assert skip.tokens_per_night > sisy.tokens_per_night > union.tokens_per_night


def test_ram_budget_is_derived_not_asserted():
    # a 128 GB box cannot give experts 100 GB once the 60 GB trunk and KV are resident
    assert cache_budget_gb(128, 32, 768) < 60
    assert cache_budget_gb(128, 32, 768, trunk_on_gpu=True) > 100
    assert cache_budget_gb(32, 8, 768) == 0.0            # cannot even hold the trunk
    cache, streams = MACHINES["32"].budget(8, 768)
    assert streams and cache > 0                         # so the trunk streams
    assert hot_set_gb(0.10) + TRUNK_GB > 128             # the hot set + trunk need > 128 GB


def test_incoherent_routing_destroys_the_convoy_premise():
    mach = MACHINES["192"]
    a = _cube_and_calls(16, 4)
    rm_i = RoutingModel(**BRACKETS["expected"], coherent=False)
    calls_i = [Call(f"c{i}", [], domain="code") for i in range(16)]
    cube_i = rm_i.build_cube(calls_i, 4)
    coh = measure("decay", a[1], a[2], mach, 4, rm=a[0])
    inc = measure("decay", calls_i, cube_i, mach, 4, rm=rm_i)
    assert inc.mb_per_token > 1.5 * coh.mb_per_token


def test_determinism_across_processes():
    """The old hash()-seeded model gave different bytes per process. Stable seeding must
    give identical bytes under two different PYTHONHASHSEED values."""
    code = ("from sisyphus.engine_sim import measure, MACHINES\n"
            "from sisyphus.routing import RoutingModel, BRACKETS\n"
            "from sisyphus.coordinator import Call\n"
            "rm=RoutingModel(**BRACKETS['expected']); calls=[Call(f'c{i}',[],'code') for i in range(8)]\n"
            "cube=rm.build_cube(calls,3); m=measure('skip',calls,cube,MACHINES['192'],3,rm=rm)\n"
            "print(repr(m.bytes_mb))")
    outs = []
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True,
                                   text=True, env=env, cwd=os.path.dirname(os.path.dirname(__file__))
                                   ).stdout.strip())
    assert outs[0] == outs[1] and outs[0]
    assert stable_seed("a", 1) == stable_seed("a", 1) != stable_seed("a", 2)


def test_full_geometry_sanity():
    rm, calls, cube = _cube_and_calls(batch=8, tokens=4)
    m = measure("skip", calls, cube, MACHINES["192"], 4, rm=rm)
    assert m.tokens == 8 * 4
    assert 0.0 <= m.hit_rate <= 1.0 and m.mb_per_token > 0
    assert len(cube["c0"]) == 4 and len(cube["c0"][0]) == LAYERS
    assert all(abs(sum(r.values()) - 1.0) < 1e-9 for r in cube["c0"][0])   # gates normalised


# --------------------------------------------------------------------------- #
#  Tape regime and KV budget
# --------------------------------------------------------------------------- #
def test_tape_batch_is_bounded_by_kv():
    assert max_batch(128, 1024) < max_batch(192, 1024) < max_batch(256, 1024)
    assert max_batch(128, 1024) < max_batch(128, 512)
    from sisyphus.geometry import KDA_STATE_MB_PER_STREAM, KV_MB_PER_TOKEN
    assert kv_gb(1, 1024) == pytest.approx((KDA_STATE_MB_PER_STREAM + 1024 * KV_MB_PER_TOKEN) / 1024)
    assert KDA_STATE_MB_PER_STREAM > 1024 * KV_MB_PER_TOKEN   # fixed state dominates KV at 1K ctx
    p = plan_tape(128, 10_000, 1024)
    assert not p.feasible and p.tokens_per_night == 0.0
    p = plan_tape(192, 64, 1024)
    assert p.feasible and p.step_io_s > 0 and p.tok_per_s == pytest.approx(64 / p.step_s)


def test_tape_becomes_compute_bound_at_large_batch():
    small = plan_tape(256, 32, 256)
    big = plan_tape(1024, 1000, 256)      # hypothetical box, just to cross the knee
    assert not small.compute_bound and big.compute_bound


# --------------------------------------------------------------------------- #
#  Profiling round trip: synthetic log -> analyzer recovers the model -> replay matches
# --------------------------------------------------------------------------- #
def test_profiling_round_trip(tmp_path):
    from sisyphus.profiling import LoggedRouting, analyze_file, synthesize
    rm = RoutingModel(hot_frac=0.10, hot_mass=0.75, cold_weight_ratio=0.5, layers=6, seed=7)
    calls = ([Call(f"a{i}", [], "code") for i in range(6)]
             + [Call(f"b{i}", [], "law") for i in range(6)])
    path = tmp_path / "routes.jsonl.gz"
    n = synthesize(rm, calls, decode_tokens=48, path=path, prefill_tokens=8)
    assert n == 12 * (48 + 8) * 6
    prof = analyze_file(path)
    d = prof.decode
    assert d.layers == 6 and d.tokens == 12 * 48 and d.topk == pytest.approx(16.0)
    assert abs(d.hot_mass_at_frac["0.10"] - 0.75) < 0.05      # recovers hot_mass
    assert abs(d.cold_weight_ratio - 0.5) < 0.10              # recovers cold gate ratio
    assert d.coherence_same_domain > 0.8 > 0.3 > d.coherence_cross_domain
    assert prof.bracket()["coherent"] is True
    assert prof.prefill.tokens == 12 * 8
    # replay reproduces the model's routes exactly (globally-unique ids, same gates)
    log = LoggedRouting.from_file(path)
    c = Call("a0", [], "code")
    replayed = log(c, 5)
    direct = rm.route(c, 5)
    assert [set(r) for r in replayed] == [set(r) for r in direct]
    for a, b in zip(replayed, direct):
        for e in a:
            assert a[e] == pytest.approx(b[e], abs=1e-6)   # gates are logged to 6 places
    prof.to_json(tmp_path / "p.json")
    from sisyphus.profiling import RoutingProfile
    assert RoutingProfile.from_json(tmp_path / "p.json").bracket() == prof.bracket()


def test_measure_logged_matches_model_measure(tmp_path):
    from sisyphus.profiling import LoggedRouting, measure_logged, synthesize
    rm = RoutingModel(**BRACKETS["expected"], layers=5, seed=3)
    calls = [Call(f"c{i}", [], "code") for i in range(6)]
    path = tmp_path / "r.jsonl"
    synthesize(rm, calls, decode_tokens=4, path=path)
    log = LoggedRouting.from_file(path)
    via_log = measure_logged("skip", log, MACHINES["192"], tokens=4, cache_gb=50)
    cube = {c.id: [rm.route(c, t) for t in range(4)] for c in calls}
    via_model = measure("skip", calls, cube, MACHINES["192"], 4, rm=None, cache_gb=50, layers=5)
    assert via_log.bytes_mb == via_model.bytes_mb


# --------------------------------------------------------------------------- #
#  Night pipeline: decay follows a domain shift, cumulative does not
# --------------------------------------------------------------------------- #
def test_decay_beats_cumulative_on_mixed_night():
    from sisyphus.schedule import run_night, sample_jobs
    jobs = sample_jobs(per_domain=8)
    mach = Machine("test", 192.0, 10.0)
    cum = run_night(jobs, policy="sisyphus", machine=mach, convoy_size=8, tokens_per_job=6)
    dec = run_night(jobs, policy="decay", machine=mach, convoy_size=8, tokens_per_job=6)
    assert dec.convoys == cum.convoys == 2
    assert dec.mb_per_token < cum.mb_per_token


# --------------------------------------------------------------------------- #
#  Compute model (V4): FLOPs vs traversal, CPU vs GPU streaming
# --------------------------------------------------------------------------- #
def test_cpu_is_flop_bound_at_batch_32_and_batching_stops_helping():
    from sisyphus.geometry import CPU_ONLY
    s1, b1 = CPU_ONLY.step_seconds(1, 1472 * EXPERT_MB / 1024, kv_gb(1, 768))
    s32, b32 = CPU_ONLY.step_seconds(32, 200 * 92 * EXPERT_MB / 1024, kv_gb(32, 768))
    s128, b128 = CPU_ONLY.step_seconds(128, 400 * 92 * EXPERT_MB / 1024, kv_gb(128, 768))
    assert b1 == "dram" and b32 == "cpu" and b128 == "cpu"
    assert s32 / 32 == pytest.approx(s128 / 128, rel=0.01)     # per-token cost flat once FLOP-bound


def test_gpu_streaming_moves_the_wall_to_pcie_and_scales_with_batch():
    from sisyphus.geometry import GPU_STREAM_X16
    s32, b32 = GPU_STREAM_X16.step_seconds(32, 200 * 92 * EXPERT_MB / 1024, kv_gb(32, 768))
    s128, b128 = GPU_STREAM_X16.step_seconds(128, 400 * 92 * EXPERT_MB / 1024, kv_gb(128, 768))
    assert b32 == b128 == "pcie"
    assert s128 / 128 < 0.6 * (s32 / 32)                        # sublinear expert growth pays


def test_trunk_on_gpu_removes_most_cpu_flops():
    from sisyphus.geometry import CPU_ONLY, GPU_TRUNK, GFLOP_PER_TOKEN_TRUNK, GFLOP_PER_TOKEN
    cpu, _ = CPU_ONLY.step_seconds(32, 18.0, 16.0)
    trunk, _ = GPU_TRUNK.step_seconds(32, 18.0, 16.0)
    assert trunk / cpu == pytest.approx(1 - GFLOP_PER_TOKEN_TRUNK / GFLOP_PER_TOKEN, abs=0.05)


def test_gpu_prefill_is_orders_of_magnitude_cheaper():
    from sisyphus.geometry import CPU_ONLY, GPU_STREAM_X16
    cpu, cb = CPU_ONLY.prefill_seconds(32 * 512, 700.0, kv_gb(32, 512))
    gpu, gb = GPU_STREAM_X16.prefill_seconds(32 * 512, 700.0, kv_gb(32, 512))
    assert cb == "cpu" and gb in ("pcie", "gpu")
    assert cpu > 20 * gpu


def test_measure_reports_binding_resource():
    rm, calls, cube = _cube_and_calls(batch=16, tokens=4)
    m_cpu = measure("skip", calls, cube, MACHINES["192"], 4, rm=rm)
    m_gpu = measure("skip", calls, cube, MACHINES["192+GPU"], 4, rm=rm)
    assert m_cpu.bound in ("ssd", "cpu", "dram")
    assert m_gpu.bound in ("ssd", "pcie", "gpu")
    assert m_gpu.prefill_s < m_cpu.prefill_s / 5
    assert m_gpu.tokens_per_night > m_cpu.tokens_per_night
