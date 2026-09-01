"""
sisyphus.engine_sim — drive the REAL scheduler and measure bytes fetched per token.

Runs actual Call objects through the actual Coordinator + ExpertMajorScheduler +
ResidencyCache, so the numbers come out of the code that would ship. Policies are
compared on IDENTICAL routing (a route cube built once per config, replayed by every
policy):

  naive_single  — token-major: every call streams alone against a tiny per-call cache.
  union_lru     — batch: load each layer's union once, LRU eviction.
  sisyphus      — batch + cumulative demand-frequency eviction (V2.19 baseline).
  decay         — sisyphus + exponential demand decay (V3: adapts to domain shift).
  skip          — decay + residency-aware routing: cold non-resident low-gate experts
                  are skipped under a per-token gate-mass budget (V3 flagship).
  tape          — analytic full-sweep floor from sisyphus.tape (no routing assumption).

V3 also accounts for what the first simulator left out:
  * the RAM budget is derived (RAM - trunk - OS - KV), not asserted as "100 GB";
  * prefill is charged per convoy (its expert working set from the routing model);
  * KV memory scales with batch x context and competes with expert residency;
  * tokens/night is net of prefill, with the job shape stated.

Reported: MB/token, disk- vs compute-bound, resident hit rate, skipped gate mass,
decode tok/s, tokens/night. Deterministic given the seed (stable hashing, V3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .coordinator import Call, Coordinator, ExpertMajorScheduler, ResidencyCache
from .geometry import (
    CPU_ONLY, CPU_FAST_KERNELS, GPU_STREAM_X16, GPU_STREAM_X8, GPU_TRUNK, ComputeModel,
    DEFAULT_DECODE_TOKENS, DEFAULT_PROMPT_TOKENS, DRIVE_SEQUENTIAL_GBPS, EXPERTS, EXPERT_MB,
    LAYERS, MB_PER_GB, OS_RESERVE_GB, TOPK, TRUNK_GB, cache_budget_gb, experts_per_gb,
    hot_set_gb, kv_gb, night_tokens,
)
from .routing import BRACKETS, RoutingModel
from .tape import LAYER_BUFFER_GB, best_tape

POLICIES = ("naive_single", "union_lru", "sisyphus", "decay", "skip")
DECAY = 0.8                    # per-step demand retention for the V3 policies


# --------------------------------------------------------------------------- #
#  Machine budget
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Machine:
    label: str
    ram_gb: float
    drive_gbps: float            # 9.7 MB expert reads (scheduler regime)
    seq_gbps: float = DRIVE_SEQUENTIAL_GBPS
    compute: ComputeModel = CPU_ONLY

    @property
    def trunk_on_gpu(self) -> bool:
        return self.compute.trunk_on_gpu

    def budget(self, batch: int, context_tokens: int) -> tuple[float, bool]:
        """(expert cache GB, trunk_streams). If the trunk cannot stay resident beside
        the OS reserve and KV, it streams every step like the experts do — the only way
        a 32 GB machine touches this model at all."""
        cache = cache_budget_gb(self.ram_gb, batch, context_tokens, self.trunk_on_gpu)
        if cache >= LAYER_BUFFER_GB or self.trunk_on_gpu:
            return cache, False
        # trunk cannot be resident: stream it, keep what is left for experts
        left = self.ram_gb - OS_RESERVE_GB - kv_gb(batch, context_tokens) - LAYER_BUFFER_GB
        return max(0.0, left), True


MACHINES = {
    "32":        Machine("32GB · Gen3 1.5 · cpu",             32.0, 1.5, 2.0),
    "32L":       Machine("32GB · layout 3.2 · cpu",           32.0, 3.2, 3.5),
    "128":       Machine("128GB · 2 drives 10 · cpu",        128.0, 10.0),
    "128+GPU":   Machine("128GB · 2 drives · GPU x16",       128.0, 10.0, compute=GPU_STREAM_X16),
    "192":       Machine("192GB · 2 drives 10 · cpu",        192.0, 10.0),
    "192-2x":    Machine("192GB · 2 drives · cpu 2x kernels",192.0, 10.0, compute=CPU_FAST_KERNELS),
    "192+trunk": Machine("192GB · 2 drives · GPU trunk only",192.0, 10.0, compute=GPU_TRUNK),
    "192+GPU":   Machine("192GB · 2 drives · GPU x16",       192.0, 10.0, compute=GPU_STREAM_X16),
    "192+GPU4":  Machine("192GB · GPU x16 + 4 drives (M.2) 20",192.0, 20.0, 30.0, compute=GPU_STREAM_X16),
    "192+GPU8":  Machine("192GB · GPU x8 + 4 drives 25",      192.0, 25.0, 40.0, compute=GPU_STREAM_X8),
    "256":       Machine("256GB · 2 drives 10 · cpu",        256.0, 10.0),
    "256+GPU":   Machine("256GB · 2 drives · GPU x16",       256.0, 10.0, compute=GPU_STREAM_X16),
}


@dataclass
class Metrics:
    policy: str
    tokens: int
    bytes_mb: float
    mb_per_token: float
    hit_rate: float
    skipped_mass: float          # mean fraction of a token-layer's gate mass dropped
    tok_per_s: float             # steady-state decode throughput
    bound: str                   # ssd | dram | cpu | gpu | pcie — the binding resource
    tokens_per_night: float      # net of prefill, for the stated job shape
    cache_gb: float
    trunk_streams: bool
    prefill_s: float             # per convoy


# --------------------------------------------------------------------------- #
#  Policy construction
# --------------------------------------------------------------------------- #
def build(policy: str, capacity: int, *, skip_budget: float = 0.10,
          skip_max_gate: float = 0.06, decay: float = DECAY
          ) -> tuple[ResidencyCache, ExpertMajorScheduler]:
    capacity = max(1, capacity)
    if policy == "union_lru":
        cache = ResidencyCache(capacity, policy="lru")
        return cache, ExpertMajorScheduler(cache, hot_min=1)
    if policy == "sisyphus":
        cache = ResidencyCache(capacity, policy="demand", decay=1.0)
        return cache, ExpertMajorScheduler(cache, hot_min=2)
    if policy == "decay":
        cache = ResidencyCache(capacity, policy="demand", decay=decay)
        return cache, ExpertMajorScheduler(cache, hot_min=2)
    if policy == "skip":
        cache = ResidencyCache(capacity, policy="demand", decay=decay)
        return cache, ExpertMajorScheduler(cache, hot_min=2, cold_policy="skip",
                                           skip_budget=skip_budget,
                                           skip_max_gate=skip_max_gate)
    raise ValueError(f"unknown policy {policy!r}")


def _cube_routing(cube):
    """Adapt a prebuilt route cube to the Coordinator.run(routing=...) contract."""
    return lambda call, tok: cube[call.id][tok]


# --------------------------------------------------------------------------- #
#  Throughput accounting
# --------------------------------------------------------------------------- #
def prefill_seconds(rm: Optional[RoutingModel], batch: int, prompt_tokens: int,
                    cache_gb: float, machine: Machine, trunk_streams: bool,
                    expert_frac: float = 1.0) -> tuple[float, str]:
    """One convoy's prefill. Three costs overlap: SSD (touched experts not resident),
    memory traversal of every touched expert once (DRAM on CPU, PCIe when streamed to a
    GPU), and the GEMM over batch x prompt tokens on whichever device runs it. Without a
    routing model the full union (every expert) is assumed — the conservative bound."""
    n = batch * prompt_tokens
    if rm is None:
        unique = float(EXPERTS)
        resident_mb = 0.0
    else:
        unique = rm.expected_unique_experts(n * TOPK)
        resident_mb = min(hot_set_gb(rm.hot_frac), cache_gb) * MB_PER_GB
    touched_gb = LAYERS * unique * EXPERT_MB / MB_PER_GB
    ssd_mb = max(0.0, touched_gb * MB_PER_GB - resident_mb) + (TRUNK_GB * MB_PER_GB if trunk_streams else 0.0)
    ssd_s = ssd_mb / MB_PER_GB / machine.drive_gbps
    kv = kv_gb(batch, prompt_tokens)
    comp_s, comp_bound = machine.compute.prefill_seconds(n, touched_gb, kv, expert_frac)
    return (ssd_s, "ssd") if ssd_s >= comp_s else (comp_s, comp_bound)


def step_time(machine: Machine, batch: int, step_ssd_mb: float, step_expert_gb: float,
              kv: float, expert_frac: float, trunk_streams: bool) -> tuple[float, str]:
    """One lockstep decode step: SSD fetches, memory/PCIe traversal of every applied
    expert, and FLOPs, all overlapped (Spotlight ideal). Returns (seconds, bound)."""
    ssd_mb = step_ssd_mb + (TRUNK_GB * MB_PER_GB if trunk_streams else 0.0)
    ssd_s = ssd_mb / MB_PER_GB / machine.drive_gbps
    comp_s, comp_bound = machine.compute.step_seconds(batch, step_expert_gb, kv, expert_frac)
    return (ssd_s, "ssd") if ssd_s >= comp_s else (comp_s, comp_bound)


def throughput(run_tokens: int, bytes_mb: float, batch: int, machine: Machine,
               trunk_streams: bool, applied_per_step: float, expert_frac: float = 1.0,
               context_tokens: int = DEFAULT_PROMPT_TOKENS + DEFAULT_DECODE_TOKENS
               ) -> tuple[float, str]:
    """Steady-state decode tok/s and the binding resource."""
    steps = run_tokens / batch if batch else 0
    step_mb = bytes_mb / steps if steps else 0.0
    expert_gb = applied_per_step * EXPERT_MB / MB_PER_GB
    secs, bound = step_time(machine, batch, step_mb, expert_gb, kv_gb(batch, context_tokens),
                            expert_frac, trunk_streams)
    return (batch / secs if secs else 0.0), bound


def tokens_per_night(tok_per_s: float, batch: int, prefill_s: float,
                     decode_tokens: int, spec_mult: float) -> float:
    if tok_per_s <= 0:
        return 0.0
    convoy_s = prefill_s + decode_tokens * batch / tok_per_s
    return night_tokens(1.0) / convoy_s * batch * decode_tokens * spec_mult


# --------------------------------------------------------------------------- #
#  measure: one policy, one machine, one route cube
# --------------------------------------------------------------------------- #
def measure(policy: str, calls: list[Call], cube: dict, machine: Machine,
            tokens: int, *, rm: Optional[RoutingModel] = None,
            prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
            decode_tokens: int = DEFAULT_DECODE_TOKENS, spec_mult: float = 1.0,
            cache_gb: Optional[float] = None, layers: int = LAYERS,
            **policy_kw) -> Metrics:
    """Run one policy over a prebuilt route cube and return its metrics. `cache_gb`
    overrides the derived budget (for controlled experiments); by default the budget
    is derived from the machine, the batch and the job shape."""
    batch = len(calls)
    context = prompt_tokens + decode_tokens
    derived, trunk_streams = machine.budget(batch, context)
    if cache_gb is None:
        cache_gb = derived
    capacity = experts_per_gb(cache_gb)
    routing = _cube_routing(cube)

    if policy == "naive_single":
        per_call_cap = max(1, capacity // batch)
        total_bytes = 0.0
        hits = fetches = 0
        for c in calls:
            cache = ResidencyCache(per_call_cap, policy="lru")
            sched = ExpertMajorScheduler(cache, hot_min=1)
            solo = Call(id=c.id, route=[], domain=c.domain)
            r = Coordinator(cache, sched, layers).run([solo], tokens, routing=routing)
            total_bytes += r.bytes_fetched
            hits += cache.hits
            fetches += cache.fetches
        tok = tokens * batch
        hit_rate = hits / (hits + fetches) if (hits + fetches) else 0.0
        # token-major: no batching; the streams serialise, so the aggregate rate is one
        # token per single-stream step (each step applies TOPK experts per layer).
        tps, bound = throughput(tok, total_bytes, 1, machine, trunk_streams,
                                applied_per_step=TOPK * layers)
        pre1, _ = prefill_seconds(rm, 1, prompt_tokens, cache_gb, machine, trunk_streams)
        return Metrics(policy, tok, total_bytes, total_bytes / tok, hit_rate, 0.0,
                       tps, bound,
                       tokens_per_night(tps, 1, pre1, decode_tokens, spec_mult),
                       cache_gb, trunk_streams, pre1)

    cache, sched = build(policy, capacity, **policy_kw)
    fresh = [Call(id=c.id, route=[], domain=c.domain) for c in calls]
    r = Coordinator(cache, sched, layers).run(fresh, tokens, routing=routing)
    tps, bound = throughput(r.tokens, r.bytes_fetched, batch, machine, trunk_streams,
                            r.applied_experts_per_step, r.expert_flop_fraction,
                            prompt_tokens + decode_tokens)
    pre, _ = prefill_seconds(rm, batch, prompt_tokens, cache_gb, machine, trunk_streams,
                             r.expert_flop_fraction)
    return Metrics(policy, r.tokens, r.bytes_fetched, r.mb_per_token, r.hit_rate,
                   r.skipped_mass_fraction, tps, bound,
                   tokens_per_night(tps, batch, pre, decode_tokens, spec_mult),
                   cache_gb, trunk_streams, pre)


def _calls(batch: int, domains: tuple[str, ...] = ("code",)) -> list[Call]:
    return [Call(id=f"c{i}", route=[], domain=domains[i % len(domains)])
            for i in range(batch)]


def _row(label: str, m: Metrics) -> str:
    return (f"{label:40} {m.policy:12} {m.cache_gb:5.0f}{'*' if m.trunk_streams else ' '} "
            f"{m.mb_per_token:8.1f} {m.bound:>5} {m.hit_rate:5.2f} "
            f"{m.skipped_mass:5.2f} {m.tok_per_s:6.2f} {m.prefill_s / 60:6.1f} {m.tokens_per_night:12,.0f}")


HEADER = (f"{'machine':40} {'policy':12} {'cache':>6} {'MB/tok':>8} {'bound':>5} "
          f"{'hit':>5} {'skip':>5} {'tok/s':>6} {'pre m':>6} {'tok/night':>12}")


# --------------------------------------------------------------------------- #
#  Reports
# --------------------------------------------------------------------------- #
def scenario_table(tokens: int = 12, spec_mult: float = 1.0) -> str:
    out = [f"SCENARIOS — expected bracket, job = {DEFAULT_PROMPT_TOKENS} prompt + "
           f"{DEFAULT_DECODE_TOKENS} decode tokens, spec x{spec_mult}. "
           f"cache = RAM - trunk - OS - KV (derived; * = trunk streams too)",
           HEADER, "-" * len(HEADER)]
    rm = RoutingModel(**BRACKETS["expected"], coherent=True)
    plan = [
        ("32",        1,  ["naive_single"]),
        ("32",        8,  ["union_lru", "skip"]),
        ("32L",       8,  ["union_lru", "skip"]),
        ("128",       32, ["union_lru", "sisyphus", "decay", "skip"]),
        ("128+GPU",   32, ["union_lru", "decay", "skip"]),
        ("128+GPU",   96, ["decay", "skip"]),
        ("192",       32, ["union_lru", "sisyphus", "decay", "skip"]),
        ("192-2x",    32, ["decay", "skip"]),
        ("192+trunk", 32, ["decay", "skip"]),
        ("192+GPU",   32, ["union_lru", "decay", "skip"]),
        ("192+GPU",   128, ["decay", "skip"]),
        ("192+GPU4",  128, ["decay", "skip"]),
        ("192+GPU8",  128, ["decay", "skip"]),
        ("256",       64, ["union_lru", "sisyphus", "decay", "skip"]),
        ("256+GPU",   128, ["decay", "skip"]),
    ]
    for key, B, policies in plan:
        mach = MACHINES[key]
        calls = _calls(B)
        cube = rm.build_cube(calls, tokens)
        for pol in policies:
            m = measure(pol, calls, cube, mach, tokens, rm=rm, spec_mult=spec_mult)
            out.append(_row(f"{mach.label} · b{B}", m))
        tp = best_tape(mach.ram_gb, DEFAULT_PROMPT_TOKENS + DEFAULT_DECODE_TOKENS,
                       compute=mach.compute, seq_gbps=mach.seq_gbps, spec_mult=spec_mult)
        if tp.feasible:
            out.append(f"{mach.label + ' · tape b' + str(tp.batch):40} {'tape':12} "
                       f"{tp.pinned_gb:5.0f}p {tp.step_io_s * mach.seq_gbps * MB_PER_GB / tp.batch:8.1f} "
                       f"{tp.bound:>5} {'  -':>5} {'  -':>5} "
                       f"{tp.tok_per_s:6.2f} {tp.prefill_s_per_convoy / 60:6.1f} {tp.tokens_per_night:12,.0f}")
        else:
            out.append(f"{mach.label + ' · tape':40} {'tape':12}  infeasible: trunk + KV exceed RAM")
        out.append("")
    return "\n".join(out)


def sensitivity_table(tokens: int = 12, machine_key: str = "192", batch: int = 32) -> str:
    mach = MACHINES[machine_key]
    out = [f"SENSITIVITY on {mach.label}, batch {batch} — every bracketed unknown, one at a time",
           f"  {'variation':34} {'policy':9} {'MB/tok':>7} {'hit':>5} {'skip':>5} {'tok/s':>6} {'tok/night':>12}"]

    def line(label, rm, pol, **kw):
        calls = _calls(batch)
        cube = rm.build_cube(calls, tokens)
        m = measure(pol, calls, cube, mach, tokens, rm=rm, **kw)
        out.append(f"  {label:34} {pol:9} {m.mb_per_token:7.0f} {m.hit_rate:5.2f} "
                   f"{m.skipped_mass:5.2f} {m.tok_per_s:6.2f} {m.tokens_per_night:12,.0f}")

    out.append("  -- hot_mass (concentration) --")
    for name in ("weak", "expected", "strong"):
        b = dict(BRACKETS["expected"]); b["hot_mass"] = BRACKETS[name]["hot_mass"]
        for pol in ("decay", "skip"):
            line(f"{name} hot_mass={b['hot_mass']}", RoutingModel(**b), pol)
    out.append("  -- hot_frac (does the hot set fit?) --")
    for hf in (0.05, 0.10, 0.15, 0.20, 0.30):
        b = dict(BRACKETS["expected"]); b["hot_frac"] = hf
        for pol in ("union_lru", "decay", "skip"):
            line(f"hot_frac={hf} (hot set {hot_set_gb(hf):.0f} GB)", RoutingModel(**b), pol)
    out.append("  -- coherence (do same-domain jobs share a hot pool?) --")
    for coh in (True, False):
        for pol in ("decay", "skip"):
            line(f"coherent={coh}", RoutingModel(**BRACKETS["expected"], coherent=coh), pol)
    out.append("  -- cold gate weight (how skippable is the tail?) --")
    for cwr in (0.3, 0.5, 0.8, 1.0):
        b = dict(BRACKETS["expected"]); b["cold_weight_ratio"] = cwr
        line(f"cold_weight_ratio={cwr}", RoutingModel(**b), "skip")
    out.append("  -- skip budget (gate mass droppable per token-layer) --")
    rm = RoutingModel(**BRACKETS["expected"])
    for sb in (0.0, 0.03, 0.06, 0.10, 0.20):
        line(f"skip_budget={sb}", rm, "skip", skip_budget=sb)
    out.append("  -- demand decay --")
    for d in (1.0, 0.95, 0.8, 0.5):
        line(f"decay={d}", rm, "decay", decay=d)
    return "\n".join(out)


def compute_sensitivity_table(tokens: int = 8, batch: int = 32) -> str:
    """The compute model's bracketed constants, one at a time, on the 192 GB machine."""
    rm = RoutingModel(**BRACKETS["expected"])
    calls = _calls(batch)
    cube = rm.build_cube(calls, tokens)
    out = [f"COMPUTE SENSITIVITY — 192 GB, 2 drives, batch {batch}: the constants Phase 0 measures",
           f"  {'variation':44} {'policy':6} {'tok/s':>6} {'bound':>5} {'pre m':>6} {'tok/night':>12}"]

    def line(label, comp, pol):
        mach = Machine("192", 192.0, 10.0, compute=comp)
        m = measure(pol, calls, cube, mach, tokens, rm=rm)
        out.append(f"  {label:44} {pol:6} {m.tok_per_s:6.2f} {m.bound:>5} {m.prefill_s / 60:6.1f} "
                   f"{m.tokens_per_night:12,.0f}")

    out.append("  -- CPU effective TFLOPS (decode kernels) --")
    for tf in (0.5, 1.0, 2.0, 4.0):
        for pol in ("decay", "skip"):
            line(f"cpu {tf} TFLOPS", ComputeModel("cpu", cpu_tflops=tf), pol)
    out.append("  -- CPU prefill GEMM gain over decode --")
    for g in (1.0, 2.0, 4.0):
        line(f"cpu 1 TFLOPS, gemm gain x{g}", ComputeModel("cpu", cpu_gemm_gain=g), "skip")
    out.append("  -- DRAM bandwidth (dual-channel 80 vs quad-channel 200) --")
    for d in (60.0, 80.0, 200.0):
        line(f"cpu 1 TFLOPS, DRAM {d:.0f} GB/s", ComputeModel("cpu", dram_gbps=d), "skip")
    out.append("  -- GPU path: PCIe width and GPU speed --")
    for pcie, gt in ((25.0, 100.0), (50.0, 100.0), (50.0, 40.0), (60.0, 100.0)):
        for pol in ("decay", "skip"):
            line(f"GPU stream, PCIe {pcie:.0f} GB/s, {gt:.0f} TFLOPS",
                 ComputeModel("gpu", trunk_on_gpu=True, experts_on_gpu=True, pcie_gbps=pcie, gpu_tflops=gt), pol)
    return "\n".join(out)


def batch_table(tokens: int = 8) -> str:
    """Batch curve, CPU vs GPU-streaming compute, 192 GB. On the CPU the curve flattens
    once FLOPs bind (~batch 15); on the GPU path it keeps improving until PCIe traffic
    saturates. This is also the deferral curve."""
    from .tape import plan_tape
    cpu, gpu = MACHINES["192"], MACHINES["192+GPU"]
    rm = RoutingModel(**BRACKETS["expected"])
    ctx = DEFAULT_PROMPT_TOKENS + DEFAULT_DECODE_TOKENS
    out = ["BATCH CURVE — 192 GB, 2 drives, skip policy: CPU compute vs GPU streaming (x16)",
           f"  {'batch':>5} {'cache':>6} {'MB/tok':>7} | {'cpu tok/s':>9} {'bound':>5} {'/night':>9} | "
           f"{'gpu tok/s':>9} {'bound':>5} {'/night':>9} | {'tape cpu':>9} {'tape gpu':>9}"]
    for B in (16, 32, 64, 128, 192):
        cache_gb, _ = cpu.budget(B, ctx)
        if cache_gb <= 0:
            out.append(f"  {B:5d}  KV alone exceeds RAM")
            continue
        calls = _calls(B)
        cube = rm.build_cube(calls, tokens)
        mc = measure("skip", calls, cube, cpu, tokens, rm=rm)
        mg = measure("skip", calls, cube, gpu, tokens, rm=rm)
        tc = plan_tape(cpu.ram_gb, B, ctx, compute=cpu.compute, seq_gbps=cpu.seq_gbps)
        tg = plan_tape(gpu.ram_gb, B, ctx, compute=gpu.compute, seq_gbps=gpu.seq_gbps)
        out.append(f"  {B:5d} {cache_gb:6.0f} {mc.mb_per_token:7.0f} | {mc.tok_per_s:9.2f} {mc.bound:>5} "
                   f"{mc.tokens_per_night:9,.0f} | {mg.tok_per_s:9.2f} {mg.bound:>5} {mg.tokens_per_night:9,.0f} | "
                   f"{tc.tokens_per_night:9,.0f} {tg.tokens_per_night:9,.0f}")
    return "\n".join(out)


NOTE = ("NOTE: bytes come from the real ResidencyCache/scheduler, not a formula. Each step's "
        "wall is max(SSD, memory/PCIe traversal, FLOPs) — the Spotlight ideal of perfect overlap "
        "(real engines lose 15-25%). 'bound' names the binding resource. Compute is a model "
        "(geometry.ComputeModel): ~208 GFLOP/token split 92 experts / 116 trunk; CPU 1 TFLOPS "
        "effective, DRAM 80 GB/s, GPU 100 TFLOPS, PCIe x16 50 GB/s — all BRACKETED until "
        "Phase 0 measures them. 'pre m' is prefill minutes per convoy. tokens/night charges one "
        "prefill per convoy. 'skip' is the mean fraction of a token-layer's gate mass dropped "
        "by residency-aware routing — the quality proxy the profiling harness calibrates.")


if __name__ == "__main__":
    print(scenario_table())
    print(sensitivity_table())
    print()
    print(compute_sensitivity_table())
    print()
    print(batch_table())
    print()
    print(NOTE)
