"""
sisyphus.tape — the large-batch floor: stream the whole model once per decode step.

At a large enough batch the optimal policy is trivial. Every layer's expert union is
nearly the full 896, so the best you can do is read the file front to back, sequentially,
at the drive's peak bandwidth, applying each expert to every call that wants it as it
streams past — the disk becomes a tape drive. No residency cache, no eviction policy,
and, crucially, NO ROUTING ASSUMPTION: the cost is bytes-of-model / sequential-bandwidth
whatever the skew turns out to be.

That makes the tape regime two things at once:
  * the FLOOR every clever policy must beat, computed from physics alone; and
  * the regime the scheduler should degrade into as batch grows.

In the tape regime the binding constraint is no longer expert residency but per-stream
memory: the batch that fits is `(RAM - trunk - OS - layer buffer) / per-stream bytes`. On
K3 that per-stream cost is dominated by the KDA recurrent state (~217 MB, constant in
context) rather than MLA KV (0.028 MB/token) — measured from the shards, see geometry.py.
Expert residency and stream state compete for the same RAM; this module makes that trade
explicit (the original simulator had no term for either).

Prefill is one sweep per convoy regardless of prompt length: all prompt tokens of the
whole batch are pushed through a layer while its experts stream past, so the prompt is
"free" until its compute exceeds the sweep time.
"""
from __future__ import annotations

from dataclasses import dataclass

from .geometry import (
    CPU_ONLY, GPU_STREAM_X16, ComputeModel, DEFAULT_DECODE_TOKENS, DEFAULT_PROMPT_TOKENS,
    DRIVE_SEQUENTIAL_GBPS, EXPERTS, EXPERT_MB, LEAN_OS_RESERVE_GB, MB_PER_GB,
    MODEL_GB, OS_RESERVE_GB, TOPK, TRUNK_GB, kv_gb, night_tokens,
)

# One layer's full expert set must be in flight while it streams: the ring buffer.
LAYER_BUFFER_GB = EXPERTS * EXPERT_MB / MB_PER_GB          # ≈ 8.5 GB
STREAM_RING_GB = 1.5                                       # what GPU streaming actually needs


@dataclass(frozen=True)
class TapeOptions:
    """The levers against the per-stream state wall (FINDINGS.md §5). Defaults reproduce
    the plain tape regime; each field is one lever.

    lean_ram      5.3: lean OS reserve (2 GB) and a 1.5 GB expert ring instead of 8.5 GB
    vram_state_gb 5.3: idle VRAM used to hold stream states (0 = none)
    state_bytes   5.5: 2 = bf16 state (default), 1 = fp8 state storage (ASSUMED to hold)
    spec_k        5.1: drafted tokens verified per stream per sweep (1 = no speculation)
    accept        5.1: per-draft acceptance probability (ASSUMED 0.7 until measured)
    replay        5.1: True = replayable rollback (store update vectors, ~6% overhead);
                       False = keep a second full state copy (halves batch)
    sparse        5.4: read only the experts some stream routed to this layer-step
    hot_frac / hot_mass: routing skew used to size the sparse sweep (expected bracket)
    """
    lean_ram: bool = False
    vram_state_gb: float = 0.0
    state_bytes: int = 2
    spec_k: int = 1
    accept: float = 0.7
    replay: bool = True
    sparse: bool = False
    hot_frac: float = 0.10
    hot_mass: float = 0.75

    @property
    def tokens_per_stream_step(self) -> float:
        """Expected tokens per stream per sweep: the verifier's own token plus the
        accepted draft prefix, 1 + a + a^2 + ... + a^(k-1) for k-1 drafts beyond the first
        (a draft is kept only if every earlier draft was)."""
        drafts = self.spec_k - 1
        return 1.0 + sum(self.accept ** i for i in range(1, drafts + 1))

    def per_stream_gb(self, context_tokens: int) -> float:
        copies = 1.0 if (self.spec_k == 1 or self.replay) else 2.0
        drafts = self.spec_k if (self.spec_k > 1 and self.replay) else 0
        return kv_gb(1, context_tokens, self.state_bytes, copies, drafts)

    def fixed_ram_gb(self) -> float:
        return (LEAN_OS_RESERVE_GB + STREAM_RING_GB) if self.lean_ram else (OS_RESERVE_GB + LAYER_BUFFER_GB)

    def sweep_fraction(self, batch: int) -> float:
        """Share of expert bytes a sweep must read. 1.0 unless sparse; with sparse, the
        expected fraction of experts routed to by batch x spec_k x TOPK picks per layer."""
        if not self.sparse:
            return 1.0
        n = batch * self.spec_k * TOPK
        hot_n = max(1, int(EXPERTS * self.hot_frac))
        p_hot = self.hot_mass / hot_n + (1 - self.hot_mass) / EXPERTS
        p_cold = (1 - self.hot_mass) / EXPERTS
        unique = hot_n * (1 - (1 - p_hot) ** n) + (EXPERTS - hot_n) * (1 - (1 - p_cold) ** n)
        return unique / EXPERTS


PLAIN = TapeOptions()


@dataclass
class TapePlan:
    ram_gb: float
    batch: int
    context_tokens: int
    pinned_gb: float            # experts held resident (skipped by the sweep)
    kv_gb: float
    feasible: bool
    step_io_s: float            # seconds to sweep the non-resident model once from SSD
    step_compute_s: float       # seconds of memory traversal + FLOPs for the whole batch
    tok_per_s: float            # decode throughput, SSD overlapped behind compute
    prefill_s_per_convoy: float # one sweep (or its compute) per convoy prefill
    tokens_per_night: float     # decode tokens/night net of prefill, given job shape
    bound: str                  # ssd | dram | cpu | gpu | pcie
    input_tokens_per_night: float = 0.0   # prompt tokens processed per night (lever 5.2)
    sweep_fraction: float = 1.0

    @property
    def step_s(self) -> float:
        return max(self.step_io_s, self.step_compute_s)

    @property
    def compute_bound(self) -> bool:
        return self.bound != "ssd"


def sweep_seconds(model_gb: float = MODEL_GB, resident_gb: float = 0.0,
                  seq_gbps: float = DRIVE_SEQUENTIAL_GBPS) -> float:
    """Wall seconds to stream everything that is not resident, sequentially."""
    return max(0.0, model_gb - TRUNK_GB - resident_gb) / seq_gbps


def max_batch(ram_gb: float, context_tokens: int, pinned_gb: float = 0.0,
              trunk_on_gpu: bool = False, opts: TapeOptions = PLAIN) -> int:
    """Largest lockstep batch whose per-stream state fits beside trunk, OS reserve, buffer
    and pins (plus any VRAM lent to states)."""
    free = ram_gb - opts.fixed_ram_gb() - pinned_gb + opts.vram_state_gb
    if not trunk_on_gpu:
        free -= TRUNK_GB
    per_stream_gb = opts.per_stream_gb(context_tokens)
    return max(0, int(free / per_stream_gb)) if per_stream_gb > 0 else 0


def plan_tape(ram_gb: float, batch: int, context_tokens: int, *,
              pinned_gb: float = 0.0, compute: ComputeModel = CPU_ONLY,
              trunk_on_gpu: bool | None = None,
              seq_gbps: float = DRIVE_SEQUENTIAL_GBPS, model_gb: float = MODEL_GB,
              prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
              decode_tokens: int = DEFAULT_DECODE_TOKENS,
              spec_mult: float = 1.0, opts: TapeOptions = PLAIN,
              overlap: float = 1.0) -> TapePlan:
    """Throughput of the full-sweep policy for one machine and batch shape.

    `pinned_gb` is RAM spent holding the most-demanded experts permanently; the sweep
    skips them. `opts` applies the state-wall levers; `overlap` < 1 haircuts every step
    for imperfect I/O-compute overlap (0.85 is the review's working figure).
    """
    if trunk_on_gpu is None:
        trunk_on_gpu = compute.trunk_on_gpu
    kv = batch * opts.per_stream_gb(context_tokens)
    need = opts.fixed_ram_gb() + pinned_gb + max(0.0, kv - opts.vram_state_gb) + (0.0 if trunk_on_gpu else TRUNK_GB)
    feasible = need <= ram_gb and batch > 0
    frac = opts.sweep_fraction(batch)
    all_experts_gb = model_gb - TRUNK_GB
    io = (TRUNK_GB + all_experts_gb * frac - pinned_gb) / seq_gbps / overlap
    # verification of spec_k drafts multiplies the tokens computed per sweep
    comp, comp_bound = compute.step_seconds(batch * opts.spec_k, all_experts_gb * frac, kv)
    comp /= overlap
    step = max(io, comp)
    bound = "ssd" if io >= comp else comp_bound
    tok_per_sweep = batch * opts.tokens_per_stream_step
    tps = tok_per_sweep / step if (step and feasible) else 0.0
    pre_comp, _ = compute.prefill_seconds(batch * prompt_tokens, all_experts_gb, kv)
    prefill = max((TRUNK_GB + all_experts_gb) / seq_gbps, pre_comp) / overlap
    # Night accounting: each convoy prefills once, then decodes until `decode_tokens` are
    # produced per stream (fewer sweeps when speculation accepts drafts).
    sweeps = decode_tokens / opts.tokens_per_stream_step
    convoy_s = prefill + sweeps * step
    convoys_per_night = night_tokens(1.0) / convoy_s if convoy_s else 0.0
    per_night = convoys_per_night * batch * decode_tokens * spec_mult if feasible else 0.0
    inputs = convoys_per_night * batch * prompt_tokens if feasible else 0.0
    return TapePlan(
        ram_gb=ram_gb, batch=batch, context_tokens=context_tokens, pinned_gb=pinned_gb,
        kv_gb=kv, feasible=feasible, step_io_s=io, step_compute_s=comp,
        tok_per_s=tps, prefill_s_per_convoy=prefill, tokens_per_night=per_night,
        bound=bound, input_tokens_per_night=inputs, sweep_fraction=frac,
    )


def best_tape(ram_gb: float, context_tokens: int, *, compute: ComputeModel = CPU_ONLY,
              seq_gbps: float = DRIVE_SEQUENTIAL_GBPS, pin_steps_gb: float = 20.0,
              opts: TapeOptions = PLAIN, **kw) -> TapePlan:
    """Search the pin-vs-state split for the plan with the most tokens per night."""
    best: TapePlan | None = None
    pinned = 0.0
    while True:
        b = max_batch(ram_gb, context_tokens, pinned, compute.trunk_on_gpu, opts)
        if b <= 0:
            break
        p = plan_tape(ram_gb, b, context_tokens, pinned_gb=pinned, compute=compute,
                      seq_gbps=seq_gbps, opts=opts, **kw)
        if best is None or p.tokens_per_night > best.tokens_per_night:
            best = p
        pinned += pin_steps_gb
    if best is None:
        return plan_tape(ram_gb, 0, context_tokens, compute=compute)
    return best


def tape_table(context_tokens: int = 1024) -> str:
    out = [f"TAPE REGIME — whole-model sequential sweep, no routing assumption, ctx {context_tokens}, "
           f"prompt {DEFAULT_PROMPT_TOKENS} / decode {DEFAULT_DECODE_TOKENS} per job",
           f"{'RAM':>6} {'drives':>7} {'compute':>8} {'batch':>6} {'pinned':>7} {'KV GB':>6} {'ssd s':>6} "
           f"{'comp s':>6} {'tok/s':>6} {'bound':>5} {'pre m':>6} {'tok/night':>12}"]
    for ram in (128.0, 192.0, 256.0):
        for seq in (12.0, 40.0):
            for comp in (CPU_ONLY, GPU_STREAM_X16):
                p = best_tape(ram, context_tokens, compute=comp, seq_gbps=seq)
                if not p.feasible:
                    out.append(f"{ram:6.0f} {seq:7.0f} {comp.name:>8}  infeasible (trunk + KV exceed RAM)")
                    continue
                out.append(f"{ram:6.0f} {seq:7.0f} {comp.name:>8} {p.batch:6d} {p.pinned_gb:7.0f} "
                           f"{p.kv_gb:6.0f} {p.step_io_s:6.0f} {p.step_compute_s:6.0f} "
                           f"{p.tok_per_s:6.2f} {p.bound:>5} {p.prefill_s_per_convoy / 60:6.1f} "
                           f"{p.tokens_per_night:12,.0f}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
#  The operator's rig: i7-12700KF, Z690-A, RTX 3070 Ti (Gen4 x16), 990 PRO + SN570
# --------------------------------------------------------------------------- #
RIG_GPU = ComputeModel("3070ti-x16", trunk_on_gpu=True, experts_on_gpu=True,
                       pcie_gbps=22.0, gpu_tflops=40.0)
RIG_DRIVES = {"as-is (990 PRO + SN570)": 10.9, "+1 Gen4 drive, split by bandwidth": 17.9}


def rig_levers_table(prompt: int = 256, decode: int = 256, overlap: float = 0.85) -> str:
    """FINDINGS.md §5 on the actual machine, lever by lever, cumulative."""
    ctx = prompt + decode
    rows = [
        ("plain tape",                                   TapeOptions()),
        ("+ 5.3 lean RAM + 6 GB VRAM for states",       TapeOptions(lean_ram=True, vram_state_gb=6.0)),
        ("+ 5.4 sparse sweep (no speculation)",         TapeOptions(lean_ram=True, vram_state_gb=6.0, sparse=True)),
        ("5.3 + 5.1 replayable speculation k=4 @70%",   TapeOptions(lean_ram=True, vram_state_gb=6.0, spec_k=4)),
        ("   same, naive rollback (2 state copies)",     TapeOptions(lean_ram=True, vram_state_gb=6.0, spec_k=4, replay=False)),
        ("5.3 + 5.1 + 5.5 fp8 state (ASSUMED)",         TapeOptions(lean_ram=True, vram_state_gb=6.0, spec_k=4, state_bytes=1)),
    ]
    out = [f"THE RIG — K3 tape regime, 3070 Ti streaming, {prompt}/{decode} jobs, overlap {overlap:.0%}, "
           f"measured geometry (217 MB KDA state/stream)",
           f"{'RAM':>5} {'drives':38} {'lever stack':44} {'streams':>7} {'s/step':>6} {'tok/s':>6} "
           f"{'out tok/night':>13} {'in tok/night':>12}"]
    for ram in (32.0, 64.0):
        for dl, seq in RIG_DRIVES.items():
            for name, o in rows:
                p = best_tape(ram, ctx, compute=RIG_GPU, seq_gbps=seq, opts=o,
                              prompt_tokens=prompt, decode_tokens=decode, overlap=overlap)
                out.append(f"{ram:5.0f} {dl:38} {name:44} {p.batch:7d} {p.step_s:6.1f} {p.tok_per_s:6.2f} "
                           f"{p.tokens_per_night:13,.0f} {p.input_tokens_per_night:12,.0f}")
            out.append("")
    # the input-token asymmetry (5.2): long-in / short-out jobs on the $480 build
    out.append("5.2 INPUT-TOKEN ASYMMETRY — 64 GB + 1 drive, lean RAM, no speculation; job shape varies")
    out.append(f"  {'prompt/decode':>14} {'streams':>7} {'out tok/night':>13} {'in tok/night':>13} {'hosted-equiv $/night':>20}")
    for pr, de in ((256, 256), (1024, 32), (2048, 8), (4096, 4)):
        p = best_tape(64.0, pr + de, compute=RIG_GPU, seq_gbps=17.9,
                      opts=TapeOptions(lean_ram=True, vram_state_gb=6.0),
                      prompt_tokens=pr, decode_tokens=de, overlap=overlap)
        usd = p.tokens_per_night * 15 / 1e6 + p.input_tokens_per_night * 3 / 1e6
        out.append(f"  {pr:>7}/{de:<6} {p.batch:7d} {p.tokens_per_night:13,.0f} {p.input_tokens_per_night:13,.0f} {usd:20.2f}")
    return "\n".join(out)


if __name__ == "__main__":
    print(tape_table())
    print()
    print(rig_levers_table())
