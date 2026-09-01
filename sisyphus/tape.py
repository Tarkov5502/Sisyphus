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

In the tape regime the binding constraint is no longer expert residency but KV memory:
the batch that fits is `(RAM - trunk - OS - layer buffer) / (context x KV per token)`.
Expert residency and KV compete for the same RAM, and this module makes that trade
explicit (the original simulator had no KV term at all).

Prefill is one sweep per convoy regardless of prompt length: all prompt tokens of the
whole batch are pushed through a layer while its experts stream past, so the prompt is
"free" until its compute exceeds the sweep time.
"""
from __future__ import annotations

from dataclasses import dataclass

from .geometry import (
    COMPUTE_S_PER_TOKEN, DEFAULT_DECODE_TOKENS, DEFAULT_PROMPT_TOKENS,
    DRIVE_SEQUENTIAL_GBPS, EXPERTS, EXPERT_MB, MB_PER_GB, MODEL_GB, OS_RESERVE_GB,
    PREFILL_COMPUTE_S_PER_TOKEN, TRUNK_GB, KV_MB_PER_TOKEN, kv_gb, night_tokens,
)

# One layer's full expert set must be in flight while it streams: the ring buffer.
LAYER_BUFFER_GB = EXPERTS * EXPERT_MB / MB_PER_GB          # ≈ 8.5 GB


@dataclass
class TapePlan:
    ram_gb: float
    batch: int
    context_tokens: int
    pinned_gb: float            # experts held resident (skipped by the sweep)
    kv_gb: float
    feasible: bool
    step_io_s: float            # seconds to sweep the non-resident model once
    step_compute_s: float       # seconds of decode compute for the whole batch
    tok_per_s: float            # decode throughput, I/O overlapped behind compute
    prefill_s_per_convoy: float # one sweep (or its compute) per convoy prefill
    tokens_per_night: float     # decode tokens/night net of prefill, given job shape
    compute_bound: bool

    @property
    def step_s(self) -> float:
        return max(self.step_io_s, self.step_compute_s)


def sweep_seconds(model_gb: float = MODEL_GB, resident_gb: float = 0.0,
                  seq_gbps: float = DRIVE_SEQUENTIAL_GBPS) -> float:
    """Wall seconds to stream everything that is not resident, sequentially."""
    return max(0.0, model_gb - TRUNK_GB - resident_gb) / seq_gbps


def max_batch(ram_gb: float, context_tokens: int, pinned_gb: float = 0.0,
              trunk_on_gpu: bool = False) -> int:
    """Largest lockstep batch whose KV fits beside trunk, OS reserve, buffer and pins."""
    free = ram_gb - OS_RESERVE_GB - LAYER_BUFFER_GB - pinned_gb
    if not trunk_on_gpu:
        free -= TRUNK_GB
    per_stream_gb = context_tokens * KV_MB_PER_TOKEN / MB_PER_GB
    return max(0, int(free / per_stream_gb)) if per_stream_gb > 0 else 0


def plan_tape(ram_gb: float, batch: int, context_tokens: int, *,
              pinned_gb: float = 0.0, trunk_on_gpu: bool = False,
              seq_gbps: float = DRIVE_SEQUENTIAL_GBPS, model_gb: float = MODEL_GB,
              prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
              decode_tokens: int = DEFAULT_DECODE_TOKENS,
              spec_mult: float = 1.0) -> TapePlan:
    """Throughput of the full-sweep policy for one machine and batch shape.

    `pinned_gb` is RAM spent holding the most-demanded experts permanently; the sweep
    skips them. It costs KV headroom, so the right split is a trade this function lets
    you evaluate rather than assume.
    """
    kv = kv_gb(batch, context_tokens)
    need = OS_RESERVE_GB + LAYER_BUFFER_GB + pinned_gb + kv + (0.0 if trunk_on_gpu else TRUNK_GB)
    feasible = need <= ram_gb and batch > 0
    io = sweep_seconds(model_gb, pinned_gb, seq_gbps)
    compute = batch * COMPUTE_S_PER_TOKEN
    step = max(io, compute)
    tps = batch / step if (step and feasible) else 0.0
    prefill = max(io, batch * prompt_tokens * PREFILL_COMPUTE_S_PER_TOKEN)
    # Night accounting: each convoy prefills once, then decodes `decode_tokens` steps.
    convoy_s = prefill + decode_tokens * step
    convoys_per_night = night_tokens(1.0) / convoy_s if convoy_s else 0.0
    per_night = convoys_per_night * batch * decode_tokens * spec_mult if feasible else 0.0
    return TapePlan(
        ram_gb=ram_gb, batch=batch, context_tokens=context_tokens, pinned_gb=pinned_gb,
        kv_gb=kv, feasible=feasible, step_io_s=io, step_compute_s=compute,
        tok_per_s=tps, prefill_s_per_convoy=prefill, tokens_per_night=per_night,
        compute_bound=compute >= io,
    )


def best_tape(ram_gb: float, context_tokens: int, *, trunk_on_gpu: bool = False,
              seq_gbps: float = DRIVE_SEQUENTIAL_GBPS, pin_steps_gb: float = 20.0,
              **kw) -> TapePlan:
    """Search the pin-vs-KV split for the plan with the most tokens per night."""
    best: TapePlan | None = None
    pinned = 0.0
    while True:
        b = max_batch(ram_gb, context_tokens, pinned, trunk_on_gpu)
        if b <= 0:
            break
        p = plan_tape(ram_gb, b, context_tokens, pinned_gb=pinned,
                      trunk_on_gpu=trunk_on_gpu, seq_gbps=seq_gbps, **kw)
        if best is None or p.tokens_per_night > best.tokens_per_night:
            best = p
        pinned += pin_steps_gb
    if best is None:
        return plan_tape(ram_gb, 0, context_tokens, trunk_on_gpu=trunk_on_gpu)
    return best


def tape_table(context_tokens: int = 1024) -> str:
    out = [f"TAPE REGIME — whole-model sequential sweep at {DRIVE_SEQUENTIAL_GBPS:.0f} GB/s, "
           f"no routing assumption, ctx {context_tokens}, "
           f"prompt {DEFAULT_PROMPT_TOKENS} / decode {DEFAULT_DECODE_TOKENS} per job",
           f"{'RAM':>6} {'trunk':>6} {'batch':>6} {'pinned':>7} {'KV GB':>6} {'io s':>6} "
           f"{'cpu s':>6} {'tok/s':>6} {'bound':>5} {'tok/night':>12}"]
    for ram in (128.0, 192.0, 256.0):
        for gpu in (False, True):
            p = best_tape(ram, context_tokens, trunk_on_gpu=gpu)
            if not p.feasible:
                out.append(f"{ram:6.0f} {'GPU' if gpu else 'RAM':>6}  infeasible (trunk + KV exceed RAM)")
                continue
            out.append(f"{ram:6.0f} {'GPU' if gpu else 'RAM':>6} {p.batch:6d} {p.pinned_gb:7.0f} "
                       f"{p.kv_gb:6.0f} {p.step_io_s:6.0f} {p.step_compute_s:6.0f} "
                       f"{p.tok_per_s:6.2f} {'CPU' if p.compute_bound else 'I/O':>5} "
                       f"{p.tokens_per_night:12,.0f}")
    return "\n".join(out)


if __name__ == "__main__":
    print(tape_table())
