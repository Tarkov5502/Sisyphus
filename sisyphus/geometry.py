"""
sisyphus.geometry — Kimi-K3 physical constants, read off the real UD-Q2_K_XL download,
plus the machine envelope (RAM budget, drive bandwidth, compute) every model shares.

These numbers are ground truth from shard-1 metadata (project_sisyphus.md V2.1), not
guesses. Everything downstream (the scheduler, the simulator, the cost model) imports
from here so a single correction propagates everywhere.

Constants marked BRACKETED are the ones the profiling harness (sisyphus.profiling)
replaces with measured values; they are named so the assumption is visible at every
call site rather than buried in a default argument.
"""
from __future__ import annotations

import hashlib
from typing import Any

# --- Kimi-K3 geometry (parsed from shard-1 GGUF metadata) ---
LAYERS = 92               # 93 blocks total; one leading dense block is the trunk, 92 MoE layers
BLOCKS = 93               # attention blocks that hold KV (the dense trunk block has KV too)
EXPERTS = 896             # routed experts per MoE layer
TOPK = 16                 # routed experts activated per token per layer
SHARED_EXPERTS = 2        # always-on experts (part of the resident trunk)
LATENT = 3584             # MLA latent dim — why KV compresses so hard

# --- Quant sizing (derived from the real 861 GB UD-Q2_K_XL file) ---
# (861 GB file - ~60 GB always-resident trunk) / (92 layers * 896 experts) ≈ 9.7 MB/expert.
EXPERT_MB = 9.7
TRUNK_GB = 60.0           # attention + router + 2 shared experts + dense block
MODEL_GB = TRUNK_GB + LAYERS * EXPERTS * EXPERT_MB / 1024.0   # ≈ 841 GB of expert+trunk bytes

# --- KV cache (MLA latent, fp16) ---
# Per token per block the cache holds the compressed latent (LATENT fp16 values). This is
# an estimate off the latent dim; the profiling harness measures it exactly.
KV_BYTES_PER_TOKEN_BLOCK = LATENT * 2
KV_MB_PER_TOKEN = KV_BYTES_PER_TOKEN_BLOCK * BLOCKS / 1e6      # ≈ 0.67 MB per context token

# --- RAM budget ---
# What is actually left for expert residency once the machine is running. The trunk is
# pinned in RAM unless a GPU with >= TRUNK_GB of VRAM holds it (not consumer hardware).
OS_RESERVE_GB = 6.0       # kernel, page tables, the engine itself, headroom
RAM_TIERS_GB = (32.0, 128.0, 192.0, 256.0)   # consumer DDR5 boards: 4 slots x 8/32/48/64 GB

# --- Drive bandwidth envelopes (operator hardware, measured class) ---
DRIVE_SCATTERED_GBPS = 1.5   # random 9.7 MB reads on the SN570 (Gen3) before layout work
DRIVE_LAYOUT_GBPS = 3.2      # co-activation sequential layout on one drive (V2.5)
DRIVE_STRIPED_GBPS = 10.0    # both drives striped, Gen4-weighted, 9.7 MB reads (V2.5)
DRIVE_SEQUENTIAL_GBPS = 12.0 # both drives striped, pure sequential (the tape regime)

# --- Compute envelope ---
# Effective per-token core-seconds for a decode step once cores are pinned. BRACKETED:
# this is a class estimate for ~104B active params at Q2 on a desktop CPU, not measured.
COMPUTE_S_PER_TOKEN = 0.12
# Prefill runs GEMM (many tokens per expert) instead of GEMV, so per-token cost is lower.
# BRACKETED: 4x is a conservative GEMM-vs-GEMV efficiency ratio; profile it.
PREFILL_COMPUTE_S_PER_TOKEN = COMPUTE_S_PER_TOKEN / 4.0

# --- Workload shape (what a night's job looks like) ---
DEFAULT_PROMPT_TOKENS = 512   # prompt length per job (prefill cost)
DEFAULT_DECODE_TOKENS = 256   # generated tokens per job (decode cost)
NIGHT_HOURS = 12.0

MB_PER_GB = 1024.0


def experts_per_gb(gb: float) -> int:
    """How many 9.7 MB experts fit in `gb` gigabytes of cache."""
    return max(0, int(gb * MB_PER_GB / EXPERT_MB))


def hot_set_gb(hot_frac: float) -> float:
    """Bytes of the per-domain hot set across all layers for a given routing skew."""
    return LAYERS * int(EXPERTS * hot_frac) * EXPERT_MB / MB_PER_GB


def kv_gb(batch: int, context_tokens: int) -> float:
    """KV residency for `batch` concurrent streams each holding `context_tokens`."""
    return batch * context_tokens * KV_MB_PER_TOKEN / MB_PER_GB


def cache_budget_gb(ram_gb: float, batch: int, context_tokens: int,
                    trunk_on_gpu: bool = False, os_reserve_gb: float = OS_RESERVE_GB) -> float:
    """RAM left for expert residency after the trunk, the OS and the KV cache.

    This is the number the old simulator hard-coded as 100 GB for a 128 GB machine; the
    trunk alone makes that impossible unless it lives on a GPU. Returns 0.0 when the
    machine cannot even hold trunk + KV (the configuration is infeasible).
    """
    left = ram_gb - os_reserve_gb - kv_gb(batch, context_tokens)
    if not trunk_on_gpu:
        left -= TRUNK_GB
    return max(0.0, left)


def night_tokens(tok_per_s: float, spec_mult: float = 1.0, hours: float = NIGHT_HOURS) -> float:
    """Aggregate tokens across a night shift, optionally amplified by speculation."""
    return tok_per_s * spec_mult * hours * 3600.0


def stable_seed(*parts: Any) -> int:
    """Process-independent 32-bit seed from arbitrary parts.

    Python's built-in hash() of str/tuple is salted per process (PYTHONHASHSEED), which
    is why RESULTS.md and README.md disagreed with each other by a few hundred tokens.
    Everything that seeds an RNG goes through here so a run reproduces byte-for-byte.
    """
    h = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=4)
    return int.from_bytes(h.digest(), "big")
