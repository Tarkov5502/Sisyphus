"""
sisyphus.geometry — Kimi-K3 physical constants, read off the real UD-Q2_K_XL download.

These numbers are ground truth from shard-1 metadata (project_sisyphus.md V2.1), not
guesses. Everything downstream (the scheduler, the simulator, the cost model) imports
from here so a single correction propagates everywhere.
"""
from __future__ import annotations

# --- Kimi-K3 geometry (parsed from shard-1 GGUF metadata) ---
LAYERS = 92               # 93 blocks total; one leading dense block is the trunk, 92 MoE layers
EXPERTS = 896             # routed experts per MoE layer
TOPK = 16                 # routed experts activated per token per layer
SHARED_EXPERTS = 2        # always-on experts (part of the resident trunk)
LATENT = 3584             # MLA latent dim — why KV compresses so hard

# --- Quant sizing (derived from the real 861 GB UD-Q2_K_XL file) ---
# (861 GB file - ~60 GB always-resident trunk) / (92 layers * 896 experts) ≈ 9.7 MB/expert.
EXPERT_MB = 9.7
TRUNK_GB = 60.0           # attention + router + 2 shared experts + dense block; pinned in RAM

# --- Drive bandwidth envelopes (operator hardware, measured class) ---
DRIVE_SCATTERED_GBPS = 1.5   # random 9.7 MB reads on the SN570 (Gen3) before layout work
DRIVE_LAYOUT_GBPS = 3.2      # co-activation sequential layout on one drive (V2.5)
DRIVE_STRIPED_GBPS = 10.0    # both drives striped, Gen4-weighted (V2.5)

# --- Compute envelope ---
# Effective per-token core-seconds once the trunk is on the GPU and cores are pinned.
COMPUTE_S_PER_TOKEN = 0.12

MB_PER_GB = 1024.0


def experts_per_gb(gb: float) -> int:
    """How many 9.7 MB experts fit in `gb` gigabytes of cache."""
    return int(gb * MB_PER_GB / EXPERT_MB)


def night_tokens(tok_per_s: float, spec_mult: float = 1.0, hours: float = 12.0) -> float:
    """Aggregate tokens across a night shift, optionally amplified by speculation."""
    return tok_per_s * spec_mult * hours * 3600.0
