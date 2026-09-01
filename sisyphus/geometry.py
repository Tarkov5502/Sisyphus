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
from dataclasses import dataclass
from typing import Any

# --- Kimi-K3 geometry — MEASURED from the shards on D:\models (2026-09-01) ---
# Header: general.architecture = kimi-k3, 93 blocks, 1 leading dense, expert_count 896,
# expert_used_count 16, expert_shared_count 2, kv_lora_rank 512, rope dim 64 (key_length 576),
# expert_latent_length 3584 (the number V2/V3 misread as an MLA latent — it is the EXPERTS'
# low-rank dimension). head_count_kv is the array [0,0,0,1,...]: only every 4th block is MLA
# attention; the other three are KDA (Kimi Delta Attention), linear attention with a fixed-
# size recurrent state per stream. K3 is a hybrid, and that changes the memory model.
LAYERS = 92               # MoE layers (93 blocks - 1 leading dense)
BLOCKS = 93
MLA_BLOCKS = 24           # blocks with a KV cache (head_count_kv > 0)
KDA_BLOCKS = 69           # blocks with recurrent state instead of KV
EXPERTS = 896
TOPK = 16
SHARED_EXPERTS = 2
EXPERT_LATENT = 3584      # experts' low-rank dimension (was mislabelled LATENT / "MLA latent")
LATENT = EXPERT_LATENT    # kept for backwards compatibility; NOT a KV quantity

# --- Quant sizing — measured from tensor offsets across 16/19 shards (739.3 GB accounted) ---
EXPERT_MB = 9.694         # per routed expert (IQ2_XS gate/up, IQ2_XS/IQ3_XXS down); 799.1 GB total, exact from 19 shards
TRUNK_GB = 62.2           # attention 38.6 (KDA 474 MB/block, MLA 232 MB/block) + shared 12.9 + latent proj 8.2 + embed 2.5
                          # 11.1 + routed latent projections 7.1 + embed/output 2.5, scaled to 93 blocks
MODEL_GB = TRUNK_GB + LAYERS * EXPERTS * EXPERT_MB / 1000.0   # = 861.3 GB, matches the 19 shards on disk

# --- Per-stream memory: KV for MLA blocks, recurrent state for KDA blocks ---
# MLA: 576 fp16 values per token per MLA block — 4x smaller than the DeepSeek-style
# estimate because only 24 of 93 blocks cache KV.
KV_BYTES_PER_TOKEN_BLOCK = 576 * 2
KV_MB_PER_TOKEN = KV_BYTES_PER_TOKEN_BLOCK * MLA_BLOCKS / 1e6            # ≈ 0.028 MB/token
# KDA: state S in R^{head_dim x head_dim} per head, 96 heads (12288 / 128), per stream,
# CONSTANT in context length. Held in bf16 between steps (fp32 doubles it).
KDA_HEADS = 96
KDA_HEAD_DIM = 128
KDA_STATE_BYTES_PER_STREAM = KDA_BLOCKS * KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 2
KDA_STATE_MB_PER_STREAM = KDA_STATE_BYTES_PER_STREAM / 1e6                # ≈ 217 MB per stream
# This fixed per-stream cost, not per-token KV, is what caps batch on a small-RAM machine.
# Per-token update ingredients (k, v, beta, gate) for replayable speculative rollback:
KDA_UPDATE_MB_PER_TOKEN = KDA_BLOCKS * KDA_HEADS * KDA_HEAD_DIM * 2 * 2 / 1e6   # ≈ 3.4 MB

# --- RAM budget ---
# What is actually left for expert residency once the machine is running. The trunk is
# pinned in RAM unless a GPU with >= TRUNK_GB of VRAM holds it (not consumer hardware).
OS_RESERVE_GB = 6.0       # kernel, page tables, the engine itself, headroom (lean Linux: ~2)
LEAN_OS_RESERVE_GB = 2.0
RAM_TIERS_GB = (32.0, 128.0, 192.0, 256.0)   # consumer DDR5 boards: 4 slots x 8/32/48/64 GB

# --- Drive bandwidth envelopes (operator hardware, measured class) ---
DRIVE_SCATTERED_GBPS = 1.5   # random 9.7 MB reads on the SN570 (Gen3) before layout work
DRIVE_LAYOUT_GBPS = 3.2      # co-activation sequential layout on one drive (V2.5)
DRIVE_STRIPED_GBPS = 10.0    # both drives striped, Gen4-weighted, 9.7 MB reads (V2.5)
DRIVE_SEQUENTIAL_GBPS = 12.0 # both drives striped, pure sequential (the tape regime)

# --- Compute envelope (V4: a model, not a constant) ---
# Work per token is fixed by the architecture: ~104B active params x 2 FLOP/param.
# BRACKETED split between routed experts and the trunk (attention, shared experts, dense
# block, embeddings), from the parameter count per expert (~31M at 9.7 MB / ~2.5 bpw).
GFLOP_PER_TOKEN = 208.0
GFLOP_PER_TOKEN_EXPERTS = 92.0      # 16 routed experts x 92 layers x ~31M params x 2
GFLOP_PER_TOKEN_TRUNK = 116.0       # everything else that runs for every token
# What the machine can do with it. All BRACKETED until Phase 0 measures them.
CPU_TFLOPS_EFF = 1.0                # 16-core desktop, quantised kernels, effective (peak is ~2.5)
CPU_GEMM_GAIN = 2.0                 # prefill (many tokens per weight) vs decode efficiency on CPU
DRAM_GBPS = 80.0                    # dual-channel DDR5, achieved
GPU_TFLOPS_EFF = 100.0              # RTX 5090-class, dense fp16/int8, effective
GPU_VRAM_GB = 32.0
PCIE_X16_GBPS = 50.0                # Gen5 x16, achieved (63 theoretical)
PCIE_X8_GBPS = 25.0                 # Gen5 x8 — what is left when NVMe drives share the lanes
# Legacy scalar kept for reference: what V3 assumed. Equivalent to ~1.7 TFLOPS effective at
# batch 32 with the trunk on the CPU. Phase 0 replaces it with a measurement.
COMPUTE_S_PER_TOKEN = 0.12

# --- Workload shape (what a night's job looks like) ---
DEFAULT_PROMPT_TOKENS = 512   # prompt length per job (prefill cost)
DEFAULT_DECODE_TOKENS = 256   # generated tokens per job (decode cost)
NIGHT_HOURS = 12.0

MB_PER_GB = 1000.0        # decimal throughout: files, drives and DIMMs are all quoted decimal;
                          # V4 mixed 1024-based GB with decimal MB and under-counted the model by 2%


def experts_per_gb(gb: float) -> int:
    """How many 9.7 MB experts fit in `gb` gigabytes of cache."""
    return max(0, int(gb * MB_PER_GB / EXPERT_MB))


def hot_set_gb(hot_frac: float) -> float:
    """Bytes of the per-domain hot set across all layers for a given routing skew."""
    return LAYERS * int(EXPERTS * hot_frac) * EXPERT_MB / MB_PER_GB


def kv_gb(batch: int, context_tokens: int, state_bytes: int = 2, state_copies: float = 1.0,
          draft_tokens: int = 0) -> float:
    """Per-stream memory for `batch` concurrent streams each holding `context_tokens`:
    the fixed KDA recurrent state plus MLA KV for the context. On K3 the fixed term
    dominates for any realistic context (217 MB vs 0.028 MB/token).

    `state_bytes`  — 2 for bf16 (default), 1 for fp8 storage (lever 5.5).
    `state_copies` — 2.0 if speculative rollback keeps a full second copy; 1.0 with
                     replayable rollback (lever 5.1), which instead stores `draft_tokens`
                     worth of per-token update vectors.
    """
    state = KDA_STATE_MB_PER_STREAM * (state_bytes / 2.0) * state_copies
    replay = draft_tokens * KDA_UPDATE_MB_PER_TOKEN
    return batch * (state + replay + context_tokens * KV_MB_PER_TOKEN) / MB_PER_GB


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


# --------------------------------------------------------------------------- #
#  Compute model: where the FLOPs run and what the bytes traverse to get there
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ComputeModel:
    """How a step's work is split between CPU and GPU, and which bus it crosses.

    Three placements matter:
      * CPU only — experts and trunk in RAM, computed on the cores. Traversal is DRAM;
        FLOPs are the CPU's. At batch >= ~15 the CPU is FLOP-bound, not DRAM-bound, so
        batching stops helping (V4 finding).
      * trunk on GPU — attention/shared/dense resident in VRAM; experts still on CPU.
        Removes ~56% of CPU FLOPs and 60 GB of DRAM traversal per step.
      * experts streamed through GPU — experts cross PCIe from RAM each step and are
        applied to every token's hidden state in VRAM; nothing expert-shaped is
        resident. PCIe becomes the wall; GPU FLOPs are negligible.
    """
    name: str
    cpu_tflops: float = CPU_TFLOPS_EFF
    dram_gbps: float = DRAM_GBPS
    trunk_on_gpu: bool = False
    experts_on_gpu: bool = False
    pcie_gbps: float = 0.0
    gpu_tflops: float = 0.0
    cpu_gemm_gain: float = CPU_GEMM_GAIN

    def _times(self, tokens: int, expert_gb: float, kv_gb: float, expert_frac: float,
               gemm: bool) -> dict[str, float]:
        """Per-resource seconds for `tokens` tokens of work touching `expert_gb` of
        expert weights. `expert_frac` is the share of routed expert applications actually
        computed (1 - skipped). `gemm` marks prefill (many tokens per weight)."""
        t: dict[str, float] = {}
        trunk_gb = 0.0 if self.trunk_on_gpu else TRUNK_GB
        cpu_gain = self.cpu_gemm_gain if gemm else 1.0
        cpu_expert_gb = 0.0 if self.experts_on_gpu else expert_gb
        t["dram"] = (cpu_expert_gb + kv_gb + trunk_gb) / self.dram_gbps
        cpu_gflop = tokens * ((0.0 if self.experts_on_gpu else GFLOP_PER_TOKEN_EXPERTS * expert_frac)
                              + (0.0 if self.trunk_on_gpu else GFLOP_PER_TOKEN_TRUNK))
        t["cpu"] = cpu_gflop / 1000.0 / (self.cpu_tflops * cpu_gain)
        if self.trunk_on_gpu or self.experts_on_gpu:
            gpu_gflop = tokens * ((GFLOP_PER_TOKEN_EXPERTS * expert_frac if self.experts_on_gpu else 0.0)
                                  + (GFLOP_PER_TOKEN_TRUNK if self.trunk_on_gpu else 0.0))
            t["gpu"] = gpu_gflop / 1000.0 / self.gpu_tflops
            # experts cross PCIe when streamed; activations crossing for a split model are
            # ~7 KB/token/layer and negligible next to 9.7 MB experts.
            t["pcie"] = (expert_gb / self.pcie_gbps) if self.experts_on_gpu else 0.0
        return t

    def step_seconds(self, batch: int, expert_gb: float, kv_gb: float,
                     expert_frac: float = 1.0) -> tuple[float, str]:
        """One lockstep decode step, excluding SSD. Returns (seconds, binding resource)."""
        t = self._times(batch, expert_gb, kv_gb, expert_frac, gemm=False)
        k = max(t, key=t.get)
        return t[k], k

    def prefill_seconds(self, tokens: int, touched_expert_gb: float, kv_gb: float,
                        expert_frac: float = 1.0) -> tuple[float, str]:
        """One convoy's prefill, excluding SSD: every touched expert traverses once and is
        applied to all `tokens` (GEMM)."""
        t = self._times(tokens, touched_expert_gb, kv_gb, expert_frac, gemm=True)
        k = max(t, key=t.get)
        return t[k], k


CPU_ONLY = ComputeModel("cpu")
CPU_FAST_KERNELS = ComputeModel("cpu-2x", cpu_tflops=2.0)
GPU_TRUNK = ComputeModel("gpu-trunk", trunk_on_gpu=True, gpu_tflops=GPU_TFLOPS_EFF,
                         pcie_gbps=PCIE_X16_GBPS)
GPU_STREAM_X16 = ComputeModel("gpu-x16", trunk_on_gpu=True, experts_on_gpu=True,
                              gpu_tflops=GPU_TFLOPS_EFF, pcie_gbps=PCIE_X16_GBPS)
GPU_STREAM_X8 = ComputeModel("gpu-x8", trunk_on_gpu=True, experts_on_gpu=True,
                             gpu_tflops=GPU_TFLOPS_EFF, pcie_gbps=PCIE_X8_GBPS)
