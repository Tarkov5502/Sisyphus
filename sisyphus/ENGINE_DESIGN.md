# Sisyphus tape engine — design

*Draft 1, 2026-09-01. The engine that turns the numbers in FINDINGS.md into tokens. Scope:
Kimi-K3 UD-Q2_K_XL (861.3 GB, 19 shards) on the operator's rig — i7-12700KF, 32→64 GB
DDR5, RTX 3070 Ti 8 GB on PCIe 4.0 x16, 990 PRO 4 TB + SN570 1 TB (+1 Gen4 drive), native
Linux. Target: 500K output tokens/night on 256-in/256-out jobs; ~6M input tokens/night on
extraction-shaped jobs.*

---

## 0. One paragraph

The model does not fit anywhere, so the model moves and the work stays still. Every decode
step, the entire 861 GB flows SSD → RAM ring → PCIe → VRAM, expert by expert; each expert
is applied to every stream that routed to it while it is in VRAM, then discarded. What
persists between steps is only per-stream state: 217 MB of KDA recurrent state per stream
in RAM (the bound on batch), tiny MLA KV, and the hidden states. Throughput is
`streams × tokens-per-stream-per-sweep / sweep-seconds`; the levers are RAM for streams,
drives for sweep time, and speculation for tokens per sweep. The GPU is never a storage
tier. It is a pump with FLOPs.

## 1. What a sweep is

One sweep = one forward pass of all B streams through all 93 blocks, in block order, with
weights streamed. Per block:

```
KDA block (69 of them)                      MLA block (24)                        every block
─────────────────────                       ───────────                           ───────────
load attn q/k/v/g, ssm_* (474 MB)  ─┐       load kv_a/q_a/q_b/k_b/v_b/out (232 MB) load router + shared experts (~140 MB)
S_i ← KDA(S_i, x_i)  for all i      │       attn over KV_i for all i              gate: top-16 per stream → expert set E_L
                                    ▼                                             stream 896 experts (8.7 GB): for e in E_L:
                                                                                     y_i += g_ie · W_e(x̃_i)  for i ∈ callers(e)
                                                                                  x ← x + shared(x) + routed(y)
```

The weights for block L+1 are being read from SSD while block L computes. Nothing is
random-access: the file is laid out block-major, expert-major, so a sweep is one sequential
pass over each drive.

## 2. Data flow and where every byte lives

```
 SSD (861 GB, split by bandwidth across drives)
   990 PRO ~7.4 GB/s  ─┐
   new Gen4 ~7.0      ─┼─ io_uring, O_DIRECT, 4-8 MB reads, QD 32/drive ──▶ RAM RING (1.5 GB, pinned)
   SN570   ~3.5       ─┘                                                          │ cudaMemcpyAsync H2D, 22 GB/s
                                                                                  ▼
                                                                          VRAM (8 GB)
                                                                           ├ expert/attn chunk in flight   ~1.0 GB (2 × 0.5 GB double-buffer)
                                                                           ├ hidden states x, y  B×7168×2×fp16 ~ 10 MB @ B=300
                                                                           ├ per-block scratch (KDA S_i for the ACTIVE block only) B×3.1 MB ~ 0.9 GB
                                                                           └ states parked from RAM (lever 5.3)              ~5–6 GB
 RAM (64 GB)
   ├ OS                        ~2 GB
   ├ ring                       1.5 GB
   ├ KDA states  B × 69 × 3.15 MB (bf16)   = B × 217 MB     ← the budget
   ├ MLA KV      B × ctx × 27.6 KB
   ├ draft replay vectors (5.1)  B × k × 3.4 MB
   └ hidden states + logits mirror            small
```

The KDA state for block L is copied RAM→VRAM when block L begins (B × 3.15 MB ≈ 0.9 GB at
B=300, 40 ms over PCIe), updated in fp32 on the GPU, written back bf16 (or fp8, lever 5.5)
when the block ends. The 69 copies per sweep total ~65 GB of PCIe traffic each way, ~6 s of
a 55 s sweep, overlapped with the expert stream.

## 3. Memory budget (64 GB, lean)

| item | size |
|---|---|
| OS + engine | 2.0 GB |
| RAM ring | 1.5 GB |
| VRAM lent to states | −6.0 GB (adds to budget) |
| **available for streams** | **66.5 GB** |
| per stream, plain | 217 MB + ctx × 0.028 MB ≈ 231 MB @ ctx 512 |
| per stream, with k=4 replay vectors | + 13.6 MB ≈ 245 MB |
| **streams** | **~290 plain, ~270 with speculation** (fp8 state: ~500) |

## 4. The step, in order

```
for sweep in night:
  for L in 0..92:
     wait(weights[L] in VRAM)                     # prefetch issued two blocks ahead
     if L is KDA: S[L] RAM→VRAM (async, issued at L-1)
     attention/KDA over all streams                # one kernel launch per op, B rows
     if L is KDA: S[L] VRAM→RAM (async)
     router → E_L, per-expert caller lists (CSR)
     for chunk in experts[L] (≈40 experts / 400 MB per chunk):
        apply chunk: grouped GEMM over callers   # cuBLAS grouped / custom; B×16 rows total
     residual, norms
  logits → sample (or verify drafts, §6) → next tokens
  retire finished streams, admit new ones (their prefill runs inside the next sweep, §7)
```

Kernel count per sweep is small (tens of thousands) and every kernel has B rows of work, so
launch overhead is irrelevant at B ≥ 100.

## 5. Sparse sweep (lever 5.4)

Because the router for block L runs before block L's experts stream, the engine knows E_L
before reading. With B=290 and no speculation, |E_L| ≈ 650–700 of 896 under expected skew:
issue reads only for those (the layout keeps each expert contiguous, so this is 650 × 9.7 MB
sequential runs, still near peak on NVMe). Saves ~25% of sweep time. Disabled automatically
when speculation makes |E_L| → 896.

## 6. Replayable speculation (lever 5.1)

**Draft.** A small model (1–3B, fully resident in VRAM, ~2–4 GB at Q4 — this competes with
lever 5.3's VRAM for states; budget accordingly) proposes k=4 tokens per stream after each
sweep. Cost: 290 streams × 4 tokens of a 3B model ≈ negligible against the sweep.

**Verify.** The next sweep processes k+1 positions per stream (the last accepted token plus
k drafts) as a short sequence: attention/KDA in chunked form, experts applied to B × (k+1)
rows. Expert union → ~all 896 (sparse sweep off). FLOPs: 290 × 5 × 208 GFLOP ≈ 300 TFLOP
≈ 8 s of a 55 s sweep on the 3070 Ti. Standard rejection sampling on the verifier's
distributions accepts a prefix of n ≤ k drafts plus one verifier token.

**State rollback without a copy.** The KDA update is `S ← S·diag(α) + β·k·vᵀ` (gated
delta rule). During verification the block computes S_1..S_{k+1} transiently in VRAM and
*discards* them; it keeps the canonical S_0 in RAM and stores, per stream per drafted
position, the vectors (k, v, β, α) — 3.4 MB per token across 69 blocks. After acceptance
(known only after block 92), a tiny kernel replays the n+1 accepted updates onto S_0 for all
69 blocks: rank-1 updates, no weights, ~B × 5 × 3.4 MB of reads, milliseconds. MLA KV for
rejected positions is simply truncated. Memory overhead ~6%, versus 100% for a second state
copy. This is the mechanism that makes the 510K row possible on 64 GB.

**Acceptance is the open number.** 70% per draft is assumed. It depends on the draft model
and the job domain; measure it first with the draft model against hosted K3 outputs on your
prompts before any engine work depends on it.

## 7. Prefill inside the sweep

A newly admitted stream's prompt (256–2048 tokens) is processed by the *same* sweep that
decodes everyone else: its block-L computation is a chunked-recurrence over its prompt
tokens (KDA) or full attention over them (MLA), then its expert rows join the grouped GEMM.
Prefill therefore costs FLOPs, not sweeps: 290 × 256 prompt tokens ≈ 15 PFLOP ≈ 6 min of GPU
time per convoy, hidden behind SSD time where possible. This is why input tokens are ~40×
cheaper than output tokens here (lever 5.2) — a 2048-token prompt costs the same number of
sweeps as an 8-token one: zero extra.

**Continuous admission.** Streams finish at different times; the engine keeps B full by
admitting new jobs each sweep, so the night is one long sweep sequence, not discrete
convoys. Tokens/night = sweeps/night × Σ tokens per sweep.

## 8. File layout and the drive split

- Re-pack the 19 shards into one **block-major, expert-major** image: for each block, the
  attention/shared/router tensors, then experts 0..895 contiguous. A sweep is then a single
  forward pass over the image.
- **Split the image across drives by bandwidth**, at expert granularity, round-robin
  weighted 7.4 : 7.0 : 3.5 (990 PRO : new Gen4 : SN570). Each drive is read sequentially in
  its own io_uring queue; the ring reassembles order. Aggregate ≈ 17–18 GB/s.
- The KDA/MLA/shared/router tensors (62 GB) are read every sweep too; pin them in RAM only
  if RAM ever exceeds the state budget (it does not at 64 GB).
- Reads are O_DIRECT + io_uring, 4–8 MB, QD ≥ 32 per drive; the page cache is bypassed (it
  would evict itself uselessly every sweep). Heatsinks on the M.2 drives; ~800 TB of reads
  per night.

## 9. Kernels needed (the actual work)

| kernel | exists? | notes |
|---|---|---|
| IQ2_XS / IQ3_XXS dequant → fp16 in VRAM, per chunk | yes (ggml-cuda) | reuse |
| grouped expert GEMM over CSR caller lists | partly (ggml `mul_mat_id`) | needs the streaming variant: weights transient, B×16 rows |
| KDA chunked recurrence (delta rule with gate), fp32 accumulate | yes in FLA / Kimi Linear repo; llama.cpp has kimi-linear support to check | port; must expose (k, v, β, α) for replay |
| MLA attention over per-stream KV in RAM | yes (ggml) | KV H2D per block; or keep MLA KV in VRAM (small: B × ctx × 27.6 KB ≈ 4 GB at B=290, ctx 512 — competes with parked states) |
| replay kernel: apply n rank-1 gated updates to S | trivial | new |
| draft model runner | yes (llama.cpp) | in-process, VRAM-resident |
| io_uring split-drive streamer + ring | new | the core of the engine |
| router + top-k + CSR build | yes | reuse |

## 10. Build order (each step yields a number)

1. **Streamer alone.** Read the image end-to-end across three drives into the ring, discard.
   Number: GB/s sustained over 12 h, and drive temperatures. Target ≥ 17 GB/s.
2. **Sweep without experts.** Trunk only (attention/KDA/shared), B streams, weights streamed.
   Number: KDA state H2D/D2H cost per block, correctness vs llama.cpp on a few streams.
3. **Full sweep, B=32.** Add the expert stream + grouped GEMM. Number: sweep seconds vs the
   model's 55 s; token agreement with llama.cpp on the same prompts (this is the correctness
   gate for the whole engine).
4. **Scale B to the memory bound.** Number: tokens/night plain. Target ≥ 220K at 64 GB.
5. **Sparse sweep.** Number: sweep seconds at B≈290. Target −25%.
6. **Speculation.** Draft model, verification, replay. Numbers: acceptance rate, tokens per
   sweep. Target ≥ 2.5 tokens per sweep → ≥ 500K/night.
7. **fp8 state (5.5)**, gated on the Kimi Linear agreement test. Target: streams ×2.

Steps 1–4 are a few weeks of C++/CUDA for someone who knows ggml; 5 is days; 6 is the hard
part (weeks); 7 is a flag once 6 works. Nothing here needs hardware beyond the $480 build.

## 11. Bases to evaluate before writing from scratch

- **llama.cpp / ggml-cuda**: has the quant kernels, `mul_mat_id`, kimi-linear KDA (verify
  K3's variant), a draft-model speculative path (`llama-speculative`), and a CUDA backend
  that already streams *layers*. Missing: expert-granularity weight streaming with
  transient weights, per-stream state in host RAM, io_uring multi-drive reads. Likely fork
  target.
- **ktransformers**: CPU/GPU MoE split with experts on CPU; not a streaming design, but its
  attention-on-GPU / experts-elsewhere plumbing is relevant.
- **MoE-Infinity**: SSD expert offload for PyTorch; token-major, not tape, but its prefetch
  and expert-cache code is instructive.
- **flash-linear-attention (FLA)**: reference chunked KDA / gated delta-rule kernels.

## 12. Risks, in the order they would bite

1. Draft acceptance below ~55% makes speculation roughly break-even with sparse sweep.
2. io_uring + O_DIRECT across three heterogeneous drives sustaining 17 GB/s for 12 hours
   (thermal throttling, chipset DMI contention).
3. KDA kernel correctness for K3's specific variant (`situ` activation, `attn_res` blocks,
   `kda.gate_lower_bound` — see shard-1 metadata); validate against llama.cpp token-for-token.
4. VRAM pressure at 8 GB once a draft model, parked states, MLA KV and the expert chunk all
   want to live there — the budget in §2 is tight and will need tuning.
5. Q2 quality for the jobs. Measured only against hosted K3 on the real prompts.
