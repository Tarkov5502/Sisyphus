# Sisyphus tape engine — design

*Draft 2, 2026-09-03 (Draft 1: 2026-09-01). The engine that turns the numbers in FINDINGS.md into
tokens. Scope: Kimi-K3 UD-Q2_K_XL (861.3 GB, 19 shards) on the operator's rig — i7-12700KF,
32→64 GB DDR5, RTX 3070 Ti 8 GB on PCIe 4.0 x16, 990 PRO 4 TB (measured 7.07 GB/s) + SN570 1 TB
(3.47 GB/s) (+1 Gen4 drive, assumed 6.5), native Linux (Ubuntu 26.04). Targets: 500K output
tokens/night on 256-in/256-out jobs with measured levers only; ~1.5M/night if the state-compression
levers (§6c) hold; ~6M+ unique input tokens/night on extraction-shaped jobs (§7b).*

*Draft 2 adds: deferred commit replacing replay vectors (§6a), tree verification (§6b), low-rank +
fp8 state (§6c), shared-prefix forking (§7b), entropy-coded expert blocks (§8b), the quant-quality
oracle (§13), and folds them into the memory budget, kernel inventory, build order and risks.*

---

## 0. One paragraph

The model does not fit anywhere, so the model moves and the work stays still. Every decode step,
the entire 861 GB flows SSD → RAM ring → PCIe → VRAM, expert by expert; each expert is applied to
every stream that routed to it while it is in VRAM, then discarded. What persists between steps is
only per-stream state: the KDA recurrent state (217 MB bf16 per stream, the bound on batch; §6c
shrinks it), tiny MLA KV, and the hidden states. Throughput is
`streams × tokens-per-stream-per-sweep / sweep-seconds`; the levers are state compression and RAM
for streams, drives and fewer bytes for sweep time, and speculation for tokens per sweep. The GPU
is never a storage tier. It is a pump with FLOPs, and the FLOPs it has left over are spent buying
back the other two terms (§6b, §8b).

## 1. What a sweep is

One sweep = one forward pass of all B streams through all 93 blocks, in block order, with weights
streamed. Per block:

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
random-access: the file is laid out block-major, expert-major, so a sweep is one sequential pass
over each drive.

## 2. Data flow and where every byte lives

```
 SSD (861 GB, split by measured bandwidth across drives)
   990 PRO 7.07 GB/s (measured)  ─┐
   new Gen4 ~6.5 (assumed)       ─┼─ io_uring, O_DIRECT, 8 MB reads, QD 32/drive ──▶ RAM RING (1.5 GB, pinned)
   SN570   3.47 (measured)       ─┘                                                        │ cudaMemcpyAsync H2D, 22 GB/s
                                                                                           ▼
                                                                                   VRAM (8 GB)
                                                                                    ├ expert/attn chunk in flight   ~1.0 GB (2 × 0.5 GB double-buffer)
                                                                                    ├ rANS decode scratch (§8b)     ~0.1 GB
                                                                                    ├ hidden states x, y  B×rows×7168×2×fp16   ~50 MB @ B=300, 24 rows (§6b)
                                                                                    ├ per-block scratch: expanded KDA S for the ACTIVE block  B×3.1 MB ~0.9 GB @ B=300 (fp32 accumulate)
                                                                                    ├ draft model (§6)                                       ~2–3 GB at Q4 (or on CPU)
                                                                                    └ states parked from RAM (lever 5.3)                     remainder, ~2–3 GB
 RAM (64 GB)
   ├ OS                        ~2 GB
   ├ ring                       1.5 GB
   ├ KDA states  B × 69 × per-block bytes     ← the budget: bf16 full 3.15 MB/block (217 MB/stream);
   │                                             fp8 full 1.57 MB (108 MB); fp8 rank-32 factors 0.79 MB (54 MB); fp8 rank-16 0.39 MB (27 MB)  (§6c)
   ├ MLA KV      B × ctx × 27.6 KB  (fp8: 13.8 KB)
   ├ prefix checkpoints (§7b)  one state + KV per active job template, ~230 MB each, few templates
   └ hidden states + logits mirror            small
```

The KDA state for block L is copied RAM→VRAM when block L begins, expanded (if factored, §6c) and
updated in fp32 on the GPU, then re-truncated/re-quantized and written back when the block ends.
With bf16 full states the 69 round trips per sweep total ~65 GB of PCIe traffic each way at
B=300 (~6 s of a 55 s sweep, overlapped with the expert stream); factored states cut that
proportionally (rank-16 fp8: ~8 GB).

## 3. Memory budget (64 GB, lean)

| item | size |
|---|---|
| OS + engine | 2.0 GB |
| RAM ring | 1.5 GB |
| VRAM lent to states | −2 to −6 GB (depends on draft-model placement) |
| **available for streams** | **~62–66 GB** |
| per stream, bf16 full state | 217 MB + ctx × 0.028 MB ≈ 231 MB @ ctx 512 |
| per stream, speculation with deferred commit (§6a) | + 0 MB (replay vectors eliminated) |
| per stream, fp8 full state | 108 MB + KV ≈ 122 MB |
| per stream, fp8 rank-32 (§6c, ASSUMED) | 54 MB + KV (fp8) ≈ 61 MB |
| per stream, fp8 rank-16 (§6c, ASSUMED) | 27 MB + KV (fp8) ≈ 34 MB |
| **streams** | **~285 bf16 · ~520 fp8 · ~1,050 r32 · ~1,900 r16** (before the drives/PCIe/compute bind) |

Past ~1,000 streams the binding constraint moves: PCIe (861 GB / 22 GB/s = 39 s per sweep) and
verification compute (§6b) both approach the sweep time, and MLA KV becomes a visible share of
RAM. The plan's ceiling on this rig is therefore ~1.5–2M tokens/night, not "more RAM".

## 4. The step, in order

```
for sweep in night:
  for L in 0..92:
     wait(weights[L] in VRAM)                     # prefetch issued two blocks ahead; rANS-decoded on arrival (§8b)
     if L is KDA: factors[L] RAM→VRAM (async, issued at L-1); S = expand(factors)          (§6c)
     attention/KDA over all streams' rows          # rows = [committed n+1 | tree nodes N] per stream (§6a, §6b)
     if L is KDA: S_committed ← S after committed rows only; truncate/quantize; VRAM→RAM (async)
     router → E_L over all rows, per-expert caller lists (CSR)
     for chunk in experts[L] (≈40 experts / 400 MB per chunk):
        apply chunk: grouped GEMM over callers    # cuBLAS grouped / custom; B×rows×16 row-expert pairs
     residual, norms
  logits at tree nodes → multi-candidate rejection sampling down each stream's tree → accepted path (n ≤ depth) + 1 verifier token
  draft model proposes next tree per stream from its new tail
  retire finished streams, admit new ones (prefix fork §7b, then prefill of the unique suffix inside the next sweep §7)
```

Kernel count per sweep is small (tens of thousands) and every kernel has B×rows of work, so
launch overhead is irrelevant at B ≥ 100.

## 5. Sparse sweep (lever 5.4)

Because the router for block L runs before block L's experts stream, the engine knows E_L before
reading. With B=290 and no speculation, |E_L| ≈ 650–700 of 896 under expected skew: issue reads
only for those (the layout keeps each expert contiguous, so this is 650 × 9.7 MB sequential runs,
still near peak on NVMe). Saves ~25% of sweep time. Disabled automatically when speculation makes
|E_L| → 896; at B ≥ 500 with trees it is always off. The two levers are alternatives, not a stack.

## 6. Speculation (lever 5.1) — draft, verify, commit

**Draft.** A same-tokenizer model proposes tokens per stream after each sweep. Measured:
Moonlight-16B-A3B-Instruct Q4 gives top-1 acceptance 0.56 on job-shaped prompts (top-3 0.71,
top-8 0.80); a better draft (Kimi Linear 48B-A3B, or a specialist distilled from the night's own
K3 outputs) moves these. The draft runs in VRAM (2–3 GB at Q4, competing with parked states) or on
the CPU (3B-active is ~40 tok/s on 12 cores; 300 streams × 6 tokens = 45 s — too slow for the CPU
alone at large B, so VRAM is the default and the CPU takes overflow).

**Verify.** The next sweep processes each stream's committed tail plus its draft tokens as a short
chunk (§6a/§6b). Standard rejection sampling on the verifier's distributions accepts a prefix plus
one verifier token. Greedy jobs (temperature 0) use exact match, which is what the acceptance test
measured.

### 6a. Deferred commit (replaces Draft 1's replay vectors)

Draft 1 kept the canonical S_0 in RAM and stored, per drafted position, the update vectors
(k, v, β, α) — 3.4 MB per token per stream — to replay the accepted prefix after block 92. With
trees (§6b) that would be 20+ positions × 3.4 MB ≈ 70 MB per stream, a third of the state. Draft
2 stores nothing:

- During the sweep, block L computes the chunk `[committed c_1..c_{n+1} | tree nodes t_1..t_N]`
  from S_0 with the chunked delta rule. Outputs are needed only at tree nodes (their logits decide
  acceptance); the committed rows are recomputed purely for their state contribution.
- The state written back for block L is `S_1 = S_0 after c_1..c_{n+1}` — the committed tokens
  only. The tree's effect on the state is never materialised outside the block's scratch.
- After block 92 the accepted path becomes next sweep's committed rows. Nothing to replay.

Cost: (n+1) extra rows per stream per sweep (≈ 2.3–3 at measured acceptance) on top of the N tree
rows — about 10% more verification FLOPs, zero bytes of RAM, and the replay kernel disappears.
MLA KV for committed tokens is written at commit time (their K/V are produced by the same rows);
KV for rejected nodes is never written. This is exactly the chunked-recurrence formulation the
prefill path (§7) already needs, so it is one kernel, not two.

### 6b. Tree verification (lever 5.7)

**Why.** Top-1 acceptance is 0.56 but K3's token is in the draft's top-3 71% of the time and
top-8 80%. Verifying several candidates per position converts the top-k rate into the effective
acceptance. The sweep's I/O time is fixed; the extra verification rows cost only GPU FLOPs the
sweep is not using.

**Tree shape.** Static per sweep, chosen from the measured top-k profile: branching factors
`[3, 2, 1, 1, 1, 1]` (3 candidates at depth 1, 2 children each at depth 2, then chains) = 3 + 6 +
6 + 6 + 6 + 6 = 33 nodes, depth 6; or the leaner `[3, 2, 1, 1]` = 21 nodes, depth 4. Expected
accepted length with per-depth acceptance ≈ (0.71, 0.71·0.66, then 0.56 per extra depth): ≈ 2.9–3.1
tokens per sweep for the 33-node tree vs 2.32 for the k=4 chain. Node budget is adaptive: the
scheduler sets total rows R per sweep so that verification FLOPs ≤ 0.8 × (sweep I/O seconds ×
40 TFLOPS); at B=270 that is ~11K rows (≈ 40 per stream); at B=1,000 it is ~11 per stream, so the
tree shrinks toward a chain and streams with high historical acceptance get deeper trees.

**Attention with a tree mask.**
- MLA blocks: standard tree attention — a node attends to the stream's committed KV plus its
  ancestors in the chunk (ancestor mask instead of causal mask). One kernel, mask as a small
  bitset per stream.
- KDA blocks (the reason this is not free elsewhere): the chunked gated delta rule computes chunk
  outputs as `O = Q·S_0-term + (intra-chunk term)`, where the intra-chunk term uses the causal
  matrix of decays and a lower-triangular solve `(I − tril(A))^{-1}`. For a tree, replace
  "predecessor" with "ancestor": the decay from node j to node i is the product of α along the
  path (cumulative log-decay stored per node = parent's + own), and the ancestor matrix is
  strictly lower-triangular in any topological order, so it is nilpotent and the same triangular
  solve applies unchanged. Every node's output equals what a plain chain containing only its
  root-to-node path would produce — which is the correctness condition for speculative
  verification. The committed rows form the chain prefix that every tree node descends from.
- The state written back is the committed-prefix state (§6a); no per-branch state ever exists.

**Acceptance walk.** Multi-candidate rejection sampling (SpecInfer-style): at each depth, sample
from the verifier distribution restricted by the residual rule over the available children; if a
child is accepted descend, else stop and emit the verifier's corrected token. Greedy mode: descend
while the verifier argmax is among the children.

**What it costs.** At B=270, 33-node trees + ~3 committed rows: ~9.7K rows × 208 GFLOP ≈ 2 PFLOP
≈ 50 s on the 3070 Ti — right at the 60 s sweep on the $480 build, so the adaptive budget will
settle around 20–25 nodes there. On the current 32 GB box with its 96 s sweep, the full tree fits
easily. Draft cost: 33 draft tokens per stream per sweep from a 3B-active model in VRAM ≈ 9K
draft tokens/sweep ≈ 10–15 s at batch — acceptable, and it overlaps the tail of the sweep.

### 6c. Compressed KDA state: fp8 (lever 5.5) and low-rank (lever 5.8)

Both are ASSUMED until the Kimi Linear 48B test passes; both share one implementation seam: the
state leaves VRAM through a `pack()` and enters through an `unpack()`, and the recurrence itself
always runs on a full fp32 128×128 per head in block scratch.

**fp8 (E4M3, per-head absmax scale).** `pack`: scale = amax/448, q = round(S/scale), store q
(int8) + scale (fp32) per head; 16 KB + 4 B per head, 108 MB per stream. `unpack`: S = q·scale.
Two trivial kernels; the KV cache path in every major engine does the same thing. Error model:
one rounding per step, damped by the gate's decay; the test (below) measures whether the
accumulated error stays below the model's own noise floor.

**Low-rank factors.** Per head keep `A (128×r)`, `B (128×r)` with `S ≈ A·Bᵀ`, factors in fp8
(per-column scales) or bf16.
- `unpack`: `S = A·Bᵀ` — one small GEMM per head (128×128×r).
- Update inside the block: run the chunk on the dense S as usual; the delta rule with m tokens
  adds at most m to the rank: `S_1 = S_0·D + Σ β_j k_j v_jᵀ` (D = product of decays), i.e.
  exactly `[A | K]·[D·B | βV]ᵀ` — the true post-chunk state has rank ≤ r+m.
- `pack` (re-truncate): thin QR of `[A | K]` → Q_a R_a; thin QR of `[D·B | βV]` → Q_b R_b; SVD of
  the small `(r+m)×(r+m)` core `R_a R_bᵀ`; keep the top r singular triplets; `A' = Q_a U_r Σ_r^½`,
  `B' = Q_b V_r Σ_r^½`. Per head per stream this is O(128·(r+m)²) ≈ 0.5 MFLOP at r=32, m=30;
  across 69 blocks × 96 heads × 1,000 streams ≈ 3.3 TFLOP per sweep — under 0.1 s. Batched small
  QR/SVD kernels exist (cuSOLVER batched, or a hand-written Jacobi for 64×64).
- Telemetry that makes it safe: record per head the discarded energy `Σ_{i>r} σ_i² / Σ σ_i²`
  every pack. If a layer's heads consistently discard more than a threshold (say 1e-3), that layer
  is kept at full rank (fp8 dense) — mixed precision per layer, chosen by measurement, not by
  hand. Expect the earliest KDA layers and any "copy-heavy" heads to demand full rank.
- Why it plausibly works: every step multiplies old directions by the gate decay α ∈ (0,1) before
  adding one new rank-1 term, so the singular spectrum is a decaying sequence of past keys — a
  soft window. The state's *exact* rank after t tokens from zero-init is ≤ t, so for t ≤ r the
  factorization is lossless by construction (prefill of short prompts is exact).

**The test (both levers, one harness).** Kimi Linear 48B-A3B (same KDA layers, fits 64 GB at Q4;
also a candidate draft model, so it is downloaded anyway). Generate 256 tokens greedily on the 50
job prompts with bf16 dense state; repeat with fp8 dense, fp8 r=32, fp8 r=16, and per-layer mixed.
Pass: token agreement > 99% and mean log-prob delta < 1%. Also log the discarded-energy profile
per layer — that profile is what the engine's per-layer rank table is initialised from. An
afternoon per configuration once the RAM is in.

**Payoff (from the tape model, 64 GB, measured speculation).** Drives as-is: bf16 272K → fp8
475K → fp8 r32 756K → fp8 r16 1.07M/night. +1 drive: 428K → 735K → 1.14M → 1.58M. +2 drives:
498K → 849K → 1.31M → 1.79M. With tree verification at 3.0 tokens/sweep on top: ~1.2M / 1.76M /
1.98M. At that point PCIe and verification compute are within 2× of binding.

## 7. Prefill inside the sweep

A newly admitted stream's prompt (256–2048 tokens) is processed by the *same* sweep that decodes
everyone else: its block-L computation is a chunked recurrence over its prompt tokens (KDA) or
full attention over them (MLA), then its expert rows join the grouped GEMM. Prefill therefore
costs FLOPs, not sweeps: 290 × 256 prompt tokens ≈ 15 PFLOP ≈ 6 min of GPU time per convoy,
hidden behind SSD time where possible. This is why input tokens are ~40× cheaper than output
tokens here (lever 5.2) — a 2048-token prompt costs the same number of sweeps as an 8-token one.

**Continuous admission.** Streams finish at different times; the engine keeps B full by admitting
new jobs each sweep, so the night is one long sweep sequence, not discrete convoys. Prefill rows
share the per-sweep row budget with verification rows (§6b); the scheduler admits as many new
streams per sweep as the FLOP budget allows, oldest jobs first.

### 7b. Shared-prefix forking (lever 5.9)

Jobs of one type share their instructions (system prompt, schema, few-shot examples), often
1,000–1,500 of a 2,048-token prompt. The state after the shared prefix is identical for every such
stream, so:

- **Template registry.** A job declares `(template_id, prefix_tokens, suffix_tokens)`; the
  engine enforces that the prefix ends on a token boundary the template controls (a delimiter
  token), so tokenization of the suffix cannot merge into the prefix.
- **Prefix checkpoint.** The first stream with a new template is prefilled normally; at commit,
  its state after the prefix (all 69 KDA blocks, packed as the engine stores states) plus its MLA
  KV over the prefix are snapshotted as a checkpoint: ~217 MB (bf16) or ~30–110 MB (compressed)
  + 27.6 KB × prefix tokens. Checkpoints persist on disk (one file per template per model
  image) and are cached in RAM while the template is active.
- **Fork on admission.** A new stream of that template starts by `memcpy` of the checkpoint into
  its state slot and KV region; prefill covers only the suffix (~500 tokens instead of 2,048).
- **Exactness.** The forward pass is deterministic given the same kernels and reduction order,
  so a forked stream is bit-identical to a fully prefilled one; the checkpoint records the image
  hash and kernel version and is invalidated on either change.
- **Payoff.** Prefill FLOPs per job drop by the shared fraction (~4× at 1,500/2,048). Since prefill
  compute is what binds input tokens, unique input per night rises accordingly (the 6.1M/night
  figure counts all prompt tokens; forked prefixes stop counting against the budget). No change to
  sweep bytes or per-stream state.

## 8. File layout and the drive split

- Re-pack the 19 shards into one **block-major, expert-major** image: for each block, the
  attention/shared/router tensors, then experts 0..895 contiguous. A sweep is then a single
  forward pass over the image. `tools/repack_plan.py` emits the manifest.
- **Split the image across drives by measured bandwidth**, at expert granularity, round-robin
  weighted 7.07 : 6.5 : 3.47 (990 PRO : new Gen4 : SN570) — the 990 PRO holds ~42%, the new
  drive ~38%, the SN570 ~20%. Each drive is read sequentially in its own io_uring queue; the ring
  reassembles order. Aggregate ≈ 17 GB/s (10.55 measured today with two drives, no contention
  between the CPU M.2 slot and the chipset).
- The trunk (62 GB) is read every sweep too. Pinning it in RAM was evaluated and rejected: each GB
  pinned displaces ~4.6 streams to save 0.12% of sweep bytes.
- Reads are O_DIRECT + io_uring, 8 MB, QD 32 per drive (the exact pattern diskspd measured at
  full spec on both drives); the page cache is bypassed. Heatsinks on the M.2 drives; ~700–800 TB
  of reads per night, so the drives are consumables and their SMART read counters are logged.

### 8b. Entropy-coded expert blocks (lever 5.10)

Quantized expert blocks are codebook indices plus scales; the indices are not uniformly
distributed, so a lossless entropy coder shrinks the bytes the drives must deliver, and the GPU
has spare cycles to undo it.

- **Measure before building.** Compress one shard's expert tensors with a prototype rANS coder
  using per-tensor static frequency tables (and, as a floor, `zstd -19` on the raw blocks). If the
  saving is < 4%, drop the lever; the design estimate is 3–8%.
- **Format.** The image stores, per expert, the scale stream raw and the index stream rANS-coded
  in 32 interleaved sub-streams (one per lane of a warp), with the frequency table in the image
  header per tensor type. Expert boundaries stay aligned to 4 KB so the io_uring layout is
  unchanged.
- **Decode.** On H2D arrival, a warp-per-expert rANS decode kernel writes the reconstructed
  quantized block into the double buffer the existing dequant/GEMM kernels already read. nvCOMP's
  ANS and the DFloat11 decoder are the reference designs; throughput per SM is far above the
  ≤ 22 GB/s stream rate, so the decode is hidden behind the next chunk's transfer.
- **Payoff.** Every SSD-bound row improves by the compression ratio (a 6% saving is ~+6%
  tokens/night). Zero quality risk; last in the build order because it is the smallest.

## 9. Kernels needed (the actual work)

| kernel | exists? | notes |
|---|---|---|
| IQ2_XS / IQ3_XXS dequant → fp16 in VRAM, per chunk | yes (ggml-cuda) | reuse |
| grouped expert GEMM over CSR caller lists | partly (ggml `mul_mat_id`) | streaming variant: weights transient, B×rows×16 row-expert pairs |
| KDA chunked gated delta rule, fp32 accumulate, **ancestor mask + per-node cumulative decay** | chain version in FLA / Kimi Linear repo / llama.cpp kimi-k3 | port; add tree mask (§6b); output committed-prefix state (§6a) |
| MLA tree attention over committed KV + ancestor mask | yes (ggml, mask variant) | KV H2D per block; or keep MLA KV in VRAM (B × ctx × 27.6 KB) |
| state `pack`/`unpack`: fp8 quant/dequant; low-rank expand (A·Bᵀ) and re-truncate (batched thin QR + small SVD) | fp8: trivial; low-rank: cuSOLVER batched or hand-written 64×64 Jacobi | new (§6c); telemetry of discarded energy per head |
| multi-candidate rejection sampling down a tree | small | new (§6b) |
| prefix checkpoint snapshot / fork (memcpy + KV splice) | trivial | new (§7b) |
| rANS decode, warp-per-expert, 32 interleaved streams | reference designs exist (nvCOMP ANS, DFloat11) | new (§8b), gated on the offline compression measurement |
| draft model runner (tree proposals) | yes (llama.cpp; tree/EAGLE-style drafting exists) | in-process, VRAM-resident |
| io_uring split-drive streamer + ring | new | the core of the engine |
| router + top-k + CSR build | yes | reuse |

Removed from Draft 1: the replay kernel and the per-token (k, v, β, α) buffers (§6a).

## 10. Build order (each step yields a number)

1. **Correctness oracle.** Stock llama.cpp on the GGUF, mmap from disk, batch 1
   (`tools/first_token.ps1`, `tools/linux_setup.sh`). Numbers: it runs; s/token; the greedy
   tokens that every later step must reproduce.
2. **Streamer alone.** Read the image end-to-end across the drives into the ring with io_uring +
   O_DIRECT, discard. Number: GB/s sustained over 12 h and drive temperatures. Target: ≥ 10.5 GB/s
   on the two drives now, ≥ 17 GB/s with the third.
3. **Sweep without experts.** Trunk only (attention/KDA/shared), B streams, weights streamed.
   Number: state H2D/D2H cost per block; token agreement with step 1 on a few streams.
4. **Full sweep, B=32.** Expert stream + grouped GEMM. Number: sweep seconds vs the model's
   prediction; token agreement with step 1 (the correctness gate for the engine).
5. **Scale B to the memory bound.** Number: tokens/night plain. Target ≥ 90K at 64 GB as-is.
6. **Sparse sweep.** Number: sweep seconds at B≈290. Target −25%.
7. **Speculation with deferred commit** (§6, §6a), chain k=4 first. Numbers: acceptance,
   tokens/sweep (target 2.3, matching the offline measurement), RAM overhead (target 0).
8. **Tree verification** (§6b). Number: tokens/sweep (target ≥ 2.8) and GPU seconds per sweep.
9. **Shared-prefix forking** (§7b). Number: prefill FLOPs per admitted job on templated work.
10. **fp8 state, then low-rank** (§6c), each gated on the Kimi Linear agreement test. Numbers:
    streams, agreement with step 1 on the engine itself.
11. **Entropy-coded blocks** (§8b), gated on the offline compression measurement.

Steps 2–5 are a few weeks of C++/CUDA for someone who knows ggml; 6 is days; 7–8 are the hard
part (weeks); 9 is days; 10 is a week each including the tests; 11 is a week. Nothing needs
hardware beyond the $480 build; the 48B test model needs the 64 GB.

## 11. Bases to evaluate before writing from scratch

- **llama.cpp / ggml-cuda**: quant kernels, `mul_mat_id`, kimi-k3 KDA (merged upstream, b10448+),
  a speculative path, tree-style drafting in newer builds, a CUDA backend that already streams
  *layers*. Missing: expert-granularity weight streaming with transient weights, per-stream state
  in host RAM, io_uring multi-drive reads, tree-masked KDA, packed states. Likely fork target.
- **ktransformers**: CPU/GPU MoE split with experts on CPU; not a streaming design, but its
  attention-on-GPU / experts-elsewhere plumbing is relevant.
- **MoE-Infinity**: SSD expert offload for PyTorch; token-major, not tape, but its prefetch and
  expert-cache code is instructive.
- **flash-linear-attention (FLA)**: reference chunked KDA / gated delta-rule kernels; the tree
  mask is a modification of its intra-chunk term.
- **SpecInfer / Medusa / EAGLE**: tree verification and multi-candidate sampling references.
- **nvCOMP ANS / DFloat11**: GPU entropy decode references.

## 12. Risks, in the order they would bite

1. io_uring + O_DIRECT across heterogeneous drives sustaining full rate for 12 hours (thermal
   throttling — 700+ TB/night of reads; chipset DMI contention once two drives share it). Step 2
   answers this before any model code exists.
2. KDA kernel correctness for K3's specific variant (`situ` activation, `attn_res` blocks,
   `kda.gate_lower_bound` — see shard-1 metadata); validate against llama.cpp token-for-token at
   every step. The tree mask adds a second correctness surface: verify that each node's output
   equals the chain result for its path (unit test with random trees vs. per-path chains).
3. Compressed state (fp8, low-rank) drifting over long generations. Gated on the 48B test; the
   per-layer discarded-energy telemetry gives an in-engine early warning, and any layer can fall
   back to dense fp8 or bf16 individually.
4. VRAM pressure at 8 GB: draft model, parked states, MLA KV, expanded block state, expert
   double-buffer and rANS scratch all want to live there. The budget in §2 is tight; the draft
   model may end up on the CPU at small B and in VRAM at large B.
5. Verification compute becoming the bound at B ≥ 800 (§6b, §6c): the adaptive row budget keeps
   the GPU under the sweep time, at the cost of shallower trees — throughput plateaus rather than
   collapses.
6. Draft acceptance on the real job mix (measured 0.56 top-1 on the sample); a domain with lower
   agreement lowers every speculation row proportionally. Tree verification and a better draft
   are the two answers; both are measurable offline before engine work.
7. Q2 quality for the jobs — measured only against hosted K3 on the real prompts (§13).

## 13. Quant-quality oracle (lever 5.11) — the experiment that picks the tape

The hosted-K3 greedy continuations cached by `tools/acceptance_test.py` (`k3_targets_cache.json`)
are a reference for the *full-precision* model on the operator's own prompts. Any local quant can
be scored against them without an engine, one prefill sweep per prompt:

- For each prompt, teacher-force the local model on `prompt + K3 target` and record, per target
  position, whether the local argmax equals K3's token and the log-prob of K3's token.
- Report agreement and mean log-prob for UD-Q2_K_XL (861 GB) and UD-IQ1_S (594 GB), and any other
  candidate image. At batch 1 via llama.cpp mmap this is ~5–10 minutes per prompt today (one pass
  over the model per prompt, prefill-only), i.e. an overnight job for 38 prompts per quant.
- Decision rule: if IQ1_S agrees with hosted K3 within ~2 points of Q2_K_XL on the job prompts,
  IQ1_S becomes the production image (−31% bytes → 1.45× on every SSD-bound row). If not, the
  same data says which quant the 5.6 tail experiments should target. Q2_K_XL's own agreement
  number is also the honest "quality of the box" figure for anyone asking what the night's
  tokens are worth.
- Storage: IQ1_S needs a second 594 GB download onto the Linux ext4 partition (D: cannot hold
  both). `tools/quant_oracle.py` (to write) drives the run; it is the same teacher-forcing code as
  the acceptance test with the draft replaced by the local K3.
