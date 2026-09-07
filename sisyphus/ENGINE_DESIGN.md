# Sisyphus tape engine — design

*Draft 2, 2026-09-03 (Draft 1: 2026-09-01). The engine that turns the numbers in FINDINGS.md into
tokens. Scope: Kimi-K3 UD-Q2_K_XL (861.3 GB, 19 shards) on the operator's rig — i7-12700KF,
32→64 GB DDR5, RTX 3070 Ti 8 GB on PCIe 4.0 x16, 990 PRO 4 TB (measured 7.07 GB/s) + SN570 1 TB
(3.47 GB/s) (+1 Gen4 drive, assumed 6.5), native Linux (Ubuntu 26.04). Targets: ~430K output
tokens/night at the $480 build and ~500K at $610 on measured levers; ~1.1M/night if fp8 state (§6c)
holds, ~1.5M with low-rank (GPU-capped); ~3M read tokens/night at realistic GPU efficiency on
extraction-shaped jobs (§7b). Revised 2026-09-07 after review 3 (FINDINGS 5.28).*

*Draft 2 adds: deferred commit replacing replay vectors (§6a), tree verification (§6b), low-rank +
fp8 state (§6c), shared-prefix forking (§7b), entropy-coded expert blocks (§8b), the quant-quality
oracle (§13), and folds them into the memory budget, kernel inventory, build order and risks.
Draft 2.1 (2026-09-07) adds the daytime mode (§7c): draft-first streaming, front-loaded buffer,
multiplexed threads; row masking (§5b); fork-parallel answers and the stitcher (§7c.4); the
read/write asymmetry and the distillation loop (§7d, step 14). Draft 2.2 (2026-09-07 evening)
applies adversarial review 3 (FINDINGS 5.28): targets, draft footprint, tree yield, pack cost, fp8
scaling, exactness gates, daytime reads, thread modes, prefix-state caps, and sixteen new risks (§12).*

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

### 5a. Per-job expert count (lever 5.27)

The router's k (16 as trained) is a per-job parameter of the scheduler, defaulting to 16 for
generation and to the largest k the quant oracle certifies as loss-free (target 12–14) for verdict,
classification and review jobs. Expert bytes scale linearly with k; nothing else in the engine
changes (the CSR caller lists simply get shorter). Gate: token agreement and KL vs k=16 on the
acceptance set, measured on llama.cpp before the engine exists.

### 5b. Contextual sparsity inside the expert (lever 5.14, FINDINGS 5.14) — daytime only

The sparse sweep skips whole experts a batch does not route to. Below that there is a second,
finer level: within a selected expert, most FFN rows produce ~0 after the activation for any
given token (Deja Vu 2023, PowerInfer 2024 measured 80–90% of rows idle on dense LLMs; ASSUMED
for K3's 3584-latent experts until measured on the 48B stand-in). The engine exploits it as follows.

- **Predictor (re-sized, review 3).** Rows belong to *experts*, so predictors are per expert: 82,432 of
  them. A low-rank int8 sketch of each expert's gate matrix (3584×32 → ~115 KB) is ~9.5 GB for the
  model — it lives on the SSD beside the expert and is read with it (a 1.2% byte overhead), not in
  VRAM. Training data comes from activation traces on the 48B stand-in and from the night engine's
  own passes, not from the oracle's few hundred tokens. Expect 25–50% skippable rows (SwiGLU), and
  permute neurons by co-activation at repack time so masked reads cluster into ≥ 64 KB runs.
  Trained offline from activation traces of the oracle run; recall target ≥ 95% on the rows that
  carry ≥ 99% of the output energy. Down-projection columns follow the same mask for free.
- **Layout.** GGUF stores an expert as three row-major matrices whose quant blocks interleave
  along the latent dimension, so a "neuron" is not contiguous. The tape re-packs each expert
  **neuron-major** once at build time: for row i, the gate row, up row and down column are
  adjacent (one ~2.7 KB record at IQ2_XS). A masked read is then a set of contiguous runs; the
  reader coalesces runs ≤ 4 KiB apart into one request and pads to 4 KiB (O_DIRECT). Reading 15%
  of an expert costs ~1.9 MB against 9.7 MB dense, in ~25 requests of ~64 KB.
- **Where it pays.** Only at small B. Random 64 KB reads reach ~60–70% of sequential bandwidth on
  the 990 PRO (to be measured by `engine/streamer --rows`, planned), so the effective byte cut is
  ~3–5× on expert bytes at B=1..5. At B ≥ 32 the row unions of the batch cover most of every
  expert and the mask is switched off, exactly as the sparse sweep is switched off when
  speculation fills |E_L|. This is a **daytime-mode lever** (§7c), not a night lever.
- **Correctness.** Mask misses perturb the residual; the acceptance test is the same token-for-token
  oracle comparison as everything else, with a KL budget on the logits (≤ 0.02 nats/token, same as
  the fp8-state gate) and a fallback of "read the whole expert" when predictor confidence is low.
- **Native draft head (CHECK).** Before any of this, read shard 1's tensor table for a `nextn` /
  MTP block (DeepSeek-V3 and Kimi-K2 shipped one). If K3 carries it, it is a free draft model that
  already matches K3's distribution and replaces the external Kimi-Linear-48B draft in §6.

Single-stream stack (CALC, FINDINGS 5.14): 78 GB/token dense-B=1 today → sparse sweep already
included → 5.6 low-bpw tail and 5.10 rANS (−25..35%) → 5b row masking (expert bytes ÷3..5) → 5.1
speculation sharing the trunk read over ~2.5 tokens: **~10–14 GB per token, 1.0–1.4 s at 10 GB/s,
~1–2 verified tok/s** on the present drives. That is the ceiling of software on this box; beyond it
only bandwidth (drives, channels) moves the number.

## 6. Speculation (lever 5.1) — draft, verify, commit

**Draft.** A same-tokenizer model proposes tokens per stream after each sweep. Measured:
Moonlight-16B-A3B-Instruct Q4 gives top-1 acceptance 0.56 on job-shaped prompts (top-3 0.71,
top-8 0.80); a better draft (Kimi Linear 48B-A3B, or a specialist distilled from the night's own
K3 outputs) moves these. Footprint (corrected, FINDINGS 5.28): Moonlight-16B-A3B is ~9–10 GB at
Q4 and does not fit in VRAM beside the tape; Kimi Linear 48B is ~28 GB. A KV-cache draft (Moonlight)
is the right shape for tree proposals — a KDA draft would need its own rollback machinery. The
draft's own per-stream state/KV (×300 streams) is budgeted in RAM. Acceptance must be re-measured
against the *local Q2* K3, not hosted. The draft therefore runs from RAM on the CPU by default, or
is replaced by a native MTP/EAGLE head on K3's hidden states (build step 0/14). Previous text: runs
in VRAM (2–3 GB at Q4, competing with parked states) or on
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
accepted length with per-depth acceptance ≈ (0.71, 0.71·0.66, then 0.56 per extra depth): ≈ 2.72
tokens per sweep for the 33-node tree vs 2.32 for the k=4 chain (an earlier draft said 2.9–3.1;
recomputed 2026-09-07). The depth-1 candidates are scored by the root row's logits, which are
therefore needed as well. KDA's gates are per-channel (diag α), so the ancestor matrix carries a
128-vector of cumulative log-decays per node plus the erase term — a real modification of chunked
KDA, not a mask. Node budget is adaptive: the
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
  `B' = Q_b V_r Σ_r^½`. Per head per stream this is ~5 MFLOP at r=32, m=30 (two thin QRs
  plus the small SVD — an earlier draft said 0.5); across 69 blocks × 96 heads × 1,000 streams ≈
  33 TFLOP per sweep, and batched tiny QR/SVD (cuSOLVER gesvdjBatched) runs at a few percent of
  peak, so this step plausibly *exceeds the sweep* as written. Cheaper truncations first:
  Gram–Schmidt of the m new keys against A with drop-smallest, or exact rank r+km between
  truncations every k sweeps. Telemetry: retrieval error ‖(S − Ŝ)q‖ on the block's actual queries,
  not Frobenius energy (a low-energy direction can be exactly the association the next query
  needs). fp8: per-row (per key channel) scales, not per-head absmax — a fresh rounding every
  step in a channel with decay α accumulates to ~ε/√(1−α²). Test at ≥ 2K generated tokens.
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
- **Exactness (corrected).** GPU MoE inference is not batch-invariant (tile shapes, split-K and
  chunk lengths change reduction order), so a forked stream is numerically *close* to a fully
  prefilled one, not bit-identical. All correctness gates in this document are therefore
  teacher-forced per-position agreement plus a logit-KL budget, with one global KL budget
  allocated across every stacked lever (k-reduction, rows, fp8, low-rank, trunk requant).
- **Payoff.** Prefill FLOPs per job drop by the shared fraction (~4× at 1,500/2,048). Since prefill
  compute is what binds input tokens, unique input per night rises accordingly (the 6.1M/night
  figure counts all prompt tokens; forked prefixes stop counting against the budget). No change to
  sweep bytes or per-stream state.

**7b.1 Persistent, extensible prefix states (FINDINGS 5.25).** A prefix checkpoint (KDA state
217 MB + MLA KV 0.028 MB/token) is written to `states/<name>/` on the 990 PRO and reloaded on
demand; forks reference the KV read-only. Extension is append-only: new tokens (diffs, verdicts,
notes) are prefilled onto the existing state — never recomputed — so a "has read the whole repo"
state costs 45 min once and minutes per night after. Practical size is bounded by three things (review 3): K3's *trained* context length (read from
shard 1 — nothing above it, KDA or not); the KV counted against the stream budget (1M tokens =
28 GB, halving B; plan on 128–256K shared prefixes, 3.5–7 GB); and MLA attention FLOPs per row
over the prefix (~55 GFLOP/row at 1M keys, comparable to the whole model — another reason to cap).
Recall at length is measured (needle tests) before any state above 128K is trusted.
Every night job forks from a named state; verdicts produced are appended to it at dawn.

### 7c. Daytime mode: the interactive experience (FINDINGS 5.12)

The night is a throughput problem; the day is a latency problem, and on this hardware a single K3
stream tops out at ~1-1.5 tok/s after every byte-reduction lever (trunk requantization, expert
working-set cache, top-k reduction, speculation). A person reads at ~5 tok/s. The daytime mode does
not try to make one stream faster than the drives allow; it keeps the reader fed at reading speed
with three mechanisms that stack, all of them the night engine at small batch plus interface work.

**Acceptance test for the mode:** architect-scale questions (<= 500 output tokens, arbitrary
reading) answered in under ~6 minutes on the desk PC (4 with a third drive); the reader never waits
on a blank screen after the first second; verified text arrives at >= 1.4 tok/s on the active
thread (corrected 2026-09-07, FINDINGS 5.26: threads do not add aggregate speed once the trunk is
resident).

**Byte budget behind that number (FINDINGS 5.21).** With 64 GB the requantized trunk (~22 GB)
is resident and computed on the CPU from RAM (0.37 s read floor per pass; 6 GB of it in VRAM),
the SSD serves only experts (14.3 GB dense → ~3 GB after hot cache, row masking and rANS), and
speculation runs at k=2 (the expert union grows ×1.8 per pass; k=4 triples it and gains nothing
for one stream). Single thread ≈ 0.7 s/token ≈ 1.4 tok/s. (An earlier draft claimed five threads
share each pass at ×1.7 the bytes for ≈ 5 tok/s aggregate; withdrawn — with the trunk resident the
unshared expert union dominates, FINDINGS 5.26.) Ordered by size: trunk requant ×2, trunk residency ×1.7, row
masking ×1.6, hot cache ×1.3, speculation ×1.25, rANS ×1.1. The trunk is the daytime lever; the
expert-side levers together are worth less than it.

**7c.1 Draft-first streaming (ghost text).** The draft model used for speculation (a native MTP /
EAGLE-style head on K3, or Moonlight from RAM — the 48B does not fit beside a resident trunk;
15-25 tok/s on this box, faster than reading)
streams its continuation the instant the question arrives, rendered as unverified (grey). K3
verifies it in chunks behind the cursor with the same verification path as §6 (deferred commit,
tree optional), turning text solid where it agrees and rewriting the spots where it disagrees. At
the measured 0.56 acceptance most words stand; at a trained head's ~0.8 nearly all do. The reader
starts at reading speed with zero wait; frontier quality catches up underneath. Implementation:
the verifier already emits accept/reject per draft position; the UI needs a per-token state
(draft / verified / corrected) and a rewrite-in-place renderer. The quality guarantee is unchanged:
the final text is exactly what K3 would have produced alone.

**7c.2 Front-loaded buffer.** For answers the user wants to see only verified, generation runs
ahead and the display starts once the buffer can play out at reading speed without stalling:
head start = output_tokens x (1/gen_rate - 1/read_rate). A 500-token answer at 1.25 tok/s needs a
~300 s head start. The buffer moves the wait to the front; it does not shrink it. It guarantees
continuous reading once started. Combine with 7c.1: ghost text during the head start.

**7c.3 Multiplexed threads (scope corrected 2026-09-07, FINDINGS 5.26).** Several conversations
advance in the same step and share the trunk read. That is a real gain only while the trunk is
*streamed* (32 GB, or an unrequantized trunk): five threads then cost ~1.7x one thread's bytes for
~2.5x the aggregate — of a very low base (0.2 → 0.5 tok/s). With the trunk resident (64 GB) the
shared part is a 0.42 s RAM read and the unshared expert unions dominate: a second thread lowers
aggregate throughput, and it does not recover until the bulk regime (≥ ~50 streams, where the
engine is simply the night scheduler at small batch). Multiplexing is therefore a *convenience*
(several conversations open, each slow) and a bulk mechanism — not a speed lever. Per-thread
speed ≈ 1.4 / n at small n. While the reader consumes one thread's answer, the others mature slowly. Scheduling: the night scheduler at B = 3-8 with per-thread
priority (the thread being read gets the tree-verification budget); prefix forking (§7b) makes
threads that share a codebase context cheap to open. The interface presents threads as parallel
cards with per-thread progress and a "ready to read" queue.

**Economics of the mode.** Bytes per step at B threads ~ 62 GB (trunk, or ~25 after requant) +
union of experts; single-stream floor after all levers ~1-1.5 tok/s per thread on two drives,
~2-3x that on the used-EPYC hot rod (FINDINGS 5.12), where one thread alone reaches reading speed.
The bootstrap loop: the night mode generates K3 hidden states over a few nights, which trains the
EAGLE-style draft head, which raises daytime acceptance and therefore daytime speed.

**7c.4 Fork-parallel answers — REMOVED from the daytime design (FINDINGS 5.26/5.28).** Forks do not
share expert bytes; 8 forks at 0.11 tok/s each take longer than one stream, read ~10× the bytes
and add a stitcher failure mode. Structure comes from "skeleton → sections" on *one* stream. The
mechanism below is kept only as a night-time structural pattern for large documents (where the
sweep is paid anyway). Original text: For structured outputs the scheduler runs one
conversation as several streams: skeleton first (one stream, ~30 tokens, a JSON list of sections
with the shared decisions — names, interfaces, order); fork the KDA state and MLA KV once per
section (5.9's fork primitive; 217 MB each); the forks decode as multiplexed threads (7c.3) and
the UI renders each section's ghost text in its slot as it arrives; stitch in skeleton order.
Sync points every ~100 tokens: each fork prefills its siblings' committed text as a bracketed
context block (prefill is GPU compute, 190 tok/s, no door cost), so cross-references resolve. The
router decides fork count from the skeleton (1 below ~150 expected tokens; up to 8 by day). Four
**Speed claim withdrawn (FINDINGS 5.26):** forks do not share expert bytes, so they do not speed an
answer up on this hardware; the pattern is kept for structure, coherence and to keep K3's output
tokens on judgment. Acceptance:
structure matches a single-stream reference; no duplicated sections; names consistent across forks.

*Stitcher (7c.4b).* The seams between forks are glued by the resident fast model, not by K3: it
receives the skeleton plus every fork's section and is allowed exactly three operations — rename
for consistency with the skeleton's declared names, delete duplicated material, and write
transitions/imports/wiring between sections. It may not change logic or claims; its output is a
diff against the forks' text, and the diff is size-capped (e.g. ≤ 10% of tokens) so a "helpful"
rewrite is rejected mechanically. Then **K3 reads the stitched result** — reading is prefill,
190 tok/s of GPU compute, no door cost — and emits a short verdict (≤ 30 tokens): accept, or name
the seam that is wrong, which re-forks that section only. Cost on a 500-token answer: stitcher
~15–25 s at 20–40 tok/s (overlappable per section as forks finish), K3 read ~3 s + verdict ~15 s.
This is the architect/coder split (7d) applied inside one answer: K3 decides and verifies, the
cheap model types the connective tissue, and K3's expensive output tokens are spent only on
content and judgment. It replaces most sync points (keep one mid-answer sync for long outputs).
Failure mode to test for: the stitcher "fixing" correct code — caught by the diff cap and by
K3's read; measure the rate on the acceptance set.

**7c.5 Thread modes: conversation vs bulk (asked 2026-09-07).** The daytime scheduler owns a
fixed pool of stream slots (default 8 by day at 64 GB) and runs them in one of three modes,
switchable per request and automatically:
- **Conversation.** One stream, all of the drives' bandwidth and the tree-verification budget on
  the active conversation: ~1.4 tok/s verified (2.1 with three drives), ghost text from the first
  second. Other threads pause (states parked in RAM, 217 MB each, not lost). Forking is used only
  when the answer's *structure* wants it (7c.4), not for speed (FINDINGS 5.26).
- **Bulk.** The night scheduler at whatever batch RAM allows (≥ 64 streams by day at 64 GB after
  the resident trunk): each stream slow (~0.03–0.1 tok/s), aggregate 2–4 tok/s rising toward the
  night rate as B grows; maximum jobs finished per hour. No forking, no stitcher; verdict-shaped
  jobs preferred (reads are cheap).
- **Mixed (redefined, review 3).** One extra stream adds ~15 GB of unshared expert reads per pass
  (a 2.7× foreground slowdown), so "foreground within 10–20% and background never idles" cannot
  both hold. Mixed means: bulk runs only in foreground idle gaps (between the user's turns, while
  they read), and is pre-empted the moment a foreground answer starts.
Switching is a scheduler flag, not a restart: forks are just streams, so "consolidate" means
"fork the foreground conversation into the free slots" and "spread" means "stop forking and pull
from the queue". Automatic policy: keyboard/interaction within the last 90 s → Conversation/Mixed; idle > 10 min →
Bulk at small batch (same memory layout). The *night* layout (~60 GB of stream states, trunk
streamed) is a different world: switching to it re-prefills ~270 streams (tens of minutes of GPU),
so the day→night hand-off uses hysteresis of hours and a schedule, and night states are
checkpointed to the 990 PRO (62 GB, ~10 s) so an interruption resumes without re-prefill. The UI shows the mode and the slot allocation.

### 7d. Target workloads: what the machine is for

The engine has two regimes and the workloads must match them; mismatched use is where the
"borderline useless" feeling comes from.

**Night (throughput, B ~ 270+, 60-90 s steps).** Anything read-heavy, independent, and not
waited for: review every open PR or module with a written verdict; generate test scaffolds per
function; security-review a repository file by file; write docs; triage and label issues;
migrate independent files; classify/extract/score documents (the original 5.2 shape). 500K
output tokens ~ 2,000 jobs of 256 tokens, or ~50 medium agentic engineering tasks if K3 does the
typing; ~6M input tokens of reading. Latency per stream is ~90 tokens/hour, so a 300-token answer
takes ~3 h - throughput is aggregate, never per conversation.

**Day (latency, B = 1-8, §7c).** K3 as the architect/reviewer: reads everything (cheap, prefix
forking §7b makes shared context near-free), reasons (free), writes short (<= 500 tokens, minutes).
A fast local coder model (any tokenizer; 7-14B, resident on the GPU by day or on the CPU by
night) does the verbose, interactive tool loop: edit, run, read errors, retry. K3 issues orders;
the coder executes; K3 re-reviews. Cadence: within-day for short orders, overnight for whole-repo
review. The GPU is time-sliced: coder by day, tape by night (8 GB VRAM cannot host both at once).
Quantization risk lands on the cheap, swappable model; K3's judgment over inputs degrades more
gracefully under Q2 than exact code generation would.

*Coder-model candidates (PRICE/landscape, Sep 2026; SWE-bench Verified, mostly vendor-reported,
saturating near the top):* Claude Opus 5 96%, Fable 5 95% (closed); Kimi K3 93.4% (the tape);
no open model matches Opus 5. Fast, fits this rig as-is: **Qwen3-Coder-Next 80B-A3B, 70.6%,
~18 GB at Q4** — 3B active → ~20–40 tok/s CPU+GPU, the default coder. Middle tier that only runs
here in tape/daytime mode with 64 GB: MiniMax M3 (~230B/10B active, 80.5%), DeepSeek V4 Flash
(284B/13B, 79%) — ~6–8 GB of experts per token from SSD → ~2–4 tok/s; GLM-5.2 (744B/40B, strong on
Terminal-Bench) ~2 s/token. The gap K3 → Qwen-Next (23 points) is the reason the architect is K3
and the coder is not asked to make judgment calls.

**The asymmetry that shapes both regimes (FINDINGS 5.23, bounded by 5.28).** *Inside a night sweep*
K3 reading costs only GPU compute (~3M tokens per night at realistic efficiency, shared with
verification; ~6M at spec); K3 writing costs drive bandwidth (~500K per night). By day a read is
a near-dense pass (~80 s per 4K tokens) — reads are queued for the night. So: the coder writes
~0.9–1.3M tokens by day; K3 reads most of it by night in ~700 review jobs of 4K tokens with
128-token verdicts (each carrying 113 MB of KV, so ~190 streams on review nights), alongside
~250K tokens of K3-authored generation. Never have K3 generate what a cheap model can type; always have K3
read what the cheap model typed. Output tokens buy judgment and hard content; input is spent freely.

**Distillation loop (FINDINGS 5.24, corrected 5.28; build step 14).** Two separate paths. *Draft:*
an EAGLE-style head on K3's hidden states (or a Kimi-vocabulary model) trained on the night's
(hidden state, next token) pairs — a 7–8B coder with its own tokenizer cannot be the draft
(rejection sampling needs a shared vocabulary). *Coder:* LoRA on a ≤ 8B dense coder (the 80B-A3B
Qwen cannot be QLoRA'd in 8 GB) from (context, K3 output) traces; at ~300–700 train tok/s on the
3070 Ti and millions of context tokens per night, this is hours per night, run on a schedule, with
a held-out acceptance set to catch forgetting. Track acceptance nightly — it is the free metric of how much K3 has been
absorbed — and expect 0.56 → 0.7+ over weeks (ASSUMED), i.e. ×1.2–1.4 on night throughput and a
draft small enough (~4–5 GB) to sit in VRAM beside the resident trunk by day. Also store the
traces with embeddings for retrieval into the coder's prompt (a K3-authored handbook of the
operator's codebases).

**What neither regime does.** K3 typing a 4,000-token agentic turn while someone watches:
~1 h on this PC at the daytime ceiling, ~10-15 min on the used-EPYC hot rod, ~2 min hosted. The
interactive frontier assistant is not replaced; it is complemented by a night shift and an
architect with a 5-minute turnaround.

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
  full spec on both drives); the page cache is bypassed. Heatsinks on the M.2 drives; ~390 TB (two drives) to ~730 TB (three)
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
2. **Streamer alone** — `engine/streamer.c` (written 2026-09-07, `tools/stream_run.sh` runs it).
   io_uring + O_DIRECT, one ring per drive, pinned 8 MiB slots at QD 32; file mode (whole shards)
   and **sweep mode** (parses the GGUF tensor tables and reads the model in block order, trunk then
   expert slabs, with `--experts N` sampling a batch's routing union so the sparse regime's 3.2 MB
   slab reads are measured, not assumed). Numbers: GB/s per drive and aggregate; seconds per dense
   step and per sparse step (B=1 and B=32 unions). Target: ≥ 9.5 GB/s aggregate on the two drives
   (fio), ≥ 17 GB/s with the third; a B=1 sparse step faster than llama.cpp's 26.6 s token.
   Engineering notes from three adversarial review rounds (39 findings fixed, ASan/UBSan clean on
   hostile synthetic GGUFs): O_DIRECT needs 4 KiB-aligned offsets, lengths and buffers, and
   short-reads at EOF (requests are aligned outward and clamped; mid-file short reads are
   re-issued and counted); ntfs3 has no NOWAIT path so every read is punted to io-wq workers
   (harmless: reads are not inode-serialized; bio construction is ~100 µs); RLIMIT_MEMLOCK is
   8 MiB by default so mlock is best-effort (O_DIRECT pins per DMA anyway); devices must be keyed
   by parent block device (partitions of one NVMe share a ring) and NVMe names swap across boots;
   a serial FNV hash in the I/O thread caps at ~5 GB/s per core (verification is four-lane and
   labelled a lower bound); qd x bs x devices is refused above half of RAM; the GGUF tensor
   table is sorted by offset per shard and sizes are offset deltas (alignment pad < 32 B), so
   per-expert slabs are exact only when pad < n_expert (checked); expert tensors are 3-D with the
   expert index slowest; block 0 of K3 is a dense FFN block with no `_exps` tensors; shared
   experts (`_shexp`), router (`ffn_gate_inp`, `exp_probs_b`) are trunk. Correctness gate for
   the engine: token-for-token agreement with the llama.cpp oracle (" Paris. It is").
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
12. **Daytime mode** (§7c): small-batch scheduler with per-thread priority, draft-first streaming
    UI, front-loaded buffer, thread modes (7c.5). Numbers: verified tok/s on the active thread
    (target ≥ 1.4 on two drives), time-to-first-visible-text (target < 1 s), and the measured
    routing-union growth vs thread count (the number that decides whether 7c.3/7c.4 ever add speed). Needs 7 and the resident draft; gains from 10.
13. **Row masking** (§5b), gated on an activation-trace measurement on the 48B stand-in
    (fraction of rows idle per token) and a random-64 KB read benchmark. Numbers: expert bytes per
    token at B=1, s/token, KL vs the oracle. Target: B=1 ≤ 2 s/token on the present drives.
14. **Distillation loop** (§7d): trace logging from night one; QLoRA pipeline for the 7–8B coder;
    nightly acceptance chart. Numbers: acceptance vs weeks; coder pass rate on the operator's own
    acceptance set vs weeks.
0. (before 1) **GGUF metadata check** for a native MTP/`nextn` head in shard 1 — one command,
    decides whether §6 needs an external draft model at all.

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

*Added 2026-09-07 from review 3 (FINDINGS 5.28), ahead of the original list because several are
cheap to measure before any CUDA is written:* (a) the VRAM budget in §2 is wrong — the active
block's fp32 KDA state is 6.3 MB/stream (1.9 GB at B=300, ×2 double-buffered), dequant must fuse
into the GEMM (a 40-expert chunk in fp16 is 1.6 GB), rANS needs input+output scratch, the CUDA
context is ~0.4 GB: there is no VRAM to lend to states and none for a resident draft; (b) achieved
GPU rate for ragged grouped GEMM + chunked KDA is 15–25 TFLOPS, not 40 — prefill, tree budget and
read capacity all scale with it; (c) reads and verification share one GPU budget; tree tiers
leave ~no prefill; (d) daytime reads are near-dense passes (~80 s per 4K tokens on two drives);
(e) day and night memory layouts are incompatible — switching re-prefills ~270 streams (tens of
minutes): hysteresis of hours and night-state checkpoints to the 990 PRO; (f) daytime RAM/GPU is
over-subscribed (trunk + hot cache + coder 18 GB + draft > 64 GB; coder decode and trunk compute
contend for the 60 GB/s RAM door); (g) bf16 between-step state is untested (ggml keeps F32);
(h) the sparse sweep's router dependency forbids expert prefetch across blocks: 92 × 50–100 ms
of drive idle per sweep unless routing is predicted a block ahead; (i) PCIe co-binds on three
drives once ~65 GB/sweep of state/KV traffic is added, and pinning ~60 GB of host RAM is fragile;
(j) sweep time = max_i(bytes_i/bw_i) under a static split — one throttling drive stalls all
(rebalance nightly from measured rates; the budget Gen4 drive sustains ~5–5.5 GB/s, not 6.5);
(k) nowhere to write the repacked 861 GB image (ext4 300 GB, C: NTFS): grow ext4 or stage on
the new drive; ntfs3 is a dev path only; (l) read-disturb/read-reclaim and thermals (M2_1 sits
under the GPU exhaust), not TBW, are the sustained-read risks; (m) IOMMU map/unmap at 2,300
8-MiB requests/s costs 1–2 cores (`iommu=pt`, huge-page ring); (n) extraction-shaped nights are
compute-bound (150K prefill rows/sweep); (o) K3's trained context length and tokenizer identity
must be read from shard 1 before prefix states above 256K or any draft are built; (p) BaM/P2P
NVMe→GPU is likely infeasible on this chipset — plan three-drive RAM traffic inside the RAM door.


1. io_uring + O_DIRECT across heterogeneous drives sustaining full rate for 12 hours (thermal
   throttling and read-reclaim — 390–730 TB/night of reads; chipset DMI contention once two drives share it). Step 2
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
