# Sisyphus — findings ledger

*Consolidated record of what has been measured, what has been derived, and what remains
conditional, as of 2026-09-01. Later entries supersede earlier ones. Each finding names its
source: SHARDS (read off the GGUF on D:\), RIG (the machine's own reporting), SIM (the
scheduler simulator in this repo), CALC (arithmetic from measured constants), PRICE (a
dated web source), or ASSUMED (not yet measured).*

---

## 1. The model, measured (SHARDS)

| quantity | value | note |
|---|---|---|
| architecture | `kimi-k3` | 93 blocks: 1 leading dense, 92 MoE |
| attention | **24 MLA blocks + 69 KDA blocks** | `head_count_kv = [0,0,0,1,...]`; hybrid linear attention |
| experts / active / shared | 896 / 16 / 2 | |
| expert latent dim | 3584 | `expert_latent_length`; V2/V3 misread this as an MLA latent |
| MLA KV per token | 576 fp16 values per MLA block → **0.028 MB/token** | `kv_lora_rank 512 + rope 64` |
| **KDA state per stream** | 69 × 96 heads × 128 × 128 × 2 B → **217 MB (bf16)** | constant in context; the binding memory bound |
| expert size | **9.68 MB** (IQ2_XS gate/up, IQ2_XS or IQ3_XXS down) | from tensor offsets |
| trunk | **62 GB** | KDA block 474 MB, MLA block 232 MB, shared experts 11 GB, latent proj 7 GB, embed 2.5 GB |
| model total | **860 GB**, 19 shards | 16 present (739 GB) on D: (the SN570 1 TB, not the 990 PRO); shards 16/18/19 missing; 245 GB of stale partials deleted 2026-09-01, 238 GB free |
| work per token | ~208 GFLOP (CALC: 104B active × 2) | ~92 experts, ~116 trunk (of which KDA attention is large) |

## 2. The rig (RIG)

i7-12700KF (8P+4E, no AVX-512) · MSI PRO Z690-A WIFI (4 DIMM, 2 free; M.2: M2_1 CPU Gen4,
M2_2 chipset Gen4, M2_3 chipset Gen3, M2_4 chipset Gen4; chipset uplink ~13 GB/s achievable)
· 2×16 GB DDR5-4800 · RTX 3070 Ti 8 GB on PCIe 4.0 x16 (~22 GB/s, ~40 TFLOPS effective)
· Samsung 990 PRO 4 TB (Gen4 ~7.4 GB/s) + WD SN570 1 TB (Gen3 ~3.5 GB/s) · Windows (dev);
production must be native Linux.

## 3. The bounds (CALC)

For a target of T tokens/night with lockstep batch B and sweep time s:

- **SSD:** tape regime reads the whole model per step: `s ≥ 860 GB / bandwidth`. As-is ~11
  GB/s → 78 s; +1 Gen4 drive with the model split by bandwidth ~18 GB/s → 48 s; ×1/0.85 for
  realistic overlap.
- **Per-stream memory:** `B ≤ (RAM − OS − buffers) / (217 MB + ctx × 0.028 MB)`. 32 GB →
  ~77; 64 GB → ~219; 128 GB → ~500; 192 GB → ~760. **This is the wall.**
- **Compute:** GPU FLOPs ≈ 40 TFLOPS → 190 tokens/s ceiling for *any* token; irrelevant for
  output tokens (SSD/state bind first), binding for input tokens (§6).
- **PCIe:** 860 GB / 22 GB/s = 39 s < SSD time; not binding on this rig.

Tokens/night ≈ 43,200 × B × (tokens per stream per sweep) / s, minus prefill.

## 4. K3 on this rig — honest numbers (CALC, 85% overlap, 256/256 jobs)

| config | cost | streams | s/step | tokens/night |
|---|---|---|---|---|
| 32 GB, drives as-is | $0 | 77 | 91 | 36K |
| 32 GB + 1 Gen4 drive | $130 | 77 | 55 | 60K |
| 64 GB, drives as-is | $350 | 219 | 91 | 103K |
| **64 GB + 1 drive** | **~$480** | 219 | 55 | **167K** |
| 128 GB + 1 drive | ~$1,530 | 502 | 55 | 372K |
| 192 GB + 1 drive | ~$2,930 | 760 | 55 | ~560K |

Within $500 the plain K3 ceiling is ~170K/night. 500K plain needs ~145 GB free for state.

## 5. The levers against the state wall (CALC unless noted)

**5.1 Replayable speculative decoding.** In tape mode the sweep already touches every
expert, so verifying k drafted tokens per stream per sweep multiplies tokens per sweep by
`1 + a + a² + … + a^(k−1)` (2.77 at a=0.7, k=4) at no SSD cost; GPU has the FLOPs (219 × 4 × 208 GFLOP ≈ 5 s per
sweep). Rollback of the KDA state on rejection does NOT need a second state copy: the update
is a gated rank-1 rule `S ← S·decay + β·k·vᵀ`, so store the canonical S plus the per-token
(k, v, β, gate) for the k drafts (~3.4 MB/token/stream) and replay the accepted prefix after
acceptance. Overhead ~6% memory. **64 GB + 1 drive, k=4, 70% accept → ~510K/night** (with 5.3).
Assumed: 70% acceptance with a 1–3B draft model (measure).

**5.2 Workload shape: input tokens are ~40× cheaper than output tokens.** One sweep
processes all prompt tokens of all streams; only the GPU's FLOPs bind. ~150–190 input
tokens/s → **6–8 M input tokens/night** vs ~170K output tokens. Long-in / short-out jobs
(classify, extract, score, audit-and-verdict) get ~40× the hosted-equivalent value from the
same box. Needs nothing invented.

**5.3 RAM reclaim.** GPU streaming needs a ~1–2 GB expert ring in RAM, not an 8.5 GB layer
buffer; lean Linux idles at ~2 GB, not 6; ~6 GB of idle VRAM can hold ~27 stream states.
**+55–60 streams at 64 GB (~+25%), $0.**

**5.4 Sparse sweep.** The router for layer L runs before layer L's experts stream, so the
needed set is known before reading. At ~219 streams ~650/896 experts are touched per layer
under expected skew → read ~27% fewer bytes → ~1.3× on sweep time. Free. Mutually exclusive
with 5.1 at k≥4 (the union becomes everything).

**5.5 fp8 KDA state.** Store the per-stream state in fp8 (E4M3, per-head absmax scale)
between steps; compute in fp32/bf16 on the GPU. 217 → 108 MB/stream → **~2× streams**.
Risk: rounding injected every step; the gated decay should damp it (analogous to fp8 KV
cache, which is standard), but it is ASSUMED until measured. Test path: Kimi Linear 48B-A3B
(same KDA layers, fits 64 GB at Q4) — bf16 vs fp8 state over 256-token generations, token
agreement >99% and perplexity <1% delta to pass.

**5.6 Lower-precision tail experts.** Re-quantize cold experts IQ2_XS → IQ1_S/M: sweep
bytes ~−30%. Quality cost unmeasured; needs the routing profile to choose hot vs cold.

**Stack, modelled (`sisyphus.tape.rig_levers_table`, 64 GB + 1 drive, 256/256 jobs, 85%
overlap; acceptance per sweep is geometric, 1 + a + a² + a³ = 2.77 at a=0.7, k=4):**

| lever stack | streams | tok/s | out tokens/night |
|---|---|---|---|
| plain tape | 219 | 4.0 | 167K |
| + 5.3 lean RAM + 6 GB VRAM for states | 294 | 5.3 | 222K |
| + 5.4 sparse sweep (no speculation) | 294 | 6.9 | 284K |
| 5.3 + 5.1 replayable speculation (k=4, 70%) | 278 | 12.7 | **510K** |
| same with naive rollback (2 state copies) | 151 | 6.9 | 287K |
| 5.3 + 5.1 + 5.5 fp8 state (ASSUMED) | 499 | 22.9 | 866K |

At 32 GB (no RAM purchase) the same stack reaches 332K with speculation, 576K with fp8.
Input tokens (5.2), same build, 2048/8 jobs: ~6.2 M/night (~$19/night hosted-equivalent).
Regenerate with `python -m sisyphus.results`; the tables live in RESULTS.md.

## 6. Earlier findings that still stand (SIM)

- 84% of scheduler-regime bytes are single-use cold experts; only skip/substitute/tape
  touch them.
- Demand decay fixes the domain-shift squat (−11% bytes on mixed nights).
- Residency-aware skip halves bytes *only if* cold picks are low-gate
  (`cold_weight_ratio ≤ ~1.0`); unverified; the profile's one column decides it.
- On a CPU past batch ~15 the wall is FLOPs, not bytes (~0.2 s/token on this class of
  chip); GPU streaming moves it to PCIe/SSD and makes prefill ~free.
- The scheduler regime is irrelevant on this rig at ≤64 GB: there is no expert cache worth
  having. K3 here is tape + GPU streaming.
- Ordinary speculative decoding in the *scheduler* regime is ~worthless (union grows); in
  the *tape* regime it is the main lever (5.1).

## 7. Corrections log (things we got wrong, then fixed)

100 GB cache on a 128 GB box · process-salted RNG seeds · aging/deferral could not change a
byte · "2.1× at 128 GB" was at 192 GB · "pennies per million" hosted (it is $3/$15) · "$30
profiling" · compute as a scalar · KV from `LATENT=3584` (an expert dimension) · no
per-stream state term at all · `max_batch` ignoring that state · RAM ranked below drives
(in tape mode RAM is the multiplier) · assuming the download was complete.

## 8. Prices used (PRICE, Sep 2026)

Hosted K3 $3 / $15 per M in/out (OpenRouter) · DDR5 16 GB ~$175/stick, 48 GB ~$700 ·
Gen4 NVMe 1 TB ~$130, Gen5 2 TB ~$400 · RTX 3090 used ~$1,300, 4090 used ~$2,500,
5090 ~$5,000 · B200 8-GPU node $48–57/hr · large-RAM CPU instance ~$8–15/hr.

## 9. Order of operations

0. ~~Delete stale partials~~ done (245 GB freed). Finish shards 16, 18, 19:
   `tools\finish_download.ps1`. Then move/split the model onto the 990 PRO (C:) — today it
   sits entirely on the Gen3 SN570. **$0.**
1. Buy 2×16 GB DDR5 (~$350) → 64 GB. Buy one Gen4 1 TB NVMe (~$130) for M2_2; split the
   model across the three drives by bandwidth. **~$480.**
2. Test 5.5 on Kimi Linear 48B locally (fp8 vs bf16 state agreement). **$0.**
3. Engine: tape-mode GPU streaming (expert chunks SSD→RAM ring→VRAM, all-stream hidden
   states in VRAM, KDA states in RAM/VRAM, MLA KV in RAM), then 5.3, then 5.4 or 5.1.
   Months, not dollars. Evaluate ktransformers / MoE-Infinity as bases first.
4. Route extraction-shaped jobs to the box first (5.2): highest value per sweep, no
   speculation needed.
5. Profile routes (skip's `cold_weight_ratio`) only when the scheduler regime becomes
   relevant, i.e. at ≥128 GB RAM.
