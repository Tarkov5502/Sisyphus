# Adversarial review of Sisyphus — is any of this real? (V3 review, V4 recalculation)

*September 2026. Written to break the plan, not to defend it. Every attack below was either
run against the simulator or checked against a primary source. Verdicts are: **REAL**
(holds under attack), **CONDITIONAL** (holds only if a named unknown comes out a named way),
**WRONG** (an error in our own work, now corrected), or **UNKNOWN** (cannot be settled
without the model or the hardware).*

---

## 0. Scorecard

| claim | verdict | what would kill it |
|---|---|---|
| Expert-major batching + refcounted residency cache reduce bytes vs token-major | REAL | nothing; it is arithmetic on the union |
| 84% of bytes are the single-use cold tail | REAL under the model | only if routing is *more* concentrated than modeled (then the tail is smaller and the win is smaller) |
| Demand eviction beats LRU when the hot set fits | REAL | hot set > cache (e.g. `hot_frac` ≥ 0.15 at 192 GB) |
| Demand decay fixes the domain-shift squat | REAL | nothing; −11% bytes on mixed nights, neutral on single-domain |
| Residency-aware skip halves bytes | **CONDITIONAL** | cold picks are *not* low-gate. At `cold_weight_ratio` ≥ 1.5 the win is gone entirely (§2.1) |
| Skip costs only ~7% quality | **UNKNOWN** | error compounds over 92 layers; only an eval on the real model can price it |
| Compute ceiling 8.3 tok/s (0.12 s/token) | **WRONG as a constant** — V4 replaced it with a FLOP/traversal model; on a CPU the floor is ~0.2 s/token and prefill is 27 min/convoy (§2.3, §10) | CPU effective TFLOPS is still unmeasured; Phase 0 |
| Tape regime ≈ 100K/night at 192 GB | CONDITIONAL on the KV constant, which may be 6× too pessimistic (§2.6) |
| 128 GB RAM unlocks the policy | **WRONG** (V2 claim) — trunk leaves ~46 GB; the tier is 192 GB | — |
| Profiling costs ~$30 | **WRONG** (my claim) — a GPU path is $200–500; a CPU path is $50–150 (§4) | — |
| Hosted K3 is "pennies per million" | **WRONG** (my claim) — it is $3 / $15 per M in/out (§3) | — |
| The scheduler can be bound to llama.cpp as-is | **WRONG-ish** — llama.cpp has no residency hook; a fork is needed (§5) | — |

Net: the mechanics are real. The *numbers* rest on four unknowns, two of which you can
measure this week for nothing, and the economics are better than I first said but only
because hosted K3 is expensive, not because local is cheap.

---

## 1. What survives attack

**The byte anatomy.** Splitting the scheduler's traffic into hot-pass and cold-tail is not
a model assumption; it falls out of top-k routing with any long tail at all. With 16 picks
over 896 experts and any realistic concentration, most *unique* experts per layer-step are
wanted by exactly one caller. Batching cannot dedupe them and no cache can hold 800 GB of
them. Anything that does not touch the tail is optimising 16% of the problem. This is the
one finding I would bet on.

**The tape floor.** Sequential streaming of the whole model once per step needs no routing
assumption and its cost is bytes ÷ bandwidth. It is the correct baseline and V2 did not
have one.

**Determinism and generated results.** Fixed. Byte counts are identical across
`PYTHONHASHSEED` values in-process and on the user's machine.

**The convoy premise is *testable*.** `coherence_same_domain` vs `cross_domain` in the
profile is a direct measurement. If they are close, batch by arrival and stop pretending.

---

## 2. The attacks that landed

### 2.1 Skip assumes cold ⇒ low gate. Nobody has checked.

The routing model gives cold picks half the gate weight of hot picks. That is a guess with
an intuitive story ("the tail is noise") and a plausible counter-story: a rarely-routed
expert may be a *specialist* that fires hard when it fires. Under the counter-story:

```
cold_weight_ratio   decay MB/tok   skip MB/tok   skip tokens/night
0.5 (assumed)         3,403          1,419          217,532
1.0                   3,403          2,518          141,222
1.5                   3,403          3,188          116,333
2.0                   3,403          3,372          110,970   <- skip does nothing
```

The `skip_max_gate` cap does exactly what it should — refuses to drop heavy picks — and
in doing so removes the byte savings. **The headline V3 win is conditional on one
correlation that the profile measures in a single column (`cold_weight_ratio`).** Until
that column exists, the honest V3 number is the `decay` row: ~113K/night at 192 GB.

### 2.2 Skip's quality cost is not 7%.

7% is gate mass dropped per token-layer. The residual stream dampens it, but it is applied
at 92 layers and the model was not trained with it. Published expert-dropping results on
Mixtral-class models show perplexity rising by several percent at 10–20% of experts
dropped; K3-class results do not exist. The simulator cannot price this. Only an eval on
the real weights can, and until then "skip" should be labelled *lossy, unpriced*.

### 2.3 The compute constant is unmeasured and now load-bearing.

`COMPUTE_S_PER_TOKEN = 0.12` was a class estimate. The DRAM-traversal check (251 GB/step at
~80 GB/s ⇒ ~0.10 s/token) is consistent with it *if CPU kernels run near DRAM bandwidth*,
which quantised Q2 matmuls on desktop CPUs often do not (30–60% is typical). Sensitivity:

```
compute s/tok   prefill       decay tok/s   skip tok/s   skip tokens/night
0.12            ÷4 (GEMM)        3.01          6.68          205,906
0.24            ÷4               3.01          4.17          120,000
0.36            ÷4               2.78          2.78           80,000   <- everything compute-bound
0.12            ÷1 (no GEMM win) 3.01          6.68          110,825   <- prefill eats the night
0.24            ÷1               3.01          4.17           60,000
```

At 0.36 s/token every SSD optimisation past ~80K/night buys nothing. **This is the single
most consequential unknown after routing, and unlike routing it can be measured today on
the existing PC with a smaller MoE** (§6, Phase 0). Prefill compute matters as much as
decode: 32 × 512 prompt tokens per convoy is 16K tokens of forward pass, and if quantised
CPU GEMM does not beat GEMV per token, prefill alone is ~33 minutes per convoy.

### 2.4 Perfect I/O–compute overlap is assumed.

`wall = max(io, compute)` is the Spotlight ideal. With no overlap it is `io + compute`:
27% loss for `decay`, 44% for `skip` (skip is closer to balanced, so it loses more). Real
engines land in between; budget a 15–25% haircut on every tok/s figure until measured.

### 2.5 Drive numbers are optimistic and OS-dependent.

10 GB/s for 9.7 MB random reads from two striped Gen4 drives is ~80% of their combined
sequential peak. Achievable on Linux mdraid with high queue depth; not on Windows Storage
Spaces, and not through WSL2's filesystem bridge. At 6 GB/s: `decay` 70K, `skip` 140K per
night. The project should assume **native Linux** for production; the development machine
is Windows.

### 2.6 The KV constant may be 6× wrong — in our favour, but wrong.

`geometry.LATENT = 3584` is labelled "MLA latent dim". In the DeepSeek-V3 / Kimi-K2 family
the cached quantity per token per layer is `kv_lora_rank (512) + rope dim (64) = 576`
values, not 3584. If K3 follows the family:

```
latent   KV per token   KV for batch 176 x 768   max tape batch @192 GB
3584     0.667 MB          88 GB                      235
 576     0.107 MB          14 GB                    1,462
```

That would make KV a non-issue, push the tape regime straight to the compute ceiling, and
make the *cheap* path (more drives, no RAM) far stronger. It is also the kind of number
that is one `gguf-dump` away — the shard is on disk (§6, Phase 0). Two other geometry
constants share the same status: `TRUNK_GB = 60` ("~60") and the 861 vs 918 GB file size
that appears in two places in the old README.

### 2.7 The engine does not exist.

llama.cpp with `--parallel 32` and mmap already *is* the `union_lru` row: the OS page cache
is roughly LRU, and `mul_mat_id` computes the per-layer union for the batch. Sisyphus's
deltas over it need code that does not exist in any maintained engine:

- demand/decay eviction — no residency hook in llama.cpp; the cheap approximation is
  **pinning the profiled hot set with `mlock`** and letting the page cache take the tail;
- residency-aware skip — a patch to the MoE graph: after top-k, mask experts that are cold
  *and* not resident against a residency bitmap, renormalise (moderate C++);
- the tape regime — a different execution order (expert-outer, token-inner); no engine
  does this; it is a new backend.

Honest effort: a llama.cpp fork with pinning + masking + route logging is weeks of C++.
The tape engine is months. `MoE-Infinity` and `ktransformers` are the nearest existing
codebases for SSD/CPU expert offload and should be evaluated before writing anything.

### 2.8 Q2 is not K3.

Every token produced locally is from a 2-bit quantisation of a 2.8T model. The hosted
comparison is against FP8. The quality gap is unmeasured for K3 and is a real cost of
the whole approach that no scheduler improves. It is measurable in the same rented
session as the profile (§4).

### 2.9 Smaller things

- 12 simulated decode steps: checked at 48 — results improve slightly (decay 3,284 → 3,111
  MB/tok), so short runs are conservative, not flattering.
- `hot_min = 2`: an expert wanted by 2 of 32 callers counts as hot. Sensible; not swept.
- The routing model has no within-call autocorrelation. Real consecutive tokens share
  experts more than independent draws, which helps caches and the draft/verify idea.
  Conservative direction.
- Speculative decoding (`spec_mult 1.4`) charged honestly is ~1.0× in the I/O-bound regime.
  Already removed from the plan.
- The convoy composer's prompt-shingle similarity is a placeholder that trivially clusters
  the sample jobs. Real job text will need route-history similarity, which needs the log.

---

## 3. Economics, with real prices (September 2026)

**Hosted K3** (OpenRouter, 1 Sep 2026): $3.00 / $15.00 per M tokens in/out at most
providers; cheapest $2.55 / $12.75. Not "pennies" — I was wrong about that earlier, and it
changes the conclusion in the project's favour.

**A night's output**, using the `skip` row at 192 GB (218K output tokens, ≈850 jobs × 512
prompt tokens ≈ 435K input tokens): hosted equivalent ≈ 218K × $15 + 435K × $3 ≈
**$4.60/night ≈ $1,700/year**. Using the conditional-free `decay` row (113K): ≈ $2.60/night
≈ $950/year.

**Electricity**: ~400 W × 12 h ≈ 4.8 kWh ≈ $0.80/night at $0.17/kWh ≈ **$300/year**.

**Hardware, at today's inflated prices** (Tom's Hardware trackers, 24–27 Aug 2026; DDR5 is
described as having tripled to quadrupled since October 2025 and NAND as "skyrocketing"):

| item | price now | note |
|---|---|---|
| DDR5 48 GB module | ~$701 | 192 GB = 4 × 48 ≈ **$2,800** |
| DDR5 96 GB kit | ~$1,849 | |
| Gen5 NVMe 2 TB (T705 / 9100 Pro / SN8100 / MP700 Pro) | ~$393–437 | |
| Gen5 NVMe 4 TB | ~$567–1,128 | |
| B200 8-GPU node, specialist cloud | ~$48–57/hr | hyperscalers $114–129/hr |
| H200, on demand | ~$4–11/GPU-hr | |

**Verdict**: local K3 is not cheaper than hosted on electricity alone (it never was), but
hosted K3 is expensive enough that the *storage-only* build pays back in about a year on
the conditional-free number and in months on the optimistic one. The **RAM build does not
pay back inside three years at current DDR5 prices** and should wait for the profile and
for prices to normalise.

---

## 4. Rent-to-profile, revised

The H100/B200 path I described first is the *wrong* rental. K3 at FP8 is 2.8 TB and needs
two 8×B200 nodes (~$100–115/hr) plus 1–2 hours of load time: $300–500 to profile the FP8
model, which is not the model you will run.

The right rental is a **large-RAM CPU instance** (1–2 TB, e.g. AWS `u-*`/`x2idn`, Hetzner
dedicated, or a bare-metal provider) at roughly $8–15/hr. It runs the *actual* Q2 GGUF
under the *actual* engine family and answers four questions in one session:

1. **Routes.** llama.cpp names the MoE graph tensors (`ffn_moe_topk-N`,
   `ffn_moe_weights-N`); the existing `eval-callback` mechanism can dump them per layer
   per token with a small patch, emitting `sisyphus.profiling.schema` records directly.
   No transformers, no FP8 checkpoint.
2. **Q2 quality** vs hosted FP8 on your prompts (§2.8).
3. **The compute constant** on a server CPU, which bounds what a desktop can do (§2.3).
4. **Skip's real cost**: rerun the eval with the sub-threshold experts masked at
   `skip_budget` 0.03 / 0.06 / 0.10 (§2.2).

Budget: 4–8 hours ≈ **$50–150**, including the model copy.

---

## 5. Production architecture (what "fully practicing it" means)

```
 jobs (JSON)  ->  convoy_composer (route-history similarity)  ->  night plan
                                                                     |
      llama.cpp fork  <--------------------------------------------- v
        * --parallel N continuous batching (the union)
        * hot-set pinning: mlock() the profiled hot experts        (decay policy, approx.)
        * residency bitmap + top-k masking + renormalise           (skip policy)
        * route + gate logging to JSONL                            (profiling, ongoing)
        * expert-major file layout on a Linux mdraid stripe        (tape-friendly)
      +  eval harness: nightly sample re-scored against hosted K3  (quality guardrail)
      +  metrics: bytes/token, hit rate, skipped mass, tok/s        (RESULTS.md, live)
```

The scheduler package stays what it is — the *model* of the engine that decides pinning,
budgets and batch shape, and regenerates the tables from the live logs.

---

## 6. Next steps, in order, with cost

### Phase 0 — measure what you already own ($0, this week)

1. **Audit the geometry from the shard.** `gguf-dump` (or the `gguf` Python package) on
   shard 1: `kv_lora_rank`, `expert_count`, `expert_used_count`, block count, and the byte
   size of every tensor. Fix `LATENT`, `TRUNK_GB`, `EXPERT_MB`, `MODEL_GB` from data. This
   alone may move the tape floor 3–6× (§2.6).
2. **Measure the compute constant.** Run llama.cpp on the current PC with a MoE that fits
   in RAM (a Q2 100B–235B-class MoE), `--parallel 32`, and record decode s/token and
   prefill s/token. Scale by active parameters to K3's 104B. Replace `COMPUTE_S_PER_TOKEN`
   and `PREFILL_COMPUTE_S_PER_TOKEN`. If the answer is ≥ 0.3 s/token, stop optimising
   the SSD path and start on compute (GPU prefill, §7).
3. **Measure the drives.** `fio` with 9.7 MB random reads, QD 32, on the current stripe,
   on Linux. Replace `DRIVE_*_GBPS`.
4. Regenerate `RESULTS.md`. Every number after this step is conditional only on routing.

### Phase 1 — profile ($50–150, one weekend)

5. Large-RAM CPU instance, Q2 GGUF, llama.cpp with the route-dump patch. Bring home
   `routes.jsonl.gz`, a quality table (Q2 vs hosted on 50 of your prompts), the server-CPU
   compute constant, and the skip-cost table.
6. `python -m sisyphus.profiling.analyze` → three numbers: `hot_frac_at_mass[0.75]` (RAM
   tier), coherence (convoy premise), `cold_weight_ratio` (is skip alive?).
7. Regenerate `RESULTS.md` from `measure_logged`. This is the first honest number.

### Phase 2 — storage ($800–1,700)

8. Two to four Gen5 2 TB drives (~$400 each). Check the board's bifurcation first; a
   4×M.2 carrier card is ~$50–100 if the slot supports x4/x4/x4/x4. Linux mdraid stripe,
   expert-major layout. This is the lever that pays regardless of §6.6's answer.

### Phase 3 — engine (time: 4–8 weeks part-time, $0)

9. Fork llama.cpp: hot-set `mlock` pinning from the profile; residency bitmap and top-k
   masking; JSONL route logging. Evaluate `MoE-Infinity` / `ktransformers` first — if one
   already does 80% of this, fork that instead.
10. Wire `convoy_composer` to real job text via route-history similarity from the log.
11. Nightly eval harness against hosted K3 on a fixed prompt set. Skipping stays off
    until this harness says the cost is acceptable.

### Phase 4 — RAM, only if the profile says so (~$2,800 for 192 GB at today's prices)

12. Buy 192 GB only if `hot_set_gb(hot_frac_at_mass[0.75]) + 60 < ~170`. Otherwise the
    RAM does not unlock the policy and the money belongs in drives or a GPU. At current
    DDR5 prices this step does not pay back inside three years; defer unless the profile is
    strongly favourable and prices fall.

### Phase 5 — production

13. Job queue → convoy plan → nightly run → results + metrics + eval, on a schedule.
    Regenerate `RESULTS.md` from live logs weekly so the model and the machine never
    disagree again.

---

## 7. If compute is the wall

If Phase 0 puts the compute constant at ≥ 0.3 s/token, the SSD work is finished and the
next lever is compute, in this order: (a) prefill on a GPU by streaming experts through
VRAM once per convoy (GEMM over 16K tokens per expert is exactly what a GPU is for, and it
needs no VRAM residency); (b) lower expert precision in RAM to cut DRAM traversal (§ deep
dive B — it helps both tiers); (c) larger batch, now that KV is cheaper than we thought.
A consumer GPU for the trunk is 24% of DRAM traffic and does not fit a consumer card;
not the first move.

---

## 8. Budget summary

| phase | cost | gates |
|---|---|---|
| 0 measure locally | $0 | nothing; do first |
| 1 profile (CPU rental) | $50–150 | Phase 0 done |
| 2 storage (2–4 Gen5 drives + carrier) | $800–1,700 | none; pays regardless |
| 3 engine fork | $0, 4–8 weeks | Phase 1 profile |
| 4 RAM 192 GB | ~$2,800 today | profile says hot set fits; prices normalise |
| electricity | ~$300/yr | — |
| **total to production without RAM** | **$850–1,850 + time** | |
| hosted-equivalent output | $950–1,700/yr | conditional on §2.1 and §2.3 |

Payback on the no-RAM build: roughly one year at the conservative number, months at the
optimistic one, *provided* Q2 quality is acceptable for the jobs — which is the first thing
Phase 1 measures.

---

## 9. Copium check

Things this project has already been wrong about, in order of discovery: a 100 GB cache
on a 128 GB box; process-salted RNG seeds; two mechanisms that could not change a byte;
a "2.1× at 128 GB" that was 2.1× at 192 GB; "pennies per million" hosted pricing; a $30
profiling estimate; a KV constant that may be 6× off. Each was found by checking, not by
arguing. The pattern is that the *mechanisms* have held and the *constants* have not.
That is the right way round — constants are cheap to fix — but it means no number in
`RESULTS.md` should be quoted without its gate from §6, and the two that gate everything
(§2.1 cold-gate correlation, §2.3 compute) are measurable for under $150 combined.

---

## 10. V4 recalculation — the compute constant, and what replaces it

### 10.1 What was wrong

`COMPUTE_S_PER_TOKEN = 0.12` treated compute as a scalar. Working it properly: K3 does ~208
GFLOP per token (104B active × 2), fixed by the architecture. At batch 32 that is 6.7 TFLOP
per step. A 16-core desktop delivers ~1 TFLOPS effective on quantised kernels, so the FLOP
time (6.7 s) exceeds the DRAM traversal (3.1 s). **On a CPU past batch ~15 the wall is
FLOPs, not bytes; batching cannot move it; the floor is ~0.2 s/token.** Prefill is worse:
16K tokens per convoy is 3.4 PFLOP, ~27 minutes at a 2× GEMM gain, ~57 without. V3's
218K/night at 192 GB becomes **110K** once compute is modelled, and the `bound` column
reads `cpu` — more RAM cannot help that row.

### 10.2 What improves it

`geometry.ComputeModel` places the trunk and experts on CPU or GPU and computes each step
as max(SSD, memory/PCIe traversal, FLOPs). Measured on the same routes (192 GB, skip):

```
compute path                               tok/s   prefill/convoy   tokens/night   bound
cpu, 1 TFLOPS eff, batch 32                 5.1      27 min          110K          cpu
cpu, 2 TFLOPS eff (better kernels)          7.2      13 min          183K          ssd
GPU holds trunk only (56% of FLOPs off CPU) 8.1      11 min          211K          ssd
GPU streams experts over PCIe x16, b32      8.1      1.2 min         326K          ssd
GPU streams experts over PCIe x16, b128     9.3      2.2 min         386K          ssd
  + 4 drives (20 GB/s random reads)        18.5      2.2 min         746K          ssd
  same machine, decay (lossless)            8.1      2.3 min         338K          ssd
  same machine, tape (lossless, no routing) 13.6     6 min           558K          ssd
```

The GPU is not a storage tier — the trunk barely fits a consumer card and the experts
never will. It is a **streaming compute engine**: experts cross PCIe from RAM every step
(x16 Gen5 ≈ 50 GB/s achieved, close to desktop DRAM), are applied to every token in VRAM,
and leave. GPU FLOPs are 50–100× the CPU's, so they stop mattering: the sensitivity table
shows 40 vs 100 TFLOPS changes nothing and PCIe 25 vs 50 GB/s costs ~8%. **A used 24 GB
Gen4 card captures most of the win.** Prefill, the hidden killer of the CPU path, becomes a
one-minute cost. And with the trunk in VRAM the 128 GB tier recovers 60 GB of RAM budget:
`128 GB + GPU` reaches 292–322K/night, three times what 192 GB of RAM alone achieves.

The purchase order inverted. V3 said RAM (192 GB) unlocks the policy. V4 says: **GPU
first, drives second, RAM a distant third.** The `bound` column explains why — on the GPU
path every row is SSD-bound, so drives are the next lever and the tape row shows what they
buy with no routing assumption at all.

### 10.3 Two conditions on the GPU path

- **The trunk must fit VRAM.** Attention weights (~37B params) at Q4 are ~18–20 GB; shared
  experts ~3 GB; embeddings can stay on the CPU. That fits 24 GB *only if KV lives in
  RAM or the KV constant is the family's 576-value latent (§2.6) rather than 3584*. Phase 0's
  `gguf-dump` decides whether a 24 GB card is enough or a 32 GB card is needed.
- **PCIe lanes are shared with the drives.** On AM5/Z890 the x16 slot is the CPU-attached
  budget. GPU at x16 leaves the CPU M.2 slots (two Gen5) plus chipset M.2 (aggregate ~8
  GB/s) — the `192GB · GPU x16 + 4 drives (M.2) 20` row. GPU at x8 frees lanes for a
  4-drive carrier but halves PCIe: the `GPU x8` row, ~20% slower on skip. Either beats the
  CPU path by 3–6×.

### 10.4 Recalculated budget (September 2026 prices)

| item | price now | source |
|---|---|---|
| RTX 5090 32 GB, Gen5 | ~$5,000 (2.5× launch) | Tom's Hardware / TechSpot, 1 Sep 2026 |
| RTX 4090 24 GB, Gen4, used | ~$2,500 | bestvaluegpu, 1 Sep 2026 |
| RTX 3090 24 GB, Gen4, used | ~$1,300 | bestvaluegpu, 1 Sep 2026 |
| Gen5 NVMe 2 TB | ~$400 | Tom's Hardware, 24 Aug 2026 |
| DDR5 48 GB module | ~$700 (192 GB ≈ $2,800) | Tom's Hardware, 27 Aug 2026 |
| large-RAM CPU instance for profiling | ~$8–15/hr | typical |

GPU speed is irrelevant on this path and Gen4 x16 (~25 GB/s) costs ~8%, so the **used
RTX 3090 is the value pick** — provided the trunk fits 24 GB (§10.3). The 5090 buys Gen5
PCIe and 32 GB for 4× the price; not first.

| phase | what | cost | gate |
|---|---|---|---|
| **0** | `gguf-dump` shard 1 (kv_lora_rank, tensor sizes → fixes KV, trunk, expert constants). llama.cpp on a RAM-resident MoE, `--parallel 32`: decode + prefill s/token, then with attention offloaded to any GPU present. `fio` 9.7 MB QD32 on Linux. | **$0** | first |
| **1** | Large-RAM CPU instance, real Q2 GGUF, llama.cpp route dump + Q2-vs-hosted eval + skip-cost eval + server compute constant. | **$50–150** | Phase 0 |
| **2** | **GPU**: used RTX 3090 24 GB (or 4090 if the trunk needs headroom; 5090 only if Phase 0 shows PCIe-bound). Attention re-quantised to Q4 for VRAM. | **$1,300–2,500** | Phase 0 says trunk fits |
| **3** | **Drives**: 2 more Gen5 2 TB on CPU M.2 (+ chipset if available). Linux mdraid, expert-major layout. | **$800–1,700** | none |
| **4** | Engine: llama.cpp fork — attention on GPU (exists), expert streaming through GPU (new: `mul_mat_id` with host-pinned expert buffers over PCIe), hot-set `mlock`, residency-bitmap top-k masking, route logging. Evaluate ktransformers first (it already splits attention/GPU from experts/CPU). | **$0, 6–10 weeks** | Phase 1 |
| **5** | RAM 192 GB. Only if the profile's hot set does not fit the recovered 128 GB budget and prices normalise. | ~$2,800 | profile |
| — | electricity (~450 W with GPU) | ~$340/yr | — |
| **total to production, GPU + drives, no RAM** | | **$2,150–4,350 + time** | |

### 10.5 Recalculated economics

Hosted-equivalent value per night (§3 prices, 512-in / 256-out jobs):

| configuration | policy | tokens/night | hosted-equiv/night | /year |
|---|---|---|---|---|
| 192 GB, cpu (V3 plan) | skip (lossy) | 110K | $2.30 | $850 |
| 128 GB + used GPU, 2 drives | decay (lossless) | 149K | $3.15 | $1,150 |
| 128 GB + used GPU, 2 drives | skip | 322K | $6.80 | $2,480 |
| 128 GB + GPU, 4 drives | tape (lossless, no assumption) | ~360K | $7.60 | $2,770 |
| 192 GB + GPU, 4 drives | tape | 558K | $11.80 | $4,300 |
| 192 GB + GPU, 4 drives | skip | 746K | $15.70 | $5,750 |

Against electricity of ~$340/yr, the **$2,150 GPU + drives build on the existing 128 GB
pays back in 6–12 months on lossless numbers alone**, in ~4 months if skip survives the
profile. That is a different conclusion from V3's "marginal, 1–3 years", and the whole
difference is putting FLOPs on a GPU instead of buying RAM. Every row above still carries
the §6 gates: routing (§2.1), compute (Phase 0), Q2 quality (§2.8), and I/O overlap
(15–25% haircut).

### 10.6 What V4 did not fix

The engine gap got larger, not smaller: streaming experts through a GPU is a real
`mul_mat_id`-over-PCIe backend that no maintained engine ships (ktransformers is closest,
with attention on GPU and experts on CPU; the streaming variant is new work). The GPU path
also assumes KV fits beside the trunk in VRAM or streams cheaply — Phase 0 decides. And
nothing in V4 touches the routing unknowns; the profile is still the gate on every
`skip` row.

