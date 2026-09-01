# Sisyphus — the scheduler core (V4)

*An inference operating system for underutilized consumer hardware. This package is the
part that is pure algorithm and needs no 861 GB model present to build, test, or measure:
the demand-ordered expert-major scheduler, its concurrency-safe residency owner, the
assumption-free tape floor, and the round trip that replaces every guess with a
measurement.*

## The problem, in one paragraph

Kimi-K3 is a 2.8T-param MoE: 92 MoE layers, 896 experts each, 16 active per token. At
Q2 that is an 861 GB file no consumer machine holds in RAM, so experts stream from SSD.
The cost that dominates is **bytes fetched per generated token** — every expert not
already resident is a ~9.7 MB disk read. Naive inference is *token-major*: per call, per
layer, fetch its experts, scattering reads and reloading popular experts once per call.
Sisyphus inverts this into an overnight **throughput** engine that fetches each expert as
few times as physically possible — and, in V3, that decides *whether an expert is worth
fetching at all*.

## What V3 changed, and why

V2 measured a 2.1× win for demand-eviction over LRU at "128 GB". Auditing it found five
problems; V3 fixes each and measures the fix.

| # | problem in V2 | V3 fix | measured effect (192 GB, batch 32, expected bracket) |
|---|---|---|---|
| 1 | Nothing touched the cold tail — ~all remaining bytes were single-use experts no policy could cache | **Residency-aware routing**: a cold expert that is not resident and carries little gate weight is skipped under a per-token budget (`cold_policy="skip"`) | 3,284 → 1,419 MB/tok; 113K → 218K tokens/night at 7% gate mass skipped |
| 2 | Every number rested on an unmeasurable routing guess | **Rent-to-profile** (`sisyphus.profiling`): log real routes on a rented node, analyze, replay the real scheduler on them | replaces `routing.py` with data; closed-loop tested on synthetic logs |
| 3 | Cumulative demand never decayed — a stale domain squatted in the cache | **Exponential demand decay** in `ResidencyCache` (`decay=0.8`), O(1) per step | mixed night: 3,823 → 3,412 MB/tok (−11%); single domain unchanged |
| 4 | No floor that did not depend on the guess | **Tape regime** (`sisyphus.tape`): stream the whole model once per step; KV budget vs batch made explicit | 100K/night at 192 GB with zero routing assumptions |
| 5 | Simulator asserted a 100 GB cache on a 128 GB box, ignored KV and prefill, never swept `hot_frac`, seeded RNGs with process-salted `hash()`, and carried two mechanisms that could not change a byte | RAM budget derived (RAM − trunk − OS − KV); prefill charged per convoy; every unknown swept; stable seeding; aging/deferral removed with the reason documented | the 128 GB row fell from 121K to 83K/night once the trunk was charged; `RESULTS.md` is now generated, not typed |

The honest headline moved. **The 128 GB purchase does not unlock the policy** — with the
60 GB trunk resident, a 128 GB box has ~46 GB for experts and the 78 GB hot set does not
fit; the tier that does is 192 GB. What unlocks the target is #1, which makes the result
compute-bound and largely indifferent to the skew you cannot verify.

## What V4 changed: compute is a model, not a constant

V3 carried a single `COMPUTE_S_PER_TOKEN = 0.12`. Working the FLOP side properly showed why
that cannot be right on a CPU: K3 does ~208 GFLOP per token regardless of scheduling, and
at batch 32 a 16-core desktop's ~1 TFLOPS effective needs 6.7 s per step against 3.1 s of
DRAM traversal. **Past batch ~15 the CPU is FLOP-bound, batching stops helping, and the
per-token cost is a floor near 0.2 s** — worse than V3 assumed, and it makes prefill
(16K tokens per convoy) a 27-minute cost the SSD work cannot touch.

`geometry.ComputeModel` now places the trunk and the experts on CPU or GPU and computes
each step as the slowest of SSD, memory/PCIe traversal and FLOPs. The GPU's role is not a
storage tier (the trunk barely fits a consumer card) but a **streaming compute engine**:
experts cross PCIe from RAM each step, are applied to every token in the batch in VRAM,
and leave. PCIe x16 (~50 GB/s achieved) is close to desktop DRAM bandwidth, and GPU FLOPs
are 50-100x the CPU's, so the wall moves to PCIe/SSD and per-token cost falls with batch
again. Prefill drops from ~27 minutes to ~1 minute per convoy.

| 192 GB, skip policy, expected bracket | tok/s | prefill/convoy | tokens/night |
|---|---|---|---|
| cpu, batch 32 (V3 said 218K)         | 5.1  | 27 min | 110K |
| cpu with 2x better kernels           | 7.2  | 13 min | 183K |
| GPU holds trunk only                 | 8.1  | 11 min | 211K |
| GPU streams experts, x16, batch 32   | 8.1  | 1.2 min | 326K |
| GPU streams experts, x16, batch 128  | 9.3  | 2.2 min | 386K |
| + 4 drives (20 GB/s random)          | 18.5 | 2.2 min | **746K** |
| same machine, tape (lossless, no routing assumption) | 13.6 | 6 min | **558K** |
| same machine, decay (lossless)       | 8.1  | 2.3 min | 338K |

The GPU also rescues the 128 GB tier: with the trunk in VRAM the RAM budget recovers 60 GB
(cache 74-106 GB), and `128 GB + GPU x16` reaches 292-322K/night — three times what
192 GB of RAM alone does on the CPU path. **The purchase order inverted: GPU, then
drives, then RAM.** GPU speed barely matters (40 vs 100 TFLOPS changes nothing; SSD and
PCIe bind), so a used 24 GB Gen4 card is most of the win at a quarter of the price.

## What this core does

Five ideas, each a module:

- **`ResidencyCache`** — the only genuinely contended resource is scarce RAM *residency*
  (weights are read-only during inference, so concurrent convoys never corrupt each
  other's output — correctness is free). The cache is **reference-counted** (an expert a
  live call stands on cannot be evicted) with **pluggable eviction**. The Sisyphus policy
  is **decayed demand-frequency**, not LRU: a layer-sequential sweep is nearly LRU's
  worst case, and un-decayed frequency is anti-adaptive.

- **`ExpertMajorScheduler`** — per layer, across every phase-aligned call, service
  experts **hottest-first** (load once, serve the whole crowd), then sweep the cold
  single-use experts in one late pass. Under `cold_policy="skip"`, a cold expert that is
  not resident and whose gate weight is below `skip_max_gate` is dropped and its caller
  renormalises, bounded by `skip_budget` gate mass per token-layer. Skipped mass is
  reported on every run: the trade is visible, never silent.

- **`Coordinator`** — drives a lockstep batch through all layers for N decode steps,
  refreshing per-token routing and advancing demand decay once per step.

- **`tape`** — the large-batch floor. At high batch every layer's union is nearly all 896
  experts, so the optimal policy is a sequential sweep of the file: bytes/bandwidth,
  whatever the skew. It also owns the KV-vs-residency RAM split V2 did not model.

- **`profiling`** — capture (rented box), schema, analyze, replay, synthetic. The
  round trip that turns `RoutingModel(hot_frac=0.10, hot_mass=0.75)` into
  `RoutingModel(**profile.bracket())`, or drops the model and replays the log.

Plus `schedule.py`, which wires the **convoy composer** (route-aware batching +
route-chaining) to the scheduler over one shared cache: jobs → coherent convoys →
warm-road throughput, now also the place the decay fix is visible.

## The results that matter

From `RESULTS.md` (generated; expected bracket; 512-prompt + 256-decode jobs; cache
derived from RAM; compute per `ComputeModel`):

| machine | policy | cache | MB/tok | bound | tok/s | tok/night |
|---|---|---|---|---|---|---|
| 128 GB, 2 drives, cpu, b32 | union + LRU | 46 | 5,598 | ssd | 1.8 | 57K |
| 128 GB, 2 drives, cpu, b32 | skip | 46 | 3,003 | ssd | 3.4 | 88K |
| 128 GB, 2 drives, GPU x16, b96 | decay | 74 | 2,923 | ssd | 3.5 | 149K |
| 128 GB, 2 drives, GPU x16, b96 | **skip** | 74 | 1,335 | ssd | 7.7 | **322K** |
| 192 GB, 2 drives, cpu, b32 | sisyphus (V2) | 110 | 3,248 | ssd | 3.2 | 82K |
| 192 GB, 2 drives, cpu, b32 | skip | 110 | 1,419 | **cpu** | 5.1 | 110K |
| 192 GB, 2 drives, GPU x16, b128 | decay | 122 | 2,536 | ssd | 4.0 | 172K |
| 192 GB, 2 drives, GPU x16, b128 | **skip** | 122 | 1,105 | ssd | 9.3 | **386K** |
| 192 GB, GPU x16 + 4 drives, b128 | decay | 122 | 2,536 | ssd | 8.1 | 338K |
| 192 GB, GPU x16 + 4 drives, b128 | **skip** | 122 | 1,105 | ssd | 18.5 | **746K** |
| 192 GB, GPU x16 + 4 drives, tape b355 | full sweep | – | 2,252 | ssd | 13.6 | **558K** |

Read off it: on the CPU path the `bound` column says `cpu` — more RAM cannot help. On the
GPU path every row is SSD-bound — drives are the next lever, and the tape row shows what
they buy with no routing assumption at all.

**Sensitivity** (all in `RESULTS.md`): routing unknowns as before (`coherent=False`
collapses everything; `cold_weight_ratio` >= 1.5 removes the skip win). Compute unknowns:
CPU at 0.5 TFLOPS caps every policy at ~55K/night; GPU TFLOPS is irrelevant; PCIe 25 vs
50 GB/s costs 8% on the skip row.

## Run it

```bash
python -m pytest sisyphus/test_coordinator.py -q     # 33 invariant tests, ~6 s
python -m sisyphus.engine_sim                         # scenario, sensitivity, batch tables
python -m sisyphus.tape                               # the tape floor by RAM tier
python -m sisyphus.schedule                           # mixed-domain night, four policies
python -m sisyphus.results                            # regenerate RESULTS.md (~4 min)
python -m sisyphus.profiling.analyze routes.jsonl.gz  # once you have a real log
```

## What's real vs. modeled

- **Real:** the scheduler, the refcounted decayed cache, the cold-pick policy and its
  budget accounting, the byte accounting, the tape arithmetic, the profiling pipeline
  (tested end to end on synthetic logs). This is code that ships into the engine.
- **Modeled, bracketed, swept:** routing skew (`hot_frac`, `hot_mass`), domain coherence,
  cold gate weight, CPU effective TFLOPS (1.0), CPU prefill GEMM gain (2x), DRAM (80 GB/s),
  GPU TFLOPS (100), PCIe (50 / 25 GB/s), KV per token (from the MLA latent dim — possibly
  6x pessimistic, see ADVERSARIAL_REVIEW.md), OS reserve. Each is named at its call site and swept in
  `RESULTS.md`; the profiling harness replaces the routing ones with data.
- **Not modeled, and known:** the *quality* cost of skipped gate mass. The simulator
  prices it in gate units; only an eval on the real model (same prompts, those experts
  zeroed) can price it in answers. `RENT_TO_PROFILE.md` puts that eval next to the
  capture run. Also unmodeled: a real engine's ability to overlap the hot pass's compute
  with the cold pass's I/O (hot-first ordering exists to allow it).
- **Removed:** V2's aging/deferral. In a lockstep decode a call cannot proceed past a
  layer without its expert, so deferring a cold load to ride with a later convoy is
  identical in bytes to running a larger batch. The batch curve in `RESULTS.md` is the
  deferral curve; the mechanism was bookkeeping that could not move a measurement.

## Files

```
geometry.py            K3 constants, RAM budget, KV, ComputeModel (CPU / GPU-trunk / GPU-stream), stable seeding
routing.py             bracketed routing model with gate weights; BRACKETS; analytic helpers
coordinator.py         ResidencyCache (decay), DemandHistogram, ExpertMajorScheduler (skip), Coordinator
tape.py                full-sweep floor, KV-bounded batch, pin-vs-KV search
engine_sim.py          Machine tiers x compute models, policies, SSD/traversal/FLOP throughput, all tables
convoy_composer.py     route-aware batching + route-chaining (V2.11/V2.16, unchanged)
schedule.py            the night pipeline: jobs -> convoys -> throughput, four policies
results.py             regenerates RESULTS.md
profiling/
  schema.py            the JSONL route record both sides agree on
  capture.py           transformers router hook (runs on the rented box only)
  analyze.py           log -> RoutingProfile (hot_frac, hot_mass, coherence, skippable mass, ...)
  replay.py            LoggedRouting + measure_logged: the real scheduler on real routes
  synthetic.py         a log from RoutingModel, so the loop is proven before paying
  RENT_TO_PROFILE.md   the runbook
test_coordinator.py    33 invariant tests
ADVERSARIAL_REVIEW.md  what holds, what is conditional, what was wrong; roadmap and budget
RESULTS.md             generated measurements
```

## Next

0. **Measure the compute constants on the machine you have** (`ADVERSARIAL_REVIEW.md`
   Phase 0): llama.cpp on a RAM-resident MoE at `--parallel 32`, decode and prefill s/token,
   then again with attention offloaded to whatever GPU is in the box. Replace
   `CPU_TFLOPS_EFF`, `CPU_GEMM_GAIN`, and confirm the GPU-streaming path's PCIe rate.
1. **Profile.** Follow `profiling/RENT_TO_PROFILE.md`. Three numbers decide the project:
   `hot_frac_at_mass["0.75"]` (does the hot set fit the RAM tier?), same- vs cross-domain
   coherence (is the convoy premise true?), `skippable_mass_at_gate` (how much can skip
   remove?). Regenerate `RESULTS.md` from the log.
2. **Price the skip.** On the same rented session, eval quality with the sub-threshold
   experts zeroed at `skip_budget` 0.03/0.06/0.10. That number, not the simulator's, sets
   the production budget.
3. **Bind to a backend.** The `Coordinator` contract (`routing(call, tok)`,
   `acquire`/`release`, `cold_policy`) is what the llama.cpp / WASTE wrapper implements.
   Hot-first ordering means the wrapper can start compute on the hot pass while the cold
   pass's reads are in flight — the overlap this model assumes.
4. **Degrade into tape.** When the derived cache budget falls below the hot set (or the
   profile says coherence is weak), the scheduler should switch to the sweep. The
   crossover is in the batch table; wire it.
