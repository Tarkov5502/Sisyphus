# Sisyphus — the scheduler core

*An inference operating system for underutilized consumer hardware. This package is the
part that is pure algorithm and needs no 918 GB model present to build, test, or measure —
the demand-ordered expert-major scheduler and its concurrency-safe residency owner
(design doc V2.18 + V2.19).*

## The problem, in one paragraph

Kimi-K3 is a 2.8T-param MoE: ~92 MoE layers, 896 experts each, 16 active per token. At
Q2 that's an 861 GB file no consumer machine holds in RAM, so experts stream from SSD.
The cost that dominates is **bytes fetched per token** — every expert not already resident
is a ~9.7 MB disk read. Naive inference is *token-major*: per call, per layer, fetch its
experts, scattering reads and reloading popular experts once per call. Sisyphus inverts
this into an overnight **throughput** engine that fetches each expert as few times as
physically possible.

## What this core does

Three ideas, each a class:

- **`ResidencyCache`** — the only genuinely contended resource is scarce RAM *residency*
  (weights are read-only during inference, so concurrent convoys never corrupt each
  other's output — correctness is free, V2.18). The cache is **reference-counted** (an
  expert a live call stands on cannot be evicted) with **pluggable eviction**. The
  Sisyphus policy is **demand-frequency**, not LRU, because a layer-sequential sweep is
  nearly LRU's worst case — every union expert was "just used", so LRU keeps exactly the
  experts about to become useless.

- **`ExpertMajorScheduler`** (V2.19, the flagship) — per layer, across every phase-aligned
  call, service experts **hottest-first** (load once, serve the whole crowd), sweep cold
  single-use experts into **one consolidated straggler batch** at the layer's end (rare
  work done once, late, cheap), and **age** deferred calls so a rare-expert-only job can
  never starve.

- **`Coordinator`** — drives a lockstep batch through all layers for N decode steps,
  refreshing per-token routing while aging state persists on each `Call`.

Plus `schedule.py`, which wires the existing **convoy composer** (route-aware batching +
route-chaining) to the scheduler: jobs → coherent convoys → warm-road throughput.

## The result that matters

From `RESULTS.md` (real scheduler, coherent routing, expected 0.75 skew):

| hardware | policy | MB/token | hit | tok/night |
|---|---|---|---|---|
| 32 GB RAM, scattered | naive token-major | 12,982 | 0.09 | 5,111 |
| 32 GB RAM, layout | union + LRU | 7,761 | 0.00 | 18,239 |
| 32 GB RAM, layout | **Sisyphus** | 7,122 | 0.08 | 19,877 |
| 128 GB RAM, striped | union + LRU | 7,761 | 0.00 | 56,997 |
| 128 GB RAM, striped | **Sisyphus** | **3,660** | **0.53** | **120,880** |

**The headline:** demand-ordered eviction is *marginal* at 32 GB (the cache can't hold the
hot set) but a **2.1× throughput win at 128 GB** — it keeps the globally-hot experts
resident where LRU thrashes to a 0% hit rate. That is a *quantitative* argument for the
128 GB RAM purchase (doc V2.10): the RAM is what unlocks the policy. Sensitivity across the
one unknown (domain coherence) spans 127K–289K tokens/night; the profiling harness will
collapse that band to a measured number.

## Run it

```bash
python -m pytest sisyphus/test_coordinator.py -q     # 12 invariant tests
python -m sisyphus.engine_sim                          # the throughput table
python -m sisyphus.schedule                            # jobs -> convoys -> tok/night
```

## What's real vs. modeled

- **Real:** the scheduler, the refcounted cache, the eviction policies, the straggler +
  aging logic, the byte accounting — this is code that ships into the engine.
- **Modeled (honestly bracketed):** the *routing* — how concentrated K3's expert
  activation is, and how much same-domain jobs share experts. That's the one empirical
  unknown, isolated in `routing.py`, replaced the day the profiling harness logs real K3
  routes. Everything else is physics off the real download (`geometry.py`).

## Files

```
geometry.py         Kimi-K3 constants, off the real UD-Q2_K_XL download
coordinator.py      ResidencyCache, DemandHistogram, ExpertMajorScheduler, Coordinator
routing.py          bracketed routing model (the one unknown, made explicit)
engine_sim.py       drives the real scheduler; the throughput table
convoy_composer.py  route-aware batching + route-chaining (V2.11/V2.16)
schedule.py         the night pipeline: jobs -> convoys -> throughput
test_coordinator.py 12 invariant tests
RESULTS.md          captured measurements
```

## Next (when the model lands)

1. Profiling harness: log real K3 routes on the workload corpus → replace `routing.py`'s
   model with measured skew, and turn `RESULTS.md`'s band into a number.
2. Bind the `Coordinator` to a real backend (llama.cpp slot-save / WASTE streaming) so the
   scheduler drives actual fetches, not simulated ones.
3. Cross-layer convoy misalignment: the straggler consolidation's *distinct* win appears
   when convoys drift across layers — model it, then add the aligned-vs-drifted comparison.
