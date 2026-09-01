# Rent-to-profile — collapse the one unknown for the price of lunch

Every number Sisyphus produces rests on how Kimi-K3 routes tokens on *your* workload:
how concentrated expert use is, whether same-domain jobs share experts, and how much
gate weight rides on the cold tail. The laptop cannot run K3 to find out. A rented GPU
node can, in an hour or two, and the log it produces replaces `routing.py`'s guess
everywhere: the scenario tables, the skip budget, the RAM purchase argument.

This is the highest-information-per-dollar action available to the project, and it is
the prerequisite for trusting anything in `RESULTS.md`.

## What you bring home

One file, `routes.jsonl.gz`, a few hundred MB: one record per (call, token, layer) with
the routed experts and their gate weights (schema in `schema.py`). From it, the analyzer
measures every bracketed constant:

| unknown | in `routing.py` | measured by |
|---|---|---|
| hot set size | `hot_frac` | `hot_frac_at_mass` — experts needed for 60/75/85% of gate mass |
| concentration | `hot_mass` | `hot_mass_at_frac` — share of picks in the top 10% |
| convoy premise | `coherent` | `coherence_same_domain` vs `coherence_cross_domain` |
| skippability | `cold_weight_ratio` | `cold_weight_ratio`, `skippable_mass_at_gate` |
| prefill working set | `expected_unique_experts` | `unique_per_layer_at_tokens` |
| home-turf stability | (assumed) | `drift_jaccard` |

plus the replay: `measure_logged()` runs the real scheduler on the real routes.

## Before renting: prove the loop on the laptop

```bash
python - <<'EOF'
from sisyphus.routing import RoutingModel
from sisyphus.coordinator import Call
from sisyphus.profiling import synthesize, analyze_file, LoggedRouting, measure_logged
from sisyphus.engine_sim import MACHINES
rm = RoutingModel(hot_frac=0.10, hot_mass=0.75, cold_weight_ratio=0.5)
calls = [Call(f"audit{i}", [], "code") for i in range(8)] + [Call(f"dili{i}", [], "diligence") for i in range(8)]
synthesize(rm, calls, decode_tokens=64, path="dryrun.jsonl.gz", prefill_tokens=64)
prof = analyze_file("dryrun.jsonl.gz"); print(prof.report()); print(prof.bracket())
print(measure_logged("skip", LoggedRouting.from_file("dryrun.jsonl.gz"), MACHINES["192"], tokens=16))
EOF
```

The analyzer must hand back ~0.75 / ~0.5 / coherent=True. `test_profiling_round_trip`
asserts exactly this. When it passes, the only untested piece is `capture.py` against
the real model's module names — which is what the first five minutes on the box are for.

## On the box

**Hardware.** K3 at bf16 is ~5.6 TB; at FP8 ~2.8 TB. You need a node that can hold it:
8×H200 (1.1 TB) does not; 8×B200/GB200-class or a 16-GPU FP8 node does, or use the
provider's hosted-inference route if they expose router logits (most do not). Practical
path: rent the smallest multi-node config the provider offers for K3 FP8, or profile the
*published INT4/FP8 checkpoint* with vLLM/SGLang on 8 GPUs if a fitting quant exists.
Routing statistics at INT4 track bf16 closely; you are measuring the router, not the
experts. Budget for 1–3 hours including load time. Load time dominates — pick a
provider with the weights already cached on local NVMe.

**Corpus.** Bring the prompts the night will actually run: 30–50 per domain, the real
domains (`code`, `diligence`, …). Coherence is measured *between* calls of the same
domain, so fewer than ~20 per domain gives a noisy number. Generate 128–256 tokens each;
decode statistics stabilise quickly, prefill statistics need the real prompt lengths.

**Steps.**

```bash
pip install torch transformers accelerate      # box only; the core stays stdlib
python - <<'EOF'
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sisyphus.profiling.capture import discover_routers, profile_prompts

name = "moonshotai/Kimi-K3"                     # or the FP8/INT4 checkpoint you rented for
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto",
                                             trust_remote_code=True)

# 1. FIRST: confirm the router modules and what they return.
discover_routers(model, r"(gate|router)")
#    You want ~92 modules named like model.layers.N.mlp.gate returning
#    (indices [T,16] int64, weights [T,16] float). If they return logits [T,896]
#    instead, use extractor="logits" below. Adjust router_regex to what you see;
#    the regex's first group must capture the layer number.

# 2. Profile.
jobs = json.load(open("night_jobs.json"))       # [{"id","domain","prompt"}, ...]
n = profile_prompts(model, tok, jobs, "routes.jsonl.gz", max_new_tokens=192,
                    router_regex=r"^model\.layers\.(\d+)\.mlp\.gate$", extractor="tuple")
print("records:", n)
EOF
```

Copy `routes.jsonl.gz` home. Nothing else on the box is needed.

**Sanity checks before you release the node** (thirty seconds, saves a second rental):

```bash
python -m sisyphus.profiling.analyze routes.jsonl.gz profile.json
```

- `layers` should be 92 and `topk` 16.0. If `layers` is 93, the dense block's gate was
  captured — tighten the regex. If `topk` is not 16, the extractor picked up the wrong
  tensor.
- `experts_seen` near 896 and `unique experts/layer at 4096 tokens` near 896: you
  captured the real router, not a truncated one.
- Gates should sum to ~1 per record (the analyzer's `skippable mass at gate<=0.08`
  should be well below 1.0; if it is ~1.0 the weights are un-normalised logits).

## At home

```bash
python -m sisyphus.profiling.analyze routes.jsonl.gz profile.json    # the report
```

Then in `engine_sim.py`, replace the bracket:

```python
from sisyphus.profiling import RoutingProfile
rm = RoutingModel(**RoutingProfile.from_json("profile.json").bracket())
```

or skip the model entirely and regenerate every table from the routes themselves:

```python
from sisyphus.profiling import LoggedRouting, measure_logged
log = LoggedRouting.from_file("routes.jsonl.gz")
for pol in ("union_lru", "sisyphus", "decay", "skip"):
    print(pol, measure_logged(pol, log, MACHINES["192"], tokens=64))
```

Three numbers decide the project, in this order:

1. **`hot_frac_at_mass["0.75"]` × `hot_set_gb`.** If the hot set plus the 60 GB trunk
   exceeds the RAM tier you are considering, demand eviction cannot help at that tier and
   the RAM purchase argument moves up a tier (or to the tape regime).
2. **`coherence_same_domain` vs `cross_domain`.** If they are close, the convoy premise
   is false: batch by arrival, not by domain, and expect the `coherent=False` row of the
   sensitivity table.
3. **`skippable_mass_at_gate`.** This sets `skip_max_gate` and bounds `skip_budget`.
   Pair it with a quality eval (perplexity or task score on the same prompts with those
   experts zeroed) before trusting any "skip" row for real work — the simulator prices
   skipped mass, it cannot price its effect on answers.

## Cost

Node-hours dominate and depend on what fits K3; the *profiling* itself is minutes of
compute per hundred prompts once the model is loaded. Compare against the alternative:
the 128→192→256 GB RAM decision is a few hundred dollars made on a guess. One log turns
it into a measurement.
