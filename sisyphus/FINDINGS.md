# Sisyphus — findings ledger

*Consolidated record of what has been measured, what has been derived, and what remains
conditional, as of 2026-09-07. Later entries supersede earlier ones. Each finding names its
source: SHARDS (read off the GGUF on D:\), RIG (the machine's own reporting), SIM (the
scheduler simulator in this repo), CALC (arithmetic from measured constants), PRICE (a
dated web source), or ASSUMED (not yet measured).*

---

## 0. Scoreboard — where we stand (2026-09-07, after adversarial review 3, §5.28)

Three kinds of numbers, kept apart: **MEASURED** on this machine, **DERIVED** by the simulator from
measured constants (its B=1 byte count agrees with the one real token to within 19%; overlap, PCIe,
GPU efficiency and state traffic are *not* yet validated), and **ASSUMED**. Canonical throughput
numbers are RESULTS.md's; they use the Windows-measured drive speeds (10.55 GB/s aggregate) — the
Linux engine measured 9.51 (and **9.73 sustained over 8 hours**), so expect ~8% less until the third
drive changes the mix.

**Measured.**

| what | value | where |
|---|---|---|
| K3 runs on this PC, correct greedy output | " Paris. It is" | §5b, `first_token_linux.log` |
| load (mmap, cold) · prompt token · output token, B=1, one drive | 7 min 26 s · 8.6 s · **26.6 s/token** | §5b |
| drives, Windows diskspd · Linux fio io_uring O_DIRECT (ntfs3) | 7.07 + 3.47 = **10.55** · 6.2 + 3.56 = **9.51 GB/s** | §2 |
| **sustained 8 h, both drives, Linux** | **6.19 + 3.54 = 9.73 GB/s average, no sag; 279 TB read, 0 errors; 990 PRO steady 71 °C, SN570 61 °C** | §2 |
| speculation offline, **Moonlight-16B-A3B** draft vs hosted K3, k=4 | acceptance 0.56, **2.32 tok/sweep** | §5.1 |
| GGUF constants: 896/16/2 experts, 92 MoE blocks, 9.68 MB/expert, 62 GB trunk | | §1 |

Derived from those, not measured: bytes per B=1 token ~76 GB (16 routed experts; the 2 shared are in
the trunk); KDA state 217 MB/stream *in bf16* — llama.cpp keeps it in F32 (434 MB), so bf16 storage
is itself the first compression to test (§5.28).

**Derived (RESULTS, 85% overlap, 256/256 jobs, lean states, measured 2.32 tok/sweep).**

| build | streams | s/sweep | tok/s | tokens/night |
|---|---|---|---|---|
| tonight, as-is, plain tape, 32 GB | 74 | 96 | 0.8 | **33K** |
| 32 GB + lean + measured speculation | 140 | 96 | 3.4 | 143K |
| 64 GB (+$350) + lean + measured speculation | 271 | 96 | 6.6 | **272K** |
| 64 GB + 1 Gen4 drive (~$480) | 271 | 60 | 10.6 | **428K** |
| 64 GB + 2 drives (~$610) | 271 | 51 | 12.4 | **498K** |

**Assumed tiers, cumulative on the $610 row** (§8d has every build): better draft (acceptance 0.7,
2.53 tok/sweep) 543K · + tree verification (~3.0 tok/sweep) 644K · + fp8 state (×1.8 streams)
1.16M · + low-rank r=32 state (×3) ~1.5M, capped there by GPU verification compute.

**Reading it.** 500K/night is reached on measured levers only at the $610 build, and just (498K).
$480 needs one assumed lever (better draft or tree gets it to 470–550K). fp8 state is the coin that
doubles everything: physics odds ~50%, and ~35% as a tier reached within a year. Low-rank r=32:
~20%. Confidence that the *engine* beats the plain 33K: ~85%; that speculation transfers into it
at the measured rate: ~60%.

**Daytime (one thread; §5.21/5.26/5.28).** Today 26.6 s/token. Engine, both drives, sparse: ~8 s.
64 GB with a resident requantized trunk and the expert-side levers: **~0.7–0.8 tok/s verified
(1.3 s/token); ~1.1–1.3 with a third drive**. Ghost text from the first second, rendered as a short
window. Extra threads do not add speed (routing unions grow faster than they share). Reading a
4K-token document by day is a near-dense pass: ~80 s (two drives), ~40 s (three) — reads are free
only *inside a night sweep*. A single stream at 4+ tok/s needs a ~200 GB/s memory door (used
8-channel EPYC, ~$3K).

**Reads at night.** GPU-compute-bound and shared with verification: ~3M tokens/night at realistic
GPU efficiency (~6M at spec), in chain-speculation tiers only (tree verification uses the GPU);
~700 four-K-token review jobs a night realistically.

**Per day** (12 h night + 6 h bulk + 6 h interactive): measured $610 ≈ **0.76M**; fp8 ≈ 1.7M;
low-rank ≈ 2.2M.

**Not built yet.** Everything above "measured" is a plan; the streamer (step 2) exists unrun on the
real rig. First real Sisyphus token: after step 4.

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
| model total | **861 GB**, 19 shards | all 19 present on D: (the SN570 1 TB) since 2026-09-03; ~115 GB free |
| work per token | ~208 GFLOP (CALC: 104B active × 2) | ~92 experts, ~116 trunk (of which KDA attention is large) |

## 2. The rig (RIG)

i7-12700KF (8P+4E, no AVX-512) · MSI PRO Z690-A WIFI (4 DIMM, 2 free; M.2: M2_1 CPU Gen4,
M2_2 chipset Gen4, M2_3 chipset Gen3, M2_4 chipset Gen4; chipset uplink ~13 GB/s achievable)
· 2×16 GB DDR5-4800 (confirmed) · RTX 3070 Ti 8 GB, **PCIe 4.0 x16 confirmed** (nvidia-smi link
4/4, width 16/16; ~22 GB/s, ~40 TFLOPS effective) · Samsung 990 PRO 4 TB = **C:** (1,008 GB free),
**measured 7.07 GB/s** sustained unbuffered sequential (diskspd QD32, 2026-09-01; spec 7.4) · WD
SN570 1 TB = **D:**, holds the model, **measured 3.47 GB/s** (spec 3.5) · **both concurrently for
180 s: 7.07 + 3.47 = 10.55 GB/s aggregate, no contention** (`rig_diskspd.json`) · Windows (dev);
production is native Linux (io_uring + O_DIRECT is the same I/O pattern diskspd used).

**Rig operations log (2026-09-03..07).** Dual boot: Ubuntu 26.04.1 LTS (kernel 7.0) installed
alongside Windows on the 990 PRO (300 GB carved by Windows shrink after removing shadow copies and
the pagefile - Windows can only shrink to the last unmovable cluster; the storage service wedged
after the second shrink, "not enough resources", cleared by reboot). Secure Boot and Fast Startup
off. Both NTFS volumes carry the label "Windows", so mounts must be keyed by device, not label
(`tools/linux_setup.sh` bug found and fixed). NVMe device names (nvme0n1/nvme1n1) swap between
boots - key drives by serial or by which one holds the ext4 partition. ntfs3 (in-kernel) mounts
the model drive read-only and supports O_DIRECT at full speed; C: is remounted read-write only
for the hand-off (`tools/linux_handoff.sh` copies Linux results into C:\dev\sisyphus-src for
the Windows-linked Claude session). NVIDIA 595.84 + CUDA 12.4 from the Ubuntu archive; llama.cpp
builds with CUDA on the first try. RLIMIT_MEMLOCK defaults to 8 MiB (mlock of the ring is best
effort; O_DIRECT pins pages per DMA regardless). **Sustained 8-hour two-drive read test, MEASURED 2026-09-07** (`rig_fio_sustained.json`,
`drive_temps.csv`; fio io_uring O_DIRECT 8 MiB QD32, two jobs per drive, ntfs3, both drives at
once for 28,800 s): **990 PRO 6.19 GB/s average (3.09 + 3.10), SN570 3.54 (1.77 + 1.77) — 9.73 GB/s
aggregate sustained for eight hours**, matching the 180 s Linux number (9.51) — no sag over the
night. 279 TB read (178 + 102), zero errors. Per-interval minima show transient dips (990 PRO to
~2.8 GB/s combined, SN570 to ~0.2 GB/s for single intervals) — read-reclaim or housekeeping
moments, not throttling; they matter for the static drive split (sweep = max_i(bytes_i/bw_i),
§5.28) and argue for runtime rebalancing. Temperatures: **990 PRO 33 → 71 °C and steady at 71 for
hours; SN570 28 → 61 °C**. The 990 PRO's throttle threshold is ~80–85 °C and it sits in M2_1 under
the GPU; with the GPU at load all night the margin shrinks — a heatsink/airflow check is a
prerequisite for the night engine, not an option. Planning number for the Linux engine from here:
**9.7 GB/s as-is → 861 GB dense sweep 89 s, 104 s at 85% overlap** (RESULTS' 96 s uses the Windows
10.55; expect ~8% fewer tokens than its rows until the third drive changes the mix).

## 3. The bounds (CALC)

For a target of T tokens/night with lockstep batch B and sweep time s:

- **SSD:** tape regime reads the whole model per step: `s ≥ 861 GB / bandwidth`. As-is
  **measured 10.55 GB/s** → 82 s; +1 Gen4 drive (assumed 6.5) split by bandwidth ~17 GB/s → 51 s;
  +2 drives ~20 GB/s (chipset uplink ~13 GB/s shared by SN570 + new drives) → 43 s; ×1/0.85 for
  realistic overlap.
- **Per-stream memory:** `B ≤ (RAM − OS − buffers) / (217 MB + ctx × 0.028 MB)`. 32 GB →
  ~77; 64 GB → ~219; 128 GB → ~500; 192 GB → ~760. **This is the wall.**
- **Compute:** GPU FLOPs ≈ 40 TFLOPS → 190 tokens/s ceiling for *any* token; irrelevant for
  output tokens (SSD/state bind first), binding for input tokens (§6).
- **PCIe:** 860 GB / 22 GB/s = 39 s < SSD time; not binding on this rig.

Tokens/night ≈ 43,200 × B × (tokens per stream per sweep) / s, minus prefill.

## 4. K3 on this rig — honest numbers (CALC, 85% overlap, 256/256 jobs) — SUPERSEDED by §0/§8d (winsat-era drive speed; kept for the record)

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

**5.1 Replayable speculative decoding (see MEASURED note below; formula corrected 5.28: tokens/sweep = Σ_{i=0..k} aⁱ = 2.53 at a=0.7, k=4, not 2.77).** In tape mode the sweep already touches every
expert, so verifying k drafted tokens per stream per sweep multiplies tokens per sweep by
`1 + a + a² + … + a^k` (2.53 at a=0.7, k=4; an earlier draft wrote 2.77) at no SSD cost; GPU has the FLOPs (219 × 4 × 208 GFLOP ≈ 5 s per
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

**Stack, modelled (`sisyphus.tape.rig_levers_table`, MEASURED drive speeds, 64 GB, 256/256
jobs, 85% overlap; acceptance per sweep is geometric, Σ_{i=0..4} aⁱ = 2.53 at a=0.7, k=4 — the "2.77" labels below are what the simulator computed as 2.53):**

| lever stack | drives as-is (10.55 GB/s, $350: RAM only) | +1 drive (~17 GB/s, $480) | +2 drives (~20 GB/s, $610) |
|---|---|---|---|
| plain tape | 95K | 151K | 177K |
| + 5.3 lean RAM + 6 GB VRAM for states | 127K | 202K | 237K |
| + 5.4 sparse sweep (no speculation) | 164K | 261K | 305K |
| 5.3 + 5.1 replayable speculation, **MEASURED 2.32 tok/sweep** | **272K** | **428K** | **498K** |
| same, assumed 70% (2.77 tok/sweep) | 296K | 465K | 541K |
| same with naive rollback (2 state copies) | 165K | 262K | 306K |
| 5.3 + 5.1 (measured) + 5.5 fp8 state (ASSUMED) | 475K | 735K | 849K |

At 32 GB (nothing bought) the same stack is 33K / 66K / 143K (speculation) / 254K (+fp8).
Input tokens (5.2), 64 GB + 1 drive, 2048/8 jobs: ~6.1 M/night.

**What the measured drives changed (2026-09-01, twice):** winsat's 64 KB low-queue-depth
test said 5.55 + 3.0 = 8.5 GB/s and the tables were built on that. diskspd at the engine's
real pattern (8 MB reads, QD32, unbuffered, both drives at once) says **7.07 + 3.47 = 10.55
GB/s**, i.e. both drives at spec with zero contention between the CPU M.2 slot and the chipset.
Every sweep is 24% faster than the winsat-based tables. Consequence for the $500 target: with
measured acceptance, **the $610 build (RAM + 2 drives) lands at 498K with no fp8 and no better
draft**; the $480 build lands at 428K and reaches 500K with any one of: draft acceptance
0.56 → ~0.66 (top-3 ceiling is 0.71), fp8 state, or a faster new drive than the 6.5 GB/s
assumed. Regenerate with `python -m sisyphus.results`; the tables live in RESULTS.md.

**5.1 MEASURED (2026-09-01, `tools/acceptance_test.py`, Moonlight-16B-A3B-Instruct as the
same-tokenizer draft, teacher-forced against hosted K3's greedy output):**

| prompt set | prompts | tokens | acceptance | tokens/sweep k=4 (empirical) | K3 token in draft top-3 / top-8 |
|---|---|---|---|---|---|
| 2 open-ended (code audit, concept explanation) | 2 | 3,940 | 0.27 | 1.39 | — |
| **50 job-shaped** (denial/invoice/audit/scoring/transcript, `tools/make_prompts.py`) | 38 | 4,430 | **0.56** | **2.32** | 0.71 / 0.80 |

The plan assumed 0.70 / 2.53 (mislabelled 2.77). On the jobs the night would actually run, speculation is
**84% of the assumed strength**, and matches are positively correlated (2.32 empirical vs 2.05
from the flat-rate formula). The top-3/top-8 ceiling (0.71/0.80) says a better-calibrated
draft (Kimi Linear 48B-A3B, same tokenizer) or multi-candidate verification could approach
the original 0.70. Open-ended generation is a different regime (0.27) — keep speculation for
job-shaped work. 12 of 50 prompts returned empty K3 output even with reasoning excluded
(likely max_tokens on JSON-only answers with hidden reasoning); the script counts only
non-empty targets.

**Stack with the MEASURED acceptance** (64 GB, 256/256, 85% overlap):

| lever stack | +1 drive ($480) | +2 drives ($610) |
|---|---|---|
| lean + sparse (no speculation) | 216K | 283K |
| lean + speculation k=4 @ 2.32 tok/sweep | **356K** | **463K** |
| lean + speculation + fp8 state (assumed) | 617K | 793K |

With the diskspd drive numbers (below), 500K on the $610 build is speculation alone (498K);
on the $480 build it is speculation plus one more lever. Input-token path unchanged.

**Sustained streaming, first pass (2026-09-01, `tools/stream_bench.py`, Windows, buffered
reads, 4 processes per drive, 300 s, both drives concurrently):** D: 2.25 GB/s, C: 2.22 GB/s,
**aggregate 4.48 GB/s → 192 s sweep**. Both drives peaked ~3 GB/s each for the first 20 s
then dropped together to an identical 2.2 — the signature of Windows' cache manager
(buffered reads copy through the page cache; once RAM fills, eviction churn caps the
machine), not of the NAND (winsat unbuffered got 5.55 on C: alone). Consequence: **the engine
must use unbuffered high-queue-depth I/O (O_DIRECT + io_uring on Linux; FILE_FLAG_NO_BUFFERING
on Windows); a naive implementation forfeits half the bandwidth.**

**Sustained streaming, definitive (2026-09-01, `tools/diskspd_bench.ps1`: Microsoft diskspd,
8 MB sequential reads, QD32 × 2 threads per file, `-Su` unbuffered, one model shard per drive):**
C: alone 60 s **7.07 GB/s**; D: alone 60 s **3.52 GB/s**; both concurrently 180 s **C: 7.07 +
D: 3.47 = 10.55 GB/s aggregate → 82 s sweep**. Concurrent equals solo, so the CPU-attached
M2_1 and the chipset-attached SN570 do not share a bottleneck, and neither drive thermally
throttled over three minutes of full-rate reads (a 12-hour night still needs checking on
Linux, but 990 PRO 4 TB throttling under sustained *reads* is not a known problem). This
replaces the winsat 8.5 GB/s everywhere; the buffered 4.48 GB/s stands as the cost of doing
I/O the wrong way.

**Linux, the engine's platform (2026-09-06, Ubuntu 26.04.1, kernel 7.0, `tools/linux_setup.sh`,
fio io_uring + O_DIRECT, 8 MB, QD32 x 2 jobs, shards read through the in-kernel ntfs3 driver):**
SN570 alone **3.56 GB/s** (= Windows); 990 PRO alone **6.2 GB/s** (Windows: 7.07); both
concurrently **9.51 GB/s -> 90 s sweep** (Windows: 10.55). Direct I/O works on ntfs3, so the
engine can read the model where it sits. The 990 PRO's 12% gap to Windows is either ntfs3 overhead
or the placement of the two test shards on C:; the streamer (raw offsets, own queue depth) is
the tiebreaker, and the ext4 partition is the fallback. The tape model keeps 10.55 until then;
9.51 is the conservative figure.

## 5b. FIRST TOKENS — K3 ran on this machine (MEASURED, 2026-09-06)

Stock llama.cpp (upstream, CUDA build, `-ngl 0`, mmap from D: through ntfs3, batch 1, greedy),
prompt "The capital of France is" → **" Paris. It is"**. Timings: model load 7 min 26 s (mapping
19 shards); prompt eval **8.6 s/token** over 5 tokens; decode **26.6 s/token** (0.04 tok/s); 32 GB
RAM. `first_token_linux.log`. Correctness oracle for the engine: every later step must reproduce
these greedy tokens.

**What the 26.6 s says (CALC).** A full pass over 861 GB from D: at 3.5 GB/s would take ~250 s;
the token took 27 s because at batch 1 llama.cpp touches only the trunk (62 GB) plus the 18 routed
experts per layer (~16 GB), ~80–90 GB per token, i.e. it runs the sparse regime by default. The
5 prompt tokens shared one trunk read and a union of ~80 experts/layer, 43 s total — the same
arithmetic. Two consequences for the design: (1) the "861 GB per token" tape cost is the price of
batch, not a floor — the sweep must be worth it, and it is only worth it when the batch's expert
union approaches all 896 (B ≳ 150 at top-16), exactly the regime 5.4 sparse sweep interpolates;
(2) at B=1 the naive path is ~1,300 tokens/night on this box, so every engine milestone is
measured against a real number, not a model. Per-token bytes at batch B are
`62 GB + 92 x |union_B| x 9.7 MB`, from ~80 GB (B=1) to 861 GB (B → all experts).

**The simulator agrees (CALC vs MEASURED).** `TapeOptions(sparse=True).sweep_fraction(1)` gives
76 GB per step → 21.6 s at the SN570's 3.5 GB/s; measured 26.6 s, i.e. the mmap page-fault path
runs at ~81% of sequential — the gap the streamer exists to close. Now a unit test
(`test_batch1_sparse_step_matches_first_token`). The same curve says a 32-stream step reads
~237 GB (25 s at 9.5 GB/s → ~55K tokens/night on today's 32 GB box, no other levers) and a
290-stream step ~664 GB (75% of experts) under expected skew; the "plain tape" rows, which read
all 861 GB, are the naive streamer's cost, not the engine's. With k=4 speculation the union
reaches ~97%, which is why 5.1 and 5.4 are alternatives.

### 5.x Second design pass (2026-09-03): traversal distance

Distance per token = `bytes per sweep / (streams x tokens per sweep)`. Everything below moves one of
the three terms. Labels: KNOWN = established technique applied here; NEW = not found elsewhere;
ASSUMED = plausible, unmeasured; the test that settles each is named.

**5.7 Tree verification (KNOWN: Medusa/EAGLE/SpecInfer; tokens per sweep).** Verify several draft
branches per position instead of one chain. Effective acceptance becomes the top-k rate we already
measured (top-3 0.71, top-8 0.80) instead of top-1 0.56: ~2.6-3.0 tokens/sweep at k=4-6 vs 2.32.
Costs compute the sweep is not using (budget at 60 s sweep, 40 TFLOPS: ~11K verify tokens per sweep,
i.e. ~40 per stream at 270 streams; tighter at 1,000+ streams). Compatible with the KDA state because
branch outputs are computed from the starting state with the chunked delta-rule (no per-branch state
copy); the accepted path is then replayed as in 5.1. +12-30% on every row.

**5.8 Low-rank KDA state (NEW here; ASSUMED).** Per head the state is a 128x128 matrix built by
gated rank-1 updates with decay; old directions are damped every step, so its effective rank is
plausibly far below 128. Store each head's state as a rank-r factorization A*B^T (128 x r each):
bf16 factors at r=32 halve it, fp8 factors at r=32 give 4x, fp8 at r=16 give 8x. Re-expand on the
GPU before the step, re-truncate after (QR + small SVD per head: a few TFLOP per step across all
streams, well under 0.1 s). ENGINE_DESIGN.md section 6c has the implementation. Because
tokens/night scale linearly with streams until the drives bind, this is the largest single lever
left: **64 GB + 1 drive, measured speculation: bf16 428K -> fp8 735K -> 4x (fp8, r=32) 1.14M -> 8x (fp8, r=16) 1.58M**
(drives as-is: 272K -> 475K -> 756K -> 1.07M). Test: same harness as 5.5 on Kimi Linear 48B-A3B --
bf16 state vs rank-32/64 (and fp8 factors) over 256-token generations; pass = token agreement
>99%, perplexity delta <1%. If the state is not low-rank the test says so in an afternoon.

*5.8b Why the rank question is a coin flip (2026-09-07).* KDA is a gated delta rule: per head,
`S_t = S_{t-1}·diag(α_t) + β_t·(v_t − S_{t-1}k_t)·k_tᵀ` — a 128×128 fast-weight memory that stores
key→value associations. Each step adds a rank-1 term, the delta term first *erases* whatever the
current key already retrieves (so re-writing a key does not pile up rank), and the per-channel
gates α damp old directions. Case for low rank: decay means only recent or repeatedly-refreshed
associations survive at full strength, so the spectrum should fall off; the model cannot usefully
hold 128 independent associations per head across 96 heads × 69 layers. Case against: the delta
rule writes the component *orthogonal* to what is already stored, which is exactly the update
that spreads energy across new directions; a memory the model trained to use at full rank will
use it; and truncating to rank r *every step* is a repeated lossy projection whose error compounds
over a 256-token generation the way fp8 rounding does, only structurally — the state drifts
toward the span of what was recently written. No one has published a spectrum for KDA states (or
DeltaNet/GLA states at this size); the estimate is reasoning, not data. What "works" looks like:
singular values per head that carry ≥ 99% of the energy in the top 32 (≥ 97% in the top 16),
consistently across heads and layers, and a truncated run whose tokens match bf16 at > 99% with
< 1% perplexity delta. What "fails" looks like: a flat spectrum (energy spread over 60–100
directions), or a spectrum that is fine at rank 32 for most heads but a few heads per layer that
need full rank — in which case the mixed fix is per-head ranks from the measured table (store 128
for the few, 16–32 for the rest; average maybe 3–4× instead of 4–8×). Cost of the storage scheme:
factors A, B (128 × r) re-expanded before the block's step and re-truncated after via a rank-(r+1)
QR/SVD per head — the pack/unpack seam already in the design; ~1 GFLOP per stream per block. The
measurement is the first experiment after the 64 GB arrives: dump Kimi-Linear-48B states over ~50
prompts of 512–2,048 tokens, SVD every head, plot the energy curves.

*5.8c If the state is not low-rank: the ways around, priced (2026-09-07, CALC).* The wall is
"states must sit in RAM for the whole sweep". Routes that do not need the rank result:
- **fp8/int8 state (5.5), ×1.8 streams** — a separate, better-odds bet (~70%): per-row-scaled
  8-bit is standard on recurrent states elsewhere; drift is the risk, the 48B agreement test the gate.
- **More tokens per crossing (A term):** tree verification 2.32 → 3.0 tok/sweep (×1.3), a better
  draft (K3's native MTP head if present, or a tuned EAGLE head: acceptance 0.56 → 0.7, ×1.2).
  Independent of RAM entirely.
- **Time (U):** 18 h of batch per day, ×1.5. **Second drive:** ×1.18 to the 20 GB/s cap.
- **Money:** 128 GB (4 × 32 GB DDR5, ~$700–1,400 at Sep-2026 prices) → ×1.85 streams.
- **States on the SSD (rejected, with numbers).** Treating state as part of the tape lets B exceed
  RAM: each extra stream costs 217 MB read + 217 MB *written* per sweep. +230 streams = 50 GB
  written per sweep × ~800 sweeps/night = **40 TB of writes per night**; a 1 TB Gen4 drive (600 TBW)
  lasts 15 nights → ~$9/night of drive for +85% tokens, more than the hosted price of the tokens
  gained; a 990 PRO 4 TB (2,400 TBW) lasts 60 nights. Reads are free, writes are consumables;
  the state wall is really a *write-endurance* wall once RAM is full. Not viable.
- **States in VRAM:** already counted (6 GB → ~28 streams, lever 5.3).
Stacked without low-rank (figures revised 5.28): 64 GB + 2 drives, measured spec 498K/night → tree
(2.72) 585K → fp8 1.05M/night → 18 h/day **1.6M/day**. If fp8 also fails: 498K → tree 585K → 18 h
**0.9M/day**; adding 128 GB
RAM instead of fp8 gives ~1.9M/day for ~$1,400 more. So the 1.5M-class number does not depend
on low-rank alone — it depends on *either* 8-bit state, *or* RAM money, plus tree verification and
running the doors 18 hours.

**5.9 Shared-prefix state forking (KNOWN: prefix caching; input tokens).** Jobs of one type share
their instructions; the KDA state and MLA KV after the shared prefix are identical for every
stream, so compute them once and fork. Prefill compute (the binding term for input tokens) drops
by the shared fraction: at 1,500 shared of 2,048 tokens, ~4x more unique input per night. No
effect on sweep bytes or state memory (each stream diverges and needs its own state afterwards).

**5.10 Lossless entropy coding of expert blocks (KNOWN: DFloat11-style; bytes).** IQ2_XS/IQ3_XXS
blocks are codebook indices plus scales; rANS-coding them buys an estimated 3-8% fewer bytes per
sweep, decoded on the GPU at far above stream rate. Small, free of quality risk, last priority.

**5.11 Whole-model IQ1_S as a measured trade (bytes; the 5.6 question at model scale).** The
UD-IQ1_S image is 594 GB (-31% sweep bytes, so 1.45x on every SSD-bound row). Community reports
degradation; we can measure it on OUR jobs for free: the hosted-K3 greedy targets already cached
by `tools/acceptance_test.py` are a quality oracle -- teacher-force the local Q2_K_XL and IQ1_S
models on the same targets (one prefill sweep per prompt, ~5-10 min each via llama.cpp mmap, an
overnight job) and compare agreement with hosted K3. If IQ1_S agrees within a point or two of
Q2_K_XL, the cheaper tape wins outright. Requires a second 594 GB download (D: cannot hold both;
C:'s ext4 share can).

**Things that look like levers and are not (checked this pass).** Trunk residency: the 62 GB trunk
is 7% of the sweep, but every GB kept in RAM displaces ~4.6 streams (1.7% of throughput) to save
0.12% of bytes -- stream it. State recompute from token history: the state IS a function of the
tokens, but recomputing 290 streams x 300 tokens per step is ~450 s of prefill against an 82 s
sweep. Hot-expert VRAM cache: 4-5 GB of idle VRAM holds ~0.5% of expert bytes; routing skew at
batch 270 is too flat for it to matter (rent-to-profile harness would confirm). Cross-step expert
reuse: the RAM is the state wall; there is no room for an expert cache, by construction.

**Combined ceiling if 5.5 + 5.8 hold at 8x (ASSUMED) with 5.7 (3.0 tok/sweep), 64 GB:** drives
as-is ~1.2M/night, +1 drive ~1.76M, +2 drives ~1.98M -- at which point the GPU's PCIe link
(22 GB/s) and compute (~1 PFLOP of verification per sweep) are within 2x of binding and the state
wall stops being the story. Every one of those numbers depends on two experiments on a 48B model
that need the RAM to arrive.

### 5.12 Single-stream latency ("daily driver" mode): what is and is not reachable (2026-09-07, CALC)

Throughput is the night's problem; latency is the day's. For one stream, seconds per token =
active bytes per token / bandwidth of the tier that holds them. Today: ~78 GB (62 GB trunk read
every token + 18 experts x 92 layers x 9.7 MB) from SSD at 10 GB/s = ~8 s (26.6 s measured via mmap
from one drive). Datacenters hit 3 ms because the same bytes sit in 25-60 TB/s HBM; the gap is the
tier, not compute. Two knobs exist: fewer active bytes, or a faster tier for them.

**Fewer active bytes (software, this rig):**
- *Trunk requantization.* The UD image keeps attention/shared/embeddings at high precision; 62 GB
  per token is the single largest cost. IQ2/Q3 for the trunk -> ~20-25 GB (quality cost measurable
  with the quant oracle, 5.11). Then the trunk fits in 64 GB RAM.
- *Expert working-set cache.* Expert usage within one conversation is skewed by topic; the V3
  ResidencyCache is exactly this at single stream. If a 35-40 GB RAM cache (~4,000 experts) hits
  70-90% after warm-up, SSD bytes per token fall from 16 GB to 2-5 GB. Measurable NOW with the
  rent-to-profile harness on llama.cpp routing traces; the single most valuable daytime experiment.
- *Fewer active experts.* top-8 instead of top-16 at inference, or gate-mass thresholding (drop
  picks with tiny gates): expert bytes halve. Quality cost per task measurable with the oracle.
  This is the "page with one word" inefficiency: every expert costs the same 9.7 MB to read whether
  its gate weight is 0.30 or 0.01, and every token costs the same 78 GB whether it is a hard
  reasoning step or a comma. Gate-mass skipping (V3's cold_policy) attacks the first; speculation
  (a cheap draft handles the commas, K3 verifies several at once) attacks the second; layer skipping
  / early exit for easy tokens would attack it too but K3 was not trained for it (quality unknown).
- *Speculation helps a single stream only through the trunk.* Verifying k drafts in one pass reads
  the trunk ONCE plus the union of the drafts' experts: at k=4 with today's 62 GB trunk that is
  ~62 + ~60 = ~120 GB for ~2.3 accepted tokens = ~52 GB/token vs 78 plain (-33%). Once the trunk is
  requantized to ~25 GB the saving shrinks to ~5%, because the expert bytes scale with k while the
  trunk no longer dominates. (Corrected 2026-09-07: an earlier draft of this note said speculation
  never helps a single stream; it neglected that the trunk read is shared across the k drafts.)
- *Faster/more SSDs, and the platform ceiling.* Gen5 drives do ~14 GB/s each, but on this board
  the CPU M.2 slot is Gen4 (7 GB/s) and the chipset slots share a ~13 GB/s uplink; the only Gen5
  lanes are the GPU's x16 slot. Splitting it (x8 GPU + x8 for two Gen5 drives, if the BIOS
  bifurcates) caps the GPU link at ~13 GB/s, and every streamed byte must cross the GPU link. Net:
  the 12700K's lanes bound useful streaming at ~20-25 GB/s however the drives are arranged: ~3.5 s
  per single-stream token, ~40 s dense sweep, ~2.5x today for ~$500-800 of Gen5 drives + adapter.
  [WITHDRAWN 5.21: the GPU at x8 caps the sweep at 12.5 GB/s; net negative] Worth it for the night mode (it is the drives row in the tables); not a route to interactive speed.
- *Low-rank expert deltas (the only genuinely new-to-this-project idea, ASSUMED).* Experts within a
  layer share structure; a 2025 line of work decomposes MoE experts as a shared per-layer basis plus
  per-expert low-rank deltas at 30-60% compression with small loss. If K3's 896 experts per layer
  compress to a 9.7 MB shared part + ~2 MB deltas, active expert bytes drop 16 -> ~4 GB per token
  AND the whole expert set drops 799 -> ~170 GB: the entire model would fit in 192 GB of RAM
  (4 x 48 GB DDR5 on this board, ~$1,400). Test offline on Kimi Linear 48B (same expert structure):
  SVD the expert-minus-mean matrices per layer and measure the spectrum and downstream agreement.
  Quality, calibration effort and K3-specific behavior all unknown; the decomposition itself is a
  few GPU-days of SVD.

**Faster tier for the bytes:** RAM. Dual-channel DDR5 on Z690 is ~60 GB/s practical; that is the
ceiling of the platform, and the CPU must then do the math (~80 GFLOP/token, fine). Everything in
RAM at Q2-class trunk + cached/compressed experts: 20-40 GB per token / 60 GB/s = 0.35-0.7 s.

**Reachable ceiling on THIS rig (64-192 GB RAM, no new drives/GPU):** ~1-2.5 tok/s, i.e. a
300-token answer in 2-5 minutes. Steps: trunk requant (8 s -> ~3 s), expert cache (-> ~1.5 s),
top-8 (-> ~1 s), low-rank deltas + 192 GB (-> ~0.4-0.7 s). Each is a constant factor; the last is
speculative. Not interactive by API standards, but usable for "ask, get coffee".

**The cheap hardware jump, if design alone is not enough:** memory bandwidth per dollar is best in
USED server DDR4. An EPYC 7002/7003 (8-channel DDR4-3200, ~200 GB/s) with 512 GB-1 TB of used RDIMM
(~$1/GB) is a ~$2,500-3,500 machine that holds all of Q2 K3 in RAM: 40 GB/token at 200 GB/s =
0.2 s -> ~5 tok/s, ~8 tok/s with the byte reductions above. Same engine, same model, 4x the
bandwidth of the desktop for a fifth of the $15K workstation. It also runs the night mode with
NVMe drives added. This, not the Mac or the new EPYC, is the hot rod that matches the budget.

**Daytime experience (2026-09-07; ENGINE_DESIGN 7c).** A single K3 stream cannot reach reading
speed (~5 tok/s) on this PC, but the reader can be kept fed at reading speed: (1) draft-first
streaming - the resident draft model's text appears instantly as ghost text and K3 verifies and
corrects it behind the cursor (exact final output, zero initial wait); (2) a front-loaded buffer
for verified-only viewing (moves the wait to the front, guarantees no stalls); (3) multiplexed
threads - [WITHDRAWN 5.26: true only with a streamed trunk; with the trunk resident threads do not
add aggregate speed] several conversations advance per step. Acceptance test:
<= 500-token architect answers in < 5 min, first visible text < 1 s, >= 4 verified tok/s aggregate.

### 5.13 Where the bandwidth is: memory tiers, "doors vs shelves", and the state of PIM (2026-09-07)

**The one equation, restated.** Single-stream seconds per token = active bytes per token /
bandwidth of the tier holding them. Compute is irrelevant for one stream: ~160 GFLOP per token is
4 ms on the 3070 Ti and <1 ms on an H100; both are 0.05% of the token. The 78 GB fetch is the token.

| tier holding the active weights | bandwidth | 78 GB takes |
|---|---|---|
| NVMe SSD (this rig, 2 drives) | ~10 GB/s | 8 s |
| desktop DDR5, 2 channels | ~60 GB/s | 1.3 s |
| server DDR4/DDR5, 8-12 channels | 200-500 GB/s | 0.2-0.4 s |
| one consumer GPU's VRAM (capacity too small to matter) | ~900 GB/s | 0.09 s |
| 8 datacenter GPUs, HBM, combined | 25-60 TB/s | 2-3 ms |

Datacenter tokens are ~3,000x faster than this rig's for exactly one reason: the same bytes sit
in memory ~3,000x faster. Batching is the same trick on both: one fetch of the weights serves many
tokens; a datacenter node serves 50-200 users per step (each user's share ~1% of a $400K node),
Sisyphus serves ~270 streams per sweep on a 10 GB/s tier.

**Doors vs shelves.** Capacity (cells) is cheap; bandwidth (pins, traces, channels, TSVs,
interposers) is what costs. 64 GB of DDR5 holds 512 Gbit but exits through 2 x 64 wires at 4.8
GT/s = 77 GB/s; a 4 TB SSD holds 32 Tbit but exits through 8 NAND channels (~1.6 GB/s each) into a
7 GB/s link. Channel count is fixed by the CPU package and socket; software cannot add one.
Memory-bandwidth-per-dollar today: used server DDR4 RDIMM ~$1/GB at 200 GB/s per 8-channel
board; new DDR5 ~$5-7/GB at 60 (desktop) to 500 GB/s (12-channel); HBM ~$20-40/GB at 3-8 TB/s per
stack, sold only attached to $30K accelerators; NAND ~$0.13/GB at 1.6 GB/s per channel. The
transfers themselves are already ~90% efficient (990 PRO: 7.07 of 7.9 GB/s theoretical); there
is no hidden multiple in "managing the bits better". The remaining software wins are in the
bits themselves (quantization, low-rank experts, gate skipping, trunk requant): compression, which
has ~1.5-2x left before quality collapses (~2 bits/weight now; ~1-1.5 is the floor).

**Processing-in-memory / computational storage (researched 2026-09-07).** The idea "multiply
where the weights already are" is real (Samsung HBM-PIM 2021, SK hynix GDDR6-AiM 2022, UPMEM
DIMMs) and unavailable in practice: memory-process logic is slow and sparse (PIM dies halved
capacity), compute heats the memory that hates heat, no JEDEC/NVMe command exists for it, the
business case loses to selling HBM. The only purchasable computational-storage drive, the
Samsung/Xilinx SmartSSD (~$900, 3.84 TB, Kintex KU15P FPGA), reads its own flash at ~3.3 GB/s -
SLOWER than the 990 PRO over plain PCIe - because a consumer/enterprise SSD's internal NAND
bandwidth is only ~1.5-2x its external link. High Bandwidth Flash (SK hynix/SanDisk, HBM-style
stacked flash, hundreds of GB/s) is the device that would change the daytime problem; production
~2030, datacenter first. Prediction (speculative decoding, EAGLE-style hidden-state prediction,
routing prediction/prefetch) is the software counterpart: it turns one-token-per-read into a
few-tokens-per-read (2-3x), bounded because the verifier must still read the path.

### 5.14 The software equivalent of compute-in-memory: touch fewer weights (2026-09-07, CALC/ASSUMED — 5.28: SwiGLU experts give ~25–50% skippable rows, not 80–90%; effective byte factor ~0.75, not 0.4; single-stream only)

Analog CIM's payoff is that no weight bits cross a door. Software cannot change what a read does,
so the only equivalent is to make each token *touch* fewer weights, at three granularities:
experts (routing, already exploited), **neurons within a weight matrix** (new to this ledger), and
tokens per touch (speculation, 5.1/5.7). Stacked estimate at the end.

**Contextual sparsity (neuron-level; KNOWN: Deja Vu 2023, PowerInfer 2024; ASSUMED for K3).** Within
an FFN/expert, most neurons produce ~0 for a given input; a tiny per-layer predictor (a two-layer
MLP on the layer input, run before the weights are read) names the rows that will matter, and only
those rows are read. Reported on dense models: 80-90% of FFN neurons and ~50% of attention heads
skippable with negligible loss. MoE experts are already specialized, so expect less within an
expert (50-70%?); attention-head sparsity applies to the 62 GB trunk, which is the fat term.
Requirements: (a) predictors trained per layer from K3 activations (the night mode generates them);
(b) a storage layout that exposes individual neuron rows (gate/up rows contiguous; down stored
transposed) at 4 KiB granularity; (c) accepting the random-read regime: 4 KiB-16 KiB reads at QD32
run the 990 PRO at ~2.5-4 GB/s instead of 7, so a 4x byte saving nets ~2x wall time, unless the
hot rows live in RAM (they cluster by conversation topic, like experts do - the same working-set
cache of 5.12 applies at row granularity). Test: record activations for one layer on Kimi Linear
48B, measure the fraction of |neuron output| mass in the top-k rows and predictor recall.

**Native multi-token head (CHECK the GGUF).** DeepSeek-V3-family and Kimi-K2 shipped an MTP
("nextn") module trained with the model; if K3's GGUF carries one, it is a draft with ~0.8-0.9
acceptance for free, no separate draft model, no tokenizer constraint, and it raises every
speculation row. Look for `nextn`/`mtp` tensors or `*.nextn_predict_layers` in shard 1's metadata.

**Ranked, what is left to squeeze from bits per token (single stream, this rig):** trunk requant
62 -> ~25 GB; attention-head sparsity on the trunk 25 -> ~15; expert-neuron sparsity 16 -> ~4-8;
speculation k=4 at 0.7 (or native MTP): trunk shared across ~2.5 tokens. Per-token bytes ~78 -> ~10-14
GB; at 10 GB/s (two drives) ~1-1.4 s/token; three drives ~0.7 s; plus RAM-resident hot rows ~0.5 s
-> ~2 tok/s. Every term is a measured-or-measurable constant factor; the product is roughly 5-8x on
today's 8 s, which is the honest ceiling of "CIM by software" on this hardware. The remaining
gap to reading speed is the memory door, and the used-EPYC hot rod is what closes it.

**Not possible from software (checked so it is not re-asked):** multi-row activation / analog
readout on commodity DRAM or NAND (Ambit/ComputeDRAM need a custom memory controller issuing
out-of-spec timings; the CPU's controller is locked to JEDEC; BIOS timing knobs are global, not
per-command); any access to the NAND array behind SSD firmware; precomputing input-dependent
results (the token does not exist yet); reading the model in less than one pass per token
(only compression, prediction and batching change the count).

### 5.15 "Make the compute stupid easy" — why it does not buy speed, and the one thing that does (2026-09-07, CALC)

Question asked: can a low-level software structure make the arithmetic so trivial that the box
runs *as if* it had analog compute-in-memory? Answer, with the arithmetic:

- Per B=1 token today the time splits **~8,000 ms of moving bytes, ~4 ms of computing them**
  (78 GB at 10 GB/s; 208 GFLOP at ~40 TFLOPS). Compute at zero cost saves 4 ms in 8,004. CIM is
  not fast because its multiplies are cheap; it is fast because **the weights never travel** — the
  arithmetic happens in the cells that hold them. "As-if-CIM speed" therefore means "the weights
  do not travel", and the only place a weight can sit without travelling is a tier that can compute
  on it: VRAM (~900 GB/s), then DRAM (~60 GB/s here). Software cannot change which chip a byte is in.
- Lookup-table inference (LUT-GEMM, T-MAC: weights become codebook indices, multiplies become
  table reads) makes compute trivial, which is exactly the part that is already free; the index
  bytes still have to be read from the drive. Same for any "marker"/"pre-summed" layout: a sum
  that depends on the input token cannot be stored before the token exists (§5.14).
- What *does* behave like CIM at small scale is a **hot-expert cache** (PowerInfer's hot/cold
  split applied to experts): routing is skewed, so a fraction of the 82,432 experts (896 × 92)
  takes a disproportionate share of activations. The 20 GB of RAM/VRAM not used by states at B=1
  holds ~2,000 experts (2.4%). If the top 2.4% catches 20–30% of routing decisions (typical skew
  in DeepSeek-family routers; **ASSUMED for K3**, measurable by logging `ffn_gate_inp` argmax over
  a few hundred local tokens), expert bytes per token fall by that share for free — those weights
  "live where the compute is", which is the CIM property, for the experts that are hot. Stacks
  with §5.14. At night (B ≥ 200) the cache is irrelevant: every expert is read anyway.
- Ceiling restated: bytes per token ÷ door bandwidth. Software reaches ~10–14 GB/token; the
  hot cache may take that to ~8–11. Below that only more doors (drives, DRAM channels) exist.

### 5.16 The door, written as bits: the information-theoretic frame (2026-09-07, CALC)

A door is a Shannon channel of capacity C bits/s. Two parties: the shelf holds the weights W, the
compute side holds the input x, both want f(W, x) = the next hidden state. Communication needed per
step is **min(|W_needed|, |x| + |f|)** — either the ingredients cross (78 GB per token at B=1) or the
question and the answer cross (x ≈ 7,168 values ≈ 14 KB per block, f the same size: ~2.6 MB per
token in total). The second branch is only available if there is a computer on the shelf side.
That is the entire meaning of CIM, PIM and computational storage translated into 1s and 0s: the
door carries answers (KB) instead of ingredients (GB), a ratio of ~10^4–10^5 per token. No
software running on the compute side can pick the cheap branch, because the shelf (NAND behind
firmware, DRAM behind a JEDEC controller) has no computer we can program. The SmartSSD (§5.13) is
that branch made real — at 3.3 GB/s of internal flash bandwidth, slower than the door it replaces.

Given the ingredients must cross, the floor per token is H(W_needed | what the compute side
already holds) / C. Every software lever in this ledger is one term of that expression:
compression toward the entropy of the weights (5.6 low-bpw tail, 5.10 rANS; the 2-bit quant is
already close, so ~1.3–1.5× remains), shrinking W_needed (5.4 sparse sweep, 5.14 row masking),
raising what the compute side already holds (5.15 hot-expert cache; RAM/VRAM as the conditioning
set), and amortising one crossing over more tokens (5.1/5.7 speculation, batching). Multiplied
out these give the ~7–10× from 78 GB that §5.14–5.15 estimate. C itself is the only term
software does not touch.

**Compression, asked directly (2026-09-07).** A hash is one-way: it maps many inputs to one small
output and no computation recovers the input, so "hash then unhash" is not available; what is
meant is compression, and compression is bounded by the entropy of what is sent. The weights are
already 2-bit quantized (UD-Q2_K_XL, ~2.7 bits/weight with scales), and the quantized codes are
near-uniform, so lossless coding recovers ~10–20% (5.10 rANS, measurable offline on a shard).
Going below that is lossy — a lower-bpw quant — which is a quality trade the quant oracle (5.11)
exists to price; IQ1_S at ~1.6 bpw is the floor anyone has shipped with the model still coherent.
Decompression is not the constraint: rANS decode on the 3070 Ti runs well above 10 GB/s. In
Kolmogorov terms the shortest program that produces the weights is the training run itself; the
bits look like noise to any coder that does not include it. "Compression relative to what the
receiver already has" is the one remaining lever, and it is the hot-expert cache (5.15).

### 5.17 Why weights degrade under compression, and how far the field can push it (2026-09-07, KNOWN/CALC)

Two different entropies are in play. **Lossless** (5.16): once weights are symbols, the code length
cannot go below the entropy of the symbol stream; the 2-bit codes are near-uniform, so ~10–20% is
all that remains. **Lossy** is governed by rate–distortion theory: for a source with variance σ²
and a tolerated mean-squared error D, the minimum rate is R(D) = ½·log₂(σ²/D) bits per weight
(Gaussian bound). Quantizing is choosing a point on that curve; "deprecation" is D showing up in
the output. Why it shows: rounding error is noise added to every weight; the network is a 93-deep
composition so the noise compounds; a few channels carry outliers 10–100× the typical magnitude
and a uniform grid spends its levels on them; and sensitivity is uneven (the Hessian of the loss
w.r.t. weights), so equal treatment of weights wastes bits where they do not matter and starves
them where they do.

What the field does about each, in the order of how much it buys:
1. **Scalar → vector quantization** (QuIP#, AQLM, QTIP trellis codes). A uniform scalar quantizer
   sits 0.25 bit/weight (1.53 dB) above the rate–distortion bound; lattice/trellis VQ closes most
   of that gap. Result on 70B-class models: ~2 bpw near-lossless where K-quants need ~3. This is
   post-training and per-matrix, so it applies to K3's experts (9.7 MB each) without a training
   run; the cost is a slower decode kernel (codebook lookups) — irrelevant here, compute is free.
2. **Incoherence processing** (random Hadamard/rotation before quantizing): spreads outliers so
   every weight looks Gaussian and the grid is used evenly. Cheap, post-training, ~0.3–0.5 bpw.
3. **Calibration** (GPTQ/AWQ/importance matrices — Unsloth's UD-Q2_K_XL already does this):
   round with knowledge of which weights the activations actually excite.
4. **Quantization-aware training / native low precision** (BitNet b1.58: ternary weights from
   scratch matching fp16 quality at 1.58 bpw; QAT fine-tuning of existing models). This is the
   real answer to "clean up the weights so they cannot degrade": a model trained to live on a
   coarse grid loses nothing there. It needs the training run, i.e. Moonshot, not us. A partial
   version — short QAT fine-tune of only the expert matrices — is compute we do not have at 2.8T.

Numbers for Sisyphus: current tape ~2.7 bpw effective. Best post-training path (1+2+3, tested by
the 5.11 quant oracle on the 48B stand-in first) plausibly **~2.0 bpw at equal quality → −25%
bytes**; native 1.58 would be −40% but is not ours to make. Below the rate–distortion curve for
the *function* (not the weights) is unknown territory: pruning, distillation and BitNet all say
the function needs far fewer bits than the parameters hold, but extracting that requires training
compute proportional to the model, which is the one resource this project has none of.

### 5.18 Zip-in-flight and "boxes of boxes": nested precision (2026-09-07, KNOWN/CALC)

**Zip-in-flight is the design already.** The tape is stored compressed (the quant is the zip;
rANS in 5.10 tightens it), decoded on the GPU as the bytes arrive, and never stored uncompressed
anywhere — the ring holds compressed slabs, the decode kernel writes fp16/int8 tiles into shared
memory, the GEMM consumes them, and they are gone. Nothing is zipped back up because weights are
read-only: the compressed copy on disk *is* the model. Decode cost on the 3070 Ti is far above the
10 GB/s the drives supply, so the unzip is invisible. What the user proposed is what step 2–4 build.

**Boxes of boxes = residual / nested quantization** (RVQ; "Any-Precision LLM", 2024). Store each
weight as a coarse code (≈1.6 bpw) plus one or more residual codes (≈0.5–1 bpw each) that refine
it; reading only the coarse box gives an IQ1_S-class model, opening more boxes gives Q2, Q3.
Three uses, graded:
1. **Sensitivity-adaptive precision (USEFUL, new granularity for lever 5.6).** Today precision is
   chosen per tensor by the quant maker. With nested codes the engine chooses per expert, per row,
   per block *at read time*, from a sensitivity table computed once (Hessian-trace or the
   activation-weighted error the quant oracle 5.11 measures). Open the fine boxes only where the
   loss is sensitive; leave 60–80% of experts at the coarse box. Bytes per token move smoothly
   between the 1.6 and 2.7 bpw endpoints instead of jumping; at equal quality the oracle decides
   the mix. Estimate: −15..25% bytes beyond a uniform 2.7 bpw tape — overlapping with, not adding
   to, the −25% of 5.17's VQ path (both attack the same slack).
2. **One tape, many precisions (CONVENIENCE).** Night runs and daytime runs can pick different
   quality points from the same on-disk file; no second 861 GB copy for a low-bpw experiment.
3. **Self-drafting with the coarse box (DOES NOT WORK HERE).** Tempting: draft with the 1.6 bpw
   model, verify with the residuals — routing identical, no external draft model, trunk shared.
   But every draft token needs the KDA recurrence advanced through all 93 blocks, i.e. a coarse
   *sweep* per draft token: k=4 costs 4 × 0.6 + 0.4 = 2.8 full-read equivalents for ~3 tokens,
   ~1 token per full read — no better than no speculation. Self-drafting only pays on a
   KV-cache transformer, where a draft token costs compute, not a weight pass. K3's KDA state is
   why the external 3B-active draft (5.1) is the right shape.

Decode side for nested codes: two table lookups per weight instead of one; still compute-free
relative to the door. Layout: coarse plane and residual planes stored as separate contiguous
slabs per expert so a coarse-only read is one sequential run.

### 5.19 "Inject the GPU into the SSD": what moving compute to the drive can and cannot change (2026-09-07, KNOWN/CALC — 5.28: item (2) BaM/P2P is ASSUMED-INFEASIBLE on this chipset until a P2P probe passes)

Three physical readings of the request, with the number each one moves:

1. **Compute inside the SSD (the literal version).** Exists as computational storage (SmartSSD,
   §5.13: an FPGA beside the NAND; ScaleFlux; NGD). Its ceiling is the NAND's own internal
   bandwidth, which is not much wider than the door: a 990 PRO 4 TB has 8 flash channels at
   ~1.6–2.4 GT/s → ~12–19 GB/s raw from the dies, of which the controller and PCIe 4.0 x4 deliver
   7.07. A perfect processor glued to the NAND would therefore see at most **~1.7–2.7× per drive**,
   and today's products deliver *less* than the door (3.3 GB/s) because the FPGA's own memory path
   is the bottleneck. The shelf itself has a door; NAND is slow silicon. This is why every
   large-scale "GPU on the SSD" (AMD Radeon SSG, 2017: two M.2 drives on the graphics card) died —
   the drives were the limit wherever they sat.
2. **SSD → GPU without touching RAM (GPU-initiated or peer-to-peer NVMe I/O).** Real and
   software-only: BaM (NVIDIA research, ASPLOS 2023, open source: GPU threads build NVMe
   commands and DMA straight into VRAM), GPUDirect Storage (data-center GPUs officially), SPDK
   user-space NVMe with P2P DMA. Does not raise drive bandwidth by a byte — the door is the
   door — but removes the SSD→RAM→GPU double hop: bytes cross PCIe once, CPU and RAM do nothing,
   the RAM ring disappears. For Sisyphus that means (a) the ring's RAM (1–4 GB at QD32×8 MiB
   ×2 drives) returns to state → ~5–18 more streams; (b) CPU idle; (c) ~50–100 µs lower per-slab
   latency, cosmetic. PCIe was not binding (39 s of transfer vs 82 s of drive time), so the
   speed gain is ~0. Practicality on this rig: ASSUMED — GeForce lacks official P2P/GDS support;
   BaM's kernel module needs the GPU BAR mapped for DMA targets (Resizable BAR is on; works on
   some consumer boards, untested here). Worth a day *after* step 5, for RAM, not speed.
3. **The GPU as the shelf (VRAM holds weights).** 8 GB holds 0.9% of the tape. The trunk's hottest
   tensors (embedding rows in use, router matrices, shared experts for the first blocks) and the
   hot-expert cache (5.15) are the only weights that belong there; the rest cannot fit by a
   factor of 100. HBM is the real CIM-adjacent tier; it is sold by the tens of GB at $10K+ per card.

**Why (2) is not faster, precisely.** The path is a pipeline of stages in series — NAND → SSD
controller → PCIe → (RAM) → PCIe → GPU decode → GEMM — and a pipeline's throughput is the rate of
its slowest stage as long as every stage is kept busy (queue depth ≥ bandwidth × latency, which
QD32 × 8 MiB satisfies by a factor of ~10). Stage rates here: drives 10 GB/s, PCIe to the GPU
22 GB/s, RAM copy ~30–40 GB/s, decode+GEMM >100 GB/s-equivalent. Removing the RAM stage deletes
a stage that had 3–4× slack; the slowest stage is unchanged, so the rate is unchanged. What it
does cut is latency per byte (a few µs of the ~800 µs an 8 MiB slab spends in flight), which
matters only when the pipeline cannot be kept full.

Two regimes where it *would* matter, both recorded so they are not missed later:
- **Three-drive build.** At ~20 GB/s of drive input, the RAM hop costs 20 GB/s of DMA writes plus
  20 GB/s of H2D reads = 40 GB/s of memory traffic on dual-channel DDR5 that sustains ~60 for
  streaming, alongside state traffic. RAM becomes a stage with little slack; P2P removes it. So
  BaM-style I/O is a *prerequisite for scaling past two drives*, not an accelerator of two.
- **Fine-grained reads (daytime row masking, 5.14).** BaM's actual purpose: millions of 4 KiB
  reads per second, issued by GPU threads that know which rows they need, with no CPU round-trip.
  At 10 GB/s in 4 KiB pieces that is 2.5 M IOPS, beyond comfortable CPU-driven io_uring (~0.5–1 M
  IOPS per core with ntfs3's io-wq punting); at 64 KiB it is 160 K IOPS and the CPU path is fine.
  If row masking wants 4 KiB granularity to hit its 3–5× byte cut, GPU-initiated I/O is how it
  gets there — the CPU becoming the bottleneck is the only way the door could be left idle.

Net: nothing in this direction moves the per-token number on the present two drives; (2) is a
RAM lever for the night run now, and becomes necessary for the three-drive build and for
fine-grained daytime sparsity. Optional step after 5; required before 13 at 4 KiB granularity.

### 5.20 The problem in one equation, and where the slack still is (2026-09-07, CALC — a thinking map)

**The equation.** For any build of this machine,

    tok/s  =  A · U · Σ_i (bytes served by tier i per second) / (bytes a token needs)
           =  A · U · C_eff / N

where N = bytes one token must touch (78 GB dense B=1; 861 GB/B per token at batch B), C_eff = the
combined bandwidth of every tier that holds weights *and is busy* (SSD 10 GB/s; RAM ~60 GB/s ×
32 GB; VRAM ~900 GB/s × 8 GB; the doors add only if each holds a share of the bytes), U = fraction
of wall time the doors are actually moving bytes (assumed 0.85 at night, ~0.3 by day today), and
A = tokens produced per crossing of the bytes (1 plain; 2.3 measured with speculation; ×B with
batch). Every lever in this ledger is one of four moves: shrink N, widen C_eff, raise U, raise A.
Hardware money buys C. Everything else is ours.

**What you have to work with (the inventory).** 10 GB/s of NAND door; 60 GB/s of RAM door behind
32 GB of shelf; 900 GB/s of VRAM door behind 8 GB of shelf; 22 GB/s of PCIe; ~40 TFLOPS that sit
idle 99.9% of the time; 86,400 seconds per day of which the plan uses 43,200; a router that tells
you, one block ahead, which 2% of the shelf the next step needs; and the fact that every stream's
217 MB of state is a deterministic function of a few KB of tokens.

**Where the slack is, by term.**

- **N, the trunk.** At B=1 the trunk is 62 of 78 GB — 80% of a daytime token — and every lever
  aimed at experts (5.14 rows, 5.15 cache, 5.4 sparse) touches the other 20%. The daytime problem
  *is* the trunk: requantize it (5.12: 62 → 20–25 GB), share it across tokens (5.1: the reason
  speculation helps one stream), and make it resident (with 64 GB RAM the requantized trunk plus a
  hot-expert cache fits; SSD then serves ~16 GB of experts per token → ~1.6 s → with 5.14 ~0.5 s
  → with speculation **~1.4 tok/s single stream at $350** (5.21; an earlier draft said ~4 single and ~5 aggregate over
  five threads — both wrong, see 5.21 and 5.26). This chain is the most valuable unmeasured
  claim in the ledger; the quant oracle on the trunk is its first experiment.
- **C_eff, every door busy.** Today weights come only through the NAND door; RAM and VRAM hold
  states and buffers. A memory hierarchy is the discipline of making every tier serve bytes at
  once: the 20 GB of hottest weights in RAM/VRAM at 60–900 GB/s, the rest from NAND. The gain is
  the hit fraction, and the hit fraction is measurable right now from routing traces of the local
  model at 26 s/token (a few hundred tokens overnight = a histogram of 82,432 experts). Nothing in
  the project is cheaper per unit of information than that measurement.
- **C, lanes.** Doors are PCIe lanes and where they terminate. M2_1 hangs off the CPU; M2_2–4 hang
  off the chipset and share one ~13 GB/s uplink. The x16 GPU slot is 16 CPU lanes of which the
  GPU needs ~8 at our data rate. If the board bifurcates the slot (x8/x8 or x8/x4/x4 — a BIOS
  option on some Z690 boards, not all), a $20 M.2 carrier puts two more drives on CPU lanes —
  **but the GPU at x8 then receives ≤ 12.5 GB/s and every sweep byte must reach the GPU**, so
  bifurcation is net negative here (5.21). This board's ceiling is ~20 GB/s with the GPU at x16.
- **U, time.** tok/s is the wrong numerator; tok/day is the one that pays. The doors are idle 16
  hours a day. A low-priority daytime batch (the night scheduler at B=20–50 behind the
  interactive threads of 7c) turns 12 hours of door time into 24 for the same electricity — the
  cheapest 2× in this document, and pure scheduling. Also: the sweep has bubbles (router before
  expert reads; state H2D/D2H at block edges); U at night is *assumed* 0.85 and step 4 measures it.
- **A, the state wall as a cache.** The 217 MB state is a cache of ≤ a few KB of tokens. A parked
  stream (waiting on a user, or queued) can be stored as its tokens and re-prefilled when it wakes
  — prefill is compute (190 tok/s here), not a door. So the B that fits in RAM is the number of
  *active* streams, not of open jobs; admission and eviction by token replay let B stay at the
  wall while thousands of jobs are in flight. Also the reason low-rank/fp8 state (5.5/5.8) is a
  legitimate ask: the state's information content is far below 217 MB or it could not be
  recovered from KBs of tokens.
- **N again, the output.** Tokens are a proxy; jobs are the product. Structured outputs, no
  reasoning traces where they buy nothing, architect/coder splits (7d): 2–3× more jobs from the
  same tokens, no physics involved.

**What is exhausted (do not re-mine).** Making compute cheaper (5.15); reading the model in less
than one pass per token without a computer on the shelf (5.16); lossless compression beyond
~20% (5.16); compute at the drive (5.19); P2P I/O as a speed lever on two drives (5.19).

### 5.21 Deep dive on the §5.20 slack, with numbers (2026-09-07, CALC; ASSUMED items marked)

Two models, both in the repo's spirit of "one line of arithmetic per claim". **Day** = one stream,
per-block sequential (attention → router → expert reads; the SSD idles while the CPU/GPU does the
trunk unless routing is predicted, hence a utilization U per row). **Night** = the RESULTS formula
(tokens = hours × B × tok/sweep ÷ sweep-seconds × 0.93 prefill haircut), 64 GB, lean states,
measured 2.32 tok/sweep speculation.

**Day: single-stream latency, drives 10 GB/s, RAM 60 GB/s, cumulative top to bottom.**

| step | what changes | s/token | tok/s | status |
|---|---|---|---|---|
| S0 | today: llama.cpp mmap, one drive | 26.6 | 0.04 | MEASURED |
| S1 | engine, both drives, sparse expert reads (62 GB trunk + 14.3 GB experts) | 7.9 | 0.13 | CALC |
| S2 | + trunk requantized 62 → 22 GB (5.12; quant oracle gates it) | 3.9 | 0.26 | ASSUMED quality |
| S3 | + 64 GB RAM: trunk resident, **CPU computes it** from RAM (0.37 s read + ~0.15 s compute) | 2.3 | 0.43 | CALC |
| S4 | + hot-expert cache, 30% hit (5.15) | 1.8 | 0.57 | ASSUMED skew |
| S5 | + row masking, 0.4 effective bytes incl. random-read penalty (5.14) | 1.1 | 0.92 | ASSUMED |
| S6 | + rANS ×0.85 (5.10) | 1.0 | 1.0 | ASSUMED ratio |
| S7 | + 6 GB of the trunk in VRAM | 0.9 | 1.1 | CALC |
| S8 | + speculation k=2 (expert union ×1.8, ~1.7 tok/sweep) | 1.23/sweep | **1.4** | CALC from measured acceptance |
| S9 | ~~five threads multiplexed~~ WITHDRAWN — see 5.26: threads do not share expert bytes; aggregate stays ~1 | — | — | corrected |

Readings. (a) Once the trunk is resident, its RAM read (22 GB / 60 GB/s = 0.37 s) becomes the
floor of every pass — the door moved from NAND to DRAM, and DRAM is 6× wider, not infinitely. (b)
Speculation buys little for one stream once the trunk is cheap, because the expert *union* grows
with k (k=4 → ×3 expert bytes) — k=2 is the sweet spot; the 2.32 tok/sweep at k=4 gives the same
1.37 tok/s. (c) The honest single-stream ceiling of this box with every software lever and 64 GB is
**~1.4 tok/s (0.7 s/token)** on one thread; the five-thread aggregate claimed in an earlier draft is
withdrawn (5.26) — the "reads as fast as you read" experience comes from ghost text (7c.1), not from rate. §5.20's "~4 tok/s single stream"
was wrong by ~3× (it ignored the union growth and the RAM-door floor); corrected here and in §7.
(d) Gains in order of size: trunk requant ×2, trunk resident ×1.7, row masking ×1.6, cache ×1.3,
speculation ×1.25, rANS ×1.1, VRAM slice ×1.1. Steps S2–S3 are 3.4× of the total 11× from S1.

**Night: throughput, cumulative; per night = 12 h unless marked per day.**

| step | what changes | s/sweep | tok/s | tokens | status |
|---|---|---|---|---|---|
| N0 | 64 GB + 1 Gen4 drive, lean, measured speculation (RESULTS baseline) | 59.6 | 10.6 | 424K | CALC |
| N1 | + trunk requant (sweep 861 → 821 GB) | 56.8 | 11.1 | 445K | ASSUMED quality |
| N2 | + 2nd extra drive, chipset uplink caps at ~20 GB/s (+$130) | 48.3 | 13.0 | 523K | CALC |
| N3 | + sweep bubbles removed, U 0.85 → 0.95 (step 4 measures U) | 43.2 | 14.6 | 585K | ASSUMED |
| N4 | + state-as-cache admission: no idle slots, +10% | 43.2 | 14.6 | 643K | ASSUMED |
| N5 | + 24 h operation: 18 h batch + 6 h interactive → **per day** | 43.2 | 14.6 | **965K/day** (N1/N3/N4 are ASSUMED; measured-only ≈ 0.76M/day, §8c) | CALC |
| N6 | + fp8 state, B 488 (per day) | 43.2 | 26.2 | 1.74M/day | ASSUMED |
| N7 | + low-rank r=32, B 811 (per day) | 43.2 | 43.5 | 2.89M/day | ASSUMED |

Readings. (a) Trunk work is worth 5% at night (it is 7% of the sweep) — the trunk is a *daytime*
lever. (b) The hot-expert cache is worth ~0 at night (B=271 touches every expert). (c) The
biggest measured-lever night gains are the second extra drive (×1.18) and running the doors 18
hours instead of 12 (×1.5); the biggest assumed ones remain the state compressions (×1.8, ×3).
(d) Nothing on the measured-only path passes 500K *per night* below the second extra drive; N2 is
the first row over the line, at ~$610 total (RAM 350 + two drives 260).

**Lanes, corrected.** §5.20 suggested bifurcating the x16 slot to put drives on CPU lanes. The
arithmetic kills it: Alder Lake exposes 16 + 4 CPU lanes, the GPU must *receive* every sweep byte,
and at x8 Gen4 it takes ≤ 12.5 GB/s → a 66 s floor per sweep, worse than two drives through the
chipset. With a GPU that must see the whole tape, this board's door ceiling is **~20 GB/s**
(M2_1 7 + chipset uplink ~13), full stop. Wider doors mean a platform with more CPU lanes (EPYC:
128), which is the §5.12 hot-rod path, not a $20 adapter. At 20 GB/s the RAM hop carries ~40 GB/s
of traffic (5.19) — the P2P/BaM step becomes worth doing at N2, for RAM and for headroom.

**What the 5 tok/s aggregate needs (asked 2026-09-07; the aggregate itself was withdrawn in 5.26 — the memory-budget findings below still hold).** The 64 GB RAM ($350) and nothing else
bought: the trunk must be resident (22 GB) or every pass pays 2.2 s of SSD for it. As-is at 32 GB
the same software stack gives **0.36 tok/s single, ~1.6 tok/s across five threads** (trunk from
SSD each pass, small cache, no room for a draft model) — usable for parked threads, not for reading.
Memory budget at 64 GB is tighter than §7c implied: OS ~3 + trunk 22 + ring 2 + hot cache ≥ 20
leaves ~17 GB, and the Kimi-Linear-48B draft is ~28 GB at Q4 — **the 48B draft does not fit
alongside a resident trunk.** Options, in order: (1) K3's native MTP head if the GGUF has one
(build step 0; tiny); (2) a 3B-class dense draft in VRAM (~2 GB; acceptance ~0.45 assumed →
1.5 tok/sweep at k=2); (3) no speculation. The last two land within 5%: **4.1–4.2 tok/s aggregate,
1.15–1.2 single** — speculation is nearly a wash for one stream once the trunk is cheap, so the
5 tok/s figure rounds to "4–5", and it does not hinge on the draft model at all. Night mode is
unaffected (the trunk is streamed, RAM holds states, the 48B draft runs from disk between sweeps
or is replaced by the same small draft with lower tok/sweep — a −20% night risk if it is).

**Conversation speed, single thread (asked 2026-09-07).** Reading is ~250 words/min ≈ 5.5 tok/s;
"feels like conversation" is ~3–5 tok/s *on one thread*, i.e. ≤ 0.25–0.33 s per token. The 4–5
aggregate above is five threads at ~1 each and does not give that. Per-pass floors on this
platform with 64 GB: trunk from RAM (22 GB − 6 in VRAM) / 60 GB/s + compute ≈ 0.42 s; experts
after all levers ~3 GB × 1.8 (k=2 union) / drives / U. Two drives: 0.42 + 0.72 = 1.14 s per 1.7
tokens → 1.5 tok/s. Three drives (20 GB/s): 0.42 + 0.36 → **2.2 tok/s**. Plus a deeper trunk
quant to ~14 GB (IQ2-class, oracle-gated): 0.28 + 0.36 → **2.7 tok/s** — the single-thread
ceiling of this motherboard. 4–5 tok/s on one thread needs the trunk read in ≤ 0.1 s and experts
in ≤ 0.15 s: a RAM door of ~200 GB/s with the whole 78 GB working set resident, which is the
8-channel EPYC/Threadripper tier (§5.12; ~$3K used) at 4–8 tok/s. On this box the conversational
*feel* comes from 7c.1 instead: the draft's ghost text streams at 15–25 tok/s the instant you
send, and K3's verified text overwrites it at 1.5–2.7 tok/s behind — reading never stalls,
corrections are visible as they land.

**Total, if everything measured-or-plausible lands and the assumed items fail:** day 1.4 tok/s
single (revised to 0.7–0.8 in 5.28; the 5 aggregate is withdrawn); ~0.76M tokens per day at ~$610 on
measured levers, ~965K if N1/N3/N4 also hold. **If fp8 state also lands:** ~1.7M/day.
Confidence that the measured-plus-plausible column (N5, S8) is reachable within a year of
part-time engineering: ~65%; that any state compression lands: ~55%.

### 5.22 Fork-parallel answers: one conversation across many streams (2026-09-07, KNOWN + CALC)

Asked: can one answer be split across threads and stitched? Yes — Skeleton-of-Thought (2023),
APAR, and Hogwild! inference (2025) all do it on hosted stacks; on Sisyphus it is unusually cheap
because forking is a state copy (5.9) and forks share every trunk pass (7c.3). Mechanism: K3 first
emits a short skeleton (section headings, function signatures, the plan — ~30 tokens, one stream);
the engine forks the stream once per section (217 MB state copy each, ~0 compute); the forks
generate their sections concurrently as multiplexed threads; the engine stitches in skeleton order.
Optional **sync points**: every ~100 tokens, each fork prefills its siblings' text so far into its
own state (prefill is compute at 190 tok/s, not a door), so later sections can refer to earlier
ones — the coherence fix Hogwild! gets from a shared KV cache, done here with cheap prefill.

**WITHDRAWN 2026-09-07 (see 5.26):** the table below assumed forks share expert bytes (n^0.33 growth).
They do not once the trunk is resident — the union grows ~linearly at small n and row masking
collapses — so forks give **no speedup** on this hardware. Kept for the record; the structural
mechanism (skeleton → sections → stitcher → K3 read) stays in the design for coherence and for
the cheap-model-writes pattern, not for speed. Original (incorrect) estimate:

| forks | two drives: tok/s on the one conversation | answer in | three drives (20 GB/s): tok/s | answer in |
|---|---|---|---|---|
| 1 | 1.5 | 5.6 min | 2.2 | 3.9 min |
| 2 | 2.6 | 3.5 min | 3.9 | 2.3 min |
| 4 | **4.4** | 2.2 min | 6.9 | 1.5 min |
| 6 | 5.9 | 1.7 min | 9.6 | 1.1 min |
| 8 | 7.4 | 1.5 min | 12.0 | 1.0 min |

~~So the conversational 4–5 tok/s mark is reachable on one conversation on this motherboard~~ — **it is
not (5.26)**; on this box forks cost more than they return. The rest of this section (what
decomposes, coherence risk, stitcher) still applies as a structuring technique. What decomposes: documents with sections, code
files with independent functions/classes, plans, reviews of several files, lists, test suites.
What does not: a single chain of reasoning, a proof, a function whose body depends on itself,
short answers (the 30-token skeleton is overhead; below ~150 tokens run one stream). Quality
risk: forks lose each other's context between sync points — duplicated content, inconsistent
naming — mitigated by the skeleton carrying shared decisions (names, interfaces) and by sync
points; the acceptance test is agreement with a single-stream generation on structure and a
human-judged coherence pass. Memory: 8 forks = 1.7 GB of state, trivial by day. Not a night lever
(the night batch has no latency to save). Design: ENGINE_DESIGN §7c.4.

**Stitcher (asked 2026-09-07).** Glue the seams with the resident fast model, constrained to
rename / dedupe / transition with a size-capped diff, then have **K3 read the result** (prefill,
compute-cheap) and emit a ≤ 30-token verdict. K3's output tokens go only to content and judgment;
connective tissue is typed by the cheap model; correctness is kept by K3 reading, which is the
one thing this machine does nearly for free. Adds ~30–40 s to a 500-token answer, overlappable;
replaces most sync points. Design §7c.4b.

### 5.23 "K3 reads, the cheap model writes": system-level estimates (2026-09-07, CALC — **bounded by 5.28**: reads are free only inside a night sweep and share the GPU with verification; realistic ~3M read tokens/night, ~700 4K reviews; daytime reads are ~80 s per 4K tokens. Figures below are the pre-review upper bounds.)

The stitcher (5.22) generalises. On this machine K3's *reading* is GPU compute (190 tok/s, ~7M
tokens per 12 h night at 85% utilisation) and K3's *writing* is drive bandwidth (523K/night at
the $610 build on measured levers). A 13:1 asymmetry — so the system should be arranged so that
K3 reads far more than it writes, and a fast resident model (Qwen3-Coder-Next, 20–40 tok/s)
writes far more than K3 does. Three regimes, each with its own number:

**Day, one conversation (64 GB, two drives):** **~1.4 tok/s of verified K3 content** (2.1 with three
drives); a 500-token answer in ~6 min (4 min); ghost text visible from the first second. Forks do
not speed this up (5.26); the skeleton/stitcher pattern is used for structure and to keep K3's
output tokens on judgment, not for rate.

**Day, the pipeline:** the coder model drafts at 20–40 tok/s → **0.9–1.7M draft tokens per 12 h
day** on CPU+GPU (GPU shared with the daytime K3 threads; ~1.3M at 30 tok/s is the planning number).
Everything it writes is queued for K3's night read.

**Night, K3 as reviewer/judge (upper bound at spec GPU rate; ~700 jobs realistic, 5.28):** jobs shaped 4,096 in / 128 out are **prefill-bound at ~1,700
jobs per night** (7.0M tokens read, 218K written — the drives sit 60% idle, so review jobs and
generation jobs interleave: ~1,700 reviews plus ~300K tokens of fresh generation in one night).
8K-token reviews: ~850/night. 16K: ~425. Every draft the coder wrote by day is read by K3 by
night with capacity to spare (1.3M drafted vs 7M readable).

**What that is worth, restated.** Not "500K K3 tokens" but **~1.3M tokens per day of code that a
93%-SWE-bench model has read line by line and accepted or annotated**, plus ~300K tokens of
K3-authored content (specs, verdicts, hard sections), plus one conversational thread by day at
~0.8 tok/s (5.26/5.28). Against list prices that is ~$25/day of K3 reading and ~$5/day
of K3 writing for ~$1 of electricity; the reading is where the leverage is, and it is the term
hosted pricing charges for and this machine does not.

**Rule that falls out (design principle, 7d):** never have K3 generate what a cheap model can
type; always have K3 read what a cheap model typed. Output tokens buy judgment and hard content;
input tokens are the budget you spend freely.

### 5.24 Injecting K3 into the cheap model: four mechanisms, one of them a training loop (2026-09-07, KNOWN/CALC — **corrected in 5.28**: the draft path must share K3's vocabulary (EAGLE head or Kimi-vocab model), the coder LoRA is a separate path, and training runs hours per night, not minutes)

Asked: can K3's reasoning ride inside the fast model's output speed? Ranked by how much K3
actually gets transferred:

1. **Speculative decoding (exact; already lever 5.1).** The cheap model types, K3 accepts or
   corrects each token. Output is *exactly* K3's distribution at the cheap model's speed for the
   accepted run. Limited by acceptance (0.56 measured with the 48B draft → 2.32 tok/sweep).
2. **Plan injection (text; already 7d).** K3 writes the plan, interfaces, invariants and the
   failure modes to avoid (short, expensive); the coder executes with that in context. Planner/
   executor splits in the literature lift small-model task success by ~5–15 points; the coder
   does not become K3, but it stops making the class of mistake the plan names. Free.
3. **Retrieval of K3's past verdicts (text; new).** Every K3 verdict, plan and hard section is
   stored with an embedding; the coder's prompt pulls the nearest ones. After a few weeks the
   machine has a K3-authored handbook of *this* codebase's decisions; the coder reads it instead
   of re-deriving it. Cost ~0; it is what the night's 300K K3 tokens accumulate into.
4. **Distillation (weights; new, and the real answer).** K3 emits ~300K–500K tokens a night of
   plans, verdicts and code on the operator's own repos and style. That is training data of
   exactly the kind labs use to make small models behave like large ones. Nightly loop: K3
   writes → at dawn, QLoRA-fine-tune a 7B–8B dense coder on the night's traces (QLoRA 7B fits
   the 3070 Ti's 8 GB; ~1,000–1,500 train tok/s → 500K tokens in ~6–8 min; a month's 15M in ~3 h)
   → the distilled model is both the daytime coder *and the speculation draft*. Two payoffs:
   the coder converges on K3's decisions for these repos (domain-specific, not general frontier
   ability — it will not reach 93% on SWE-bench, it will reach "writes what K3 would have written
   here"); and the draft's acceptance rises because distillation on K3 outputs targets precisely
   the token-distribution match speculation depends on (EAGLE/Medusa heads are trained this way).
   Acceptance 0.56 → 0.70–0.80 (ASSUMED) gives 2.32 → 2.8–3.3 tok/sweep, **×1.2–1.4 on every
   night row**, compounding monthly as the draft learns the operator's work. Draft footprint
   falls from the 48B's 28 GB to ~4–5 GB, which resolves the 64 GB daytime budget conflict (5.21).
   Not transferable: K3's internal state or KV — representations differ; only text and gradients cross.

Estimate of the combined effect: coder quality on the operator's repos rising month over month
toward K3-consistency (unmeasurable in advance; the acceptance-rate curve is the proxy and it is
measured every night for free); night throughput ×1.2–1.4 from the better draft; daytime memory
freed. This turns the night run into a flywheel: every night's K3 output makes the next night
faster and the next day's coder better.

### 5.25 Pushing the read side: state as memory, verification as the unit of work (2026-09-07, CALC/ASSUMED — **bounded by 5.28**: prefix size capped by K3's trained context and by KV counted against streams (plan 128–256K, not 1M); daytime verification reads cost a near-dense pass; night read capacity ~3M realistic)

Reads cost GPU compute (190 tok/s ≈ 7M/night; ~2× with int8 tensor-core prefill, ASSUMED) and
per-token MLA KV (0.028 MB/token/stream); the KDA state does not grow with context. Everything
below spends reads to save writes.

**1. The repo-state: a K3 that has already read everything.** Prefill the *entire* codebase (plus
docs, decisions, style guide) once — 500K tokens ≈ 45 min of GPU — and snapshot the result: one
217 MB KDA state + 14 GB of MLA KV, shared read-only by every fork. Every job of the night then
starts as a model that has read the whole repo, at zero per-job cost. Because the recurrence is
causal, **the snapshot is extended, not recomputed**: append tonight's diffs and verdicts to the
prefix and only the new tokens are prefilled. Bound: MLA KV for the shared prefix (0.028 MB/token
→ 1M tokens = 28 GB; 500K–1M is the practical ceiling at 64 GB); K3's trained context length and
its *effective* recall at that length (KDA blocks compress; MLA blocks do not) — measure with
needle/recall tests on the 48B stand-in and on K3 itself before trusting > 256K.

**2. A state library.** Snapshots are files: "has read repo X", "has read the business plan and
every past decision", "has read this week's tickets". 217 MB + KV each on the 990 PRO; fork any job
from any of them; extend them nightly with that day's outputs. This is the "funnel past outputs
into reads" loop made cheap: K3's own verdicts and plans become part of the state its next jobs
start from, so the machine accumulates judgment about *this* work without re-reading it.

**3. Generate once, verify many ways.** Verification is a read plus a ≤ 30-token verdict; the
output budget alone allows ~16K verdicts a night, so reads bind first (~1,700 4K-token checks).
Run every artifact past K3 several times under different lenses — spec conformance, test
adequacy, security checklist, performance, style — each a separate fork from the repo-state, each
answering one question. Chain-of-verification at frontier quality for the price of prefill.

**4. Best-of-N with K3 as judge.** The cheap coder writes N candidate implementations (N = 8 costs
it ~15K tokens, minutes); K3 reads all eight and picks or ranks in ~40 output tokens. Reranking
with a strong verifier is one of the most reliable quality gains in the literature; here the
generation is nearly free and the judgment is a read. Expect the coder+K3-judge pair to beat the
coder alone by a wide margin on the operator's acceptance set (ASSUMED; measure).

**5. Spec expansion.** Since a 10K-token spec costs the same read as a 500-token one, let the
coder *expand* every request into an exhaustive spec — schemas, invariants, examples,
counterexamples, edge cases, the tests that must pass — and let K3 read the expanded spec and
correct it (≤ 100 tokens) before anything is built. Specificity is free on the input side; the
only cost is the coder's time to write it.

**6. Multi-granularity reading.** Read the same code at file, module and cross-module levels in
separate passes (each a prefill), producing K3 notes that go back into the repo-state. The 7M
tokens a night covers a 500K-token repo at all three levels every night with room left.

Bounds to respect: K3 output budget still ~500K–1.5M/night (verdicts are cheap, code is not);
MLA KV per unique context; effective long-context recall (unmeasured); prefill at 190 tok/s
means a fresh 500K-token state is 45 min — build states incrementally, never from scratch
nightly. Design implications: ENGINE_DESIGN §7b (prefix forking) gains snapshot persistence and
incremental extension; a `states/` directory in the file layout (§8).

### 5.26 CORRECTION — threads and forks do not multiply daytime speed once the trunk is resident (2026-09-07)

Asked "how many threads, and what determines it" — and the arithmetic exposed an error that runs
through today's daytime numbers. The claim that five threads cost ~1.7× the bytes of one came from
§5.12, where the 62 GB trunk was *streamed from SSD* every pass: five streams' 5 × 14 GB of experts
on top of a shared 62 GB trunk is 132 vs 76 GB, 1.7×. **With the trunk resident (the 64 GB plan),
the shared part is a 0.42 s RAM read and the expert bytes — which are not shared — dominate.**
Two streams' routing overlaps little: expected union of n streams with 1.8 rows each is
896 × (1 − (1 − 16/896)^(1.8n)) experts per block, i.e. 3.2% of experts at n=1, 6.3% at n=2,
12% at n=4, 22% at n=8. Row masking (5.14) collapses even faster: rows touched go from 25% of an
expert at n=1 to 44% at n=2 to ~70% at n=4, at which point the random-read penalty erases it.

Per-pass model, 64 GB, resident trunk (16 GB in RAM + 6 in VRAM → 0.42 s), k=2 speculation
(1.7 tok/pass), hot cache 0.7, rANS 0.85, drives 10 GB/s (three drives 20):

| threads | expert GB/pass | s/pass | per-thread tok/s | **aggregate tok/s** | three drives: per-thread · aggregate |
|---|---|---|---|---|---|
| 1 | 6 | 1.2 | 1.4 | **1.4** | 2.1 · 2.1 |
| 2 | 21 | 3.2 | 0.5 | 1.0 | 0.9 · 1.9 |
| 4 | 58 | 8.1 | 0.2 | 0.8 | 0.4 · 1.6 |
| 8 | 109 | 15 | 0.11 | 0.9 | 0.22 · 1.8 |
| 16 | 193 | 23 | 0.07 | 1.2 | 0.14 · 2.3 |
| 64 | 416 | 49 | 0.03 | 2.2 | 0.07 · 4.4 |
| 271 (night regime) | 476 | 56 | 0.03 | 8.2 | 0.06 · 16 |

**Consequences.** (a) The daytime optimum on this box is **one thread at ~1.4 tok/s (2.1 with
three drives)**; a second thread *lowers* aggregate throughput until the bulk regime past ~50
streams. (b) **Fork-parallel answers (5.22) give no speedup here** — four forks read 10× the
expert bytes of one stream for 4× the tokens. The skeleton/stitcher machinery still has value for
*structure* and for the K3-reads-cheap-model-writes pattern, not for speed. (c) "4–5 tok/s
aggregate over five threads", "4.4 tok/s on one structured answer", and the 7c.3 multiplexing
gain are **withdrawn for the resident-trunk configuration**. They remain roughly right where the
trunk is streamed (32 GB, or an unrequantized 62 GB trunk): there five threads do get ~2.5× the
aggregate of one — but one is 0.2 tok/s, so the aggregate is ~0.5. (d) The conversational
4–5 tok/s mark is **not reachable on this motherboard by any thread arrangement**; single-stream
1.4–2.1 verified tok/s with ghost text in front of it is the daytime ceiling. 4+ on one stream is
the 200 GB/s memory-door tier (§5.12). (e) What *determines* the thread count: not RAM (17 GB free
→ ~80 threads) but routing-union growth, which is set by 16-of-896 routing and cannot be changed
from outside the model. The one unknown that could soften it: routing correlation across streams
working on the same repo (topical skew). Under strong skew the union grows slower and forks recover
some benefit — the routing-trace histogram experiment (5.20) measures this too. Until then, plan
for one interactive thread by day, bulk mode at ≥ 64 streams when nobody is at the keyboard, and
the night engine as the real throughput.

Daytime per-day numbers in §8c revise accordingly: 6 h interactive ≈ 1.4 tok/s × 21,600 ≈ 30K
(not 100K); the 6 h daytime *bulk* window is unchanged (it runs the night regime). Day totals fall
~8%: column A 0.81M/day.

### 5.27 Fewer active experts: k is a runtime knob (2026-09-07, KNOWN elsewhere; ASSUMED for K3)

`expert_used_count = 16` is read from the GGUF header (SHARDS) and the router is a fixed top-k, so
16 is exact for the model as trained — but k is a load-time setting, not a law: llama.cpp accepts
`--override-kv kimi-k3.expert_used_count=int:12`, and the engine can do the same per job. Running an
MoE below its trained k is a known speed-for-quality trade (modest loss at k−2, steeper beyond,
on DeepSeek/Qwen-class MoEs; unpublished for K3). Expert bytes per token scale linearly with k, so
k=12 is **−25% expert bytes for every token, day and night** — at night that is −23% of sweep bytes
(the trunk is 7%), a lever as large as rANS + hot cache together, and it needs no engine work.
Test, runnable now on llama.cpp against the cached hosted greedy targets: same prompts at k = 16,
14, 12, 10, 8; measure token agreement and KL vs the k=16 local run and vs hosted. Expectations
(ASSUMED): 14 ≈ free, 12 the interesting case, 8 not. Per-job k is also a *quality dial* — verdict
and classification jobs may run at k=10 while code generation stays at 16 — and sits naturally in
the scheduler beside the precision dial of 5.18. Caveat: fewer experts per token also shrinks the
routing union per stream, which helps the multi-thread arithmetic of 5.26 proportionally.

### 5.28 Adversarial review 3 of the documents themselves (2026-09-07 evening): findings and dispositions

Two independent cold reads of FINDINGS.md and ENGINE_DESIGN.md, one for numerical consistency (53
items) and one for physical/engineering plausibility (35). Dispositions below; §0, §8c and §8d were
regenerated from the corrected constants, and stale passages elsewhere are marked rather than
silently rewritten so the record of what was believed when stays legible.

**Withdrawn or downgraded claims (beyond §5.21/5.26):**
- **"K3 reads cheaply" holds only inside a night sweep.** A daytime read of T tokens at B=1 must
  fetch the union of experts those tokens route to: 83% of experts at T=100, ~all at T=500 — a
  near-dense ~800 GB pass, ~80 s on two drives. The stitcher's "3 s read", the daytime review
  pipeline and the 7c acceptance test's "arbitrary reading" were wrong. Night reads ride on a
  sweep that is happening anyway and cost only GPU compute.
- **GPU compute is one budget.** 7M read tokens/night = 1.46 EFLOP against 1.73 EFLOP of 12 h at
  the 40 TFLOPS *spec*; tree verification at B=270 is another ~80%. Realistic achieved rate for
  ragged grouped GEMM + chunked KDA is 15–25 TFLOPS. Reads/night ≈ 3M realistic (6M at spec), and
  only in chain-speculation tiers; low-rank tiers are GPU-capped near 1.5M tokens/night. "40 TFLOPS
  effective" is relabelled spec; step 4 measures achieved TFLOPS on `mul_mat_id` at M ∈ {1, 16, 170}.
- **Speculation arithmetic.** Expected tokens per sweep is Σ_{i=0..k} aⁱ (k drafts + the verifier
  token): 2.53 at a=0.7, k=4, not 2.77 — RESULTS' "70%" rows already compute 2.53 (their label was
  wrong). The stated tree profile (0.71, 0.71·0.66, then 0.56/depth) yields 2.72 tok/sweep, not
  2.9–3.1; "3.0" is kept only as the tree-with-better-draft figure. Distilled-draft tier ×1.09, not ×1.3.
- **The measured draft was Moonlight-16B-A3B** (~9–10 GB at Q4), not Kimi Linear 48B; the scoreboard
  row was mislabelled. Acceptance was measured against *hosted* K3, not the local Q2 model; the
  Q2 argmax differs from hosted at an unmeasured rate — re-measure against the local run.
- **Sparse sweep at large B.** 896·(1−(1−16/896)^(B·16)) is 891 experts at B=290 under independent
  routing; the "650–700 under expected skew" behind the −25% rows (164K/261K/305K) is unsupported
  until the routing histogram exists. Treat the sparse-sweep saving as ~0 at B ≥ 150 for planning.
- **Row masking (5.14/§5b):** SwiGLU experts show ~25–50% safely skippable rows (TEAL/CATS), not the
  80–90% of ReLU models; per-expert predictors are needed (82K of them), scattered 2.8 KB records
  straddle 4 KiB pages, and co-activation permutation is required for any clustering. Effective
  factor revised 0.4 → ~0.75; predictor sizing in §5b corrected.
- **Trunk requantization** to 20–25 GB has the *worst* prior of the daytime chain: the trunk is
  mostly attention/recurrence projections, the tensors every study finds most quantization-
  sensitive. Expect Q4-class (~35–40 GB); odds of ≤ 25 GB at acceptable KL ~40%, not 75%. At
  38 GB the trunk still fits resident in 64 GB but leaves no room for a hot cache or the coder.
- **Daytime single stream** therefore: ~0.7–0.8 tok/s on two drives, 1.1–1.3 on three (§0),
  down from 1.4/2.1. The 5.21 ladder's rows S2–S8 are superseded by this paragraph.
- **rANS on IQ2/IQ3 codebook indices:** E8-lattice codes chosen by importance search are near-uniform;
  3–5%, not 10–20% (ENGINE §8b already said 3–8%).
- **fp8 state:** a fresh rounding every step in a channel with decay α accumulates to ~ε/√(1−α²) —
  ~22× a single rounding at α = 0.999, exactly the long-memory channels; per-head absmax scaling
  zeroes weak associations. Use per-row scales or int8; test at ≥ 2K generated tokens. Physics
  odds 50%, not 70%.
- **Low-rank pack step:** ~5 MFLOP per head (two thin QRs + small SVD), ~33 TFLOP per sweep at 1,000
  streams, and batched tiny SVDs run at a few percent of peak — plausibly longer than the sweep.
  Cheaper truncations (Gram–Schmidt of new keys, truncate every k sweeps) before the lever counts.
  Telemetry must be retrieval error on the block's real queries, not Frobenius energy. 5.8b's
  "~1 GFLOP per stream per block" → ~50 MFLOP.
- **bf16 between-step state is itself untested:** ggml holds recurrent states in F32 (434 MB/stream).
  Every stream count assumes bf16 is lossless; it is the first item on the 48B harness.
- **Deferred-commit / tree kernel:** depth-1 candidates need the root row's logits too; KDA gates are
  per-channel (diag α), so the ancestor matrix carries a 128-vector of cumulative log-decays per
  node plus the erase term — a substantive modification of chunked KDA, not "add a mask".
- **"Bit-identical" forks and token-for-token gates** are not achievable on GPU MoE (batch-dependent
  reduction order). Gates become teacher-forced per-position agreement + a logit-KL budget, and one
  *global* KL budget is allocated across all stacked levers (k-reduction, rows, fp8, low-rank, trunk).
- **Ghost text at 0.56 acceptance churns:** expected run between rejections ~1.3 tokens; render an
  8–12-token window beyond the verified frontier, not a full in-place tail.
- **Fork-parallel answers (5.22, §7c.4)** are removed from the daytime design (no speedup, 10× bytes,
  new failure modes); "skeleton → sections on one stream" gives the structure. Sync-point prefill
  changes the fork's sequence — it is a different prompt, not synchronization.
- **Distillation loop:** a 7–8B dense coder with its own tokenizer cannot be K3's draft (rejection
  sampling needs a shared vocabulary) — the draft path is an EAGLE-style head on K3's hidden
  states or a Kimi-vocab model; the coder path is a separate LoRA. QLoRA-7B on the 3070 Ti runs
  ~300–700 train tok/s and the training set is the contexts (millions of tokens), not the outputs:
  hours per night, not minutes.
- **Repo-state (5.25/§7b.1):** 1M-token prefix = 28 GB KV that halves B, ~55 GFLOP/row of MLA
  attention (comparable to the whole model) and 28 GB of PCIe per sweep. Cap at K3's trained
  context (read it from the GGUF) and count the KV against the stream budget.
- **BaM / P2P NVMe→GPU:** needs the drive unbound from the kernel driver and root-complex P2P between
  chipset-attached NVMe and a GeForce — unsupported on this platform. ASSUMED-INFEASIBLE until a
  P2P probe passes; plan the three-drive RAM traffic inside the 60 GB/s door.
- **Bifurcation** (5.20) — still withdrawn; §5.12's "worth it for the night mode" sentence is marked.

**Engineering issues added to the risk list (ENGINE §12):** VRAM budget (fp32 active-block state is
6.3 MB/stream, double-buffered ~3.8 GB at B=300 → no VRAM to lend to states; dequant must fuse
into the GEMM; rANS scratch; CUDA context); day/night memory layouts incompatible (switching costs
a re-prefill of ~270 streams — hysteresis of hours, checkpoint night states to disk); the daytime
GPU/RAM triple-booking (coder 18 GB + trunk + cache + draft > 64 GB; coder decode saturates the
RAM door the trunk needs); Mixed thread mode redefined as "bulk only in foreground idle gaps";
sparse sweep's per-block router dependency forbids expert prefetch across blocks (5–9 s/sweep of
bubbles unless routing is predicted a block ahead); PCIe co-binding on three drives once state/KV
traffic (~65 GB/sweep) is added, and pinning ~60 GB of host RAM; static drive split → sweep =
max_i(bytes_i/bw_i), so one throttling drive stalls all; no filesystem to hold the repacked image
(ext4 is 300 GB; C: is NTFS) — a staging plan and an ext4 production path are required; read-disturb
and thermals, not TBW, are the sustained-read risks; IOMMU/pinning CPU cost at 2,300 req/s;
extraction-shaped nights are compute-bound (150K prefill rows/sweep stretches the sweep to
~13–25 min); K3's trained context length and tokenizer identity must be read from shard 1.

**Ledger hygiene fixed:** §1 shard row (19/19 present); §4 and the second §5 stack table labelled
superseded (winsat era); the corrections-log entry that still quoted "5 across five threads";
§5.12's multi-thread paragraph; §5.23's "5 threads at 4+ each"; RAM prices unified ($175/16 GB,
$700/48 GB; 128 GB ≈ $1,400 extra); prefill capacity 7M at spec / 3M realistic; 32 GB daytime
aggregate 0.2 single → 0.5; 700–800 TB/night of reads is the three-drive figure (387 TB as-is).
Confidence figures are judgment, not calibration; they are labelled as such.

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
(in tape mode RAM is the multiplier) · assuming the download was complete · winsat's 8.5 GB/s as
the drive number (diskspd/fio at the engine's pattern: 10.55/9.51) · "861 GB per token" as a
floor (it is the price of batch; B=1 reads ~78 GB, measured 26.6 s) · rank-32 factors "4x, 8x
with fp8" (2x/4x; 8x needs r=16) · "speculation never helps a single stream" (the trunk read is
shared across drafts: -33% with today's trunk) · mount folders keyed by NTFS label (both volumes
are "Windows") · the streamer's thread re-join (would have hung on two drives; caught in audit). · "~4 tok/s single stream with a resident trunk" (5.20 draft; the expert union grows with k and
the resident trunk's DRAM read is a 0.37 s floor → ~1.4 single, 5.21; the five-thread aggregate
was itself withdrawn in 5.26, and 1.4 became 0.7–0.8 in 5.28) ·
"bifurcate the x16 slot for drives on CPU lanes" (the GPU at x8 caps the sweep at 12.5 GB/s; net
negative, 5.21). · "five threads ≈ 1.7× bytes, 3× aggregate; forks give 4.4 tok/s on one answer" (true only
with a streamed trunk; with the trunk resident the unshared expert union dominates and threads/forks
add nothing — the daytime ceiling is one thread at 1.4–2.1 tok/s, 5.26). · review 3 (5.28): "K3 reads
cheaply" by day (near-dense pass, ~80 s per 4K tokens) · GPU budget double-booked between reads and
tree verification · 40 TFLOPS as effective (spec; 15–25 achieved) · 2.77 tok/sweep at 70% (2.53) ·
tree 3.0 (2.72) · draft mislabelled Kimi Linear (Moonlight) · sparse-sweep −25% at B=290 (~0 without
skew) · row masking 0.4 (~0.75) · trunk requant to 22 GB as likely (Q4-class ~38 GB more likely) ·
rANS 10–20% (3–5%) · fp8 odds 70% (50%) · bf16 state as lossless baseline (untested; ggml is F32) ·
bit-identical forks / token-for-token gates (teacher-forced + KL) · 7B distilled coder as draft
(tokenizer) · BaM "worth a day" (likely infeasible on this chipset) · reads 7M/night (3M realistic).

## 8. Prices used (PRICE, Sep 2026)

Hosted K3 $3 / $15 per M in/out (OpenRouter) · DDR5 16 GB ~$175/stick, 48 GB ~$700 ·
Gen4 NVMe 1 TB ~$130, Gen5 2 TB ~$400 · RTX 3090 used ~$1,300, 4090 used ~$2,500,
5090 ~$5,000 · B200 8-GPU node $48–57/hr · large-RAM CPU instance ~$8–15/hr.

## 8b. What the tokens are worth (CALC, list prices; not financial advice)

Hosted-equivalent value of a night: 500K output tokens x $15/M = $7.50; 849K = $12.74; the
input-heavy shape (~6M input tokens/night at $3/M) adds ~$18. Electricity ~450 W x 12 h at
$0.17/kWh = ~$0.90/night. At 849K/night the $610 build pays back in ~48 nights; steady-state cost
~$1.06/M output tokens vs $15 hosted (~14x); year one including hardware ~$3.45/M (~4x). At 498K:
~8x steady state, ~2.5x year one. In things: 500K output tokens = ~375K words = ~2,000 jobs of 256
tokens or ~10,000 of 50; the input path reads ~3,000 two-thousand-token documents a night.

What the multiple is NOT: a comparison of cost to cost. Hosted list price carries the provider's
margin; on datacenter hardware the tape regime's advantage on batch work is ~2-3x capex per token
at lower quality and latency, and it is zero once a node holds the model in HBM. The advantage is
largest when the token buyer and the hardware owner are the same person; every step toward
selling tokens to strangers hands it back. Interactive use (agentic SWE turns: ~4,000 output
tokens, ~100K input, 10 tool calls) is ~2 min hosted, ~45-75 min on this PC at the daytime
ceiling, ~10-15 min on the used-EPYC hot rod; the fit is the architect/coder split (ENGINE_DESIGN
7d) and the night review shift, not K3 typing code.

## 8c. The full quantification (revised 2026-09-07 after review 3; the $610 build)

Day = 12 h night batch + 6 h daytime batch at the night rate + 6 h interactive (one thread, ~0.8
tok/s ≈ 17K). Conversions: 0.75 words/token; 11 tokens/line; 40% of code tokens kept; 500
words/page; 90K words/book. Hosted $15/M out, $3/M in (reads at 40% for cache hits). Power ~500 W
× 24 h ≈ $1.80/day at $0.15/kWh. Columns cumulative left → right; odds are "tier reached within a
year of part-time work".

| | A measured (60%) | B + better draft & tree (45%) | C + fp8 state (35%) | D + low-rank r32 (20%) |
|---|---|---|---|---|
| K3 output / night | **498K** | 644K | 1.16M | ~1.5M (GPU-capped) |
| K3 output / day | **0.76M** | 0.98M | 1.75M | 2.27M |
| / week · / month · / year | 5.3M · 23M · 277M | 6.9M · 29M · 358M | 12M · 53M · 640M | 16M · 68M · 830M |
| words / day · pages · books | 570K · 1,140 · 6.3 | 735K · 1,470 · 8.2 | 1.3M · 2,630 · 14.6 | 1.7M · 3,400 · 18.9 |
| code lines / day, gross → kept | 69K → 28K | 89K → 36K | 159K → 64K | 206K → 83K |
| jobs / day: 256-tok · 500-tok · 2K-tok | 2,970 · 1,520 · 380 | 3,830 · 1,960 · 490 | 6,840 · 3,500 · 875 | 8,870 · 4,540 · 1,135 |
| average rate over 24 h · night rate | 8.8 · 12.4 tok/s | 11 · 16 | 20 · 29 | 26 · 38 |
| hosted-equivalent output value / day | ~$11 | ~$15 | ~$26 | ~$34 |
| all-in $/M output, year 1 (hardware + power) | $4.55 | $3.50 | $1.96 | $1.51 |
| share of the operator's daily output at work (real output est. 1–6M of a 120M meter) | 13–76% | 16–98% | 29–175% | 38–227% |

**Reading (compute-bound; shares the GPU with verification):** ~3M tokens/night at realistic GPU
efficiency (6M at spec), columns A–B only (tree and low-rank tiers consume the GPU); plus ~1.5M in
the 6 h day batch → **~4.5M tokens read per day**, ~400K lines gross, 150–200K with context;
**~700 four-K-token reviews per night**. Hosted-equivalent ~$5/day. Each 4K review carries 113 MB
of MLA KV, so review-heavy nights run ~190 streams, not 271. The resident coder writes ~0.9–1.3M
tokens/day; K3 reads most of it, not all.

**Daytime feel:** first text < 1 s (ghost text, short window); ~0.8 tok/s verified on one thread
(1.1–1.3 with three drives); a 500-token answer in ~10 min (7); reading a 4K document ~80 s (40).

**Fix capacity (A):** K3 writing corrected code ≈ 20–25K lines/night; K3 writing 100-token fix
instructions for the coder ≈ 1,500–2,500 directed fixes/night.

## 8d. The whole chart: hardware × lever tier, tokens per night (revised 2026-09-07)

12-hour night, RESULTS conventions. Cumulative left → right: measured speculation (2.32) → better
draft (acceptance 0.7 → 2.53) → tree (~3.0) → fp8 state (×1.8 streams) → low-rank r32 (×3,
capped by GPU verification compute at ~1.5M on this GPU). Odds of reaching a tier within a year,
given the hardware: plain 85% · measured spec 60% · draft+tree 45% · fp8 35% · low-rank 20%.

| hardware | plain lean | measured spec | + draft 0.7 | + tree | + fp8 | + low-rank r32 |
|---|---|---|---|---|---|---|
| as-is, 32 GB, two drives ($0) | 66K | 143K | 156K | 185K | 333K | 555K |
| 64 GB ($350) | 127K | 272K | 297K | 352K | 633K | 1.06M |
| 64 GB + 1 drive ($480) | 202K | 428K | 467K | 553K | 1.00M | ~1.5M (cap) |
| **64 GB + 2 drives ($610)** | 236K | **498K** | 543K | 644K | **1.16M** | **~1.5M (cap)** |
| 128 GB + 2 drives (~$1,900) | 480K | 1.03M | 1.12M | 1.33M | ~1.5M (cap) | ~1.5M (cap) |

The cap: verification rows × 208 GFLOP per sweep against a realistic 15–25 TFLOPS. Past ~800
streams the GPU, not the drives, binds; more RAM beyond that buys nothing on this GPU. Multiply
by ~1.5 for the 24-hour pattern of §8c.

**Daytime by hardware:** as-is 32 GB: ~0.2 tok/s single (threads add ~2.5× aggregate here, to
~0.5). 64 GB + two drives: 0.7–0.8 single; threads do not add (5.26). Three drives: 1.1–1.3. Used
8-channel EPYC, 1 TB RAM (~$3K): 4–8 tok/s single; no better at night (PCIe/compute bind).

**Realistic expectation, stated once.** The $610 build lands at ~500K a night (~0.75M/day) with
~60% confidence; ~1.1M a night with ~35%; ~1.5M with ~20%. If speculation fails to transfer into
the engine: ~240–300K (85%).

## 9. Order of operations

0. ~~Delete stale partials, finish the download~~ done (19/19). ~~Measure drives (diskspd,
   fio)~~ done. ~~Linux installed, first K3 tokens generated~~ done (2026-09-06). Next $0 steps:
   run `tools/stream_run.sh` (streamer vs fio; sparse B=1 vs llama.cpp's 26.6 s); read the
   sustained 8 h drive test; move/split the model onto the 990 PRO's ext4 partition by measured
   bandwidth (today it sits entirely on the Gen3 SN570).
1. Buy 2×16 GB DDR5-4800 (~$350) → 64 GB. Buy one Gen4 1 TB NVMe (~$130) for M2_2 (a second
   for M2_4 later, +$130, is what carries 500K without fp8). Split the model across the drives
   by measured bandwidth: 990 PRO 7.07 : new ~6.5 : SN570 3.47 (`tools/repack_plan.py` takes the
   measured numbers). C: has 1,008 GB free. **~$480**; the second drive (**$610 total**) is what
   carries 500K on measured acceptance alone.
2. Test 5.5 on Kimi Linear 48B locally (fp8 vs bf16 state agreement). **$0.**
3. Engine: tape-mode GPU streaming (expert chunks SSD→RAM ring→VRAM, all-stream hidden
   states in VRAM, KDA states in RAM/VRAM, MLA KV in RAM), then 5.3, then 5.4 or 5.1.
   Months, not dollars. Evaluate ktransformers / MoE-Infinity as bases first.
4. Route extraction-shaped jobs to the box first (5.2): highest value per sweep, no
   speculation needed.
5. Profile routes (skip's `cold_weight_ratio`) only when the scheduler regime becomes
   relevant, i.e. at ≥128 GB RAM.
