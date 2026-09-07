#!/usr/bin/env python3
"""tools/route_stats.py - analyse a Sisyphus routing trace (route_k*.bin from the patched llama-perplexity).

    python3 tools/route_stats.py route_k16.bin [--json out.json] [--n-expert 896]

Answers, from real K3 routing on real job text (FINDINGS 5.20/5.26/5.30/5.31):
  * expert frequency skew per layer: how uneven is routing? (sparse sweep at large B; expert pruning)
  * union growth: how many distinct experts do n independent tokens touch? (threads/forks by day)
  * sequence-local reuse: LRU hit rate for caches of C experts as a conversation proceeds (5.30 D5)
  * global hot-cache hit rate for the top-C experts (5.15)
  * next-block predictability: how much does block L's set predict block L+1's? (5.30 D4, upper bound)
Tokens in the trace are in evaluation order: the perplexity tool feeds chunks of n_ctx consecutive
tokens of the eval text, so consecutive records within a chunk ARE a conversation-like sequence.
"""
import sys, struct, json, math, random
from collections import defaultdict, Counter, OrderedDict

MAGIC = 0x53525431

def read_trace(path):
    """-> dict layer -> list of (ids[n_used], weights[n_used]) per token, aligned across layers; n_expert guess.
    llama-perplexity only computes the last half of each chunk through the final blocks (output pruning), so
    later layers can carry fewer tokens per batch than earlier ones; within a batch the tokens present are the
    LAST n of the chunk, so each batch is trimmed to its minimum token count from the tail."""
    batches = []          # list of dict layer -> (n_tok, ids, ws)
    cur = {}; last_layer = -1; max_id = 0
    with open(path, "rb") as f:
        while True:
            hdr = f.read(16)
            if len(hdr) < 16:
                break
            magic, layer, n_tok, n_used = struct.unpack("<IiII", hdr)
            if magic != MAGIC:
                raise SystemExit(f"bad magic at offset {f.tell()-16}")
            ids = struct.unpack(f"<{n_used*n_tok}i", f.read(4 * n_used * n_tok))
            ws = struct.unpack(f"<{n_used*n_tok}f", f.read(4 * n_used * n_tok))
            if layer <= last_layer and cur:
                batches.append(cur); cur = {}
            cur[layer] = (n_tok, n_used, ids, ws); last_layer = layer
            if ids: max_id = max(max_id, max(ids))
    if cur: batches.append(cur)
    per_layer = defaultdict(list)
    for b in batches:
        m = min(v[0] for v in b.values())
        if m < 8:   # warm-up / single-token decodes: skip
            continue
        for layer, (n_tok, n_used, ids, ws) in b.items():
            for t in range(n_tok - m, n_tok):
                per_layer[layer].append((ids[t*n_used:(t+1)*n_used], ws[t*n_used:(t+1)*n_used]))
    return per_layer, max_id + 1

def gini(counts):
    xs = sorted(counts)
    n = len(xs); s = sum(xs)
    if s == 0: return 0.0
    cum = 0.0; g = 0.0
    for i, x in enumerate(xs, 1):
        cum += x; g += cum
    return 1.0 - 2.0 * g / (n * s) + 1.0 / n

def main():
    path = sys.argv[1]
    out_json = None
    if "--json" in sys.argv:
        out_json = sys.argv[sys.argv.index("--json") + 1]
    per_layer, n_expert = read_trace(path)
    if "--n-expert" in sys.argv: n_expert = int(sys.argv[sys.argv.index("--n-expert") + 1])
    else: n_expert = max(n_expert, 896)   # K3; the trace only shows ids that were used
    layers = sorted(per_layer)
    n_tok = min(len(per_layer[l]) for l in layers)
    n_used = len(per_layer[layers[0]][0][0])
    print(f"trace: {len(layers)} MoE layers, {n_tok} tokens, k={n_used}, n_expert≈{n_expert}")
    R = {"layers": len(layers), "tokens": n_tok, "k": n_used, "n_expert": n_expert}

    # 1. frequency skew per layer --------------------------------------------------------------
    ginis = []; top_share = defaultdict(list); active_frac = []
    for l in layers:
        c = Counter()
        for ids, _ in per_layer[l]:
            c.update(ids)
        counts = [c.get(e, 0) for e in range(n_expert)]
        ginis.append(gini(counts))
        tot = sum(counts); srt = sorted(counts, reverse=True)
        for frac in (0.05, 0.10, 0.25, 0.50):
            m = max(1, int(n_expert * frac))
            top_share[frac].append(sum(srt[:m]) / tot)
        active_frac.append(sum(1 for x in counts if x > 0) / n_expert)
    print("\n== 1. routing skew (mean over layers) ==")
    print(f"  Gini of expert usage: {sum(ginis)/len(ginis):.3f}  (0 = perfectly balanced, 1 = one expert)")
    for frac in (0.05, 0.10, 0.25, 0.50):
        v = sum(top_share[frac]) / len(layers)
        print(f"  top {int(frac*100):2d}% of experts take {v*100:5.1f}% of routing decisions")
    print(f"  experts touched at all in {n_tok} tokens: {sum(active_frac)/len(active_frac)*100:.1f}% per layer")
    R["gini_mean"] = sum(ginis) / len(ginis)
    R["top_share"] = {str(k): sum(v)/len(v) for k, v in top_share.items()}
    R["active_frac_mean"] = sum(active_frac) / len(active_frac)

    # 2. union growth: n tokens sampled far apart (independent 'streams') -----------------------
    print("\n== 2. expert union vs number of independent tokens (mean over layers; ×1 = k experts) ==")
    rnd = random.Random(1)
    R["union"] = {}
    for n in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        if n > n_tok: break
        vals = []
        for l in layers:
            rows = per_layer[l]
            for _ in range(8):
                pick = rnd.sample(range(n_tok), n)
                u = set()
                for t in pick: u.update(rows[t][0])
                vals.append(len(u))
        mean_u = sum(vals) / len(vals)
        indep = n_expert * (1 - (1 - n_used / n_expert) ** n)
        print(f"  n={n:3d}: union {mean_u:6.1f} experts ({mean_u/n_used:5.2f}× one token; independent-routing model {indep:6.1f}, {mean_u/indep*100:5.1f}% of it)")
        R["union"][n] = mean_u

    # 3. sequence-local LRU cache hit rates ------------------------------------------------------
    print("\n== 3. sequence-local reuse: LRU cache of C experts per layer, hit rate along the token order ==")
    R["lru_hit"] = {}
    for C in (16, 32, 64, 128, 256):
        hits = 0; total = 0
        for l in layers:
            lru = OrderedDict()
            for ids, _ in per_layer[l]:
                for e in ids:
                    total += 1
                    if e in lru:
                        hits += 1; lru.move_to_end(e)
                    else:
                        lru[e] = 1
                        if len(lru) > C: lru.popitem(last=False)
        print(f"  C={C:3d} experts/layer ({C*len(layers)*9.7/1000:5.1f} GB for all layers): hit rate {hits/total*100:5.1f}%")
        R["lru_hit"][C] = hits / total

    # 4. global hot cache (top-C by frequency, measured on the first half, tested on the second) --
    print("\n== 4. global hot cache: top-C experts per layer by frequency (fit on first half of tokens, test on second) ==")
    R["hot_hit"] = {}
    half = n_tok // 2
    for C in (16, 32, 64, 128, 256):
        hits = 0; total = 0
        for l in layers:
            c = Counter()
            for ids, _ in per_layer[l][:half]: c.update(ids)
            hot = set(e for e, _ in c.most_common(C))
            for ids, _ in per_layer[l][half:]:
                for e in ids:
                    total += 1; hits += e in hot
        print(f"  C={C:3d} ({C*len(layers)*9.7/1000:5.1f} GB): hit rate {hits/total*100:5.1f}%")
        R["hot_hit"][C] = hits / total

    # 5. next-block predictability (same-token, adjacent layers): overlap of expert sets ---------
    print("\n== 5. adjacent-layer overlap for the same token (an upper bound on trivial next-block prediction) ==")
    ov = []
    for a, b in zip(layers, layers[1:]):
        ra, rb = per_layer[a], per_layer[b]
        s = 0
        for t in range(n_tok):
            s += len(set(ra[t][0]) & set(rb[t][0]))
        ov.append(s / n_tok / n_used)
    print(f"  mean fraction of block L+1's experts already in block L's set: {sum(ov)/len(ov)*100:.1f}%  (independent: {n_used/n_expert*100:.1f}%)")
    R["adjacent_overlap"] = sum(ov) / len(ov)

    # 6. routing weight concentration (how much of the gate mass is in the top experts) --------------
    print("\n== 6. routing weight concentration within a token's k (mean) ==")
    wsum = [0.0] * n_used; cnt = 0
    for l in layers:
        for ids, ws in per_layer[l]:
            srt = sorted(ws, reverse=True)
            for i in range(n_used): wsum[i] += srt[i]
            cnt += 1
    tot = sum(wsum)
    if tot > 0:
        cum = 0.0
        for i in range(n_used):
            cum += wsum[i]
            if i in (3, 7, 11, n_used - 1):
                print(f"  top {i+1:2d} of {n_used}: {cum/tot*100:5.1f}% of gate mass")
        R["gate_mass_top"] = [w / tot for w in wsum]
    else:
        print("  (weights not captured)")

    if out_json:
        json.dump(R, open(out_json, "w"), indent=1)
        print(f"\nwrote {out_json}")

if __name__ == "__main__":
    main()
