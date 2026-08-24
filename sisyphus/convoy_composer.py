"""Atlas convoy composer — route-aware batch composition + route-chaining (V2.11.2 + V2.16.4).
Groups jobs into convoys maximizing route overlap; orders convoys so each inherits
the previous convoy's warm roads. Stdlib only.

Job dict: {"id": str, "domain": str, "prompt": str, "route": optional list[int]}
  - route: expert ids from history (bus-route replay) when available
  - fallback similarity: prompt token-shingle Jaccard within/across domains
"""
from collections import Counter

def _shingles(text, k=4):
    toks = [t for t in text.lower().split() if t]
    return {" ".join(toks[i:i+k]) for i in range(max(1, len(toks)-k+1))}

def _jaccard(a, b):
    if not a or not b: return 0.0
    i = len(a & b)
    return i / (len(a) + len(b) - i)

def job_similarity(j1, j2):
    """Route overlap when both have history; else domain + content similarity."""
    r1, r2 = j1.get("route"), j2.get("route")
    if r1 and r2:
        return _jaccard(set(r1), set(r2))
    s = 0.6 if j1.get("domain") and j1.get("domain") == j2.get("domain") else 0.0
    return s + 0.4 * _jaccard(_shingles(j1.get("prompt","")), _shingles(j2.get("prompt","")))

def compose(jobs, convoy_size=8, min_coherence=0.15):
    """Greedy agglomerative: grow each convoy around a seed by best-similarity,
    refusing riders below min_coherence (admission control)."""
    remaining = list(jobs)
    convoys = []
    while remaining:
        seed = remaining.pop(0)
        convoy = [seed]
        while len(convoy) < convoy_size and remaining:
            best_i, best_s = None, -1.0
            for i, cand in enumerate(remaining):
                s = sum(job_similarity(cand, m) for m in convoy) / len(convoy)
                if s > best_s: best_i, best_s = i, s
            if best_s < min_coherence: break          # admission control: wrong car waits
            convoy.append(remaining.pop(best_i))
        convoys.append(convoy)
    return convoys

def convoy_signature(convoy):
    """Aggregate route/content signature for chaining."""
    routes = [set(j["route"]) for j in convoy if j.get("route")]
    if routes:
        return set().union(*routes), "route"
    sig = set()
    for j in convoy: sig |= _shingles(j.get("prompt",""))
    return sig, "content"

def chain(convoys):
    """Route-chaining: greedy nearest-neighbor order so consecutive convoys
    maximize shared signature (warm-road inheritance)."""
    if len(convoys) <= 2: return convoys
    sigs = [convoy_signature(c)[0] for c in convoys]
    order = [0]; used = {0}
    while len(order) < len(convoys):
        last = sigs[order[-1]]
        best_i, best_s = None, -1.0
        for i in range(len(convoys)):
            if i in used: continue
            s = _jaccard(last, sigs[i])
            if s > best_s: best_i, best_s = i, s
        order.append(best_i); used.add(best_i)
    return [convoys[i] for i in order]

def plan_night(jobs, convoy_size=8, min_coherence=0.15):
    """Full pipeline: compose -> chain -> flat schedule with convoy indices."""
    convoys = chain(compose(jobs, convoy_size, min_coherence))
    return [{"convoy": ci, "position": pi, "id": j["id"], "domain": j.get("domain")}
            for ci, c in enumerate(convoys) for pi, j in enumerate(c)], convoys
