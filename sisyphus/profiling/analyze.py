"""
sisyphus.profiling.analyze — turn a route log into the numbers routing.py guesses.

For each bracketed unknown in `sisyphus.routing` this produces a measured value:

  hot_frac            smallest fraction of experts (per layer, per domain) that carries
                      `mass` of the routed gate weight — reported at 0.60/0.75/0.85 so
                      the "does the hot set fit in RAM" question has a real answer
  hot_mass            fraction of routed PICKS that land in the top `hot_frac` (default
                      10%) of experts — count-based, the quantity RoutingModel.hot_mass
                      means; the gate-weighted share is reported beside it
  coherence           mean Jaccard of hot sets between calls of the SAME domain vs
                      calls of DIFFERENT domains — the convoy premise, measured
  cold_weight_ratio   mean gate of a pick outside the hot set / mean gate inside it
  skippable_mass      gate mass that rides on picks with gate <= g, for several g —
                      the ceiling on residency-aware skipping at each quality budget
  unique_per_layer    distinct experts touched per layer by N tokens (the prefill
                      working set; check against RoutingModel.expected_unique_experts)
  drift               how much a call's hot set changes from the first to the second
                      half of its decode — whether "domain has a home turf" holds
                      within a call

Everything is computed for decode and prefill separately; decode is what the
scheduler runs, prefill is what it survives.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from ..geometry import EXPERTS
from .schema import RouteRecord, read_log

MASS_LEVELS = (0.60, 0.75, 0.85)
GATE_LEVELS = (0.02, 0.04, 0.06, 0.08)


@dataclass
class PhaseProfile:
    phase: str
    records: int = 0
    tokens: int = 0
    layers: int = 0
    experts_seen: int = 0
    topk: float = 0.0
    hot_frac_at_mass: dict[str, float] = field(default_factory=dict)  # "0.75" -> frac (gate mass)
    hot_mass_at_frac: dict[str, float] = field(default_factory=dict)  # "0.10" -> pick fraction
    hot_gate_at_frac: dict[str, float] = field(default_factory=dict)  # "0.10" -> gate-mass share
    cold_weight_ratio: float = 0.0
    coherence_same_domain: float = 0.0
    coherence_cross_domain: float = 0.0
    skippable_mass_at_gate: dict[str, float] = field(default_factory=dict)
    skippable_picks_at_gate: dict[str, float] = field(default_factory=dict)
    unique_per_layer_at_tokens: dict[str, float] = field(default_factory=dict)
    drift_jaccard: float = 0.0
    per_domain_calls: dict[str, int] = field(default_factory=dict)


@dataclass
class RoutingProfile:
    source: str
    decode: PhaseProfile
    prefill: PhaseProfile

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def from_json(path: str | Path) -> "RoutingProfile":
        d = json.loads(Path(path).read_text())
        return RoutingProfile(d["source"], PhaseProfile(**d["decode"]), PhaseProfile(**d["prefill"]))

    # -- the bridge back into the simulator --------------------------------------
    def bracket(self, hot_frac: float = 0.10) -> dict:
        """kwargs for RoutingModel(**profile.bracket()) — the measured bracket."""
        p = self.decode
        return dict(hot_frac=hot_frac,
                    hot_mass=p.hot_mass_at_frac.get(f"{hot_frac:.2f}", 0.0),
                    cold_weight_ratio=p.cold_weight_ratio,
                    coherent=p.coherence_same_domain > 2.0 * max(p.coherence_cross_domain, 1e-9))

    def report(self) -> str:
        out = [f"ROUTING PROFILE — {self.source}"]
        for p in (self.decode, self.prefill):
            if not p.records:
                continue
            out.append(f"\n[{p.phase}]  {p.records:,} records, {p.tokens:,} tokens, "
                       f"{p.layers} layers, {p.experts_seen} distinct experts, top-k {p.topk:.1f}")
            out.append("  hot_frac needed for mass: " + "  ".join(
                f"{m}->{f:.3f}" for m, f in p.hot_frac_at_mass.items()))
            out.append("  hot_mass (picks) at frac: " + "  ".join(
                f"{f}->{m:.3f}" for f, m in p.hot_mass_at_frac.items()))
            out.append("  gate share at frac:       " + "  ".join(
                f"{f}->{m:.3f}" for f, m in p.hot_gate_at_frac.items()))
            out.append(f"  cold_weight_ratio:        {p.cold_weight_ratio:.3f}")
            out.append(f"  coherence same/cross:     {p.coherence_same_domain:.3f} / "
                       f"{p.coherence_cross_domain:.3f}   (calls per domain: {p.per_domain_calls})")
            out.append("  skippable mass at gate<=: " + "  ".join(
                f"{g}->{m:.3f}" for g, m in p.skippable_mass_at_gate.items()))
            out.append("  skippable picks at gate<=:" + "  ".join(
                f"{g}->{m:.3f}" for g, m in p.skippable_picks_at_gate.items()))
            out.append("  unique experts/layer at N tokens: " + "  ".join(
                f"{n}->{u:.0f}" for n, u in p.unique_per_layer_at_tokens.items()))
            out.append(f"  within-call drift (1 - Jaccard first/second half): {p.drift_jaccard:.3f}")
        return "\n".join(out)


# --------------------------------------------------------------------------- #
#  Analysis
# --------------------------------------------------------------------------- #
def _hot_frac_for_mass(mass_by_expert: Counter, total: float, target: float, n_experts: int) -> float:
    acc = 0.0
    for i, (_, m) in enumerate(mass_by_expert.most_common(), start=1):
        acc += m
        if acc >= target * total:
            return i / n_experts
    return 1.0


def _top_set(mass_by_expert: Counter, frac: float, n_experts: int) -> set[int]:
    k = max(1, int(n_experts * frac))
    return {e for e, _ in mass_by_expert.most_common(k)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _analyze_phase(recs: list[RouteRecord], phase: str, n_experts: int,
                   hot_frac: float = 0.10) -> PhaseProfile:
    p = PhaseProfile(phase=phase)
    if not recs:
        return p
    p.records = len(recs)
    layers = sorted({r.layer for r in recs})
    p.layers = len(layers)
    p.tokens = len({(r.call, r.token) for r in recs})
    p.topk = sum(len(r.experts) for r in recs) / len(recs)
    seen: set[tuple[int, int]] = set()
    for r in recs:
        seen.update((r.layer, e) for e in r.experts)
    p.experts_seen = len({e for _, e in seen})

    # mass per (domain, layer, expert) and per (call, layer, expert)
    dom_layer: dict[tuple[str, int], Counter] = defaultdict(Counter)
    dom_layer_n: dict[tuple[str, int], Counter] = defaultdict(Counter)   # pick counts
    call_layer: dict[tuple[str, int], Counter] = defaultdict(Counter)
    call_domain: dict[str, str] = {}
    call_half: dict[tuple[str, int, int], Counter] = defaultdict(Counter)
    max_tok: dict[str, int] = defaultdict(int)
    for r in recs:
        max_tok[r.call] = max(max_tok[r.call], r.token)
    for r in recs:
        call_domain[r.call] = r.domain
        for e, g in zip(r.experts, r.gates):
            dom_layer[(r.domain, r.layer)][e] += g
            dom_layer_n[(r.domain, r.layer)][e] += 1
            call_layer[(r.call, r.layer)][e] += g
            half = 0 if r.token <= max_tok[r.call] // 2 else 1
            call_half[(r.call, r.layer, half)][e] += g
    p.per_domain_calls = dict(Counter(call_domain.values()))

    # hot_frac at mass levels / hot_mass at fracs, averaged over (domain, layer)
    fracs = {m: [] for m in MASS_LEVELS}
    masses = {f: [] for f in (0.05, hot_frac, 0.15, 0.20)}
    gates = {f: [] for f in masses}
    for key, cnt in dom_layer.items():
        total = sum(cnt.values())
        n_cnt = dom_layer_n[key]
        n_total = sum(n_cnt.values())
        for m in MASS_LEVELS:
            fracs[m].append(_hot_frac_for_mass(cnt, total, m, n_experts))
        for f in masses:
            top = _top_set(cnt, f, n_experts)
            masses[f].append(sum(n_cnt[e] for e in top) / n_total if n_total else 0.0)
            gates[f].append(sum(cnt[e] for e in top) / total if total else 0.0)
    p.hot_frac_at_mass = {f"{m:.2f}": sum(v) / len(v) for m, v in fracs.items()}
    p.hot_mass_at_frac = {f"{f:.2f}": sum(v) / len(v) for f, v in masses.items()}
    p.hot_gate_at_frac = {f"{f:.2f}": sum(v) / len(v) for f, v in gates.items()}

    # cold weight ratio: mean gate of picks outside the domain's top-hot_frac set vs inside
    hot_sets = {key: _top_set(cnt, hot_frac, n_experts) for key, cnt in dom_layer.items()}
    in_g: list[float] = []
    out_g: list[float] = []
    skip_mass = {g: 0.0 for g in GATE_LEVELS}
    skip_picks = {g: 0 for g in GATE_LEVELS}
    total_mass = 0.0
    total_picks = 0
    for r in recs:
        hs = hot_sets[(r.domain, r.layer)]
        for e, g in zip(r.experts, r.gates):
            (in_g if e in hs else out_g).append(g)
            total_mass += g
            total_picks += 1
            for lvl in GATE_LEVELS:
                if g <= lvl:
                    skip_mass[lvl] += g
                    skip_picks[lvl] += 1
    mi = sum(in_g) / len(in_g) if in_g else 0.0
    mo = sum(out_g) / len(out_g) if out_g else 0.0
    p.cold_weight_ratio = (mo / mi) if mi else 0.0
    p.skippable_mass_at_gate = {f"{g:.2f}": (skip_mass[g] / total_mass if total_mass else 0.0)
                                for g in GATE_LEVELS}
    p.skippable_picks_at_gate = {f"{g:.2f}": (skip_picks[g] / total_picks if total_picks else 0.0)
                                 for g in GATE_LEVELS}

    # coherence: per layer, Jaccard of per-call hot sets, same vs cross domain
    calls = sorted(call_domain)
    same: list[float] = []
    cross: list[float] = []
    for L in layers:
        sets = {c: _top_set(call_layer[(c, L)], hot_frac, n_experts)
                for c in calls if (c, L) in call_layer}
        cs = list(sets)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                j_ = _jaccard(sets[cs[i]], sets[cs[j]])
                (same if call_domain[cs[i]] == call_domain[cs[j]] else cross).append(j_)
    p.coherence_same_domain = sum(same) / len(same) if same else 0.0
    p.coherence_cross_domain = sum(cross) / len(cross) if cross else 0.0

    # drift within a call: hot set of first half vs second half of its tokens
    drift: list[float] = []
    for c in calls:
        for L in layers:
            a, b = call_half.get((c, L, 0)), call_half.get((c, L, 1))
            if a and b:
                drift.append(1.0 - _jaccard(_top_set(a, hot_frac, n_experts),
                                            _top_set(b, hot_frac, n_experts)))
    p.drift_jaccard = sum(drift) / len(drift) if drift else 0.0

    # unique experts per layer touched by the first N tokens (pooled over calls)
    by_layer_tokens: dict[int, list[tuple[int, set[int]]]] = defaultdict(list)
    for r in recs:
        by_layer_tokens[r.layer].append((r.token, set(r.experts)))
    for n in (16, 64, 256, 1024, 4096):
        vals = []
        for L in layers:
            items = sorted(by_layer_tokens[L], key=lambda t: t[0])[:n]
            if len(items) >= min(n, 16):
                u: set[int] = set()
                for _, s in items:
                    u |= s
                vals.append(len(u))
        if vals:
            p.unique_per_layer_at_tokens[str(n)] = sum(vals) / len(vals)
    return p


def analyze(records: Iterable[RouteRecord], source: str = "log",
            n_experts: int = EXPERTS, hot_frac: float = 0.10) -> RoutingProfile:
    dec: list[RouteRecord] = []
    pre: list[RouteRecord] = []
    for r in records:
        (dec if r.phase == "decode" else pre).append(r)
    return RoutingProfile(source,
                          _analyze_phase(dec, "decode", n_experts, hot_frac),
                          _analyze_phase(pre, "prefill", n_experts, hot_frac))


def analyze_file(path: str | Path, **kw) -> RoutingProfile:
    return analyze(read_log(path), source=str(path), **kw)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m sisyphus.profiling.analyze routes.jsonl[.gz] [profile.json]")
        sys.exit(2)
    prof = analyze_file(sys.argv[1])
    print(prof.report())
    if len(sys.argv) > 2:
        prof.to_json(sys.argv[2])
        print(f"\nwrote {sys.argv[2]}; RoutingModel(**profile.bracket()) = {prof.bracket()}")
