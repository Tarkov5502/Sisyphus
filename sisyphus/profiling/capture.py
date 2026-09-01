"""
sisyphus.profiling.capture — log real routes from a running Hugging Face MoE model.

This is the one file in the package that runs on the RENTED box, not the laptop. It
needs torch + transformers there; nothing else in Sisyphus imports it, so the scheduler
core stays stdlib-only.

How it works. Every MoE layer has a router module that, given the token hidden states,
produces the top-k expert indices and their normalised weights. We register a forward
hook on each router, read those two tensors off its output, and write one RouteRecord
per token per layer. Module names and output shapes differ between model families, so:

  * `discover_routers(model)` prints every module whose name matches the router regex,
    with the shapes it returns on a one-token dry run — run this FIRST and set
    `router_regex` / `extractor` to what you see;
  * `extractor` turns a router's output into (indices[T,k], weights[T,k]) tensors.
    Two are built in: "tuple" (output is (topk_idx, topk_weight, ...)) and "logits"
    (output is [T, E] logits or probs -> softmax -> topk). Pass a callable for
    anything else.

Profile with batch size 1 (one prompt at a time). Routers see hidden states flattened
to [tokens, hidden], so with batch > 1 the token->call mapping is ambiguous; at batch 1
it is exact, and a profiling run is not a throughput run.

Untested against Kimi-K3 specifically (no weights here); the regex default targets the
`mlp.gate` naming used by the DeepSeek/Kimi family and `discover_routers` exists
precisely so the first five minutes on the box are spent confirming it, not debugging.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .schema import RouteLogWriter, RouteRecord

DEFAULT_ROUTER_REGEX = r"^model\.layers\.(\d+)\.mlp\.gate$"


def _import_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("sisyphus.profiling.capture needs torch + transformers "
                           "(install them on the rented box only)") from e


# --------------------------------------------------------------------------- #
#  Extractors: router output -> (indices [T,k], weights [T,k])
# --------------------------------------------------------------------------- #
def extract_tuple(output, topk: int | None = None):
    """Router returns a tuple whose first two tensor entries are (topk_idx, topk_w)."""
    tensors = [o for o in output if hasattr(o, "shape")]
    if len(tensors) < 2:
        raise ValueError("expected (indices, weights, ...) tuple from router")
    a, b = tensors[0], tensors[1]
    # some routers return (weights, indices): the integer one is the indices
    if a.dtype.is_floating_point and not b.dtype.is_floating_point:
        a, b = b, a
    return a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])


def extract_logits(output, topk: int = 16):
    """Router returns logits or probabilities [T, E]; take softmax top-k, renormalise."""
    torch = _import_torch()
    x = output[0] if isinstance(output, (tuple, list)) else output
    x = x.reshape(-1, x.shape[-1]).float()
    probs = torch.softmax(x, dim=-1) if (x.min() < 0 or x.sum(-1).mean() > 1.5) else x
    w, idx = torch.topk(probs, topk, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    return idx, w


EXTRACTORS: dict[str, Callable] = {"tuple": extract_tuple, "logits": extract_logits}


# --------------------------------------------------------------------------- #
#  Discovery
# --------------------------------------------------------------------------- #
def discover_routers(model, router_regex: str = r"(gate|router)", limit: int = 12) -> list[str]:
    """Print candidate router modules and their output shapes on a 1-token dry run."""
    torch = _import_torch()
    pat = re.compile(router_regex)
    names = [n for n, _ in model.named_modules() if pat.search(n)]
    print(f"{len(names)} modules match /{router_regex}/; first {limit}:")
    shapes: dict[str, str] = {}
    handles = []

    def mk(name):
        def hook(_m, _i, out):
            def sh(o):
                return "x".join(str(s) for s in o.shape) + f":{o.dtype}" if hasattr(o, "shape") else type(o).__name__
            shapes[name] = ", ".join(sh(o) for o in out) if isinstance(out, (tuple, list)) else sh(out)
        return hook

    for n, m in model.named_modules():
        if n in names[:limit]:
            handles.append(m.register_forward_hook(mk(n)))
    try:
        dev = next(model.parameters()).device
        with torch.no_grad():
            model(input_ids=torch.tensor([[1]], device=dev))
    finally:
        for h in handles:
            h.remove()
    for n in names[:limit]:
        print(f"  {n:60} -> {shapes.get(n, '(not called)')}")
    return names


# --------------------------------------------------------------------------- #
#  Capture
# --------------------------------------------------------------------------- #
@dataclass
class RouterCapture:
    """Hooks every router matching `router_regex` and streams RouteRecords to `writer`.

    Drive it with `begin_call()` before each prompt, `set_phase()` when generation
    starts, and it does the token bookkeeping: a forward pass that sees T tokens is
    prefill (tokens 0..T-1 of the prompt), each later 1-token pass is one decode step.
    """
    model: object
    writer: RouteLogWriter
    router_regex: str = DEFAULT_ROUTER_REGEX
    extractor: str | Callable = "tuple"
    topk: int = 16
    _handles: list = field(default_factory=list)
    _call: str = ""
    _domain: str = ""
    _phase: str = "prefill"
    _tok_base: dict[str, int] = field(default_factory=dict)      # phase -> tokens so far
    _fired: int = 0                                              # routers seen this pass
    records: int = 0

    def __post_init__(self):
        pat = re.compile(self.router_regex)
        fn = EXTRACTORS[self.extractor] if isinstance(self.extractor, str) else self.extractor
        matched = 0
        for name, mod in self.model.named_modules():
            m = pat.search(name)
            if not m:
                continue
            layer = int(m.group(1)) if m.groups() else matched
            self._handles.append(mod.register_forward_hook(self._hook(layer, fn)))
            matched += 1
        if not matched:
            raise ValueError(f"no modules matched /{self.router_regex}/ — run discover_routers()")
        self.layers = matched

    def _hook(self, layer: int, fn: Callable):
        def hook(_mod, _inp, out):
            idx, w = fn(out) if fn is not extract_logits else fn(out, self.topk)
            idx = idx.detach().cpu().tolist()
            w = w.detach().cpu().float().tolist()
            base = self._tok_base.get(self._phase, 0)
            for t, (ei, wi) in enumerate(zip(idx, w)):
                self.writer.write(RouteRecord(self._call, self._domain, self._phase,
                                              base + t, layer, tuple(int(e) for e in ei),
                                              tuple(float(g) for g in wi)))
                self.records += 1
            # advance the token base once every router has reported for this pass
            # (MoE layer indices need not start at 0 — K3's first block is dense)
            self._fired += 1
            if self._fired >= self.layers:
                self._fired = 0
                self._tok_base[self._phase] = base + len(idx)
        return hook

    def begin_call(self, call_id: str, domain: str) -> None:
        self._call, self._domain = call_id, domain
        self._phase = "prefill"
        self._tok_base = {}
        self._fired = 0

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def profile_prompts(model, tokenizer, jobs: Sequence[dict], out_path: str,
                    max_new_tokens: int = 128, router_regex: str = DEFAULT_ROUTER_REGEX,
                    extractor: str | Callable = "tuple", topk: int = 16) -> int:
    """Run each job {"id", "domain", "prompt"} through the model at batch 1, logging
    routes. Prefill is one forward pass; decode is greedy, one token per pass, driven
    manually so the phase boundary is exact. Returns the record count."""
    torch = _import_torch()
    dev = next(model.parameters()).device
    total = 0
    with RouteLogWriter(out_path) as w:
        cap = RouterCapture(model, w, router_regex, extractor, topk)
        try:
            for j in jobs:
                cap.begin_call(j["id"], j.get("domain", ""))
                ids = tokenizer(j["prompt"], return_tensors="pt").input_ids.to(dev)
                t0 = time.time()
                with torch.no_grad():
                    out = model(input_ids=ids, use_cache=True)
                    past = out.past_key_values
                    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                    cap.set_phase("decode")
                    for _ in range(max_new_tokens):
                        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                        past = out.past_key_values
                        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                        if tokenizer.eos_token_id is not None and int(nxt) == tokenizer.eos_token_id:
                            break
                print(f"{j['id']:16} {j.get('domain',''):12} {ids.shape[1]:5d} prompt tok  "
                      f"{time.time() - t0:6.1f}s  records so far {cap.records:,}")
            total = cap.records
        finally:
            cap.close()
    return total


if __name__ == "__main__":  # pragma: no cover
    print(__doc__)
