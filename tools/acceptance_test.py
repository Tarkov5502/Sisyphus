"""
tools/acceptance_test.py — measure the draft-acceptance rate that FINDINGS.md §5.1 assumes.

The 387K/503K rows in RESULTS.md assume a draft model's tokens are accepted 70% of the time
by K3. This script measures that on YOUR prompts, without the engine, for ~$0.25 of API:

  1. get K3's greedy continuation for each prompt from a hosted endpoint (OpenRouter);
  2. run a same-tokenizer draft model locally (Moonlight-16B-A3B-Instruct, Moonshot's own
     3B-active MoE — same 163,840-token vocab as K3; a draft with a different tokenizer
     cannot be used for speculative decoding at all);
  3. teacher-force: at every position of K3's continuation, ask the draft for its argmax
     given the prompt + K3's tokens so far; a match is an "accepted draft";
  4. report per-position acceptance and, the number the tape model needs, the empirical
     expected accepted tokens per sweep for k drafts: mean over positions of the run length
     of consecutive matches (capped at k), plus one for the verifier's own token.

Greedy-vs-greedy match is a slightly conservative proxy for rejection-sampling acceptance.

Setup (Windows, CPU is fine for the draft; ~10 GB download):
    pip install llama-cpp-python openai huggingface_hub
    $env:OPENROUTER_API_KEY = "sk-or-..."
    python tools/acceptance_test.py --prompts my_prompts.jsonl --n 50 --k 4

prompts.jsonl: one {"prompt": "..."} per line — use the real jobs the night would run.
Without --prompts a small built-in sample is used (measures the mechanism, not your domain).
Outputs acceptance_results.json next to the repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SAMPLE_PROMPTS = [
    "Audit the following Python function for concurrency bugs and list each with a fix:\n\n"
    "def transfer(a, b, amt):\n    if a.balance >= amt:\n        a.balance -= amt\n        b.balance += amt\n",
    "You are scoring an HVAC services company for acquisition. Revenue $4.2M, EBITDA $610K, 3 technicians, "
    "owner-operated for 19 years, 62% residential. Give a 1-10 score and a two-sentence rationale.",
    "Classify this insurance denial letter's primary reason code and suggest the strongest appeal argument:\n"
    "\"Claim denied: procedure 97110 not medically necessary per plan guidelines; documentation does not support frequency.\"",
    "Extract the vendor, invoice number, total, and due date as JSON from:\n"
    "\"ACME Industrial Supply — Invoice #A-77812 — Total due $12,480.00 — Net 30 from 08/14/2026\"",
    "Explain, for an engineer, why a linear-attention recurrent state cannot be rolled back after speculative decoding without either a copy or a replay of the accepted updates.",
]


def get_k3_targets(prompts, model, max_tokens, cache_path):
    from openai import OpenAI
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    out = []
    for i, p in enumerate(prompts):
        if p in cache:
            out.append(cache[p]); continue
        # K3 is a reasoning model: without this it spends max_tokens thinking and returns
        # empty content. Ask OpenRouter to exclude reasoning; fall back to the reasoning
        # text if content is still empty so nothing is silently dropped.
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": p}],
                                           temperature=0.0, max_tokens=max_tokens,
                                           extra_body={"reasoning": {"exclude": True, "effort": "low"}})
        msg = r.choices[0].message
        txt = msg.content or getattr(msg, "reasoning", None) or ""
        if not txt:
            print(f"  K3 target {i+1}: EMPTY response (skipped)", flush=True)
        cache[p] = txt; out.append(txt)
        json.dump(cache, open(cache_path, "w"), indent=1)
        print(f"  K3 target {i+1}/{len(prompts)}: {len(txt)} chars", flush=True)
    return out


def load_draft(repo, filename, n_ctx, n_gpu_layers):
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    path = filename if os.path.exists(filename) else hf_hub_download(repo, filename)
    print(f"draft model: {path}", flush=True)
    return Llama(model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, logits_all=True, verbose=False)


def chat_prompt(llm, user):
    """Render with the draft's chat template if it has one; else a plain fallback."""
    try:
        meta = llm.metadata
        tmpl = meta.get("tokenizer.chat_template")
    except Exception:
        tmpl = None
    if tmpl:
        try:
            from jinja2 import Template
            return Template(tmpl).render(messages=[{"role": "user", "content": user}], add_generation_prompt=True,
                                         bos_token="", eos_token="")
        except Exception:
            pass
    return f"User: {user}\nAssistant: "


def measure(llm, prompt, target, k):
    import numpy as np
    p_ids = llm.tokenize(chat_prompt(llm, prompt).encode("utf-8"), add_bos=True, special=True)
    t_ids = llm.tokenize(target.encode("utf-8"), add_bos=False, special=False)
    if not t_ids:
        return None
    # one forward pass over prompt + target with logits at every position (teacher forcing)
    ids = p_ids + t_ids
    llm.reset()
    llm.eval(ids)
    logits = np.asarray(llm.scores[: len(ids)])          # [T, vocab]
    # prediction for target position j comes from logits at position len(p_ids)+j-1
    lg = logits[len(p_ids) - 1: len(ids) - 1]
    tgt = np.asarray(t_ids)
    pred = lg.argmax(axis=1)
    match = (pred == tgt).astype(int)
    # is K3's token within the draft's top-3 / top-8? (how much a better-calibrated draft
    # or a multi-candidate verifier could recover)
    top8 = np.argpartition(-lg, 8, axis=1)[:, :8]
    in_top8 = (top8 == tgt[:, None]).any(axis=1)
    top3 = np.argsort(-lg, axis=1)[:, :3]
    in_top3 = (top3 == tgt[:, None]).any(axis=1)
    # empirical accepted-run length for k drafts, at every position
    runs = []
    for s in range(len(match)):
        r = 0
        while r < k and s + r < len(match) and match[s + r]:
            r += 1
        runs.append(r)
    return {"n": len(match), "match": match.tolist(), "runs": runs,
            "top3": int(in_top3.sum()), "top8": int(in_top8.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", help="jsonl with {'prompt': ...} per line")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--k3-model", default="moonshotai/kimi-k3")
    ap.add_argument("--draft-repo", default="mmnga/Moonlight-16B-A3B-Instruct-gguf")
    ap.add_argument("--draft-file", default="Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
                    help="a local .gguf path or a filename inside --draft-repo (verify the repo/file exists)")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--out", default="acceptance_results.json")
    a = ap.parse_args()

    if a.prompts:
        prompts = [json.loads(l)["prompt"] for l in open(a.prompts, encoding="utf-8") if l.strip()][: a.n]
    else:
        prompts = SAMPLE_PROMPTS
        print(f"NOTE: using the {len(SAMPLE_PROMPTS)} built-in sample prompts (not repeated); pass --prompts "
              f"with your real jobs for a domain number.")
    prompts = list(dict.fromkeys(prompts))          # dedupe: repeats would only inflate token counts
    if "OPENROUTER_API_KEY" not in os.environ:
        sys.exit("set OPENROUTER_API_KEY (hosted K3 is the reference; ~$0.25 for 50 prompts)")

    print(f"1/3 fetching {len(prompts)} greedy K3 continuations ({a.k3_model})...")
    targets = get_k3_targets(prompts, a.k3_model, a.max_tokens, "k3_targets_cache.json")

    print("2/3 loading draft model...")
    llm = load_draft(a.draft_repo, a.draft_file, a.n_ctx, a.n_gpu_layers)
    vocab = llm.n_vocab()
    print(f"   draft vocab {vocab} (K3: 163840){'  OK' if vocab == 163840 else '  MISMATCH — not usable as a K3 draft'}")

    print("3/3 teacher-forced acceptance...")
    total = matched = top3 = top8 = 0; runs_all = []; per_prompt = []
    t0 = time.time()
    for i, (p, t) in enumerate(zip(prompts, targets)):
        r = measure(llm, p, t, a.k)
        if not r or not t.strip():
            continue
        m = sum(r["match"]); total += r["n"]; matched += m; runs_all += r["runs"]
        top3 += r["top3"]; top8 += r["top8"]
        per_prompt.append({"i": i, "tokens": r["n"], "acceptance": m / r["n"]})
        print(f"  {i+1:3d}/{len(prompts)}  {r['n']:4d} tok  acceptance {m/r['n']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    acc = matched / total if total else 0.0
    exp_run = sum(runs_all) / len(runs_all) if runs_all else 0.0
    geo = sum(acc ** j for j in range(1, a.k))      # what the tape model assumes from a flat rate
    res = {"prompts": len(per_prompt), "tokens": total, "acceptance": acc, "k": a.k,
           "tokens_per_sweep_empirical": 1 + exp_run, "tokens_per_sweep_geometric_from_rate": 1 + geo,
           "draft": a.draft_file, "k3_model": a.k3_model, "per_prompt": per_prompt}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nacceptance {acc:.3f} over {total} tokens from {len(per_prompt)} prompts; tokens per sweep at k={a.k}: "
          f"{1+exp_run:.2f} empirical (geometric-from-rate {1+geo:.2f}); tape model assumed 2.77")
    print(f"K3's token in draft's top-3: {top3/total:.3f}, top-8: {top8/total:.3f}  "
          f"(ceiling for a better-calibrated draft / multi-candidate verification)")
    res["top3_rate"] = top3 / total if total else 0.0; res["top8_rate"] = top8 / total if total else 0.0
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
