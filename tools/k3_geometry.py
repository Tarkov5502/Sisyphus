"""
tools/k3_geometry.py — read the real Kimi-K3 constants off the GGUF shards.

Usage:  python tools/k3_geometry.py <path-to-any-shard-or-its-folder> [out.json]

Reads ONLY headers (no weights are loaded). From shard 1 it takes the architecture
metadata (block count, expert count, top-k, kv_lora_rank, rope dim, ...). From every
shard it can find in the same folder it sums tensor byte sizes, splitting routed-expert
tensors (*_exps.*) from everything else, which gives EXPERT_MB and TRUNK_GB exactly.
Writes a JSON the scheduler repo can ingest, and prints a ready geometry.py block.
"""
import glob, json, os, re, sys
from collections import defaultdict

try:
    from gguf import GGUFReader
except ImportError:
    sys.exit("pip install gguf")

def kv_scalar(reader, key):
    f = reader.fields.get(key)
    if f is None:
        return None
    v = f.parts[f.data[0]]
    try:
        return v.tolist()[0] if hasattr(v, "tolist") else v
    except Exception:
        return None

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    p = sys.argv[1]
    folder = p if os.path.isdir(p) else os.path.dirname(p)
    shards = sorted(glob.glob(os.path.join(folder, "*.gguf")))
    if not shards:
        sys.exit(f"no .gguf files in {folder}")
    first = [s for s in shards if re.search(r"-00001-of-", s)] or shards[:1]
    r = GGUFReader(first[0])
    arch = kv_scalar(r, "general.architecture")
    if isinstance(arch, (bytes, bytearray)):
        arch = arch.decode()
    if hasattr(arch, "tobytes"):
        arch = arch.tobytes().decode(errors="ignore")
    meta = {}
    for k in list(r.fields):
        if k.startswith(f"{arch}.") or k in ("general.file_type", "general.name", "split.count", "split.tensors.count"):
            v = kv_scalar(r, k)
            if isinstance(v, (int, float)):
                meta[k] = v
    # tensor sizes across all shards
    by_kind = defaultdict(int); per_expert_bytes = []; n_tensors = 0
    expert_count = None
    for s in shards:
        rr = r if s == first[0] else GGUFReader(s)
        for t in rr.tensors:
            n = t.name; nbytes = int(t.n_bytes); n_tensors += 1
            if "_exps" in n:                 # ffn_gate_exps / ffn_up_exps / ffn_down_exps: [E, ...]
                by_kind["experts"] += nbytes
                e = int(t.shape[-1]) if len(t.shape) == 3 else None   # gguf shape is reversed; experts is last
                if e: expert_count = expert_count or e
            elif "shexp" in n:
                by_kind["shared_experts"] += nbytes
            elif "attn" in n:
                by_kind["attention"] += nbytes
            elif "token_embd" in n or "output" in n:
                by_kind["embed_output"] += nbytes
            elif "ffn" in n:
                by_kind["dense_ffn"] += nbytes
            else:
                by_kind["other"] += nbytes
    total = sum(by_kind.values())
    a = arch
    g = lambda k, d=None: meta.get(f"{a}.{k}", d)
    blocks = g("block_count"); E = g("expert_count") or expert_count; K = g("expert_used_count")
    leading_dense = g("leading_dense_block_count", 0) or 0
    moe_layers = (blocks - leading_dense) if blocks else None
    kv_lora = g("attention.kv_lora_rank"); rope = g("rope.dimension_count")
    kv_per_tok_layer_vals = (kv_lora + rope) if (kv_lora and rope) else None
    out = {
        "architecture": a, "shards_found": len(shards), "shards_expected": meta.get("split.count"),
        "tensors": n_tensors, "meta": meta,
        "bytes_by_kind": dict(by_kind), "total_gb": total / 1e9,
        "blocks": blocks, "leading_dense_blocks": leading_dense, "moe_layers": moe_layers,
        "experts": E, "topk": K, "shared_experts": g("expert_shared_count"),
        "kv_lora_rank": kv_lora, "rope_dim": rope,
        "kv_values_per_token_layer": kv_per_tok_layer_vals,
        "kv_mb_per_token_fp16": (kv_per_tok_layer_vals * 2 * blocks / 1e6) if (kv_per_tok_layer_vals and blocks) else None,
        "expert_mb": (by_kind["experts"] / 1e6 / (moe_layers * E)) if (moe_layers and E and by_kind["experts"]) else None,
        "trunk_gb": (total - by_kind["experts"]) / 1e9 if total else None,
        "complete": meta.get("split.count") in (None, len(shards)),
    }
    dst = sys.argv[2] if len(sys.argv) > 2 else "k3_geometry.json"
    json.dump(out, open(dst, "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in out.items() if k not in ("meta",)}, indent=2, default=str))
    print(f"\nwrote {dst}")
    if not out["complete"]:
        print(f"NOTE: only {len(shards)} of {meta.get('split.count')} shards present — expert/trunk byte totals are partial; metadata is complete.")
    if out["kv_mb_per_token_fp16"]:
        print(f"\n# geometry.py\nBLOCKS = {blocks}\nLAYERS = {moe_layers}\nEXPERTS = {E}\nTOPK = {K}\n"
              f"KV_VALUES_PER_TOKEN_BLOCK = {kv_per_tok_layer_vals}   # kv_lora_rank {kv_lora} + rope {rope}\n"
              f"KV_MB_PER_TOKEN = {out['kv_mb_per_token_fp16']:.4f}")
        if out["expert_mb"]:
            print(f"EXPERT_MB = {out['expert_mb']:.2f}\nTRUNK_GB = {out['trunk_gb']:.1f}")

if __name__ == "__main__":
    main()
