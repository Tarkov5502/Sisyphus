"""
tools/repack_plan.py — plan the tape-engine image from the real GGUF shards (read-only).

ENGINE_DESIGN.md §8: the engine sweeps a BLOCK-MAJOR, EXPERT-MAJOR image, split across
drives in proportion to measured bandwidth so every sweep reads all drives at once. This
tool builds that plan from the shards' headers without copying a byte:

  1. parse every shard's tensor table (name, offset, size, shard);
  2. order tensors block-major: for each block, trunk tensors first (attention/KDA, router,
     shared experts, norms), then routed experts gate/up/down for expert 0..895 — a sweep is
     then one forward pass over the image;
  3. assign each expert (and each trunk tensor) to a drive, weighted round-robin by measured
     bandwidth, so the per-drive byte shares match the speed shares;
  4. write a manifest (JSON): per-image-file tensor list with source shard/offset/size and
     destination offset, plus the split summary and the predicted sweep seconds.

The manifest is what `repack` (the copier, not written yet) and the streamer consume. It is
also a check on the model's own numbers: bytes per block, per expert, per drive.

    python tools/repack_plan.py <shard-dir> --drives "990pro:5.55,new:5.5,sn570:3.0"
                                [--out repack_manifest.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
import sys
from collections import defaultdict

GGUF_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def parse_shard(path: str):
    """Tensor table with exact byte sizes (from offsets) for one shard."""
    size = os.path.getsize(path)
    f = open(path, "rb")

    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    def rstr():
        n = rd("Q")
        return f.read(n).decode("utf-8", "replace")

    def skip(t):
        if t == 8:
            rstr()
        elif t == 9:
            et = rd("I"); n = rd("Q")
            for _ in range(n):
                skip(et)
        else:
            f.read(struct.calcsize(GGUF_TYPES[t]))

    assert f.read(4) == b"GGUF", path
    rd("I"); nt = rd("Q"); nkv = rd("Q")
    align = 32
    for _ in range(nkv):
        k = rstr(); t = rd("I")
        if k == "general.alignment":
            align = rd("I")
        else:
            skip(t)
    tens = []
    for _ in range(nt):
        name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; typ = rd("I"); off = rd("Q")
        tens.append({"name": name, "dims": dims, "type": typ, "rel_off": off})
    hdr = f.tell()
    data = (hdr + align - 1) // align * align
    tens.sort(key=lambda t: t["rel_off"])
    for i, t in enumerate(tens):
        end = tens[i + 1]["rel_off"] if i + 1 < len(tens) else size - data
        t["size"] = end - t["rel_off"]
        t["src_off"] = data + t["rel_off"]
        t["shard"] = os.path.basename(path)
    return tens


def classify(name: str):
    """(block, kind, expert_group). kind: trunk | experts | global."""
    m = re.match(r"blk\.(\d+)\.(.*)", name)
    if not m:
        return -1, "global", None
    blk, rest = int(m.group(1)), m.group(2)
    if "_exps" in rest:
        return blk, "experts", rest.split(".")[0]      # ffn_gate_exps / ffn_up_exps / ffn_down_exps
    return blk, "trunk", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--drives", default="990pro:5.55,new:5.5,sn570:3.0",
                    help="name:GBps,... measured sequential read speeds")
    ap.add_argument("--experts", type=int, default=896)
    ap.add_argument("--out", default="repack_manifest.json")
    ap.add_argument("--overlap", type=float, default=0.85)
    a = ap.parse_args()

    drives = [(n, float(b)) for n, b in (d.split(":") for d in a.drives.split(","))]
    total_bw = sum(b for _, b in drives)

    shards = sorted(glob.glob(os.path.join(a.shard_dir, "*.gguf")))
    if not shards:
        sys.exit("no shards")
    tens = []
    for s in shards:
        tens.extend(parse_shard(s))
    print(f"{len(tens)} tensors from {len(shards)} shards, {sum(t['size'] for t in tens)/1e9:.1f} GB")

    # ---- order: global (embeddings) first, then blocks; within a block trunk then experts
    by_block = defaultdict(lambda: {"trunk": [], "experts": defaultdict(list), "global": []})
    for t in tens:
        blk, kind, grp = classify(t["name"])
        if kind == "experts":
            by_block[blk]["experts"][grp].append(t)
        else:
            by_block[blk][kind].append(t)
    blocks = sorted(b for b in by_block if b >= 0)

    # ---- weighted round-robin assignment to drives (deficit round-robin on bytes)
    credit = {n: 0.0 for n, _ in drives}
    placed = {n: [] for n, _ in drives}
    dest_off = {n: 0 for n, _ in drives}
    order = []                       # image order (for the sweep) as (drive, tensor)

    def place(t):
        # give the drive whose byte share is furthest below its bandwidth share
        for n, b in drives:
            credit[n] += b / total_bw * t["size"]
        n = max(credit, key=credit.get)
        credit[n] -= t["size"]
        t["drive"] = n; t["dst_off"] = dest_off[n]; dest_off[n] += t["size"]
        placed[n].append(t); order.append(t)

    for t in by_block[-1]["global"]:
        place(t)
    expert_slices = 0
    for blk in blocks:
        for t in sorted(by_block[blk]["trunk"], key=lambda t: t["name"]):
            place(t)
        # experts are one big tensor per (gate|up|down) in GGUF ([E, in, out] contiguous);
        # the image keeps each of the three tensors whole (a 3.2 GB contiguous run) and the
        # streamer reads expert e as three sub-ranges — sequential enough at 9.7 MB total.
        for grp in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            for t in by_block[blk]["experts"].get(grp, []):
                place(t); expert_slices += 1

    # ---- summaries
    total = sum(t["size"] for t in tens)
    exp_bytes = sum(t["size"] for t in tens if classify(t["name"])[1] == "experts")
    trunk_bytes = total - exp_bytes
    print(f"blocks {len(blocks)}, expert tensors {expert_slices}, experts {exp_bytes/1e9:.1f} GB, trunk {trunk_bytes/1e9:.1f} GB")
    print(f"per expert: {exp_bytes/(len([b for b in blocks if by_block[b]['experts']])*a.experts)/1e6:.3f} MB")
    print("\nsplit by measured bandwidth:")
    slowest = 0.0
    for n, b in drives:
        gb = dest_off[n] / 1e9
        t_read = gb / b
        slowest = max(slowest, t_read)
        print(f"  {n:8} {b:5.2f} GB/s  {gb:6.1f} GB  ({gb/total*1e9*100:4.1f}% of bytes vs {b/total_bw*100:4.1f}% of bandwidth)  reads in {t_read:5.1f} s")
    print(f"\npredicted sweep: {slowest:.1f} s at full overlap, {slowest/a.overlap:.1f} s at {a.overlap:.0%}  "
          f"(aggregate {total/1e9/slowest:.2f} GB/s)")

    manifest = {
        "source_dir": os.path.abspath(a.shard_dir), "shards": [os.path.basename(s) for s in shards],
        "drives": [{"name": n, "gbps": b, "bytes": dest_off[n], "image": f"k3_{n}.img"} for n, b in drives],
        "blocks": len(blocks), "experts_per_block": a.experts,
        "total_bytes": total, "expert_bytes": exp_bytes, "trunk_bytes": trunk_bytes,
        "predicted_sweep_s": slowest, "predicted_sweep_s_overlap": slowest / a.overlap,
        "order": [{"name": t["name"], "block": classify(t["name"])[0], "kind": classify(t["name"])[1],
                   "shard": t["shard"], "src_off": t["src_off"], "size": t["size"],
                   "drive": t["drive"], "dst_off": t["dst_off"], "type": t["type"], "dims": t["dims"]}
                  for t in order],
    }
    json.dump(manifest, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out} ({len(order)} entries)")


if __name__ == "__main__":
    main()
