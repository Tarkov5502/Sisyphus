#!/usr/bin/env bash
# tools/k_sweep_linux.sh - ONE overnight run that yields three measurements at once:
#   1. lever 5.27: perplexity + KL vs k=16 for expert_used_count k = 16, 14, 12, 10, 8 (teacher-forced
#      on the project's own job prompts + hosted-K3 answers);
#   2. the routing trace (every MoE block's selected experts + weights for every token, every k) ->
#      route_k<N>.bin, analysed by tools/route_stats.py: expert frequency skew (sparse sweep), LRU
#      hit rates (hot / sequence-local cache), union growth vs streams (threads/forks), next-block
#      predictability;
#   3. a second measurement of B=512 prompt-eval throughput through mmap (the dense pass).
#
#   bash tools/k_sweep_linux.sh                 # all k, 6 chunks x 512 tokens each (~1 h per k)
#   bash tools/k_sweep_linux.sh "16 12" 4       # subset of k, 4 chunks
#
# Builds a patched llama.cpp (tools/sisyphus_route.patch adds the trace hook to llama-perplexity) in
# ~/dev/llama.cpp-sisyphus (CPU build: the runs are -ngl 0 anyway; ~3 min). Keeps the machine awake
# with systemd-inhibit. Hands results to the Windows folder at the end.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
KS="${1:-16 14 12 10 8}"
CHUNKS="${2:-6}"
CTX=512
THREADS="${THREADS:-16}"
LL="$HOME/dev/llama.cpp-sisyphus"
PIN="${LLAMA_CPP_COMMIT:-f114f91}"   # the commit tools/sisyphus_route.patch was made against

# ---- model -------------------------------------------------------------------------------------------
FIRST="$(find /mnt/win -maxdepth 6 -name 'Kimi-K3-UD-Q2_K_XL-00001-of-*.gguf' 2>/dev/null | head -1)"
[ -n "$FIRST" ] || { echo "model not found under /mnt/win - run tools/linux_setup.sh (mounts) first"; exit 1; }
echo "model: $FIRST"

# ---- patched llama.cpp ---------------------------------------------------------------------------------
command -v cmake >/dev/null || { echo "need: sudo apt install -y cmake build-essential git"; exit 1; }
if [ ! -d "$LL" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LL" || exit 1
    (cd "$LL" && git fetch -q --depth 1 origin "$PIN" && git checkout -q FETCH_HEAD) || echo "(could not pin $PIN; using master)"
fi
if ! grep -q sisyphus_route_cb "$LL/tools/perplexity/perplexity.cpp"; then
    (cd "$LL" && git apply "$REPO/tools/sisyphus_route.patch") || { echo "patch did not apply to this llama.cpp - pin LLAMA_CPP_COMMIT to the commit in tools/sisyphus_route.patch's header"; exit 1; }
fi
if [ ! -x "$LL/build/bin/llama-perplexity" ]; then
    (cd "$LL" && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null \
        && cmake --build build --target llama-perplexity -j "$(nproc)" 2>&1 | tail -3) || { echo "build failed"; exit 1; }
fi
PPL="$LL/build/bin/llama-perplexity"
echo "llama-perplexity: $PPL ($(cd "$LL" && git rev-parse --short HEAD))"

# ---- eval text: prompts + hosted K3 answers -------------------------------------------------------------
# prompts.jsonl / k3_targets_cache.json are not in git (they hold hosted outputs); fetch them from the Windows repo folder
for f in prompts.jsonl k3_targets_cache.json; do
    [ -f "$f" ] && continue
    src="$(find /mnt/win -maxdepth 4 -path '*/dev/sisyphus-src/'"$f" 2>/dev/null | head -1)"
    [ -n "$src" ] && cp "$src" . && echo "copied $f from the Windows repo folder" || { echo "missing $f (not in the Linux clone and not found under /mnt/win)"; exit 1; }
done
if [ ! -f k_sweep_text.txt ]; then
python3 - <<'EOF'
import json
t=json.load(open("k3_targets_cache.json")); out=[]; n=0
for line in open("prompts.jsonl", encoding="utf-8"):
    line=line.strip()
    if not line: continue
    p=json.loads(line)["prompt"]; a=t.get(p)
    if a: out.append(p+"\n\n"+a+"\n\n----\n\n"); n+=1
open("k_sweep_text.txt","w",encoding="utf-8").write("".join(out))
print(f"eval text: {n} prompt+answer pairs, {sum(map(len,out))} chars")
EOF
fi

# ---- passes ---------------------------------------------------------------------------------------------
set -- $KS
case " $KS " in *" 16 "*) ;; *) set -- 16 "$@";; esac      # base pass first, always
KS="$*"
rm -f k_sweep_linux.json
echo "k = $KS | ctx $CTX x $CHUNKS chunks | $(date)"
for k in $KS; do
    log="k_sweep_k$k.log"; route="route_k$k.bin"; rm -f "$route"; [ "$k" = 16 ] && rm -f k_sweep_logits.bin
    args=(-m "$FIRST" -f k_sweep_text.txt -c $CTX -b $CTX --chunks "$CHUNKS" -t "$THREADS" -ngl 0
          --override-kv "kimi-k3.expert_used_count=int:$k" --kl-divergence-base k_sweep_logits.bin)
    [ "$k" != 16 ] && args+=(--kl-divergence)
    echo "[$(date +%H:%M)] k=$k -> $log (trace $route)"
    t0=$(date +%s)
    SISYPHUS_ROUTE_LOG="$route" systemd-inhibit --what=idle:sleep --why="sisyphus k sweep" "$PPL" "${args[@]}" > "$log" 2>&1
    rc=$?; secs=$(( $(date +%s) - t0 ))
    ppl=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' "$log" | tail -1 | awk '{print $5, $7}')
    kld=$(grep -oE 'Mean +KLD: +[0-9.]+' "$log" | tail -1 | awk '{print $NF}')
    top=$(grep -oE 'Same top p: +[0-9.]+' "$log" | tail -1 | awk '{print $NF}')
    echo "        exit $rc, ${secs}s, PPL ${ppl:-?}, mean KLD ${kld:-n/a}, same top-1 ${top:-n/a}%, trace $(du -h "$route" 2>/dev/null | cut -f1)"
    python3 - "$k" "$rc" "$secs" "$ppl" "$kld" "$top" <<'EOF'
import json,sys,os
k,rc,secs,ppl,kld,top=sys.argv[1:7]
f="k_sweep_linux.json"; d=json.load(open(f)) if os.path.exists(f) else {"ctx":512,"results":[]}
r={"k":int(k),"exit_code":int(rc),"wall_seconds":int(secs)}
if ppl: r["ppl"],r["ppl_err"]=map(float,ppl.split())
if kld: r["mean_kld"]=float(kld)
if top: r["same_top1_pct"]=float(top)
d["results"].append(r); json.dump(d,open(f,"w"),indent=1)
EOF
    [ "$k" = 16 ] && [ ! -s k_sweep_logits.bin ] && echo "WARNING: base logits not written; KL for later k will be missing (check $log)"
done
python3 tools/route_stats.py route_k16.bin --json route_stats_k16.json > route_stats_k16.txt 2>&1 && head -40 route_stats_k16.txt
echo; echo "handing results to the Windows folder..."
sudo -v 2>/dev/null && bash tools/linux_handoff.sh >/dev/null && echo "(handed off)" || echo "(handoff skipped - run: bash tools/linux_handoff.sh)"
echo "K SWEEP DONE $(date)"
