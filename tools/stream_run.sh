#!/usr/bin/env bash
# tools/stream_run.sh - build and run the Sisyphus streamer (engine/streamer) against the real model.
#
#   bash tools/stream_run.sh          # all three measurements below (~8 min), results -> rig_streamer*.json
#   bash tools/stream_run.sh both     # 1. timed 120 s, every visible shard, both drives busy (aggregate GB/s)
#   bash tools/stream_run.sh sweep    # 2. one DENSE step in sweep order: trunk + all 896 experts per block
#   bash tools/stream_run.sh sparse   # 3. sparse steps: the routing union of a 32-stream batch (~195 experts)
#                                     #    and of a single stream (~15 experts), 3.2 MB slab reads
#
# Needs the Windows drives mounted (tools/linux_setup.sh mounts them; D: holds the 19 shards) and no other
# drive benchmark running. Each run hands its JSON to the Windows folder via tools/linux_handoff.sh.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
MODE="${1:-all}"

if pgrep -x fio >/dev/null; then echo "fio is running (the sustained test?) - wait for it, or stop it: sudo pkill fio"; exit 1; fi
command -v gcc >/dev/null || { echo "gcc missing: sudo apt install build-essential liburing-dev"; exit 1; }
[ -f /usr/include/liburing.h ] || { echo "liburing.h missing: sudo apt install liburing-dev"; exit 1; }
make -s -C engine || exit 1

mapfile -t SHARDS < <(find /mnt/win -maxdepth 6 -name "Kimi-K3-UD-Q2_K_XL-*-of-*.gguf" -size +1G 2>/dev/null | sort)
[ "${#SHARDS[@]}" -gt 0 ] || { echo "no shards found under /mnt/win - run tools/linux_setup.sh first"; exit 1; }
# the directory that holds the complete model (19 shards)
MODEL_DIR=""
while IFS= read -r d; do
    n=$(find "$d" -maxdepth 1 -name "Kimi-K3-UD-Q2_K_XL-*-of-*.gguf" | wc -l)
    extra=$(find "$d" -maxdepth 1 -name "*.gguf" ! -name "Kimi-K3-UD-Q2_K_XL-*-of-*.gguf" | wc -l)
    if [ "$n" -eq 19 ]; then
        [ "$extra" -gt 0 ] && echo "warning: $extra other .gguf file(s) in $d would be swept too - move them out first" && continue
        MODEL_DIR="$d"; break
    fi
done < <(printf '%s\n' "${SHARDS[@]%/*}" | sort -u)
echo "${#SHARDS[@]} shard files visible; complete model in: ${MODEL_DIR:-<none>}"
ulimit -l unlimited 2>/dev/null || true
rc=0

if [ "$MODE" = all ] || [ "$MODE" = both ]; then
    echo; echo "== 1. timed 120 s over every visible shard: both drives busy, aggregate GB/s =="
    engine/streamer --seconds 120 --json rig_streamer.json "${SHARDS[@]}" || rc=1
fi
if [ -n "$MODEL_DIR" ] && { [ "$MODE" = all ] || [ "$MODE" = sweep ]; }; then
    echo; echo "== 2. one dense step in sweep order (trunk + every expert), from $MODEL_DIR =="
    engine/streamer --sweep "$MODEL_DIR" --json rig_streamer_sweep.json || rc=1
fi
if [ -n "$MODEL_DIR" ] && { [ "$MODE" = all ] || [ "$MODE" = sparse ]; }; then
    echo; echo "== 3a. sparse step: 32-stream routing union (~195 experts/block) =="
    engine/streamer --sweep "$MODEL_DIR" --experts 195 --json rig_streamer_sparse32.json || rc=1
    echo; echo "== 3b. sparse step: one stream (~15 experts/block) - compare with llama.cpp's 26.6 s/token =="
    engine/streamer --sweep "$MODEL_DIR" --experts 15 --json rig_streamer_sparse1.json || rc=1
fi
[ -z "$MODEL_DIR" ] && [ "$MODE" != both ] && echo "(sweep/sparse skipped: no directory with all 19 shards is mounted)"
echo; echo "handing results to the Windows folder (may ask for your password)..."
if sudo -v 2>/dev/null && bash tools/linux_handoff.sh >/dev/null; then echo "(results handed to the Windows folder)"; else echo "(handoff skipped/failed - run: bash tools/linux_handoff.sh)"; fi
exit $rc
