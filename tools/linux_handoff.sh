#!/usr/bin/env bash
# tools/linux_handoff.sh - copy the Linux-side results into the Windows repo folder so the Windows-linked
# Claude session can read them, and print them to the terminal.
#
#   bash tools/linux_handoff.sh
#
# Results live in the Ubuntu repo (ext4, invisible from Windows). Windows' C: is NTFS, which Linux can
# write (Fast Startup is off), so this mounts C: read-write and drops the files into
# C:\dev\sisyphus-src\ (they are gitignored there, so nothing gets committed by accident).

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILES=(rig_fio.json first_token_linux.json first_token_linux.log rig_fio_sustained.json drive_temps.csv rig_streamer.json rig_streamer_sweep.json rig_streamer_sparse32.json rig_streamer_sparse1.json acceptance_results_512.json quant_oracle.json)

echo "== results in $REPO =="
for f in "${FILES[@]}"; do
    [ -f "$REPO/$f" ] || continue
    echo "--- $f"
    case "$f" in
        *.json) command -v jq >/dev/null && jq -c 'del(.phases)' "$REPO/$f" 2>/dev/null || head -c 2000 "$REPO/$f"; echo ;;
        *.log)  grep -E "eval time|prompt eval|load time|^[^l].{0,200}$" "$REPO/$f" | grep -v "^llama_model\|^ggml_\|^print_info\|^load" | tail -15 ;;
    esac
done
if [ -f "$REPO/rig_fio.json" ] && command -v jq >/dev/null; then
    echo "--- fio per phase (GB/s)"
    jq -r '.phases[] | .phase as $p | .fio.jobs[] | "\($p): \(.jobname) \((.read.bw_bytes/1e9*100|round)/100)"' "$REPO/rig_fio.json"
fi

# ---- find or mount the Windows partition that holds C:\dev\sisyphus-src ---------------------------
WINREPO=""
for m in /mnt/win/* /media/*/*; do
    [ -d "$m/dev/sisyphus-src" ] && { WINREPO="$m/dev/sisyphus-src"; break; }
done
if [ -z "$WINREPO" ]; then
    sudo mkdir -p /mnt/win
    # mount candidates read-only first; only the one holding the repo is remounted read-write below
    while read -r dev fstype; do
        [ "$fstype" = "ntfs" ] || continue
        mp="/mnt/win/$(basename "$dev")"; mine=0
        mountpoint -q "$mp" || { sudo mkdir -p "$mp"; sudo mount -t ntfs3 -o ro,noatime,uid=$(id -u),gid=$(id -g) "$dev" "$mp" 2>/dev/null || sudo mount -t ntfs-3g -o ro "$dev" "$mp" 2>/dev/null || continue; mine=1; }
        if [ -d "$mp/dev/sisyphus-src" ]; then WINREPO="$mp/dev/sisyphus-src"; break; fi
        [ "$mine" = 1 ] && sudo umount "$mp" 2>/dev/null      # only what this script mounted; never the model drive
    done < <(lsblk -rno PATH,FSTYPE | awk '$2=="ntfs"')
fi
if [ -z "$WINREPO" ]; then echo "could not find C:\\dev\\sisyphus-src on any NTFS partition"; exit 1; fi

mp="$(findmnt -no TARGET -T "$WINREPO")"
if findmnt -no OPTIONS "$mp" | head -1 | grep -qE '(^|,)ro(,|$)'; then
    sudo mount -o remount,rw "$mp" || { echo "could not remount $mp read-write (Windows Fast Startup / hibernation still on?)"; exit 1; }
fi
n=0
for f in "${FILES[@]}"; do [ -f "$REPO/$f" ] && { cp -f "$REPO/$f" "$WINREPO/" && n=$((n+1)); }; done
sync
echo "copied $n file(s) to $WINREPO  (C:\\dev\\sisyphus-src). Reboot into Windows and tell Claude they are there."
