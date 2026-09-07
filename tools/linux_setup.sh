#!/usr/bin/env bash
# tools/linux_setup.sh - first boot of Ubuntu on the Sisyphus rig: drivers, tools, mount the Windows
# drives, measure them the way the engine will read them (io_uring + O_DIRECT), build llama.cpp with
# CUDA, and generate the first K3 tokens on Linux.
#
#   bash tools/linux_setup.sh            # everything (asks for sudo once)
#   bash tools/linux_setup.sh --no-token # skip the llama.cpp first-token run
#   bash tools/linux_setup.sh --bench    # only re-run the drive bench
#
# Outputs (in the repo root): rig_fio.json, first_token_linux.json, first_token_linux.log.
# Re-runnable: every step skips what is already done.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
DO_TOKEN=1; ONLY_BENCH=0
for a in "$@"; do case "$a" in --no-token) DO_TOKEN=0;; --bench) ONLY_BENCH=1;; esac; done
say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
MODEL_FILE="Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf"
WSL=0; grep -qi microsoft /proc/version 2>/dev/null && WSL=1
[ "$WSL" = 1 ] && echo "(WSL2 detected: dev environment mode - Windows NVIDIA driver, no native drive numbers unless a disk is attached with wsl --mount)"

if [ "$ONLY_BENCH" = 0 ]; then
# ---- 1. packages ---------------------------------------------------------------------------------
say "packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential cmake ninja-build git curl jq htop ccache pkg-config \
    python3 python3-venv python3-pip \
    liburing-dev fio nvme-cli smartmontools ntfs-3g libcurl4-openssl-dev \
    linux-tools-common linux-tools-generic >/dev/null
echo "kernel: $(uname -r)   (io_uring + ntfs3 need >= 5.15; 26.04 ships far newer)"

# ---- 2. NVIDIA driver + CUDA ---------------------------------------------------------------------
say "NVIDIA"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv
elif [ "$WSL" = 1 ]; then
    echo "no GPU visible in WSL: update the NVIDIA driver on the Windows side (any recent driver includes WSL support)."
else
    echo "installing the recommended NVIDIA driver (ubuntu-drivers)..."
    sudo ubuntu-drivers install 2>&1 | tail -3
    if [ "${PIPESTATUS[0]}" = 0 ]; then NEED_REBOOT=1; else echo "  driver install FAILED - see above"; fi
    echo "  -> driver installed; a REBOOT is needed before the GPU is usable. Re-run this script after."
fi
if ! command -v nvcc >/dev/null; then
    echo "installing CUDA toolkit (Ubuntu package; enough to build llama.cpp)..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-cuda-toolkit >/dev/null 2>&1 || \
        echo "  nvidia-cuda-toolkit not installable from the archive; llama.cpp will be built CPU-only for now."
fi
command -v nvcc >/dev/null && nvcc --version | tail -1

# ---- 3. mount the Windows drives (read-only, ntfs3 kernel driver) ---------------------------------
say "Windows drives"
sudo mkdir -p /mnt/win
while read -r dev fstype label; do
    [ "$fstype" = "ntfs" ] || continue
    # mount folder named by DEVICE, not NTFS label: on this rig C: and D: both carry the label "Windows"
    # (and lsblk -r escapes spaces in labels as \x20, so labels make poor directory names anyway)
    mp="/mnt/win/$(basename "$dev")"
    if ! mountpoint -q "$mp"; then
        sudo mkdir -p "$mp"
        sudo mount -t ntfs3 -o ro,noatime,uid=$(id -u),gid=$(id -g) "$dev" "$mp" 2>/dev/null \
            || sudo mount -t ntfs-3g -o ro "$dev" "$mp" 2>/dev/null \
            || { echo "  could not mount $dev ($label) - Fast Startup still on in Windows? (in WSL: only disks attached with wsl --mount --bare show up here)"; continue; }
    fi
    echo "  $dev ($label) -> $mp  [$(findmnt -no FSTYPE "$mp")]"
done < <(lsblk -rno PATH,FSTYPE,LABEL | awk '$2=="ntfs"')
fi  # ONLY_BENCH

# locate the model shards on whatever is mounted
mapfile -t SHARD1 < <(find /mnt/win /media "$HOME" -maxdepth 6 -name "$MODEL_FILE" 2>/dev/null)
[ "${#SHARD1[@]}" = 0 ] && [ "$WSL" = 1 ] && mapfile -t SHARD1 < <(ls /mnt/[a-z]/models/Kimi-K3-GGUF/UD-Q2_K_XL/$MODEL_FILE 2>/dev/null)
if [ "${#SHARD1[@]}" = 0 ]; then echo "model not found (looked for $MODEL_FILE under /mnt/win, /media, ~)"; MODEL_DIR=""; else
    MODEL_DIR="$(dirname "${SHARD1[0]}")"; echo "model: $MODEL_DIR ($(ls "$MODEL_DIR"/*.gguf | wc -l) shards)"; fi

# ---- 4. drive bench: the Linux diskspd (fio, io_uring, O_DIRECT, 8 MB, QD32 x 2 jobs) ---------------
say "drive bench (fio io_uring O_DIRECT)"
# one big file per physical NVMe: the shards on each mounted Windows drive, else the raw block device
declare -A FILE_FOR_DEV
for mp in /mnt/win/*; do
    mountpoint -q "$mp" || continue
    src=$(findmnt -no SOURCE "$mp"); pk=$(lsblk -no PKNAME "$src" 2>/dev/null | head -1); [ -n "$pk" ] || pk=$(basename "$src")
    f=$(find "$mp" -maxdepth 6 -name "*.gguf" -size +10G 2>/dev/null | head -1)
    [ -z "$f" ] && f=$(find "$mp" -maxdepth 3 -type f -size +8G 2>/dev/null | head -1)
    [ -n "$f" ] && FILE_FOR_DEV[$pk]="$f"
done
if [ "${#FILE_FOR_DEV[@]}" = 0 ]; then echo "no large files found on mounted Windows drives; skipping bench"; else
    run_fio() { # seconds dev=file ...
        local secs=$1; shift; local jobs=""
        for pair in "$@"; do jobs+=$'\n'"[${pair%%=*}]"$'\n'"filename=${pair#*=}"$'\n'; done
        fio --output-format=json - <<EOF 2>/dev/null
[global]
rw=read
bs=8M
iodepth=32
numjobs=2
ioengine=io_uring
direct=1
time_based
runtime=$secs
thread
group_reporting=1
$jobs
EOF
    }
    summarize() { jq -r '.jobs[] | "\(.jobname): \((.read.bw_bytes/1e9*100|round)/100) GB/s"' ; }
    : > /tmp/fio_all.json
    for dev in "${!FILE_FOR_DEV[@]}"; do
        echo "-- $dev alone, 60 s: ${FILE_FOR_DEV[$dev]}"
        out=$(run_fio 60 "$dev=${FILE_FOR_DEV[$dev]}")
        if [ -z "$out" ] || ! echo "$out" | jq -e '.jobs[0].error == 0 and .jobs[0].read.bw_bytes > 0' >/dev/null 2>&1; then
            echo "   O_DIRECT refused on this filesystem; retrying buffered (numbers then include the page cache)"
            out=$(fio --output-format=json --name="$dev" --filename="${FILE_FOR_DEV[$dev]}" --rw=read --bs=8M --iodepth=32 --numjobs=2 --ioengine=io_uring --direct=0 --time_based --runtime=60 --thread 2>/dev/null)
        fi
        if [ -n "$out" ] && echo "$out" | jq -e .jobs >/dev/null 2>&1; then
            echo "$out" | summarize | sed 's/^/   /'; echo "{\"phase\":\"$dev alone\",\"fio\":$out}" >> /tmp/fio_all.json
        else echo "   fio produced no usable output for $dev"; fi
    done
    if [ "${#FILE_FOR_DEV[@]}" -gt 1 ]; then
        echo "-- all drives concurrently, 180 s"
        pairs=(); for dev in "${!FILE_FOR_DEV[@]}"; do pairs+=("$dev=${FILE_FOR_DEV[$dev]}"); done
        out=$(run_fio 180 "${pairs[@]}")
        if [ -n "$out" ] && echo "$out" | jq -e .jobs >/dev/null 2>&1; then
            echo "$out" | summarize | sed 's/^/   /'
            agg=$(echo "$out" | jq '[.jobs[].read.bw_bytes] | add / 1e9')
            printf '   aggregate %.2f GB/s -> 861 GB sweep = %.0f s\n' "$agg" "$(awk -v a="$agg" 'BEGIN{print (a>0)?861/a:0}')"
            echo "{\"phase\":\"concurrent\",\"fio\":$out}" >> /tmp/fio_all.json
        else echo "   fio produced no usable output for the concurrent run"; fi
    fi
    if [ -s /tmp/fio_all.json ]; then
        jq -s '{date: (now|todate), kernel: "'"$(uname -r)"'", phases: .}' /tmp/fio_all.json > "$REPO/rig_fio.json" && echo "wrote $REPO/rig_fio.json"
    else echo "no fio results to write"; fi
fi
[ "$ONLY_BENCH" = 1 ] && exit 0

# ---- 5. repo sanity ------------------------------------------------------------------------------
say "repo"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate && pip install -q pytest numpy >/dev/null 2>&1
python -m pytest -q sisyphus 2>&1 | tail -1

# ---- 6. llama.cpp + first token on Linux ----------------------------------------------------------
if [ "$WSL" = 1 ] && [ "$DO_TOKEN" = 1 ] && [[ "$MODEL_DIR" == /mnt/[a-z]/* ]]; then
    echo "skipping first-token run: $MODEL_DIR is a 9p (drvfs) mount, ~1 GB/s - run tools\\first_token.ps1 on Windows instead"; DO_TOKEN=0
fi
if [ "$DO_TOKEN" = 1 ] && [ -n "$MODEL_DIR" ]; then
    say "llama.cpp"
    LL="$HOME/dev/llama.cpp"
    if [ ! -d "$LL" ]; then git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$LL"; fi
    if [ ! -x "$LL/build/bin/llama-completion" ] && [ ! -x "$LL/build/bin/llama-cli" ]; then
        CUDA_FLAG="-DGGML_CUDA=OFF"; command -v nvcc >/dev/null && CUDA_FLAG="-DGGML_CUDA=ON"
        echo "building ($CUDA_FLAG)..."
        cmake -S "$LL" -B "$LL/build" -G Ninja -DCMAKE_BUILD_TYPE=Release $CUDA_FLAG -DLLAMA_CURL=OFF >/dev/null && \
        cmake --build "$LL/build" --target llama-completion llama-cli llama-server 2>&1 | tail -1
    fi
    # raw completion (-no-cnv): K3's chat template needs --jinja and we want the bare prompt anyway.
    # -no-cnv exists only for llama-completion; the older llama-cli layout has no conversation mode to disable.
    BIN="$LL/build/bin/llama-completion"; EXTRA="-no-cnv"
    [ -x "$BIN" ] || { BIN="$LL/build/bin/llama-cli"; EXTRA=""; }
    say "first K3 tokens on Linux (mmap from $MODEL_DIR, batch 1 - minutes per token)"
    LOG="$REPO/first_token_linux.log"
    t0=$(date +%s)
    "$BIN" -m "$MODEL_DIR/$MODEL_FILE" -p "The capital of France is" -n 4 -c 512 -t 16 --temp 0 --no-warmup -ngl 0 $EXTRA 2>&1 | tee "$LOG"
    secs=$(( $(date +%s) - t0 ))
    spt=$(grep -oP 'eval time\s*=\s*[\d.]+ ms /\s*\d+ runs\s*\(\s*\K[\d.]+' "$LOG" | tail -1)
    jq -n --arg spt "${spt:-}" --arg secs "$secs" --arg dir "$MODEL_DIR" --arg k "$(uname -r)" \
        '{date:(now|todate), kernel:$k, model_dir:$dir, wall_seconds:($secs|tonumber),
          seconds_per_token:(if $spt=="" then null else ($spt|tonumber/1000) end),
          effective_gbps:(if $spt=="" then null else (861/($spt|tonumber/1000)) end)}' > "$REPO/first_token_linux.json"
    echo "wrote $REPO/first_token_linux.json"
fi

say "done"
[ "${NEED_REBOOT:-0}" = 1 ] && echo "REBOOT now (new NVIDIA driver), then re-run: bash tools/linux_setup.sh"
# hand the results to the Windows side (C:\dev\sisyphus-src) so the Windows-linked Claude session can read them
[ "$WSL" = 0 ] && bash "$REPO/tools/linux_handoff.sh" || true
echo "tell Claude: rig_fio.json and first_token_linux.json are there."
