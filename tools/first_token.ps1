# tools/first_token.ps1 - generate the first real Kimi-K3 tokens on this rig (ENGINE_DESIGN step 1:
# the correctness oracle), using stock llama.cpp with the model memory-mapped straight from disk.
#
#   powershell -ExecutionPolicy Bypass -File tools\first_token.ps1
#   powershell -ExecutionPolicy Bypass -File tools\first_token.ps1 -Tokens 16 -Prompt "..."
#
# What it does: downloads the latest llama.cpp Windows CPU release (upstream added the kimi-k3 text
# model in b10448), points it at the 19 shards, and asks for a few greedy tokens. The model is 861 GB
# and the box has 32 GB, so every token is one full pass over the drive through Windows' buffered mmap
# path - expect MINUTES per token (this is the naive tape regime, batch 1, buffered I/O: the slowest
# possible version of Sisyphus). That is fine: the point is the first real K3 output produced locally,
# the s/token number for the ledger, and confirming the GGUF + arch load at all. Writes first_token.log
# and first_token.json in the repo root.

param(
    [string]$ModelDir = "D:\models\Kimi-K3-GGUF\UD-Q2_K_XL",
    [string]$Prompt = "The capital of France is",
    [int]$Tokens = 8,
    [int]$Threads = 16,
    [int]$Ctx = 512,
    [switch]$Cuda
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$tooldir = Join-Path $root "tools\llama.cpp"
$log = Join-Path $root "first_token.log"
$outjson = Join-Path $root "first_token.json"

# ---- 1. llama.cpp binaries -------------------------------------------------------------------
$exe = Get-ChildItem -Path $tooldir -Recurse -Filter "llama-completion.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $exe) { $exe = Get-ChildItem -Path $tooldir -Recurse -Filter "llama-cli.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 }
if (-not $exe) {
    Write-Host "Downloading latest llama.cpp Windows release from github.com/ggml-org/llama.cpp ..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" -Headers @{ "User-Agent" = "sisyphus" }
    $tag = $rel.tag_name
    Write-Host "  release $tag"
    $want = if ($Cuda) { "*-bin-win-cuda-12.4-x64.zip" } else { "*-bin-win-cpu-x64.zip" }
    $assets = @($rel.assets | Where-Object { $_.name -like $want })
    if ($Cuda) { $assets += @($rel.assets | Where-Object { $_.name -like "cudart-*-win-cuda-12.4-x64.zip" }) }
    if ($assets.Count -eq 0) {
        Write-Host "no asset matched $want; assets in this release:"
        $rel.assets | ForEach-Object { Write-Host "  $($_.name)" }
        exit 1
    }
    New-Item -ItemType Directory -Force -Path $tooldir | Out-Null
    foreach ($a in $assets) {
        $zip = Join-Path $tooldir $a.name
        Write-Host "  $($a.name) ($([math]::Round($a.size/1MB)) MB)"
        Invoke-WebRequest -Uri $a.browser_download_url -OutFile $zip -UseBasicParsing
        Expand-Archive -Path $zip -DestinationPath $tooldir -Force
        Remove-Item $zip
    }
    Set-Content -Path (Join-Path $tooldir "VERSION.txt") -Value $tag
    $exe = Get-ChildItem -Path $tooldir -Recurse -Filter "llama-completion.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $exe) { $exe = Get-ChildItem -Path $tooldir -Recurse -Filter "llama-cli.exe" | Select-Object -First 1 }
}
$version = if (Test-Path (Join-Path $tooldir "VERSION.txt")) { (Get-Content (Join-Path $tooldir "VERSION.txt")).Trim() } else { "?" }
Write-Host "llama.cpp: $($exe.FullName) ($version)"

# ---- 2. model --------------------------------------------------------------------------------
$first = Get-ChildItem -Path $ModelDir -Filter "*-00001-of-*.gguf" | Select-Object -First 1
if (-not $first) { Write-Host "no *-00001-of-*.gguf in $ModelDir"; exit 1 }
$shards = @(Get-ChildItem -Path $ModelDir -Filter "*.gguf")
$gb = [math]::Round(($shards | Measure-Object -Property Length -Sum).Sum / 1e9, 1)
Write-Host "model: $($first.Name) (+ $($shards.Count - 1) sibling shards, $gb GB total, memory-mapped from $ModelDir)"
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Host "RAM: $ram GB -> every token streams the whole model through the page cache; expect minutes per token."
Write-Host ""

# ---- 3. run ----------------------------------------------------------------------------------
$args_ = @("-m", $first.FullName, "-p", $Prompt, "-n", "$Tokens", "-c", "$Ctx", "-t", "$Threads",
           "--temp", "0", "--no-warmup", "-ngl", "0")
if ($exe.Name -eq "llama-cli.exe") { $args_ += @("-no-cnv") }   # older layout: llama-cli is the completion tool
Write-Host ("> " + $exe.Name + " " + ($args_ -join " "))
Write-Host "(output is also captured to first_token.log)"
Write-Host "------------------------------------------------------------------"
$t0 = Get-Date
& $exe.FullName @args_ 2>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE
$secs = [math]::Round(((Get-Date) - $t0).TotalSeconds)
Write-Host "------------------------------------------------------------------"
Write-Host "exit $code after $secs s"

# ---- 4. summary ------------------------------------------------------------------------------
$text = Get-Content $log -Raw
$res = [ordered]@{ date = (Get-Date).ToString("s"); llama_cpp = $version; binary = $exe.Name; model_dir = $ModelDir;
                   model_gb = $gb; ram_gb = $ram; prompt = $Prompt; n_tokens = $Tokens; threads = $Threads;
                   exit_code = $code; wall_seconds = $secs }
$m = [regex]::Match($text, "prompt eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens\s*\(\s*([\d.]+) ms per token")
if ($m.Success) { $res.prompt_eval_ms = [double]$m.Groups[1].Value; $res.prompt_tokens = [int]$m.Groups[2].Value }
$m = [regex]::Match($text, "\beval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) runs\s*\(\s*([\d.]+) ms per token")
if ($m.Success) {
    $res.eval_ms = [double]$m.Groups[1].Value; $res.eval_runs = [int]$m.Groups[2].Value
    $res.seconds_per_token = [math]::Round([double]$m.Groups[3].Value / 1000, 1)
    $res.effective_gbps = [math]::Round($gb / $res.seconds_per_token, 2)
    Write-Host ("decode: {0} s per token  ->  {1} GB/s effective model streaming (861 GB per token, batch 1, buffered mmap)" -f $res.seconds_per_token, $res.effective_gbps)
}
if ($text -match "unknown model architecture|unknown architecture") {
    Write-Host "This llama.cpp build does not know the kimi-k3 architecture; the release picked was $version."
}
$res | ConvertTo-Json | Set-Content -Path $outjson -Encoding ASCII
Write-Host "wrote $outjson and $log - tell Claude they are there."
