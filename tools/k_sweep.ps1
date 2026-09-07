# tools/k_sweep.ps1 - lever 5.27: how many routed experts does K3 really need?
#
#   powershell -ExecutionPolicy Bypass -File tools\k_sweep.ps1               # k = 16 (base), 14, 12, 10, 8
#   powershell -ExecutionPolicy Bypass -File tools\k_sweep.ps1 -Ks 16,12     # subset
#   powershell -ExecutionPolicy Bypass -File tools\k_sweep.ps1 -Chunks 4     # shorter (4 x 512 tokens)
#
# Teacher-forced test with stock llama.cpp (llama-perplexity): the eval text is the project's own
# job prompts + hosted-K3 answers (prompts.jsonl + k3_targets_cache.json), so the numbers are about
# OUR jobs. First pass runs the model as trained (expert_used_count = 16) and saves its logits; each
# later pass overrides k at load time (--override-kv) and reports perplexity, mean KL divergence vs
# the k=16 logits, and top-1 agreement. Overnight job: ~40-50 min per k (each 512-token chunk is a
# near-dense pass over the model from the SN570 plus CPU compute), ~4 h for five k values.
# Writes k_sweep.json (summary), k_sweep_k<N>.log (raw), k_sweep_text.txt, k_sweep_logits.bin (~2 GB).

param(
    [string]$ModelDir = "D:\models\Kimi-K3-GGUF\UD-Q2_K_XL",
    [int[]]$Ks = @(16, 14, 12, 10, 8),
    [int]$Chunks = 6,
    [int]$Ctx = 512,
    [int]$Threads = 16
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$tooldir = Join-Path $root "tools\llama.cpp"
$text = Join-Path $root "k_sweep_text.txt"
$logits = Join-Path $root "k_sweep_logits.bin"
$outjson = Join-Path $root "k_sweep.json"
if ($Ks[0] -ne 16) { $Ks = @(16) + @($Ks | Where-Object { $_ -ne 16 }) }   # the base pass must come first

# ---- 1. llama.cpp binaries (same release logic as first_token.ps1) ------------------------------
$exe = Get-ChildItem -Path $tooldir -Recurse -Filter "llama-perplexity.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $exe) {
    Write-Host "Downloading latest llama.cpp Windows CPU release ..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" -Headers @{ "User-Agent" = "sisyphus" }
    $assets = @($rel.assets | Where-Object { $_.name -like "*-bin-win-cpu-x64.zip" })
    if ($assets.Count -eq 0) { Write-Host "no win-cpu-x64 asset in $($rel.tag_name)"; exit 1 }
    New-Item -ItemType Directory -Force -Path $tooldir | Out-Null
    foreach ($a in $assets) {
        $zip = Join-Path $tooldir $a.name
        Write-Host "  $($a.name) ($([math]::Round($a.size/1MB)) MB)"
        Invoke-WebRequest -Uri $a.browser_download_url -OutFile $zip -UseBasicParsing
        Expand-Archive -Path $zip -DestinationPath $tooldir -Force
        Remove-Item $zip
    }
    Set-Content -Path (Join-Path $tooldir "VERSION.txt") -Value $rel.tag_name
    $exe = Get-ChildItem -Path $tooldir -Recurse -Filter "llama-perplexity.exe" | Select-Object -First 1
}
$version = if (Test-Path (Join-Path $tooldir "VERSION.txt")) { (Get-Content (Join-Path $tooldir "VERSION.txt")).Trim() } else { "?" }
Write-Host "llama.cpp: $($exe.FullName) ($version)"

# ---- 2. model + eval text --------------------------------------------------------------------
$first = Get-ChildItem -Path $ModelDir -Filter "*-00001-of-*.gguf" | Select-Object -First 1
if (-not $first) { Write-Host "no *-00001-of-*.gguf in $ModelDir"; exit 1 }
if (-not (Test-Path $text)) {
    $targets = Get-Content (Join-Path $root "k3_targets_cache.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    $sb = New-Object System.Text.StringBuilder
    $n = 0
    foreach ($line in Get-Content (Join-Path $root "prompts.jsonl") -Encoding UTF8) {
        if (-not $line.Trim()) { continue }
        $p = ($line | ConvertFrom-Json).prompt
        $t = $targets.PSObject.Properties | Where-Object { $_.Name -eq $p } | Select-Object -First 1
        if ($t -and $t.Value) {
            [void]$sb.Append($p).Append("`n`n").Append($t.Value).Append("`n`n----`n`n"); $n++
        }
    }
    [IO.File]::WriteAllText($text, $sb.ToString(), (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "eval text: $n prompt+answer pairs, $($sb.Length) chars -> $text"
}
$needTok = $Chunks * $Ctx
Write-Host "model: $($first.Name) from $ModelDir | ctx $Ctx x $Chunks chunks = $needTok tokens per pass | k = $($Ks -join ', ')"
Write-Host "Each pass loads the model (~7 min) and streams it once per chunk. Leave it alone."
Write-Host ""

# ---- 3. passes -------------------------------------------------------------------------------
$results = @()
foreach ($k in $Ks) {
    $log = Join-Path $root "k_sweep_k$k.log"
    $args_ = @("-m", $first.FullName, "-f", $text, "-c", "$Ctx", "-b", "$Ctx", "--chunks", "$Chunks", "-t", "$Threads", "-ngl", "0",
               "--override-kv", "kimi-k3.expert_used_count=int:$k", "--kl-divergence-base", $logits)
    if ($k -ne 16) { $args_ += @("--kl-divergence") }
    Write-Host ("[{0}] k={1}  > llama-perplexity {2}" -f (Get-Date).ToString("HH:mm"), $k, ($args_ -join " "))
    $t0 = Get-Date
    $eap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $exe.FullName @args_ 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log | Out-Null
    $code = $LASTEXITCODE
    $ErrorActionPreference = $eap
    try { (Get-Content $log) | Set-Content -Path $log -Encoding UTF8 } catch {}
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds)
    $raw = Get-Content $log -Raw
    $r = [ordered]@{ k = $k; exit_code = $code; wall_seconds = $secs }
    $m = [regex]::Match($raw, "Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")
    if ($m.Success) { $r.ppl = [double]$m.Groups[1].Value; $r.ppl_err = [double]$m.Groups[2].Value }
    $m = [regex]::Match($raw, "Mean\s+KLD:\s*([\d.]+)")
    if ($m.Success) { $r.mean_kld = [double]$m.Groups[1].Value }
    $m = [regex]::Match($raw, "99\.9%\s+KLD:\s*([\d.]+)")
    if ($m.Success) { $r.kld_p999 = [double]$m.Groups[1].Value }
    $m = [regex]::Match($raw, "Same top p:\s*([\d.]+)")
    if ($m.Success) { $r.same_top1_pct = [double]$m.Groups[1].Value }
    $m = [regex]::Match($raw, "Mean\s+.p:\s*([-\d.]+)")
    if ($m.Success) { $r.mean_delta_p = [double]$m.Groups[1].Value }
    $results += [pscustomobject]$r
    Write-Host ("        k={0}: exit {1}, {2} s, PPL {3}, mean KLD {4}, same top-1 {5}%" -f $k, $code, $secs, $r.ppl, $r.mean_kld, $r.same_top1_pct)
    if ($k -eq 16 -and -not (Test-Path $logits)) { Write-Host "base logits were not written - later passes cannot compute KL; check $log"; }
    [pscustomobject]@{ date = (Get-Date).ToString("s"); llama_cpp = $version; model_dir = $ModelDir; ctx = $Ctx; chunks = $Chunks;
                       tokens_per_pass = $needTok; results = $results } | ConvertTo-Json -Depth 4 | Set-Content -Path $outjson -Encoding ASCII
}
Write-Host ""
Write-Host "K SWEEP DONE - wrote $outjson (and k_sweep_k*.log). Tell Claude."
