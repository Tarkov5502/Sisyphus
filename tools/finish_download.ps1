# tools\finish_download.ps1 — detect which Kimi-K3 shards are missing, fetch exactly those,
# resuming partials, then verify every shard's header and the total size.
#     powershell -ExecutionPolicy Bypass -File tools\finish_download.ps1
param([string]$Dest = "D:\models\Kimi-K3-GGUF", [string]$Repo = "unsloth/Kimi-K3-GGUF",
      [string]$Quant = "UD-Q2_K_XL", [int]$Total = 19, [double]$ExpectedGB = 860)

$ErrorActionPreference = "Stop"
python -c "import huggingface_hub" 2>$null
if ($LASTEXITCODE -ne 0) { pip install -U huggingface_hub | Out-Null }
$env:HF_XET_HIGH_PERFORMANCE = "1"
$cli = if (Get-Command hf -EA SilentlyContinue) { "hf" } else { "huggingface-cli" }
$dir = Join-Path $Dest $Quant

function Missing {
    $have = @{}
    Get-ChildItem "$dir\Kimi-K3-$Quant-*-of-$('{0:d5}' -f $Total).gguf" -EA SilentlyContinue | ForEach-Object {
        if ($_.Name -match '-(\d{5})-of-') { $have[[int]$Matches[1]] = $_ }
    }
    1..$Total | Where-Object { -not $have.ContainsKey($_) }
}

$miss = @(Missing)
if ($miss.Count -eq 0) { Write-Host "No shards missing." }
else {
    Write-Host "Missing shards: $($miss -join ', ')`nDownloading to $Dest via huggingface_hub (Python API, explicit filenames)..."
    $py = @"
import sys, os
from huggingface_hub import hf_hub_download
repo, dest, quant, total = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
for n in sys.argv[5:]:
    fn = f"{quant}/Kimi-K3-{quant}-{int(n):05d}-of-{total:05d}.gguf"
    print("->", fn, flush=True)
    p = hf_hub_download(repo_id=repo, filename=fn, local_dir=dest)
    print("   ok", p, round(os.path.getsize(p)/1e9, 1), "GB", flush=True)
"@
    $tmp = Join-Path $env:TEMP "k3_fetch.py"; Set-Content -Path $tmp -Value $py -Encoding UTF8
    python $tmp $Repo $Dest $Quant $Total @($miss | ForEach-Object { "$_" })
    if ($LASTEXITCODE -ne 0) { Write-Error "download failed (rerun to resume)"; exit 1 }
}

Write-Host "`nVerifying..."
$all = Get-ChildItem "$dir\Kimi-K3-$Quant-*-of-$('{0:d5}' -f $Total).gguf" | Sort-Object Name
$sumGB = ($all | Measure-Object Length -Sum).Sum / 1e9
Write-Host ("  {0}/{1} shards, {2:n1} GB (expected ~{3} GB)" -f $all.Count, $Total, $sumGB, $ExpectedGB)
$bad = @()
foreach ($f in $all) {
    $fs = [IO.File]::OpenRead($f.FullName); $b = New-Object byte[] 4; $fs.Read($b, 0, 4) | Out-Null; $fs.Close()
    if ([Text.Encoding]::ASCII.GetString($b) -ne "GGUF") { $bad += "$($f.Name): bad header" }
    elseif ($f.Name -notmatch "-00001-of" -and $f.Length -lt 1GB) { $bad += "$($f.Name): only $([math]::Round($f.Length/1e9,1)) GB" }
}
$still = @(Missing)
if ($still.Count -eq 0 -and $bad.Count -eq 0 -and [math]::Abs($sumGB - $ExpectedGB) -lt 15) {
    Write-Host "OK: all $Total shards present, headers valid, total matches."
} else {
    if ($still.Count) { Write-Host "  still missing: $($still -join ', ')" }
    $bad | ForEach-Object { Write-Host "  $_" }
    if ([math]::Abs($sumGB - $ExpectedGB) -ge 15) { Write-Host "  total size off by $([math]::Round($sumGB-$ExpectedGB,1)) GB" }
}
