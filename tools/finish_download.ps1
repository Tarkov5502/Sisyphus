# tools\finish_download.ps1 — fetch the three missing Kimi-K3 shards (16, 18, 19) into the
# existing local-dir layout, resuming any partial, then verify every shard's header.
#     powershell -ExecutionPolicy Bypass -File tools\finish_download.ps1
# Optional: -Dest "C:\models\Kimi-K3-GGUF" to land them on the 990 PRO instead (the engine
# will read from a bandwidth-split layout across drives anyway).
param([string]$Dest = "D:\models\Kimi-K3-GGUF", [string]$Repo = "unsloth/Kimi-K3-GGUF",
      [string]$Quant = "UD-Q2_K_XL", [int[]]$Shards = @(16, 18, 19))

$ErrorActionPreference = "Stop"
python -c "import huggingface_hub" 2>$null
if ($LASTEXITCODE -ne 0) { pip install -U "huggingface_hub[hf_transfer]" | Out-Null }
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"          # multi-connection downloads; falls back if unavailable

$cli = if (Get-Command hf -EA SilentlyContinue) { "hf" } else { "huggingface-cli" }
$inc = $Shards | ForEach-Object { "$Quant/Kimi-K3-$Quant-{0:d5}-of-00019.gguf" -f $_ }
Write-Host "Downloading to $Dest :`n  $($inc -join "`n  ")"
& $cli download $Repo --local-dir $Dest --include @inc
if ($LASTEXITCODE -ne 0) { Write-Error "download failed (rerun to resume)"; exit 1 }

Write-Host "`nVerifying all 19 shards..."
$dir = Join-Path $Dest $Quant
$files = Get-ChildItem "$dir\Kimi-K3-$Quant-*-of-00019.gguf" | Sort-Object Name
Write-Host ("  {0} shards, {1:n1} GB" -f $files.Count, (($files | Measure-Object Length -Sum).Sum / 1GB))
$bad = @()
foreach ($f in $files) {
    $fs = [IO.File]::OpenRead($f.FullName); $b = New-Object byte[] 4; $fs.Read($b, 0, 4) | Out-Null; $fs.Close()
    if ([Text.Encoding]::ASCII.GetString($b) -ne "GGUF") { $bad += $f.Name }
    elseif ($f.Name -notmatch "00001-of" -and $f.Length -lt 40GB) { $bad += "$($f.Name) (only $([math]::Round($f.Length/1GB,1)) GB)" }
}
if ($files.Count -eq 19 -and $bad.Count -eq 0) { Write-Host "OK: all 19 shards present with valid GGUF headers." }
else { Write-Host "PROBLEMS:"; $bad | ForEach-Object { Write-Host "  $_" }; if ($files.Count -ne 19) { Write-Host "  expected 19 shards, found $($files.Count)" } }
