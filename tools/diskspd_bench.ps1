# tools\diskspd_bench.ps1 - the definitive Windows drive test: unbuffered, high queue depth,
# each drive alone then both concurrently. Downloads Microsoft diskspd if needed.
#     powershell -ExecutionPolicy Bypass -File tools\diskspd_bench.ps1
# Writes rig_diskspd.json into the repo root. Read-only against the shard files.
param(
  [string]$FileC = "C:\models\Kimi-K3-GGUF\UD-Q2_K_XL\Kimi-K3-UD-Q2_K_XL-00002-of-00019.gguf",
  [string]$FileD = "D:\models\Kimi-K3-GGUF\UD-Q2_K_XL\Kimi-K3-UD-Q2_K_XL-00005-of-00019.gguf",
  [int]$SoloSeconds = 60, [int]$BothSeconds = 180
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$dir = Join-Path $PSScriptRoot "diskspd"
$exe = Get-ChildItem -Path $dir -Recurse -Filter diskspd.exe -EA SilentlyContinue | Where-Object { $_.FullName -match "amd64" } | Select -First 1

if (-not $exe) {
    Write-Host "Downloading diskspd from github.com/microsoft/diskspd ..."
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $rel = Invoke-RestMethod "https://api.github.com/repos/microsoft/diskspd/releases/latest" -Headers @{ "User-Agent" = "sisyphus" }
    $asset = $rel.assets | Where-Object { $_.name -match "\.zip$" } | Select -First 1
    if (-not $asset) { throw "no zip asset in latest diskspd release" }
    $zip = Join-Path $dir "diskspd.zip"
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $dir -Force
    $exe = Get-ChildItem -Path $dir -Recurse -Filter diskspd.exe | Where-Object { $_.FullName -match "amd64" } | Select -First 1
    if (-not $exe) { throw "diskspd.exe (amd64) not found after extract" }
}
Write-Host "diskspd: $($exe.FullName)"
foreach ($f in @($FileC, $FileD)) { if (-not (Test-Path $f)) { throw "missing test file: $f" } }

function Run-Test([string]$label, [int]$seconds, [string[]]$files) {
    Write-Host ("`n== {0}: {1}s, 8 MB sequential, QD32 x 2 threads per file, unbuffered ==" -f $label, $seconds)
    $args = @("-b8M", "-si", "-o32", "-t2", "-Su", "-w0", "-L", "-d$seconds", "-Rxml") + $files
    $xml = & $exe.FullName @args
    [xml]$doc = ($xml -join "`n")
    $secs = [double]$doc.Results.TimeSpan.TestTimeSeconds
    $per = @{}
    foreach ($th in $doc.Results.TimeSpan.Thread) {
        foreach ($t in $th.Target) {
            $p = $t.Path; if (-not $per.ContainsKey($p)) { $per[$p] = 0.0 }
            $per[$p] += [double]$t.ReadBytes
        }
    }
    $res = @{}
    $agg = 0.0
    foreach ($p in $per.Keys) {
        $gbps = $per[$p] / $secs / 1e9; $agg += $gbps
        $drive = (Split-Path -Qualifier $p)
        $res[$drive] = [math]::Round($gbps, 2)
        Write-Host ("  {0}  {1,6:n2} GB/s   ({2:n0} GB in {3:n0} s)" -f $drive, $gbps, ($per[$p] / 1e9), $secs)
    }
    if ($files.Count -gt 1) { Write-Host ("  aggregate {0,6:n2} GB/s  -> 861 GB sweep = {1:n0} s" -f $agg, (861 / $agg)) }
    $res["aggregate"] = [math]::Round($agg, 2)
    return $res
}

$out = [ordered]@{ timestamp = (Get-Date).ToString("s"); tool = $exe.FullName; params = "-b8M -si -o32 -t2 -Su -w0" }
$out.C_alone = Run-Test "C: alone (990 PRO)" $SoloSeconds @($FileC)
$out.D_alone = Run-Test "D: alone (SN570)" $SoloSeconds @($FileD)
$out.both    = Run-Test "C: + D: concurrently" $BothSeconds @($FileC, $FileD)

$out | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $repo "rig_diskspd.json") -Encoding UTF8
Write-Host "`nwrote $repo\rig_diskspd.json - tell Claude it's there."
