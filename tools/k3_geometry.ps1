# tools\k3_geometry.ps1 - find the Kimi-K3 GGUF shards on any local drive, read their
# headers, and write C:\dev\sisyphus-src\k3_geometry.json.  Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File tools\k3_geometry.ps1
param([string]$Path = "")

$ErrorActionPreference = "SilentlyContinue"
$repo = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command python -EA SilentlyContinue)) { Write-Error "python not on PATH"; exit 1 }
python -c "import gguf" 2>$null; if ($LASTEXITCODE -ne 0) { pip install gguf | Out-Null }

if ($Path -eq "") {
    Write-Host "Searching fixed drives for *.gguf shards (large files only; a few minutes)..."
    $drives = Get-Volume | Where-Object { $_.DriveLetter -and $_.DriveType -eq 'Fixed' } | ForEach-Object { "$($_.DriveLetter):\" }
    $hits = foreach ($d in $drives) {
        Get-ChildItem -Path $d -Recurse -Filter "*.gguf" -File -EA SilentlyContinue |
            Where-Object { $_.Length -gt 1GB -and $_.Name -match 'K3|kimi' }
    }
    if (-not $hits) {
        $hits = foreach ($d in $drives) {
            Get-ChildItem -Path $d -Recurse -Filter "*.gguf" -File -EA SilentlyContinue | Where-Object { $_.Length -gt 1GB }
        }
    }
    if (-not $hits) { Write-Error "No .gguf shards over 1 GB found on any fixed drive. Is the model downloaded?"; exit 2 }
    $groups = $hits | Group-Object DirectoryName | Sort-Object Count -Descending
    foreach ($g in $groups) {
        $gb = [math]::Round(($g.Group | Measure-Object Length -Sum).Sum / 1GB, 1)
        Write-Host ("  {0,4} shards  {1,8} GB   {2}" -f $g.Count, $gb, $g.Name)
    }
    $Path = $groups[0].Name
}
Write-Host "`nReading headers in: $Path"
python "$PSScriptRoot\k3_geometry.py" "$Path" "$repo\k3_geometry.json"
Write-Host "`nDone. Tell Claude: 'geometry json is in the repo' and it will patch geometry.py."
