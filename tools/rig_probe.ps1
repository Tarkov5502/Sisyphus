# tools\rig_probe.ps1 - measure the rig instead of assuming it. ~3-5 minutes. Read-only.
#     powershell -ExecutionPolicy Bypass -File tools\rig_probe.ps1
# Writes rig_probe.json into the repo root for Claude to fold into geometry.py.
param([string]$ModelDir = "D:\models\Kimi-K3-GGUF\UD-Q2_K_XL", [int]$ReadGB = 20)
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$out = [ordered]@{ timestamp = (Get-Date).ToString("s") }

Write-Host "== GPU / PCIe link =="
$smi = & nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,driver_version --format=csv,noheader 2>$null
Write-Host "  $smi"; $out.gpu = "$smi"

Write-Host "== RAM =="
$dimms = Get-CimInstance Win32_PhysicalMemory | Select DeviceLocator, @{n='GB';e={$_.Capacity/1GB}}, Speed, ConfiguredClockSpeed
$dimms | Format-Table -AutoSize | Out-String | Write-Host
$out.ram = $dimms | ForEach-Object { "$($_.DeviceLocator): $($_.GB) GB @ $($_.ConfiguredClockSpeed) MT/s" }

Write-Host "== Volumes =="
$vols = Get-Volume | Where-Object DriveLetter | Select DriveLetter, FileSystemLabel, @{n='SizeGB';e={[math]::Round($_.Size/1GB)}}, @{n='FreeGB';e={[math]::Round($_.SizeRemaining/1GB)}}
$vols | Format-Table -AutoSize | Out-String | Write-Host
$out.volumes = $vols | ForEach-Object { "$($_.DriveLetter): $($_.FileSystemLabel) $($_.SizeGB) GB, $($_.FreeGB) GB free" }
$disks = Get-PhysicalDisk | Select FriendlyName, BusType, @{n='SizeGB';e={[math]::Round($_.Size/1GB)}}
$out.disks = $disks | ForEach-Object { "$($_.FriendlyName) $($_.SizeGB) GB" }
# which physical disk backs which letter
$map = Get-Partition | Where-Object DriveLetter | ForEach-Object { $d = Get-Disk -Number $_.DiskNumber; "$($_.DriveLetter): -> $($d.FriendlyName)" }
$map | ForEach-Object { Write-Host "  $_" }; $out.letter_to_disk = $map

Write-Host "== Sequential read benchmark ($ReadGB GB per drive, 8 MB blocks, unbuffered) =="
$py = @"
import sys, os, time, ctypes, mmap
path, gb = sys.argv[1], float(sys.argv[2])
# Python 'rb' with a large block; FILE_FLAG_NO_BUFFERING via os.O_BINARY isn't exposed, so
# read more than RAM to defeat the page cache: gb should exceed free RAM for a true number.
blk = 8 * 1024 * 1024
total = 0; t0 = time.perf_counter(); best = 0.0
with open(path, 'rb', buffering=0) as f:
    tb = time.perf_counter(); n = 0
    while total < gb * 1e9:
        b = f.read(blk)
        if not b: break
        total += len(b); n += len(b)
        if n >= 2e9:
            dt = time.perf_counter() - tb; best = max(best, n / dt / 1e9); tb = time.perf_counter(); n = 0
dt = time.perf_counter() - t0
print(f"{total/1e9:.1f} GB in {dt:.1f} s -> {total/dt/1e9:.2f} GB/s average, {best:.2f} GB/s best 2-GB window")
"@
$tmp = Join-Path $env:TEMP "seqread.py"; Set-Content $tmp $py -Encoding UTF8
# D: - read a shard (cold: pick the one least likely cached)
$shard = Get-ChildItem "$ModelDir\*00007-of-*.gguf" | Select -First 1
if ($shard) { Write-Host "  D: ($($shard.Name)):"; $r = python $tmp $shard.FullName $ReadGB; Write-Host "    $r"; $out.seq_read_D = "$r" }
# C: - need a big file; use the largest file found under C:\ common spots, else the pagefile is off-limits: create a temp 20 GB file once
$big = Get-ChildItem C:\Users\$env:USERNAME\Downloads, C:\Windows\Installer -Recurse -File -EA SilentlyContinue | Sort-Object Length -Descending | Select -First 1
if ($big -and $big.Length -gt 4GB) { $target = $big.FullName } else {
    $target = "C:\seqread_test.bin"
    if (-not (Test-Path $target)) { Write-Host "  creating 12 GB test file on C: (one-off)..."; fsutil file createnew $target 12884901888 | Out-Null; fsutil file setvaliddata $target 12884901888 | Out-Null }
}
Write-Host "  C: ($target):"; $r = python $tmp $target $ReadGB; Write-Host "    $r"; $out.seq_read_C = "$r"; $out.seq_read_C_file = $target

Write-Host "== CPU =="
$cpu = Get-CimInstance Win32_Processor | Select Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed
$out.cpu = "$($cpu.Name) $($cpu.NumberOfCores)C/$($cpu.NumberOfLogicalProcessors)T @ $($cpu.MaxClockSpeed) MHz"; Write-Host "  $($out.cpu)"

$out | ConvertTo-Json -Depth 3 | Set-Content (Join-Path $repo "rig_probe.json") -Encoding UTF8
Write-Host "`nwrote $repo\rig_probe.json - tell Claude it's there."
