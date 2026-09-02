# tools/wsl_setup.ps1 - interim Linux dev environment WITHOUT a USB stick: WSL2 Ubuntu, with the option
# of handing the model drive (D:) to Linux as a raw block device so io_uring/O_DIRECT can be tested for real.
#
#   powershell -ExecutionPolicy Bypass -File tools\wsl_setup.ps1              # install WSL2 + Ubuntu (reboot once)
#   powershell -ExecutionPolicy Bypass -File tools\wsl_setup.ps1 -MountD      # give D: (whole NVMe) to WSL, bare
#   powershell -ExecutionPolicy Bypass -File tools\wsl_setup.ps1 -UnmountD    # give D: back to Windows
#
# Run from an ADMINISTRATOR PowerShell.
#
# What WSL2 is good for here: building and debugging the engine (io_uring streamer, CUDA kernels via the
# Windows NVIDIA driver) months before the box is rebooted into native Ubuntu. What it is NOT: the
# production path. /mnt/c and /mnt/d inside WSL go through a 9p file server and are slow; -MountD
# bypasses that by attaching the physical disk to the VM (Hyper-V SCSI passthrough), which is close to
# native but still not the number the engine ships on. Native Linux via linux_prep.ps1 when a USB stick
# (any 8 GB+) is available remains the plan.
#
# While D: is mounted into WSL it is OFFLINE in Windows (Explorer will not show it). -UnmountD restores it.

param([switch]$MountD, [switch]$UnmountD, [string]$Drive = "D")
$ErrorActionPreference = "Stop"
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { Write-Host "run from an administrator PowerShell."; exit 1 }

function Get-DiskNumberForLetter($letter) { (Get-Partition -DriveLetter $letter -ErrorAction Stop).DiskNumber }

if ($UnmountD) {
    $n = (Get-Disk | Where-Object { $_.IsOffline } | Select-Object -First 1).Number
    wsl --unmount 2>$null | Out-Null
    if ($null -ne $n) { Set-Disk -Number $n -IsOffline $false; Write-Host "disk $n back online in Windows." } else { Write-Host "no offline disk found." }
    Get-Volume -DriveLetter $Drive -ErrorAction SilentlyContinue | Format-Table -AutoSize
    exit 0
}

if ($MountD) {
    $n = Get-DiskNumberForLetter $Drive
    $d = Get-Disk -Number $n
    Write-Host ("{0}: is disk {1}: {2}, {3} GB" -f $Drive, $n, $d.FriendlyName, [math]::Round($d.Size/1GB))
    $ans = Read-Host "Take it OFFLINE in Windows and attach it to WSL as a raw block device? (nothing is written; -UnmountD reverses it) [y/N]"
    if ($ans -ne "y") { exit 0 }
    Set-Disk -Number $n -IsOffline $true
    wsl --mount "\\.\PHYSICALDRIVE$n" --bare
    Write-Host "attached. Inside WSL:"
    Write-Host "   lsblk -f                                   # find the new /dev/sdX with an ntfs partition"
    Write-Host "   sudo mkdir -p /mnt/win/D && sudo mount -t ntfs3 -o ro /dev/sdX1 /mnt/win/D"
    Write-Host "   bash ~/dev/sisyphus-src/tools/linux_setup.sh --bench    # fio io_uring O_DIRECT on the real disk"
    Write-Host "When done: powershell -File tools\wsl_setup.ps1 -UnmountD"
    exit 0
}

# ---- install WSL2 + latest Ubuntu -------------------------------------------------------------
Write-Host "== WSL status =="
$status = (wsl --status 2>&1 | Out-String)
$status | Write-Host
$online = (wsl --list --online 2>&1 | Out-String)
$ubuntu = ($online -split "`n" | Where-Object { $_ -match "^\s*Ubuntu-\d\d\.\d\d\s" } | ForEach-Object { ($_ -split "\s+")[1] } | Sort-Object -Descending | Select-Object -First 1)
if (-not $ubuntu) { $ubuntu = "Ubuntu" }
$installed = (wsl --list --quiet 2>&1 | Out-String)
if ($installed -match [regex]::Escape($ubuntu)) {
    Write-Host "$ubuntu already installed."
} else {
    Write-Host "installing WSL2 + $ubuntu (a reboot may be required; re-run this script after)..."
    wsl --install -d $ubuntu
    Write-Host "When Ubuntu opens and asks for a username/password, pick any. Then, inside Ubuntu:"
}
Write-Host @"

== inside WSL Ubuntu ==
   git clone https://github.com/Tarkov5502/Sisyphus.git ~/dev/sisyphus-src
   bash ~/dev/sisyphus-src/tools/linux_setup.sh --no-token
      (WSL-aware: skips the NVIDIA driver - the Windows driver is used - installs the CUDA toolkit,
       builds llama.cpp, runs the fio bench on whatever is mounted under /mnt/win)
   nvidia-smi                                   # GPU visible through the Windows driver

For a real drive number under WSL: powershell -File tools\wsl_setup.ps1 -MountD   (then -UnmountD)
The first_token run under WSL would go through 9p at ~1 GB/s - do that one on Windows (tools\first_token.ps1)
or native Linux instead.
"@
