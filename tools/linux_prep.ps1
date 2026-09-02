# tools/linux_prep.ps1 - get this Windows box ready to dual-boot Ubuntu for the Sisyphus engine.
#
#   powershell -ExecutionPolicy Bypass -File tools\linux_prep.ps1                 # check + download only
#   powershell -ExecutionPolicy Bypass -File tools\linux_prep.ps1 -ShrinkGB 300   # also carve 300 GB off C:
#   powershell -ExecutionPolicy Bypass -File tools\linux_prep.ps1 -DisableFastStartup
#
# Run from an ADMINISTRATOR PowerShell for the shrink / fast-startup steps (checks work without).
#
# What it does:
#   1. Checks the things that break dual-boot: BitLocker on C:, Secure Boot, Fast Startup (hibernation
#      locks NTFS so Linux mounts it read-only), free space on C:, and whether a USB stick is plugged in.
#   2. Downloads Ubuntu 26.04.1 LTS desktop ISO (6 GB) + verifies SHA256, and Rufus (portable) to
#      C:\dev\linux\. Nothing is written to any disk except that folder.
#   3. With -ShrinkGB N: shrinks C: by N GB, leaving unallocated space for the Ubuntu installer to
#      use ("Install alongside Windows" picks it up). Prompts before doing it.
#   4. Writes linux_prep.json in the repo root and prints the exact next steps.
#
# Why native Linux and not WSL: the engine's whole speed comes from O_DIRECT + io_uring reads at
# QD32 straight off the NVMe drives into pinned buffers and onto the GPU. WSL2 puts a virtual disk
# and a virtualised PCIe path in the middle of that. diskspd proved the drives do 10.55 GB/s; the
# engine needs the same path on Linux, which means a real kernel on real hardware.

param(
    [int]$ShrinkGB = 0,
    [switch]$DisableFastStartup,
    [string]$Dest = "C:\dev\linux"
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$outjson = Join-Path $root "linux_prep.json"
$res = [ordered]@{ date = (Get-Date).ToString("s") }
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$res.admin = $isAdmin
Write-Host ("admin: {0}" -f $isAdmin)

# ---- 1. checks -------------------------------------------------------------------------------
Write-Host "`n== checks =="
try {
    $bl = Get-BitLockerVolume -MountPoint "C:" -ErrorAction Stop
    $res.bitlocker_C = "$($bl.ProtectionStatus) / $($bl.VolumeStatus)"
} catch {
    $mb = (manage-bde -status C: 2>$null | Out-String)
    if ($mb -match "Protection Status:\s*(.+)") { $res.bitlocker_C = $Matches[1].Trim() } else { $res.bitlocker_C = "unknown (run as admin)" }
}
Write-Host ("BitLocker on C:      {0}" -f $res.bitlocker_C)
if ("$($res.bitlocker_C)" -match "^On|Protection On") {
    Write-Host "  !! C: is BitLocker-encrypted. Before dual-booting: Settings > Privacy & security > Device encryption"
    Write-Host "     (or BitLocker) -> turn OFF and let it finish decrypting, OR at minimum save your recovery key."
    Write-Host "     Installing a second bootloader changes the boot measurements and BitLocker WILL ask for the key."
}
try { $sb = Confirm-SecureBootUEFI } catch { $sb = "unknown" }
$res.secure_boot = "$sb"
Write-Host ("Secure Boot:         {0}" -f $sb)
if ("$sb" -eq "True") {
    Write-Host "  Ubuntu installs fine with Secure Boot on, but the NVIDIA driver then needs a MOK password enrolled"
    Write-Host "  on first boot. Simplest: disable Secure Boot in BIOS (MSI: Del at boot > Settings > Security)."
}
$hb = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power" -Name HiberbootEnabled -ErrorAction SilentlyContinue
$res.fast_startup = if ($hb) { [bool]$hb.HiberbootEnabled } else { "unknown" }
Write-Host ("Fast Startup:        {0}" -f $res.fast_startup)
if ($DisableFastStartup) {
    if ($isAdmin) {
        Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power" -Name HiberbootEnabled -Value 0
        powercfg /h off | Out-Null
        $res.fast_startup = $false
        Write-Host "  -> Fast Startup disabled, hibernation off (Linux can now mount C: and D: read-write)."
    } else { Write-Host "  -> -DisableFastStartup needs an administrator PowerShell." }
} elseif ($res.fast_startup -eq $true) {
    Write-Host "  !! Fast Startup is on: Windows leaves NTFS 'dirty' at shutdown and Linux mounts it read-only."
    Write-Host "     Re-run with -DisableFastStartup from an admin PowerShell."
}
$c = Get-Volume -DriveLetter C
$res.c_free_gb = [math]::Round($c.SizeRemaining / 1GB); $res.c_size_gb = [math]::Round($c.Size / 1GB)
Write-Host ("C: free:             {0} GB of {1} GB" -f $res.c_free_gb, $res.c_size_gb)
$part = Get-Partition -DriveLetter C
$disk = Get-Disk -Number $part.DiskNumber
$res.c_disk = "$($disk.FriendlyName) (disk $($disk.Number), $($disk.PartitionStyle))"
Write-Host ("C: lives on:         {0}" -f $res.c_disk)
$unalloc = [math]::Round(($disk.Size - ($disk | Get-Partition | Measure-Object -Property Size -Sum).Sum) / 1GB)
$res.unallocated_gb_on_c_disk = $unalloc
Write-Host ("unallocated there:   {0} GB" -f $unalloc)
$usb = @(Get-Disk | Where-Object { $_.BusType -eq "USB" })
$res.usb_disks = @($usb | ForEach-Object { "$($_.FriendlyName) $([math]::Round($_.Size/1GB)) GB" })
Write-Host ("USB sticks:          {0}" -f $(if ($usb.Count) { $res.usb_disks -join "; " } else { "none plugged in (need one, 8 GB+, it will be ERASED)" }))
$res.ram_gb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)

# ---- 2. downloads ----------------------------------------------------------------------------
Write-Host "`n== downloads -> $Dest =="
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$isoName = "ubuntu-26.04.1-desktop-amd64.iso"
$isoUrl = "https://releases.ubuntu.com/26.04.1/$isoName"
$iso = Join-Path $Dest $isoName
$sums = Join-Path $Dest "SHA256SUMS"
Invoke-WebRequest -Uri "https://releases.ubuntu.com/26.04.1/SHA256SUMS" -OutFile $sums -UseBasicParsing
$want = ((Get-Content $sums) | Where-Object { $_ -match [regex]::Escape($isoName) } | Select-Object -First 1) -split "\s+" | Select-Object -First 1
if (Test-Path $iso) {
    Write-Host "ISO already present, verifying..."
} else {
    Write-Host "Downloading $isoName (about 6 GB; this is the slow part)..."
    $ProgressPreference = "SilentlyContinue"
    try { Start-BitsTransfer -Source $isoUrl -Destination $iso -DisplayName "Ubuntu ISO" -ErrorAction Stop }
    catch { Invoke-WebRequest -Uri $isoUrl -OutFile $iso -UseBasicParsing }
    $ProgressPreference = "Continue"
}
$have = (Get-FileHash -Algorithm SHA256 -Path $iso).Hash.ToLower()
$res.iso = $iso; $res.iso_sha256_ok = ($have -eq $want.ToLower())
Write-Host ("ISO SHA256 {0}" -f $(if ($res.iso_sha256_ok) { "OK" } else { "MISMATCH - delete it and re-run" }))

$rufusExe = Get-ChildItem -Path $Dest -Filter "rufus*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $rufusExe) {
    Write-Host "Downloading Rufus (portable) from github.com/pbatard/rufus ..."
    try {
        $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/pbatard/rufus/releases/latest" -Headers @{ "User-Agent" = "sisyphus" }
        $a = $rel.assets | Where-Object { $_.name -match "^rufus-[\d.]+p\.exe$" } | Select-Object -First 1
        if (-not $a) { $a = $rel.assets | Where-Object { $_.name -match "^rufus-[\d.]+\.exe$" } | Select-Object -First 1 }
        Invoke-WebRequest -Uri $a.browser_download_url -OutFile (Join-Path $Dest $a.name) -UseBasicParsing
        $rufusExe = Get-Item (Join-Path $Dest $a.name)
    } catch { Write-Host "  Rufus download failed ($_); get it from https://rufus.ie" }
}
$res.rufus = if ($rufusExe) { $rufusExe.FullName } else { "missing" }
Write-Host ("Rufus: {0}" -f $res.rufus)

# ---- 3. optional shrink ----------------------------------------------------------------------
if ($ShrinkGB -gt 0) {
    Write-Host "`n== shrink C: by $ShrinkGB GB =="
    if (-not $isAdmin) { Write-Host "needs an administrator PowerShell."; }
    else {
        $sup = Get-PartitionSupportedSize -DriveLetter C
        $minGB = [math]::Round($sup.SizeMin / 1GB); $curGB = [math]::Round($part.Size / 1GB)
        $newGB = $curGB - $ShrinkGB
        Write-Host ("C: is {0} GB, can shrink to {1} GB minimum; target {2} GB" -f $curGB, $minGB, $newGB)
        if ($newGB -lt $minGB) { Write-Host "  cannot shrink that far (unmovable files); try a smaller -ShrinkGB or defragment first." }
        else {
            $ans = Read-Host "Shrink C: to $newGB GB now? Windows keeps running; this is the standard Disk Management shrink. [y/N]"
            if ($ans -eq "y") {
                Resize-Partition -DriveLetter C -Size ($newGB * 1GB)
                $unalloc = [math]::Round(($disk.Size - (Get-Disk -Number $disk.Number | Get-Partition | Measure-Object -Property Size -Sum).Sum) / 1GB)
                $res.unallocated_gb_on_c_disk = $unalloc
                Write-Host ("  done: {0} GB unallocated on disk {1} for Ubuntu." -f $unalloc, $disk.Number)
            } else { Write-Host "  skipped." }
        }
    }
}

$res | ConvertTo-Json | Set-Content -Path $outjson -Encoding ASCII
Write-Host "`nwrote $outjson"

# ---- 4. next steps ---------------------------------------------------------------------------
Write-Host @"

== next steps ==
 1. If BitLocker is ON above: turn it off (or save the recovery key) BEFORE anything else.
 2. If Fast Startup is True above: re-run this script as admin with -DisableFastStartup.
 3. If 'unallocated there' is 0: re-run as admin with -ShrinkGB 300 (Ubuntu root + a 600 GB ext4 model
    partition is the plan; 300 GB is enough to start - the model partition can be carved later).
 4. Plug in a USB stick (8 GB+, will be erased), run Rufus: Device = the stick, SELECT = the ISO,
    Partition scheme = GPT, Target = UEFI (non CSM), START, accept 'ISO image mode'.
 5. Reboot. Tap Del for the MSI BIOS: Settings > Security > Secure Boot -> Disabled (saves the MOK dance),
    Boot > Fast Boot -> Disabled. Save (F10), then F11 at the next boot and pick the USB stick (UEFI entry).
 6. Ubuntu installer: Interactive, Default selection, tick 'Install third-party software' (NVIDIA driver),
    Installation type: 'Install Ubuntu alongside Windows' -> it uses the unallocated space. Do NOT pick
    'Erase disk'. Username: whatever; hostname suggestion: sisyphus.
 7. First boot into Ubuntu: open a terminal and run
       bash /media/*/*/dev/sisyphus-src/tools/linux_setup.sh      (if the Windows disk auto-mounted)
    or: git clone https://github.com/Tarkov5502/Sisyphus.git ~/dev/sisyphus-src && bash ~/dev/sisyphus-src/tools/linux_setup.sh
    It installs drivers and tools, mounts C: and D:, runs the fio (Linux diskspd) bench, builds llama.cpp
    and generates the first K3 token on Linux, writing rig_fio.json and first_token_linux.json.
"@
