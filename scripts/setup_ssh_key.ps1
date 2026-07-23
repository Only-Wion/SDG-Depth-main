param([switch]$Install, [switch]$Check)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\codex_remote_config.ps1"

function Ensure-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) { throw "Missing command: $Name" }
}

function Ensure-KnownHost {
    $sshDir = Split-Path -Parent $Script:RemoteSshKey
    New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
    & ssh-keygen -F "[$Script:RemoteHost]:$Script:RemotePort" *> $null
    if ($LASTEXITCODE -eq 0) { return }

    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $scan = & ssh-keyscan -p $Script:RemotePort $Script:RemoteHost 2>$null
        if ($LASTEXITCODE -eq 0 -and $scan) {
            Add-Content -Path (Join-Path $sshDir "known_hosts") -Value $scan
        }
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

Ensure-Command ssh
Ensure-Command ssh-keygen

if (Test-RemotePasswordless) {
    Write-Host "Passwordless SSH is already working for $Script:RemoteTarget."
    exit 0
}
if ($Check) { Write-Error "Passwordless SSH is not configured for $Script:RemoteTarget." }

$sshDir = Split-Path -Parent $Script:RemoteSshKey
New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
if (-not (Test-Path -LiteralPath $Script:RemoteSshKey) -or -not (Test-Path -LiteralPath "$Script:RemoteSshKey.pub")) {
    Write-Host "Creating SSH key: $Script:RemoteSshKey"
    & ssh-keygen -t ed25519 -f $Script:RemoteSshKey -N "" -C "codex-remote-workflow"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Ensure-KnownHost

$publicKey = Get-Content -Raw -LiteralPath "$Script:RemoteSshKey.pub"
if (-not $Install) {
    Write-Host "Run this from an interactive PowerShell terminal:"
    Write-Host "  .\scripts\setup_ssh_key.ps1 -Install"
    Write-Host ""
    Write-Host "Public key:"
    Write-Host $publicKey.Trim()
    exit 1
}

$publicKeyForRemote = $publicKey.Trim().Replace("'", "'`"`"`'")
$remote = "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; grep -qxF '$publicKeyForRemote' ~/.ssh/authorized_keys || printf '%s\n' '$publicKeyForRemote' >> ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys"
$sshArgs = Get-RemoteSshArgs
& ssh @sshArgs $remote
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (Test-RemotePasswordless) { Write-Host "Passwordless SSH is ready." } else { Write-Error "SSH key installation finished, but passwordless login still failed." }