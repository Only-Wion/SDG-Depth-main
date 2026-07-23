$ErrorActionPreference = "Stop"

$Script:RemoteHost = if ($env:CODEX_REMOTE_HOST) { $env:CODEX_REMOTE_HOST } else { '222.20.99.89' }
$Script:RemoteUser = if ($env:CODEX_REMOTE_USER) { $env:CODEX_REMOTE_USER } else { 'jianglai' }
$Script:RemotePort = if ($env:CODEX_REMOTE_PORT) { $env:CODEX_REMOTE_PORT } else { '22' }
$Script:RemoteSshKey = if ($env:CODEX_REMOTE_SSH_KEY) { $env:CODEX_REMOTE_SSH_KEY } else { 'C:\Users\admin\.ssh\id_ed25519' }
$Script:RemoteProject = if ($env:CODEX_REMOTE_PROJECT) { $env:CODEX_REMOTE_PROJECT } else { '/mnt/data/jianglai/SDG-Depth-main' }
$Script:RemoteBranch = if ($env:CODEX_REMOTE_BRANCH) { $env:CODEX_REMOTE_BRANCH } else { 'dev/yuyang' }
$Script:RemoteOrigin = if ($env:CODEX_REMOTE_ORIGIN) { $env:CODEX_REMOTE_ORIGIN } else { 'git@github.com-sdg-depth-main:Only-Wion/SDG-Depth-main.git' }
$Script:RemoteSetup = if ($env:CODEX_REMOTE_SETUP) { $env:CODEX_REMOTE_SETUP } else { 'source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true' }
$Script:RemoteGithubHostAlias = if ($env:CODEX_REMOTE_GITHUB_HOST_ALIAS) { $env:CODEX_REMOTE_GITHUB_HOST_ALIAS } else { 'github.com-sdg-depth-main' }
$Script:RemoteDeployKeyPath = if ($env:CODEX_REMOTE_DEPLOY_KEY_PATH) { $env:CODEX_REMOTE_DEPLOY_KEY_PATH } else { '~/.ssh/codex_deploy_keys/sdg-depth-main_ed25519' }
$Script:RemoteTarget = "$Script:RemoteUser@$Script:RemoteHost"

function Get-RemoteSshArgs {
    param([switch]$Batch)
    $args = @("-p", $Script:RemotePort, "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", "-o", "StrictHostKeyChecking=accept-new")
    if ($Batch) { $args += @("-o", "BatchMode=yes", "-o", "ConnectTimeout=8") }
    if (Test-Path -LiteralPath $Script:RemoteSshKey) { $args += @("-i", $Script:RemoteSshKey) }
    $args += $Script:RemoteTarget
    return $args
}

function Test-RemotePasswordless {
    $sshArgs = Get-RemoteSshArgs -Batch
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & ssh @sshArgs "echo OK" 2>$null
        return ($LASTEXITCODE -eq 0 -and ($output -join "
") -match "OK")
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

function Assert-RemotePasswordless {
    if (Test-RemotePasswordless) { return }
    Write-Error "Cannot log in without a password. Run: .\scripts\setup_ssh_key.ps1 -Install"
}

function ConvertTo-BashSingleQuoted {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)
    $quote = [string][char]39
    $doubleQuote = [string][char]34
    $escapedQuote = $quote + $doubleQuote + $quote + $doubleQuote + $quote
    return $quote + $Value.Replace($quote, $escapedQuote) + $quote
}

function Invoke-RemoteBashScript {
    param([Parameter(Mandatory = $true)][string]$ScriptText)
    $sshArgs = Get-RemoteSshArgs
    $ScriptText = $ScriptText.Replace("
", "
").Replace("", "
")
    $temp = [System.IO.Path]::GetTempFileName()
    try {
        $encoding = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($temp, $ScriptText, $encoding)
        $process = Start-Process -FilePath "ssh" -ArgumentList ($sshArgs + "bash -s") -RedirectStandardInput $temp -NoNewWindow -Wait -PassThru
        if ($process.ExitCode -ne 0) { exit $process.ExitCode }
    } finally {
        Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue
    }
}

function Get-CurrentGitBranch {
    $branch = (& git symbolic-ref --quiet --short HEAD 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($branch)) {
        $branch = (& git rev-parse --abbrev-ref HEAD)
    }
    return ($branch | Select-Object -First 1).Trim()
}