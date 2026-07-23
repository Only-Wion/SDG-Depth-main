param(
    [string]$RepoPath = (Get-Location).Path,
    [string]$Branch,
    [string]$OriginUrl,
    [string]$RemoteOriginUrl,
    [string]$RemoteHost,
    [string]$RemoteUser = "root",
    [string]$RemotePort,
    [string]$RemoteRoot = "/root/autodl-tmp/projects",
    [string]$RemoteProject,
    [string]$RemoteSetup = "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true",
    [string]$RemoteGithubHostAlias,
    [string]$RemoteDeployKeyPath,
    [string]$SshKey = (Join-Path $HOME ".ssh\id_ed25519"),
    [switch]$InstallSshKey,
    [switch]$SkipSshSetup,
    [switch]$BootstrapServer,
    [switch]$SkipServerBootstrap,
    [switch]$SkipAgents
)
$ErrorActionPreference = "Stop"

function Ask-Default {
    param([string]$Prompt, [string]$Default)
    if ($Default) {
        $value = Read-Host "$Prompt [$Default]"
        if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
        return $value
    }
    return (Read-Host $Prompt)
}

function Ask-YesNo {
    param([string]$Prompt, [string]$Default = "y")
    $suffix = if ($Default -eq "y") { "Y/n" } else { "y/N" }
    $value = Read-Host "$Prompt [$suffix]"
    if ([string]::IsNullOrWhiteSpace($value)) { $value = $Default }
    return $value -match '^(y|yes)$'
}

function PsQuote {
    param([AllowEmptyString()][string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$Content)
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing command: $Name"
    }
}

$RepoPath = (Resolve-Path -LiteralPath $RepoPath).Path
if (-not (Test-Path -LiteralPath (Join-Path $RepoPath ".git"))) {
    throw "RepoPath is not a Git repository: $RepoPath"
}

Require-Command git
Require-Command ssh
Require-Command ssh-keygen

Push-Location $RepoPath
try {
    $repoName = Split-Path -Leaf $RepoPath
    $currentBranch = (& git symbolic-ref --quiet --short HEAD 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($currentBranch)) {
        $currentBranch = (& git rev-parse --abbrev-ref HEAD 2>$null)
    }
    $currentOrigin = (& git remote get-url origin 2>$null)

    if (-not $Branch) { $Branch = Ask-Default "Development branch" ($(if ($currentBranch) { $currentBranch.Trim() } else { "dev/yuyang" })) }
    if (-not $OriginUrl) { $OriginUrl = Ask-Default "Git origin URL" ($(if ($currentOrigin) { $currentOrigin.Trim() } else { "" })) }
    if (-not $RemoteHost) { $RemoteHost = Ask-Default "Remote host" "connect.nmb2.seetacloud.com" }
    if (-not $RemoteUser) { $RemoteUser = Ask-Default "Remote user" "root" }
    if (-not $RemotePort) { $RemotePort = Ask-Default "Remote SSH port" "13561" }
    if (-not $RemoteProject) { $RemoteProject = Ask-Default "Remote project path" "$RemoteRoot/$repoName" }
    if (-not $RemoteSetup) { $RemoteSetup = Ask-Default "Remote setup command" "" }
    if (-not $RemoteGithubHostAlias) { $RemoteGithubHostAlias = "github.com-$($repoName.ToLowerInvariant())" }
    if (-not $RemoteDeployKeyPath) { $RemoteDeployKeyPath = "~/.ssh/codex_deploy_keys/$($repoName.ToLowerInvariant())_ed25519" }
    if (-not $RemoteOriginUrl) {
        if ($OriginUrl -match '^git@github\.com:(.+)$') {
            $RemoteOriginUrl = "git@${RemoteGithubHostAlias}:$($Matches[1])"
        } else {
            $RemoteOriginUrl = $OriginUrl
        }
    }

    if ([string]::IsNullOrWhiteSpace($Branch)) { throw "Branch cannot be empty." }
    if ([string]::IsNullOrWhiteSpace($OriginUrl)) { throw "OriginUrl cannot be empty." }
    if ([string]::IsNullOrWhiteSpace($RemoteHost)) { throw "RemoteHost cannot be empty." }
    if ([string]::IsNullOrWhiteSpace($RemoteProject)) { throw "RemoteProject cannot be empty." }

    & git remote get-url origin *> $null
    if ($LASTEXITCODE -eq 0) {
        & git remote set-url origin $OriginUrl
    } else {
        & git remote add origin $OriginUrl
    }

    & git show-ref --verify --quiet "refs/heads/$Branch"
    if ($LASTEXITCODE -eq 0) {
        & git switch $Branch
    } else {
        & git switch -c $Branch
    }

    New-Item -ItemType Directory -Force -Path (Join-Path $RepoPath "scripts") | Out-Null

    $config = @"
`$ErrorActionPreference = "Stop"

`$Script:RemoteHost = if (`$env:CODEX_REMOTE_HOST) { `$env:CODEX_REMOTE_HOST } else { $(PsQuote $RemoteHost) }
`$Script:RemoteUser = if (`$env:CODEX_REMOTE_USER) { `$env:CODEX_REMOTE_USER } else { $(PsQuote $RemoteUser) }
`$Script:RemotePort = if (`$env:CODEX_REMOTE_PORT) { `$env:CODEX_REMOTE_PORT } else { $(PsQuote $RemotePort) }
`$Script:RemoteSshKey = if (`$env:CODEX_REMOTE_SSH_KEY) { `$env:CODEX_REMOTE_SSH_KEY } else { $(PsQuote $SshKey) }
`$Script:RemoteProject = if (`$env:CODEX_REMOTE_PROJECT) { `$env:CODEX_REMOTE_PROJECT } else { $(PsQuote $RemoteProject) }
`$Script:RemoteBranch = if (`$env:CODEX_REMOTE_BRANCH) { `$env:CODEX_REMOTE_BRANCH } else { $(PsQuote $Branch) }
`$Script:RemoteOrigin = if (`$env:CODEX_REMOTE_ORIGIN) { `$env:CODEX_REMOTE_ORIGIN } else { $(PsQuote $RemoteOriginUrl) }
`$Script:RemoteSetup = if (`$env:CODEX_REMOTE_SETUP) { `$env:CODEX_REMOTE_SETUP } else { $(PsQuote $RemoteSetup) }
`$Script:RemoteGithubHostAlias = if (`$env:CODEX_REMOTE_GITHUB_HOST_ALIAS) { `$env:CODEX_REMOTE_GITHUB_HOST_ALIAS } else { $(PsQuote $RemoteGithubHostAlias) }
`$Script:RemoteDeployKeyPath = if (`$env:CODEX_REMOTE_DEPLOY_KEY_PATH) { `$env:CODEX_REMOTE_DEPLOY_KEY_PATH } else { $(PsQuote $RemoteDeployKeyPath) }
`$Script:RemoteTarget = "`$Script:RemoteUser@`$Script:RemoteHost"

function Get-RemoteSshArgs {
    param([switch]`$Batch)
    `$args = @("-p", `$Script:RemotePort, "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", "-o", "StrictHostKeyChecking=accept-new")
    if (`$Batch) { `$args += @("-o", "BatchMode=yes", "-o", "ConnectTimeout=8") }
    if (Test-Path -LiteralPath `$Script:RemoteSshKey) { `$args += @("-i", `$Script:RemoteSshKey) }
    `$args += `$Script:RemoteTarget
    return `$args
}

function Test-RemotePasswordless {
    `$sshArgs = Get-RemoteSshArgs -Batch
    `$oldPreference = `$ErrorActionPreference
    `$ErrorActionPreference = "Continue"
    try {
        `$output = & ssh @sshArgs "echo OK" 2>`$null
        return (`$LASTEXITCODE -eq 0 -and (`$output -join "`n") -match "OK")
    } finally {
        `$ErrorActionPreference = `$oldPreference
    }
}

function Assert-RemotePasswordless {
    if (Test-RemotePasswordless) { return }
    Write-Error "Cannot log in without a password. Run: .\scripts\setup_ssh_key.ps1 -Install"
}

function ConvertTo-BashSingleQuoted {
    param([Parameter(Mandatory = `$true)][AllowEmptyString()][string]`$Value)
    `$quote = [string][char]39
    `$doubleQuote = [string][char]34
    `$escapedQuote = `$quote + `$doubleQuote + `$quote + `$doubleQuote + `$quote
    return `$quote + `$Value.Replace(`$quote, `$escapedQuote) + `$quote
}

function Invoke-RemoteBashScript {
    param([Parameter(Mandatory = `$true)][string]`$ScriptText)
    `$sshArgs = Get-RemoteSshArgs
    `$ScriptText = `$ScriptText.Replace("`r`n", "`n").Replace("`r", "`n")
    `$temp = [System.IO.Path]::GetTempFileName()
    try {
        `$encoding = New-Object System.Text.UTF8Encoding(`$false)
        [System.IO.File]::WriteAllText(`$temp, `$ScriptText, `$encoding)
        `$process = Start-Process -FilePath "ssh" -ArgumentList (`$sshArgs + "bash -s") -RedirectStandardInput `$temp -NoNewWindow -Wait -PassThru
        if (`$process.ExitCode -ne 0) { exit `$process.ExitCode }
    } finally {
        Remove-Item -LiteralPath `$temp -Force -ErrorAction SilentlyContinue
    }
}

function Get-CurrentGitBranch {
    `$branch = (& git symbolic-ref --quiet --short HEAD 2>`$null)
    if (`$LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(`$branch)) {
        `$branch = (& git rev-parse --abbrev-ref HEAD)
    }
    return (`$branch | Select-Object -First 1).Trim()
}
"@

    $setupSsh = @'
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
'@

    $serverGithubKey = @'
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\codex_remote_config.ps1"

Assert-RemotePasswordless

$alias = ConvertTo-BashSingleQuoted $Script:RemoteGithubHostAlias
$keyPath = ConvertTo-BashSingleQuoted $Script:RemoteDeployKeyPath
$origin = $Script:RemoteOrigin

Invoke-RemoteBashScript @"
set -euo pipefail
source /etc/network_turbo 2>/dev/null || true
alias_name=$alias
key_path=$keyPath
if [[ "`$key_path" == "~/"* ]]; then
  key_path="`${key_path/#\~/`$HOME}"
fi
mkdir -p ~/.ssh/codex_deploy_keys
chmod 700 ~/.ssh ~/.ssh/codex_deploy_keys
if [[ ! -f "`$key_path" ]]; then
  ssh-keygen -t ed25519 -f "`$key_path" -N "" -C "codex-deploy-`$alias_name"
fi
touch ~/.ssh/config
chmod 600 ~/.ssh/config
tmp_config="`$(mktemp)"
awk -v host="`$alias_name" '
  BEGIN { skip=0 }
  /^Host[[:space:]]+/ {
    skip=0
    for (i=2; i<=NF; i++) if (`$i == host) skip=1
  }
  !skip { print }
' ~/.ssh/config > "`$tmp_config"
cat >> "`$tmp_config" <<EOF_CONFIG
Host `$alias_name
  HostName github.com
  User git
  IdentityFile `$key_path
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
EOF_CONFIG
mv "`$tmp_config" ~/.ssh/config
chmod 600 ~/.ssh/config
echo "-----BEGIN SERVER GITHUB DEPLOY KEY-----"
cat "`$key_path.pub"
echo "-----END SERVER GITHUB DEPLOY KEY-----"
"@

Write-Host ""
Write-Host "Add the public key above to this GitHub repository as a read-only deploy key."
Write-Host "Configured server origin:"
Write-Host "  $origin"
'@

    $bootstrap = @'
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\codex_remote_config.ps1"

Assert-RemotePasswordless

$project = ConvertTo-BashSingleQuoted $Script:RemoteProject
$parent = ConvertTo-BashSingleQuoted ([System.IO.Path]::GetDirectoryName($Script:RemoteProject).Replace("\", "/"))
$branch = ConvertTo-BashSingleQuoted $Script:RemoteBranch
$origin = ConvertTo-BashSingleQuoted $Script:RemoteOrigin

Invoke-RemoteBashScript @"
set -euo pipefail
mkdir -p $parent
if [[ ! -d $project/.git ]]; then
  git clone -b $branch $origin $project
else
  cd $project
  git remote set-url origin $origin
  git fetch origin
  git checkout $branch
  git pull --ff-only origin $branch
fi
"@
'@

    $serverPull = @'
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\codex_remote_config.ps1"
Assert-RemotePasswordless
$project = ConvertTo-BashSingleQuoted $Script:RemoteProject
$branch = ConvertTo-BashSingleQuoted $Script:RemoteBranch
Invoke-RemoteBashScript @"
set -euo pipefail
cd $project
git fetch origin
git checkout $branch
git pull --ff-only origin $branch
"@
'@

    $sync = @'
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\codex_remote_config.ps1"
$branch = Get-CurrentGitBranch
if ($branch -ne $Script:RemoteBranch) { Write-Error "Current branch is '$branch', expected '$Script:RemoteBranch'." }
$dirty = & git status --porcelain
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($dirty) { Write-Error "There are uncommitted local changes. Commit before syncing." }
& git push origin $branch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& "$PSScriptRoot\server_pull.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
'@

    $serverRun = @'
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Command)
$ErrorActionPreference = "Stop"
if (-not $Command -or $Command.Count -eq 0) {
    Write-Host "Usage: .\scripts\server_run.ps1 '<server command>'"
    exit 1
}
. "$PSScriptRoot\codex_remote_config.ps1"
Assert-RemotePasswordless
$cmd = $Command -join " "
$project = ConvertTo-BashSingleQuoted $Script:RemoteProject
$setup = ConvertTo-BashSingleQuoted $Script:RemoteSetup
$remoteCommand = ConvertTo-BashSingleQuoted $cmd
Invoke-RemoteBashScript @"
set -euo pipefail
cd $project
REMOTE_SETUP=$setup
CMD=$remoteCommand
if [[ -n "`$REMOTE_SETUP" ]]; then eval "`$REMOTE_SETUP"; fi
echo "[server] pwd: `$(pwd)"
echo "[server] branch: `$(git symbolic-ref --quiet --short HEAD 2>/dev/null || git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
echo "[server] command: `$CMD"
eval "`$CMD"
"@
'@

    Write-Utf8NoBom (Join-Path $RepoPath "scripts\codex_remote_config.ps1") $config
    Write-Utf8NoBom (Join-Path $RepoPath "scripts\setup_ssh_key.ps1") $setupSsh
    Write-Utf8NoBom (Join-Path $RepoPath "scripts\server_github_key.ps1") $serverGithubKey
    Write-Utf8NoBom (Join-Path $RepoPath "scripts\server_bootstrap.ps1") $bootstrap
    Write-Utf8NoBom (Join-Path $RepoPath "scripts\server_pull.ps1") $serverPull
    Write-Utf8NoBom (Join-Path $RepoPath "scripts\sync_to_server.ps1") $sync
    Write-Utf8NoBom (Join-Path $RepoPath "scripts\server_run.ps1") $serverRun

    if (-not $SkipAgents) {
        $agents = @"
# Git-based remote workflow

- Codex edits the local repository only.
- Do not use rsync or auto-sync.
- Sync to the server through Git only.
- Work on branch ``$Branch`` and push to ``origin/$Branch``.
- Use Windows local PowerShell/Git/OpenSSH for the workflow. Do not use WSL.
- After local edits, commit and run ``.\scripts\sync_to_server.ps1`` so the server pulls the same commit.
- If passwordless SSH is not ready, run ``.\scripts\setup_ssh_key.ps1 -Install`` once from an interactive PowerShell terminal.
- If the server cannot clone a private GitHub repository, run ``.\scripts\server_github_key.ps1`` and add the printed public key as a read-only deploy key for that repository.
- Run server commands through ``.\scripts\server_run.ps1 '<command>'``.
- If GitHub or Hugging Face network access is blocked on the server, run ``source /etc/network_turbo`` in the server command before retrying.
- Do not assume local CUDA/GPU is available.
- Do not edit files directly on the server.
- Do not sync datasets or pretrained model weights through Git; use explicit artifact transfer or server-side download commands instead.
"@
        Write-Utf8NoBom (Join-Path $RepoPath "AGENTS.md") $agents
    }

    $setupSshPath = Join-Path $RepoPath "scripts\setup_ssh_key.ps1"
    if ($InstallSshKey) {
        Write-Host "Setting up passwordless SSH. If SSH asks for a password, enter the server password once."
        & $setupSshPath -Install
    } elseif (-not $SkipSshSetup) {
        $sshReady = $false
        try {
            & $setupSshPath -Check *> $null
            $sshReady = $true
        } catch {
            $sshReady = $false
        }

        if ($sshReady) {
            Write-Host "Passwordless SSH is already ready."
        } else {
            Write-Host "Passwordless SSH is not ready."
            Write-Host "This step requires manual input: enter the server password once when SSH prompts."
            if (Ask-YesNo "Set up passwordless SSH now" "y") {
                & $setupSshPath -Install
            } else {
                Write-Host "Manual next step: .\scripts\setup_ssh_key.ps1 -Install"
            }
        }
    }

    if ($BootstrapServer -and -not $SkipServerBootstrap) {
        & (Join-Path $RepoPath "scripts\server_bootstrap.ps1")
    }

    Write-Host "Codex remote workflow configured for $RepoPath"
    if (-not $BootstrapServer) {
        Write-Host "Next after SSH is ready: .\scripts\server_bootstrap.ps1"
    }
    Write-Host "Daily sync: .\scripts\sync_to_server.ps1"
    Write-Host "Server run: .\scripts\server_run.ps1 '<command>'"
} finally {
    Pop-Location
}
