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