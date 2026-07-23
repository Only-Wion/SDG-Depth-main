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