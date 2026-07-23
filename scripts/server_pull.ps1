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