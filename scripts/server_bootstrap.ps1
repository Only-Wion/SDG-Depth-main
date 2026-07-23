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