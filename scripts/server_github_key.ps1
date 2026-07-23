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