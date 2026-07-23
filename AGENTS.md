# Git-based remote workflow

- Codex edits the local repository only.
- Do not use rsync or auto-sync.
- Sync to the server through Git only.
- Work on branch `dev/yuyang` and push to `origin/dev/yuyang`.
- Use Windows local PowerShell/Git/OpenSSH for the workflow. Do not use WSL.
- After local edits, commit and run `.\scripts\sync_to_server.ps1` so the server pulls the same commit.
- If passwordless SSH is not ready, run `.\scripts\setup_ssh_key.ps1 -Install` once from an interactive PowerShell terminal.
- If the server cannot clone a private GitHub repository, run `.\scripts\server_github_key.ps1` and add the printed public key as a read-only deploy key for that repository.
- Run server commands through `.\scripts\server_run.ps1 '<command>'`.
- If GitHub or Hugging Face network access is blocked on the server, run `source /etc/network_turbo` in the server command before retrying.
- Do not assume local CUDA/GPU is available.
- Do not edit files directly on the server.
- Do not sync datasets or pretrained model weights through Git; use explicit artifact transfer or server-side download commands instead.