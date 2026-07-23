# 本地-远程协同 Codex 开发流程

这套流程适合 **Windows 本地开发 + Linux/GPU 服务器运行**。核心原则：

- Codex 只编辑 Windows 本地仓库。
- 代码通过 GitHub 同步到服务器。
- 服务器只通过 `git pull --ff-only` 拉取同一个 commit。
- 不用 WSL，不用 rsync，不自动同步，不直接在服务器改代码。
- 数据集和预训练权重不通过 Git 同步。

通用配置器固定放在：

```powershell
D:\project\codex_remote_workflow\setup_codex_remote_workflow.ps1
```

## 当前配置流程

现在的正确配置顺序是：

1. 准备本地 Git 仓库和 GitHub `origin`。
2. 运行 `setup_codex_remote_workflow.ps1` 生成仓库内脚本和 `AGENTS.md`。
3. setup 会检查本机到服务器的免密 SSH。
4. 如果免密 SSH 未配置，setup 会提醒你输入一次服务器密码，完成 `authorized_keys` 配置。
5. 如果服务器 clone 私有 GitHub 仓库失败，运行 `.\scripts\server_github_key.ps1`，把打印出的公钥添加到该 GitHub 仓库的 Deploy keys。
6. 手动运行 `.\scripts\server_bootstrap.ps1`，让服务器 clone 或更新项目。
7. 日常开发时只使用 `.\scripts\sync_to_server.ps1` 和 `.\scripts\server_run.ps1`。

注意：setup 默认**不会自动 bootstrap 服务器项目**，因为私有仓库经常需要先手动添加 Deploy key。确认免密 SSH 和 GitHub deploy key 都准备好后，再手动运行：

```powershell
.\scripts\server_bootstrap.ps1
```

如果你确定服务器 SSH 和 GitHub 权限都已准备好，也可以显式让 setup 直接 bootstrap：

```powershell
D:\project\codex_remote_workflow\setup_codex_remote_workflow.ps1 -BootstrapServer
```

## 场景一：复现别人的代码

先在 Windows 本地 clone：

```powershell
cd D:\project
git clone <别人的仓库地址> myrepo
cd myrepo
```

如果要长期开发，建议先 fork 到自己的 GitHub，然后设置：

```powershell
git remote rename origin upstream
git remote add origin <你的fork仓库地址>
git switch -c dev/yuyang
git push -u origin dev/yuyang
```

运行配置器：

```powershell
D:\project\codex_remote_workflow\setup_codex_remote_workflow.ps1 `
  -RepoPath D:\project\myrepo `
  -Branch dev/yuyang `
  -OriginUrl <你的fork仓库地址> `
  -RemoteHost connect.xxx.com `
  -RemotePort 13561 `
  -RemoteUser root `
  -RemoteProject /root/autodl-tmp/projects/myrepo
```

setup 过程中如果出现：

```text
Set up passwordless SSH now [Y/n]:
```

输入 `y` 或直接回车，然后按 SSH 提示输入一次服务器密码。

如果后续 `server_bootstrap.ps1` 报：

```text
ERROR: Repository not found.
```

通常说明服务器上的 GitHub SSH key 没有这个私有仓库权限。运行：

```powershell
.\scripts\server_github_key.ps1
```

把打印出的 `ssh-ed25519 ...` 公钥添加到 GitHub：

```text
Repository -> Settings -> Deploy keys -> Add deploy key
```

只读即可，不要勾选 `Allow write access`。添加完成后再运行：

```powershell
.\scripts\server_bootstrap.ps1
```

## 场景二：从零新建本地项目

先建本地仓库：

```powershell
mkdir D:\project\myrepo
cd D:\project\myrepo
git init
git switch -c dev/yuyang
```

在 GitHub 创建空仓库，然后绑定：

```powershell
git remote add origin git@github.com:<yourname>/myrepo.git
```

至少提交一个初始文件：

```powershell
echo "# myrepo" > README.md
git add README.md
git commit -m "init"
git push -u origin dev/yuyang
```

运行配置器：

```powershell
D:\project\codex_remote_workflow\setup_codex_remote_workflow.ps1 `
  -RepoPath D:\project\myrepo `
  -Branch dev/yuyang `
  -OriginUrl git@github.com:<yourname>/myrepo.git `
  -RemoteHost connect.xxx.com `
  -RemotePort 13561 `
  -RemoteUser root `
  -RemoteProject /root/autodl-tmp/projects/myrepo
```

之后同样按提示完成免密 SSH、GitHub Deploy key 和服务器 bootstrap。

## 日常开发

让 Codex 在 Windows 本地改代码。改完后：

```powershell
git add -A
git commit -m "describe the change"
.\scripts\sync_to_server.ps1
```

`sync_to_server.ps1` 会：

1. 检查当前分支是否是配置分支。
2. 检查本地工作区是否干净。
3. `git push origin <branch>`。
4. SSH 到服务器执行 `git pull --ff-only origin <branch>`。

服务器命令统一这样跑：

```powershell
.\scripts\server_run.ps1 'nvidia-smi'
.\scripts\server_run.ps1 'pwd && git status --short --branch'
.\scripts\server_run.ps1 'python train.py'
```

如果项目需要 conda 环境，可以在配置时设置：

```powershell
-RemoteSetup "source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv"
```

## 网络受阻

如果服务器访问 GitHub 或 Hugging Face 受阻，在服务器命令里先运行：

```bash
source /etc/network_turbo
```

示例：

```powershell
.\scripts\server_run.ps1 'source /etc/network_turbo && git pull'
.\scripts\server_run.ps1 'source /etc/network_turbo && huggingface-cli download <repo-id>'
```

这条规则也会写入每个仓库生成的 `AGENTS.md`。

## 数据集和预训练权重

数据集和预训练模型权重不能通过 Git 同步，包括但不限于：

```text
data/
datasets/
checkpoints/
weights/
pretrained/
outputs/
runs/
wandb/
*.pt
*.pth
*.ckpt
*.safetensors
*.onnx
*.bin
```

推荐方式：

- 公开权重或数据集优先让服务器自己下载。
- Hugging Face 访问受阻时先 `source /etc/network_turbo`。
- 本地已有的大文件，用显式传输命令上传到服务器项目目录外，例如 `/root/autodl-tmp/artifacts/<repo-name>/`。
- 项目 Git 目录里只放软链接、路径配置或小型示例文件。

示例：

```powershell
.\scripts\server_run.ps1 'source /etc/network_turbo && mkdir -p /root/autodl-tmp/artifacts/myrepo/pretrained && huggingface-cli download <repo-id> --local-dir /root/autodl-tmp/artifacts/myrepo/pretrained'
```

## 配置后仓库里会有什么

每个被配置的仓库会有：

```text
AGENTS.md
scripts/
  codex_remote_config.ps1
  setup_ssh_key.ps1
  server_github_key.ps1
  server_bootstrap.ps1
  server_pull.ps1
  sync_to_server.ps1
  server_run.ps1
```

通用配置器本身不放进业务仓库，统一保存在：

```text
D:\project\codex_remote_workflow
```

## 规则摘要

- 只在 Windows 本地编辑代码。
- 不用 WSL。
- 不用 rsync。
- 不自动同步。
- 不直接编辑服务器文件。
- 服务器只通过 Git 拉取已经 push 到 `origin/<branch>` 的 commit。
- 服务器 clone 私有仓库失败时，先配置 `server_github_key.ps1` 输出的 Deploy key。
- GitHub 或 Hugging Face 网络受阻时，服务器命令先运行 `source /etc/network_turbo`。
- 数据集和预训练模型权重不能通过 Git 同步。
