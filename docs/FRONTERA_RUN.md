# Frontera：本地修改，远程 pull 与运行（2026-09-23）

当前入口是本页。代码和 paper/ 在本地修改并提交；Frontera 只 `git pull --ff-only`、安装环境、启动计算。远程不 commit/push 日志，不需要 Codex。`out/`、`results/` 被忽略：**Git 同步的是代码/输入，不包含旧实验结果**。本次按 Frontera 上只有 Git 仓库处理，不要求传输结果压缩包。

研究目标与细节见 [WEIGHTED84_RUN.md](WEIGHTED84_RUN.md)；draft/PDF 是 `paper/main.tex` / `paper/main.pdf`。加权 PPO 仍是 development 候选，尚不能声称优于 fixed。

## 1. 实际运行什么

一个作业按顺序运行：

1. 若新位置没有 fixed 数据，重建 **84 节点模拟场景**下 4/16/64 三档的完整 train 和 validation：73×3=219、24×3=72，共291个任务。只从 train 生成输入缩放参考量；不跑 wait probes、等待预测器、旧 PPO 或 test。
2. 从头训练新的加权 PPO：alpha=0/0.2/0.5/0.8/1；5轮观察、400个完整任务产生共同 Tbar/Cbar，随后冻结；59轮更新，共64轮、5120个任务。最小化 `alpha*TAT/Tbar+(1-alpha)*C(rho=1)/Cbar`。
3. 验证第16/32/48/64轮，每轮24×5=120个任务；导出 Best-Fixed 比较、规模切换和时间—碳曲线。

实际 Frontera 只申请 **1个CPU节点、1个进程、单线程、4小时**，没有 GPU 请求；模拟场景的84节点和AMSP吞吐与这个计算资源申请是不同概念。4小时是初始、可恢复的申请上限，不是已测总运行时长。单线程设置沿用原实验，不擅自改变采样/并行方式。

站点设置写在 `scripts/frontera_weighted84.sh` 顶部：

| 设置 | 本次默认 |
|---|---|
| account | `deep-learning-at-sca`（用户提供的 association） |
| partition | `small` |
| nodes / tasks / cpus-per-task | `1 / 1 / 1` |
| time | `04:00:00` |
| Python | `$HOME/.conda/envs/carbon/bin/python` |
| 仓库 | `/scratch2/09796/shuyuanfan4814/carbon` |

没有强制指定 QoS：用户 association 是 qdefault，分区里的 qsmall 是分区策略，不直接当成用户 QoS 参数。`scontrol` 输出的 UNLIMITED 不代表实际无限时长；当前 [TACC 文档](https://docs.tacc.utexas.edu/hpc/frontera/running/)规定 small 用于1–2节点、最长48h，normal至少3节点，development最多2h，实时限制可用 `qlimits`。Frontera按整节点分配/收费，即使程序只使用一个核。

## 2. 本地提交这次修改

```bash
cd /Users/shuyuanfan/carbon-latest
git add AGENT.md GOAL.md README.md docs/FRONTERA_RUN.md docs/CLUSTER_RUNBOOK.md docs/RESEARCH_HANDOFF.md docs/WEIGHTED84_RUN.md scripts/frontera_env.sh scripts/frontera_weighted84.sh scripts/frontera_weighted84.py tests/test_frontera_entry.py
git commit -m "Add Frontera Slurm setup and resumable weighted PPO entry"
git push
```

显式列出文件，避免把本地无关文件带进提交。远程之后不再产生研究代码提交，从而避免两端各自提交引起分叉。

## 3. 从登录节点提交环境安装作业

```bash
cd /scratch2/09796/shuyuanfan4814/carbon &&
git pull --ff-only &&
mkdir -p out &&
sbatch scripts/frontera_env.sh
```

安装脚本申请 `deep-learning-at-sca / small / 1节点 / 1任务 / 1 CPU / 1小时`，由 Slurm 在计算节点执行，不受登录 SSH 连接中断影响。计算节点联网情况由此次作业实际验证，Conda/pip 的失败信息保留在日志中。

脚本用现有 Conda，在 `$HOME/.conda/envs/carbon` 创建 Python3.11。安装项目所需 scikit-learn/SciPy、matplotlib，以及 [PyTorch 官方 CPU 分发](https://docs.pytorch.org/get-started/previous-versions/)，不拉 CUDA toolkit。不改 `/work2/.../miniconda3/envs/carbon` 旧环境。使用二进制包。这个安装作业只创建/检查环境，不执行模拟或 PPO；已有计算节点交互 shell 也可 `bash scripts/frontera_env.sh`。

首次成功后，确切依赖写到环境中的 `carbon-pip-freeze.txt`、`carbon-conda-explicit.txt`。再次运行安装脚本只检查已冻结环境，不自动升级。中断恢复必须保持同一个环境；不同 PyTorch 版本会被实验检查拒绝。本次是新训练，不加载 Sophia 的 PPO 权重。

环境路径可在**安装和运行两边**统一通过 `export CARBON_ENV=...` 覆盖；Conda可用 `CONDA_EXE` 覆盖。默认不用 activate；运行器始终调用完整 Python 路径，并隔离旧模块/base 的 Python 包路径。安装报错时保留 `out/frontera-env.*.log`，不要接着提交作业。

## 3.1 查看安装结果和重试

提交返回 job ID 后，使用 `squeue -u "$USER"` 看状态，日志为 `out/frontera-env-JOBID.out`（替换 JOBID）。脚本另保存每次独立日志 `out/frontera-env.JOBID.UTC.UNIQUE.log`。

```bash
tail -n 60 out/frontera-env-JOBID.out
```

看到 `Ready. Python: ...` 与 `Frozen packages: ...`，且安装作业成功结束后，再执行下一节的检查和实验提交。提交成功仅代表排队，不代表安装完成。

此前 `Executing transaction` 被断开后，可能已有 `conda-meta/` 却没有 `bin/python`。安装器会检查 Python、标准库和 pip；对没有冻结记录的残缺环境，将整个目录保存在 `$HOME/.conda/envs/carbon.incomplete.XXXXXX/prefix` 后重新创建。健康环境继续使用；已有冻结记录的损坏环境拒绝自动替换。此恢复逻辑同样适用于安装作业超时。

安装失败后先查看该作业日志；需要重试时仍提交 `sbatch scripts/frontera_env.sh`，不用 `--resume`。网络问题按实际日志处理。Frontera 提供 flock 时，脚本阻止同一环境的并发安装。

这次此前中断发生在环境安装阶段，尚未启动实验，首次实验 sbatch 也不加 `--resume`。

## 4. 登录节点检查，然后提交

```bash
bash scripts/frontera_weighted84.sh --check &&
mkdir -p out &&
sbatch scripts/frontera_weighted84.sh
```

`--check` 只导入依赖、读取配置/原始数据、校验已有 fixed 文件及哈希；不会推进队列或创建实验结果。首次显示 `Fixed inputs will be built/resumed on compute` 是正常情况。

`out/` 要在 sbatch 前存在，因为 Slurm 需要先打开输出文件。安装/检查也会创建它。脚本不包含第二层 sbatch，不自行再提交任务。

在已有 Slurm **计算节点**交互 shell 中，可运行同一个脚本：

```bash
cd /scratch2/09796/shuyuanfan4814/carbon
bash scripts/frontera_weighted84.sh
```

普通登录 shell 不执行真实实验；仅有 salloc 环境变量但仍位于 login 节点也会拒绝。命令行可覆盖资源，例如 `sbatch --time=08:00:00 scripts/frontera_weighted84.sh --resume`；使用 development 时须把 time 改到2小时以内。

## 5. 中断、分阶段与状态

先确认前一个作业已结束，再恢复：

```bash
squeue -u "$USER"
sbatch scripts/frontera_weighted84.sh --resume
```

交互恢复：`bash scripts/frontera_weighted84.sh --resume`。

- fixed 按 train/validation 阶段复用；中断的阶段保留在 `.interrupted-*` 目录，然后重跑该阶段；完成的阶段校验后跳过。
- PPO 每个完整迭代保存模型、优化器、RNG和观察累计量。半个迭代中断会重跑该轮；已提交的迭代不重复。观察均值不会重复累计。
- validation 按完整 checkpoint 恢复；不因验证中断重新训练。
- 若中断时还没开始 PPO，`--resume` 完成 fixed 后正常开始新的 PPO。
- 不同时提交两个写同一输出的任务。已有结果内容/代码/配置不匹配时拒绝继续，不覆盖旧实验。

可选分阶段（默认 all 无需拆开）：

```bash
sbatch scripts/frontera_weighted84.sh --stage fixed
sbatch scripts/frontera_weighted84.sh --stage train
sbatch scripts/frontera_weighted84.sh --stage validate --resume
```

按顺序、前一阶段完成后再提交下一阶段。已有训练输出时 train 同样要加 `--resume`。

日志：`out/frontera-weighted84-JOBID.out` 是 Slurm 输出；`out/frontera-weighted84.JOBID.UTC.UNIQUE.log` 同时保存脚本完整 stdout/stderr，每次运行独立。可用 `tail -n 60 out/frontera-weighted84-JOBID.out` 看进度（替换 JOBID）。

默认结果：

- 最小 fixed：`results/amsp-frontera-7b-c84-fixed-frontera/`。
- PPO：`results/amsp-frontera-7b-c84-weighted-seed11/`。
- 进度：PPO目录的 `manifest.json`、`iteration-*/`。
- 最终报告：`weighted84-summary.md/json`、`weighted-curves.csv/png/pdf`。

可用 `CARBON_SOURCE`、`CARBON_OUTPUT` 覆盖两个结果目录。若后来找回旧 fixed 数据，指定包含 references.json、fixed-train/、fixed-validation/ 的目录即可验证复用；不要在同一次 PPO 中途换 source 路径。跨机器搬迁旧 weighted checkpoint 也涉及绝对 source 路径绑定，本次不进行该操作。

结果不提交到 Git。需要在本地分析时可用 rsync/scp 直接同步指定 out/ 日志和结果文件，无需打包；地址/连接方式确定后再给出具体同步命令。
