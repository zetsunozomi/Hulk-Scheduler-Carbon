# 当前方案与集群运行

## 当前执行进度

作者已回传 Frontera 完整 E1 的成功结束日志：`results/amsp-e1-frontera`。validation 共 368 个快照、4,416 条探针，均无删失；快照平均运行节点比例 0.5307。4/16/64/128 节点平均等待约 0.69–0.71/1.31/6.64/38.25 小时；它们是模拟器探针结果，不是历史等待预测准确率，也不是动态策略收益。后续还要核对 manifest、模型误差与日历相关性文件。

## 当前优先级：完整固定基准与主 PPO pilot

Frontera7B计时分片已完成：3个到达时间×4种规模，共12次完整结果、5.7s，chunk数5/2/1/1。此分片Fixed-64在时间和两功率端点成本上优于Fixed-128；Fixed-4成本最低。不能推广为完整日期结果。

下一条 [sophia_main_pilot.sh](MAIN_PILOT_RUN.md) 自动运行完整train固定基准（292次）、train参考量/四档预算、完整validation固定基准（96次），再跑5轮80episode、seed11的主PPO。不是正式RL训练预算，不访问test。主策略不需要等待模型。

预算范围改为从最快固定train平均TAT到Fixed-4 train第95百分位，四档s=[0,.25,.5,1]；原通用CLI的1–2倍T_ref在当前48h非终端chunk下可能过窄。来源仅train、右端不是可行性保证，各方法使用同一网格。输入/工作/开销/请求上限均不变，Frontera E1不重跑。

IW完整E1尚未回传，后续单独处理。Sophia在PBS interactive内用bash；脚本不提交另一个作业。完整stage可跳过、PPO可按轮checkpoint恢复，细节见运行说明。

## 主策略不依赖等待预测

主 actor/critic 不加载等待模型：直接读取可见队列历史、实际剩余工作/预算、节点数/速率/时长描述和未来 168h 的 28 个六小时 CI 均值。E1 的模型仅供正式规划基线 MPC、Plan-once 使用；误差再大也不构成主策略的准确率门槛。RL 仍可能学到只在训练环境成立的排队规律，已有 width/FCFS 冻结回放负责检查这个限制。不能保证 Slurm 持续不给资源时仍按期完成。

slider 控制 D 小时的完成预算，左端更紧、右端允许更多时间降低碳成本。主策略只有一阶段端到端 PPO。E3 唯一新增训练组为 Precommitted-RL：在首次提交前就确定全部规模，不读后续反馈，与主策略匹配训练网格、到达采样、seeds 和总 episode 预算；机制评估限每 panel 一个预设预算。它替换旧的等待辅助/current-CI 消融。

## 论文现在讲什么

用户在每个训练 chunk 结束后选择下次请求的节点数，在完成时间预算下减少**模型估算的相对碳排**。RL 与全部固定规模、固定混合、Plan-once、Rollout-MPC 比较。最重要的证据是它能否超过强规划基线；目前尚无正式结果。

| 项目 | 本轮固定的输入或假设 |
|---|---|
| 训练速度 | AMSP 2024 修订版图 12，LLaMA 7B/13B/30B，Our Work 系列 |
| 规模 | 4/16/64/128 节点，每节点 8 张 A800，即 32/128/512/1024 卡 |
| 模拟集群 | 声明的 128 节点场景；与提交 CPU 作业的机器无关 |
| 后台需求 | Frontera 与 iw 历史流分别使用，保留提交顺序、节点数、申请时长和占用时长；不混洗 |
| 队列状态 | 从固定前缀起点空状态重新回放；至少 28 天后才计分，不使用旧开跑时间初始化 |
| 碳强度 | 已整理的 Texas ERCOT EIA；外加地区情景，不声称 AMSP 平台位于 Texas |
| 工作量/开销 | 每模型 U=ceil(192*q4) updates；每 chunk 开销 600 秒，均为预设情景 |
| 功率 | 1 kW 只是归一化单位；绝对节点功率未知，报告相对碳排和 node-hours |
| 数据划分 | 原数据曾做 train/validation；新三段划分是回顾性 benchmark，不宣称全新未看过的 test |

**旧日志不能预测任意新机器的真实排队。** 它们在这里定义可复现的资源需求场景。等待标签由同一模拟器生成；原机器的等待时间不是真值。后台宽度、FCFS 和开销敏感性检查用来界定结论适用范围。

无需再测 Qwen、节点功率或 checkpoint。实际跨规模恢复系统不在本轮实现范围。以前执行的 synthetic 检查只验证程序，不是论文实验。

## 已准备好什么

- 公开曲线数字化记录：`data/amsp/profiles.json`；源 PDF 校验脚本：`scripts/extract_amsp.py`。
- 六个可读真实输入的配置：`configs/amsp-{frontera,iw}-{7b,13b,30b}.development.json`。输入哈希、时间解释、划分和配对 arrivals 已写入。
- 等待预测、规划、PPO、validation 选择、冻结 test、功率后处理与图表导出均有程序检查。
- `run-stress` 独立入口可固定 full/MPC 的模型和参考量，重新回放四个环境变体。一般 artifact 匹配检查仍严格保留。
- 未完成：真实规模回放的耗时/占用率/等待尾部检查、正式训练和 E1–E4 结果。环境敏感性的跨来源汇总图尚待真实输出接入。

当前配置保持 development，因为本次先检查场景和计算成本。协议冻结后才复制为正式 research 配置；`evaluation_design=retrospective_temporal` 和 `test_is_untouched=false` 仍保留。用 train 拟合、validation 选模；本轮 test 结果出来后不再据此调参。

## 1. 手动同步仓库

在本机代码仓库检查 `git status --short` 和 `git diff`，然后可以按你原来的方式手动三连：

```bash
git add -A
git commit -m "Use published AMSP profiles and portable scenario runners"
git push
```

有其他未完成修改时，请改为逐文件 add。`results/` 与 `slurm-*.out` 已忽略；原 trace 文件需在目标集群可读，若没有随 Git 分发，请按配置中的相对路径复制，保留原文件 hash。draft 在独立的 `carbon-rewrite/newest_writing` 目录，不属于这个代码仓库。

在目标集群代码仓库执行 `git pull --ff-only`。

## 2. 移植时改任务脚本顶部

直接编辑 `scripts/cluster_preflight.sh`，以后运行哪个阶段就编辑对应的 `cluster_*.sh`：

- **`#SBATCH` 块**：默认 1 个节点、1 个 task、1 CPU 核、16 GB、2 小时。按站点要求取消 account / partition / qos / constraint 示例前多余的 `#` 并填值。这是初始请求，尚非性能估计。
- **`export CARBON_PYTHON=...`**：改成新集群的 Python 路径；保留原环境作为默认示例。已 export 的 CARBON_PYTHON 优先，可用于交互环境覆盖。
- 如需 `module load`，放在紧接着的注释位置。

每个脚本都包含资源设置和执行代码。`bash` 忽略 `#SBATCH` 注释并直接运行，`sbatch` 读取资源设置后在分配节点运行相同代码。脚本内部不提交作业，不再使用外层 submit_cluster.sh 或 cluster.local.env。

Python 需 3.10+。复用现有环境；有缺失依赖才安装到所选 Python：等待模型/LP 用 `requirements-p2.txt`，PPO 用 `requirements-p3.txt`（PyTorch 2.6+，CPU 即可）。例如先在 shell 设置同一 `CARBON_PYTHON` 路径，再执行 `"$CARBON_PYTHON" -m pip install -r requirements-p2.txt`。第一次 preflight 只需标准库。

## 3. 第一次 CPU 检查：两种方式任选其一

需要计算节点是因为长历史回放和后续训练耗时。以下先运行 Frontera 的两天训练期 probes，检查输入、队列尾部、内存和耗时；不会训练 RL 或填写论文结果。**从代码仓库根目录启动。**

已申请 interactive compute 节点时，直接运行：

```bash
bash scripts/cluster_preflight.sh \
  configs/amsp-frontera-7b.development.json results/amsp-preflight-frontera
```

在 login 节点提交 batch 作业时：

```bash
sbatch scripts/cluster_preflight.sh \
  configs/amsp-frontera-7b.development.json results/amsp-preflight-frontera
```

两条执行的是同一个任务，选择一种即可。输出目录不能重复；重试使用新后缀。batch 日志直接写在提交目录的 `slurm-carbon-preflight-<jobid>.out`，无需先创建日志目录；interactive 日志默认显示在终端。

带回整个 `results/amsp-preflight-frontera/` 和 batch 日志（interactive 则保存终端输出）。失败也保留已产生的文件。成功后根据 probe 完成/删失情况和实际耗时准备后续批量命令，不要求新增 GPU 实测。

## 后续入口（第一次检查后再固定计算预算）

Sophia 在现有 PBS interactive allocation 内使用 `bash scripts/对应脚本.sh ...`。`sbatch` 只适用于 Slurm 站点，不适用于这里的 Sophia。

| 阶段 | 脚本与参数 |
|---|---|
| 等待模型 + 预测诊断 | `cluster_e1.sh CONFIG OUTPUT [PROBE_INTERVAL_SECONDS]` |
| train fixed + references | `cluster_run.sh CONFIG OUTPUT --split train`，随后 `carbon make-references` |
| PPO | `cluster_ppo.sh CONFIG OUTPUT - REFERENCES ITERATIONS [options]` |
| 冻结 test + 报告 | `cluster_test.sh CONFIG OUTPUT WAIT_MODEL SELECTION CHECKPOINT_DIR...`，详见脚本 usage |
| 环境敏感性 | `cluster_stress.sh CONFIG OUTPUT WAIT_MODEL CHECKPOINT VARIANT BETA [--miss-tolerance ...]` |

环境敏感性只用 7B、两来源、一个提前选定的预算和所有声明种子；变体为 `width2`、`fcfs`、`overhead60`、`overhead1800`。主场景的 checkpoint、预算、MPC 阈值先在 validation 冻结并归档，再运行变体。每次输出同时保留 full 与 MPC；不选最好变体，不重训或改小时预算。它单独输出 manifest/chunks/episodes，不冒充 E2 的普通 test 包。

主 PPO 命令第三个参数 `-` 表示不需要等待模型；E3 对照同样用 `-`，仅加 `--decision-mode precommitted`。底层 `train-ppo`/`run-policy` 的 `--predictor` 可省略。旧命令若仍传入路径，默认 none 模式不会读取它。v2 checkpoint 不能与旧 v1 混用；完整 test/stress 仍需 E1 模型来运行规划基线。Frontera E1 现已完成；正式 PPO 的 iterations 仍需按固定回放及短轮训练耗时确定。

`run-policy --slider-position S` 选择 checkpoint 已支持档位；通用CLI默认 beta=1/1.25/1.5/2；当前场景必须覆盖它，显式使用budget-grid.json里的train-derived beta，对应S=0/.25/.5/1。不能与 --budgets 同时使用，未验证位置不插值。所有实际输出点照常报告。预先序列保存 plans.jsonl 并纳入 test 文件校验。正式 AMSP 规模仍是 4/16/64/128；4→8→4 是机制举例，不新增 8 节点曲线输入。
