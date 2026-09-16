# P0＋P1 集群交接

这一阶段交付配置与数据校验、统一回放、固定工作量、分阶段成本日志，以及
Fixed-4/8/16/32 的配对运行入口。真实 trace 回放与实验在集群进行。
先保留 development 身份；完成来源和 untouched holdout 审核后再使用 research。

## 1. 本地：你手动执行 Git 三件套

在新代码仓库中检查并提交本次改动：

```bash
cd /Users/shuyuanfan/carbon-latest
git diff --check
git status --short
git add .gitignore README.md pyproject.toml src/carbon tests configs examples scripts/cluster_run.sh docs
git commit -m "Implement P0/P1 replay and exposure accounting"
git push
```

本实现不会替你 commit、push 或向集群提交作业。旧 `src/sim`、`src/model` 和
`src/queue_prediction` 保留；后续新方法统一使用 `src/carbon`。

## 2. 集群：拉取并验证环境

进入你已有的集群仓库目录，再执行：

```bash
git pull --ff-only
export CARBON_PYTHON=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
"$CARBON_PYTHON" --version
"$CARBON_PYTHON" -B -m unittest discover -s tests -v
"$CARBON_PYTHON" -m carbon validate --config configs/synthetic.json
```

需要 Python 3.10+；P0/P1 没有第三方运行依赖，无需安装旧 RL 代码的 PyTorch/Ray。
若使用 named timezone，系统需提供对应 IANA timezone 数据。

可用合成数据检查完整启动脚本：

```bash
bash scripts/cluster_run.sh configs/synthetic.json results/synthetic-check --nodes 4 8 16 32
```

合成输出带 `purpose=synthetic`，只能用于功能验收。

## 3. 准备真实输入

```bash
cp configs/cluster.template.json configs/development.json
```

模板中的 null 表示必须补齐。以下三项允许保持 null：未知的
`power.workload_coefficient`、未指定的 `trace.dst_fold`、不额外限制运行时长的
`execution.max_episode_seconds`。其余未填项会由 validate 一次列出。

### A. Cluster 与 trace

- 填写真实 partition、节点容量、允许的 4/8/16/32 子集、最大 walltime 与粒度。
  核对来源后填写 `provenance`。不能直接沿用旧代码的 88 节点常数。
- 原 sacct 空格分隔 `.log` 可直接读取；标准化 CSV 需要字段
  `JobID,NNodes,Submit,Start,End,TimelimitR`，其中 TimelimitR 单位为分钟。
- 明确源时区。UTC offset 时间戳优先；本地时间在 DST 回拨处需要显式 fold。
- 模板显式选择丢弃零时长行、把超出请求时长的背景作业裁到请求上限。
  这是清洗假设，需要审核；可改为 `error` 逐项处理，所有计数写入 manifest。
- `coverage_start/end` 声明背景到达完整覆盖的半开时间段。
  `coverage_attestation` 说明覆盖依据。数据文件尾部没有到达不自动代表空队列。
  同一文件需包含初始化边界前仍可能运行/排队的作业。
- 审核旧训练/验证文件之间的缺口，不能把缺口自动当零到达。若当前文件无法提供
  所需连续覆盖，需要从原数据源补齐或缩小预先定义的区间。

确认源时区后进行只读审计，例如：

```bash
# 将 TRACE_FILE、TRACE_TIMEZONE 设置为已经核实的路径和时区。
"$CARBON_PYTHON" -m carbon audit-trace --trace "$TRACE_FILE" \
  --timezone "$TRACE_TIMEZONE" --zero-duration drop --overrun clip
```

### B. 工作负载 profile

每个规模提供 updates/hour、initialization/restart/checkpoint 秒数、microbatch 和
gradient accumulation。所有规模共享 global batch、总 optimizer updates、sequence
length、精度和软件栈，并填写来源。优先找回已有 profile 与训练工程，只补缺项。

`profile_status` 可为 development 的 assumed，或 measured/published；research
禁止 synthetic/assumed profile。`correctness_artifact` 指向短恢复/切规模检查的
报告或注明 development 尚未执行。模拟器自身测试不能替代真实训练恢复检查。

### C. CI

提供已确认来源的 Texas average operational CI；本地旧 CAISO 生命周期因子
数据不能直接改名替代。文件格式：

```csv
start_utc,end_utc,gco2e_per_kwh,available_at_utc
2024-01-01T00:00:00Z,2024-01-01T01:00:00Z,400,2024-01-01T02:00:00Z
```

上面仅示范格式，不是实际观测。每行覆盖 `[start,end)`；可用时间来自真实
发布记录或明确声明的固定延迟情景。未来真实值用于事后积分，P1 的公开历史接口
只返回当时已发布且已结束的观测。缺失区间会阻止运行。

queue/CI 年份不一致时，填写明确的 calendar alignment 和有符号秒偏移。
默认偏移 0 不会自动匹配年份或季节。

### D. Splits 与 cohort

- train、validation、test 是按 UTC 定义的互不重叠半开区间。
- 已看过的旧 test 属于 development；只有完成 access-history 审核的区间才能
  声明 untouched。`research` 还要求 `holdout_audit.test_is_untouched=true`。
- cohort CSV 固定所有方法共用的初始到达及预算：

```csv
episode_id,arrival_utc,split,budget_hours
example-001,2024-01-03T00:00:00Z,train,24
```

- 为每个到达留足预热及完成所需背景覆盖。越过 split 边界会保留为 censored，
  不能按哪个方法完成来筛样本。
- 本阶段直接接受预算小时数。P2 才根据训练期 fixed 结果建立 T_ref 和预算网格。

填写真实资产 hash：

```bash
"$CARBON_PYTHON" -m carbon hash "$TRACE_FILE" "$CI_FILE" "$COHORT_FILE"
"$CARBON_PYTHON" -m carbon validate --config configs/development.json
```

配置路径相对 `root` 解析，`root` 相对配置文件目录解析；默认 `root=..` 对应
仓库根目录。来源确认后再更新 hash，不能用更新 hash 掩盖输入变动。

## 4. 提交首批集群运行

先选一个 machine–workload panel，所有 fixed 共用同一 cohort。先在 development
检查完成率、排队时间、最后一个 chunk 和成本守恒，再扩到正式实验。

可在分配到的计算节点运行：

```bash
bash scripts/cluster_run.sh configs/development.json results/p0p1-dev-001 \
  --nodes 4 8 16 32 --split train
```

或者通过 Slurm（先填写本集群允许的资源选项）：

```bash
sbatch --account="$ACCOUNT" --partition="$PARTITION" \
  --time="$WALLTIME" --mem="$MEMORY" \
  scripts/cluster_run.sh configs/development.json results/p0p1-dev-001 \
  --nodes 4 8 16 32 --split train
```

该作业运行 CPU 模拟器，`--nodes 4 8 16 32` 是**模拟的训练规模**，不是向 Slurm
申请 32 个真实节点。脚本默认单 CPU；资源额度由你按集群规则设置。

需要分片时可选 job array。所有同 episode 的 fixed 方法留在一个 shard，hash
分片不重叠；小 cohort 不宜设置过多 shard，空 shard 明确报错。

```bash
export CARBON_SHARDS=4
sbatch --array=0-3 --account="$ACCOUNT" --partition="$PARTITION" \
  --time="$WALLTIME" --mem="$MEMORY" \
  scripts/cluster_run.sh configs/development.json results/p0p1-dev-array \
  --nodes 4 8 16 32 --split train
```

不要重复使用已有输出目录。失败后保留原日志，修复后选新的 run ID。

## 5. 运行结束后检查并带回

每个 run/shard 目录带回这三个文件：

1. `manifest.json`：status 必须为 complete，核对 cohort 数 × fixed 数等于
   completed_episode_methods；保存 code/input hashes 和运行配置。
2. `episodes.jsonl`：检查每条的 final_status、completed_updates、deadline_miss、
   censor_flag；censored 的总成本是 null。
3. `chunks.jsonl`：用于核对进度、运行阶段、排队反馈和不同功率假设下的重算。

若存在 `failure.json`，同时带回它和 Slurm stdout。不要用部分日志生成“完整运行”
的论文图表。下一阶段基于这批数据实现 P2 的输入模型与强基线，再接 P3 的 RL。
