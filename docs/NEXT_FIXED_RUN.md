# 下一步：Frontera 7B 固定规模回放

## 目前到哪里了

作者回传 `E1 queue stage complete: results/amsp-e1-frontera`，Frontera 完整 E1 已完成。末尾 368 个快照 × 4 规模 × 3 请求长度 = 4,416 条 validation 探针，没有删失；快照平均运行节点比例为 0.5307。这是模拟场景诊断。等待模型的误差、artifact 哈希及日历相关性还要读结果文件；不凭这些摘要判断预测准确。IW 当前仅确认 preflight 完成，不阻塞本轮 Frontera 主线。

## 这一批做什么，为什么用集群

先用 Frontera × AMSP 7B，固定使用 4/16/64/128 节点，完成相同的 22,993 updates；引入原队列回放、Texas 碳序列和既定开销。要看的是完整任务的周转时间、相对 modeled carbon、node-hours 和 chunk 数。探针的单次等待不能替代这些结果。

完整 train cohort 有 73 个到达时间 × 4 种规模 = 292 次任务回放。先运行确定性的 hash 分片 0/16，恰好包含 3 个到达时间，共 12 次回放：

- 2020-01-29 04:00 UTC：frontera-train-0008
- 2020-02-13 16:00 UTC：frontera-train-0013
- 2020-06-06 17:00 UTC：frontera-train-0051

选择来自 episode ID 的既有哈希规则，未按等待或策略收益挑选。这是同一实验的计时分片，不新增实验条件；它不代表全部 73 个到达时间，不能据此生成正式训练参考量或宣称策略增益。保留分片数据，下一步安排完整覆盖及核对。

需要的是 Sophia CPU 计算资源。模型速度采用 AMSP 公开数据，这里不启动 LLaMA/Qwen、不申请多卡训练、不测整节点功耗。启动 Python 沿用 `/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python`，可用 CARBON_PYTHON 覆盖。无新增依赖。

## 同步和运行

代码本机只增加独立的 `scripts/sophia_fixed.sh` 与说明，没有自动 commit/push 或提交集群作业。先在本机代码仓库手动同步：

```bash
cd /Users/shuyuanfan/carbon-latest
git add scripts/sophia_fixed.sh docs/NEXT_FIXED_RUN.md docs/E1_RUN.md docs/CLUSTER_RUNBOOK.md
git commit -m "Add Sophia fixed-scale timing run after E1 completion"
git push
```

Sophia 仓库内更新：

```bash
git pull --ff-only
```

进入已有 PBS interactive allocation 后，在仓库根目录执行：

```bash
bash scripts/sophia_fixed.sh \
  configs/amsp-frontera-7b.development.json \
  results/amsp-fixed-frontera-7b/train-shard-00 \
  --split train --shard-count 16 --shard-index 0
```

脚本是一层，不调用 qsub/sbatch 或另一份 shell。Sophia batch 使用 PBS；若以后 batch 提交，需要先填本站已经确认的 #PBS 指令，不使用 sbatch。先不同时启动六个 panel 或全量 PPO。

## 看哪些输出，之后做什么

完成后终端自动打印程序耗时，以及四种规模的平均 TAT、node-hours、两个功率端点的 modeled carbon 和 chunk 数；保存到 `fixed-summary.json`。不完整结果明确计数，含不完整结果的一组不计算完整成本均值。

把终端最后的 `Fixed replay complete` 和四行 `Fixed-*` 摘要带回即可；保留整个结果目录，尤其 `manifest.json`、`episodes.jsonl`、`chunks.jsonl`、`fixed-summary.json`。E1 的 `validation-diagnostics/metrics.json`、`dependence-queue/audit.json` 也保留，后面复核时使用。

据本轮耗时安排完整 train 固定基准 → 由完整 train 生成 T_ref 和成本参考量 → 主 PPO 短轮计时 → 固定预算的正式训练/validation。T_ref 是训练集上各固定规模平均 TAT 的最小值，用于把 slider 转成具体小时数。主 PPO 和 Precommitted-RL 不加载 E1 等待模型；MPC 对照后续使用已完成的 E1 artifact。

当前 run-fixed 没有 --resume。若 allocation 中断，保留目录与日志，不给它套 probe-waits 的 --resume，也不把未完成目录当完成结果；这正是本轮只发 12 次回放的原因。一小时是否足够尚未测得，不作保证。
