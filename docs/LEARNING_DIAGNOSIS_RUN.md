# 已完成：冻结学习信号诊断

2026-09-21 更新：四轮共 256 episodes/332 chunks 全部重现。最宽松预算在第 1/4 轮与共享 actor 梯度发生局部冲突，见[结果记录](../paper/reproducibility/development_findings.md)。下一步为[独立预算 actor 对照](BUDGET_ACTOR_RUN.md)。以下保留原诊断规定；新增 actor 分支后当前核心源码哈希已不同，不在当前版本重跑本旧诊断。对应源码已校验归档到 `results/source-archives/product-shared-5e6c7ddcbb91/`。

product actor 已完成预定训练，但最终 96 次 validation 全为 `[64]`；第 10–64 轮，三个宽松预算的 2,640 次初始决策没有采样过 4/16 节点。完整负结果见 [development_findings.md](../paper/reproducibility/development_findings.md)。仅有这些频率不能区分共享梯度冲突、critic 估计、早期样本方差与探索退化的作用。

## 固定范围

- 原 Frontera–7B seed11 product actor，预先指定训练第 **1/4/8/16** 轮，合计 **256 个已记录 episode、332 个 chunk**。
- 第 1 轮重建原 seed/runtime 的初始化；其余分别读取第 3/7/15 轮 checkpoint，以取得采样时的更新前网络。
- 强制执行原日志中的动作，不重新采样动作。逐 chunk 检查公开输入哈希、概率、critic 输出及真实回放结果，逐 episode 检查回报。
- 沿用当轮冻结的 dual；按完整 episode 平均所有决策的 actor 梯度，分别计算碳、违约与熵三部分。
- 报告各预算与共享梯度的 cosine、各部分范数，以及沿各预算/共享梯度下降方向时，初始动作概率的局部变化方向。
- 没有 optimizer step，没有新模型，没有 validation 选优，不读取 test outcomes。不改 `src/carbon`，保留所有已完成运行的源码绑定。

cosine 小于零表示局部梯度冲突。这里研究的是更新开始处的完整 rollout 梯度，未重现实际 Adam 预条件、梯度裁剪与后续 shuffled minibatch/PPO epoch；正 cosine 也不能排除后续干扰，负 cosine 不能单独证明退化原因。概率变化方向是局部导数，不是已实现的新策略效果。

## 启动与恢复

登录节点提交：

```bash
qsub scripts/sophia_diagnose_learning.sh
```

已有 PBS 1 GPU 交互节点：

```bash
bash scripts/sophia_diagnose_learning.sh
```

恢复方式：

```bash
qsub -v CARBON_RESUME=1 scripts/sophia_diagnose_learning.sh
```

或在交互节点运行 `bash scripts/sophia_diagnose_learning.sh --resume`。

脚本使用已有 Python 环境、CPU 单线程；PBS 申请 Local-LLM/by-gpu、1 GPU/32 CPU/120 GB、1 小时。1 小时是申请上限，不是实测耗时。脚本检查 `PBS_JOBID`，不会在普通登录 shell 启动回放。

日志自动保存到 `out/diagnose-learning.<job>.<UTC>.<随机后缀>.log`，同时保留终端输出。agent 直接读取，无需粘贴。

结果目录 `results/amsp-learning-diagnose-frontera-7b-seed11/` 内含：

| 文件 | 含义 |
|---|---|
| `run-plan.json` | 原训练日志、所需 checkpoint、输入、runtime 与脚本哈希 |
| `round-000001/000004/000008/000016/` | 各轮检查计数、梯度报告、stage seal |
| `learning-summary.md/json` | 全部已完成轮次汇总及适用范围 |

恢复时校验并复用已完成轮次，保留中断轮次后重跑该轮。原 trial 当前只有一个完整 fresh attempt；此诊断显式要求该结构，不拼接不明的多 attempt 训练轨迹。

## 后续决策

如实归档与任一假设不符的结果。先利用诊断区分值得检验的优化机制，再决定是否需要一个有明确依据的最小训练改动。不能因为探索/梯度指标变化就声称碳或反馈收益，不能通过删除最紧预算改变原比较问题。正式 E2–E4、匹配 Precommitted-RL 和多 seed 协议仍未完成。

## 启动前验证

5 项小型合成检查通过：两轮真实合成训练日志可重现，改动输入/概率/物理回报会被拒绝，分解后的冻结梯度与 PPO actor loss 一致，源模型和训练产物不变，已完成 stage 可恢复复用，日志及退出码保留。另用保存的 manifest 和小型 cohort 校验本次真实产物的启动契约，确认 12 个绑定文件、256 episodes 与 332 chunks；这一步没有加载真实 trace 或运行真实模型。Python/Bash 语法和 paper manifest 检查通过，正式结果槽仍为空。

真实回放由用户启动，遵守 [AGENT.md](../AGENT.md)。
