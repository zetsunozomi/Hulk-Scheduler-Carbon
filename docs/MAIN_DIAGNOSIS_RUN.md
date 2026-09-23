# 已完成：冻结主策略诊断与完整规划对照

2026-09-21 更新：job `187563` 已完成本轮，见[结果解释](../paper/reproducibility/development_findings.md)。下面保留原运行说明。当前源码已新增可选 actor 交互项，与旧 checkpoint 绑定不符；不要在当前版本重跑此旧诊断，下一步使用 [actor 交互项试验](ACTOR_INTERACTION_RUN.md)。

依据 [当前结果核验](PROJECT_STATUS_2026-09-21.md)，先在 Frontera–7B seed11 development panel 分清策略退化与环境机会。此轮不训练、不修改预算/chunk 定义、不读取 test outcomes。

遵守根目录 [AGENT.md](../AGENT.md)：登录节点不运行重负载；实机实验由用户启动。下面是同一脚本的 batch 和交互用法，脚本内部不再提交作业。

## 启动

在仓库根目录，登录节点提交：

```bash
qsub scripts/sophia_diagnose_main.sh
```

已经处于 PBS 的 1 GPU 交互 allocation 时：

```bash
bash scripts/sophia_diagnose_main.sh
```

脚本检查 `PBS_JOBID`，避免在普通登录 shell 直接执行回放。PBS 资源为 Local-LLM / by-gpu、1 GPU / 32 CPU / 120 GB、4 小时，沿用已成功运行的资源形状；模拟和推理实际使用 CPU 单线程。4 小时是可恢复运行的申请上限，不是已测量的本轮耗时估计。

默认 Python：`/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python`。复用已有环境，无自动依赖安装。默认输入为原 Frontera–7B development config、完整 pilot、seed11 main 和 Frontera E1 目录。

## 本轮执行内容

1. 校验原 fixed/reference/grid、四个 PPO checkpoint、旧 validation 文件、E1 model/probes 与当前核心源码绑定。
2. 对 24 个 validation 初始状态 × 4 个原预算 × 4 个 checkpoint，共 384 个输入/概率组合进行复核。重新编码的输入哈希和网络概率必须匹配旧评估日志。报告同一到达跨预算动作分布的总变差距离（TV）及 entropy，不将其当作反馈收益证据。
3. 执行 Plan-once 与 Rollout-MPC：24 个到达 × 4 档预算 × 2 方法 = 192 个完整 episode。全部使用原未取整预算、原工作量/600 秒开销、256 planning paths、internal miss threshold=.05、seed11；二者使用相同到达种子规则。不按结果删预算或自动调整 planner 阈值。
4. 并列复用 Fixed-4/16/64/128、全部四轮 PPO 与原 validation Fixed-Mix LP。报告碳成本、miss、mean/p95 TAT、node-hours、实际规模序列、换规模、planner fallback 和耗时。未完成任务保留分母，整体碳/TAT 留空；单预算的 planner cohort 未齐时不输出部分均值。
5. 把 fixed train/validation、PPO-64 validation 和新 planner 的完整任务跨度接入已有 E1 dependence 分析。输出相关性及 span 下界，不自动选择 block 长度或宣称独立性。

这轮仍是固定 `.05` internal threshold 的 development 对比，尚未进行正式 validation 风险阈值选择。只赢该 planner 不足以证明 RL 或反馈优势；E3 的匹配 Precommitted-RL 仍需后续训练。

## 结果与恢复

默认新目录：`results/amsp-diagnose-frontera-7b-seed11/`。

当前 launcher 在 batch/交互运行时也将 stdout/stderr 保存到 `out/diagnose-main.<PBS_JOBID>.<UTC时间>.<随机后缀>.log`，同时保留终端输出；每次恢复新建日志。agent 直接读取这些文件，无需用户手动粘贴。此日志支持不改变上文所述的旧 checkpoint 源码绑定限制。

| 文件 | 用途 |
|---|---|
| `run-plan.json` | 输入、代码、预算、方法与种子的固定绑定 |
| `diagnosis-summary.md/json` | 全部 fixed/PPO、完整 planner cohort 和 Fixed-Mix 的汇总 |
| `actor/responses.jsonl` | 同一初始状态下的四档预算动作概率与输入哈希 |
| `planners/budget-XX/arrival-XXXX/` | 一个到达、一个预算、两种 planner 的 manifest/episode/chunk 日志 |
| `dependence/audit.json` | 含完整任务跨度的相关性诊断 |

共有 96 个 planner 到达×预算单元，每个单元内执行两种方法。完整单元经校验和封存后直接复用；中断单元改名保留，再重跑该单元。不会覆盖原有 pilot/main/E1 文件。

确认旧作业结束后，在登录节点恢复：

```bash
qsub -v CARBON_RESUME=1 scripts/sophia_diagnose_main.sh
```

在交互节点恢复：

```bash
bash scripts/sophia_diagnose_main.sh --resume
```

同时写同一输出目录会被锁拒绝。恢复时输入/核心源码/本编排脚本/运行时/设置必须与 `run-plan.json` 一致。需要修改实验时保留旧目录并用新的 `CARBON_OUTPUT`；batch 用 `qsub -v CARBON_OUTPUT=...` 传入。改变代码或配置可能同时使原 checkpoint 失配，新输出目录不能绕过该检查。

## 读结果后的决定

- 若输入和概率重现失败，先修复可复现性问题，不能绕过匹配检查继续比较。
- 若固定动作能随预算明显改善、PPO 却近乎不变，优先诊断共享 actor 的预算交互与训练探索。
- 若规划产生多 chunk 序列并有更好的成本/miss，环境有可利用机会，主策略需学习改进。
- 若规划也无明显增益，结合 predictor calibration 与序列结果判断；这不构成不存在动态最优策略的证明。
- 本轮结束后再决定最小方法修正和正式 E1–E4 扩展，不直接重跑 64 轮或追加 panel/seeds。

## 准备阶段验证

2026-09-21：10 项新增轻量合成/编排检查和 6 项既有规划检查通过；覆盖断点保留、完整单元复用、哈希篡改拒绝、预算配对、删失分母、动作概率重现及 PBS 启动入口。Shell 语法、默认输入路径和文档链接已检查。核心 `src/carbon/*.py` 与原 `scripts/main_train.py` 哈希未改。

这些检查没有加载真实 AMSP 场景执行回放或训练。完整真实数据运行和实际耗时仍待用户在计算节点启动后核验。
