# 项目现状核验与 E1–E4 推进顺序

初次核验日期：2026-09-21 UTC。以下为修改前的 cluster 文件与代码提交 `661c729` 的快照；初次核验未训练、未提交 PBS 作业、未评估 test，也未修改模拟器、算法或配置。

同日后续：规划诊断、actor 交互项试验及冻结梯度诊断均已完成，见[真实 development 发现](../paper/reproducibility/development_findings.md)。交互项未解决退化，发现早期局部预算梯度冲突，下一步为[独立预算 actor 对照](BUDGET_ACTOR_RUN.md)；下文的规划缺口、dependence 旧状态与推进顺序应结合该更新阅读。

## 结论

项目已具备从输入、模拟、训练到选择、test、统计和导出的软件链路，但核心研究证据尚未成立。Frontera–7B seed11 已完成 64 轮与四个预定 checkpoint 的 validation，交接时的“55/64 超时，恢复状态未知”已经过时。当前 PPO 在 validation 几乎退化为 Fixed-64，没有任务内换规模，也没有形成随预算变化的时间—碳曲线。

下一步应在同一 development panel 解决预算可达性、策略对预算不敏感和反馈机会不足这三个问题，并补强规划对照；随后才决定是否扩展六 panel、三 seed。完成论文实验允许得到负结果，不能把完成训练等同于证明主张。

## 1. 仓库结构与同步

- 最近一次 pull：2026-09-21 03:09:57 UTC，`9bae3ef → 661c729`，fast-forward。
- 该提交主要新增研究交接和 `paper/` 快照，共 36 个文件；没有改变 `src/carbon` 实现。
- 本次检查开始时 `main` 与本地 `origin/main` 一致，工作树干净；未重新 fetch 远端。
- 只读 `qstat -u "$USER"` 查询成功且无作业返回。旧 attempt manifest 的 `running` 不表示 PBS 作业仍在运行。

| 路径 | 当前角色 |
|---|---|
| `paper/` | 当前稿件、PDF、E1–E4 计划、方法规范与实验 manifest；正式结果仍占位 |
| `docs/` | 执行约定、统计约定、运行方式和交接；进度描述须与实际结果核对 |
| `src/carbon/` | 正式模拟研究实现：replay、核算、PPO、规划、选择、test、统计与导出 |
| `configs/amsp-*.development.json` | 两来源 × 三 profile 的六个 development 配置；尚无冻结 research 配置 |
| `data/` | 历史需求流、AMSP profile、Texas CI、配对 cohort |
| `scripts/` | 各阶段 runner；Sophia 用 PBS 入口，通用 cluster 脚本供对应环境使用 |
| `tests/` | 软件合成检查；本次以结果与代码核对为主，未重新运行测试套件 |
| `results/` | cluster 实验实物，Git 忽略；本次结论来自这里的 JSON/JSONL |
| `src/sim`、`src/model`、`src/queue_prediction` | 历史实现，不是当前论文执行链 |

论文主入口为 [main.tex](../paper/main.tex)，实验需求为 [7exp.tex](../paper/7exp.tex) 和 [plan_exp_list.md](../paper/plan_exp_list.md)。

## 2. 实际完成状态

| 工作 | 本次核验 |
|---|---|
| Frontera E1 train probes | complete；960 快照、11,520 条，11,505 条获得标签，15 条删失 |
| Frontera E1 wait model | complete；基于 11,505 条训练标签拟合，chronological folds 有记录 |
| Frontera E1 validation probes | complete；368 快照、4,416 条，0 删失；误差与覆盖率文件已存在 |
| Frontera E1 dependence | 队列画像存在；episode spans 尚未接入，未选定 block 长度 |
| IW | 仅 preflight：2023-02-01 至 02-03、64 条探针、0 删失；无完整 E1 目录 |
| Frontera–7B fixed | 完整 train 73×4=292 次、validation 24×4=96 次 |
| Frontera–7B 短 pilot | 已完成；正式分析优先使用后续完整 development run |
| Frontera–7B seed11 主 PPO | 64/64 轮；4,096 episode、4,270 chunk |
| 主 PPO validation | 第 16/32/48/64 轮各 96 次，共 384 次，全部完成且无删失 |
| Fixed-Mix | 四预算的 validation LP 已生成；不是独立 test 结果 |
| Plan-once / Rollout-MPC | 代码已有，当前结果目录未发现正式对比输出 |
| Precommitted-RL / 单端点训练 | 代码已有，当前结果目录未发现训练/评估输出 |
| 其他五 panel、seed23/37 | 当前结果目录未发现完成证据 |
| 冻结 selection、test、E4 stress、正式图表 | 当前结果目录未发现完成证据 |

主要证据：[validation-summary.md](../results/amsp-main-frontera-7b-seed11/validation-summary.md)、[run-plan.json](../results/amsp-main-frontera-7b-seed11/run-plan.json)、[恢复 attempt manifest](../results/amsp-main-frontera-7b-seed11/ppo-attempt-001/manifest.json)。

恢复链核验：attempt-000 的已提交轮为 1–55，含 3,520 episode / 3,691 chunk；另有第 56 轮中断前写入的 16 episode / 16 chunk，不计入已提交统计。attempt-001 从 checkpoint-55 恢复，提交 56–64 轮，新增 576 episode / 579 chunk。本次核对了全部 64 个 checkpoint JSON 和权重哈希、恢复父 checkpoint、run-plan 的核心源码/编排/配置哈希、复用的 fixed/reference/grid，以及四轮 validation 文件绑定。

## 3. 当前主结果：几乎为 Fixed-64

下表仅展示预定最后一轮 PPO-64，所有 checkpoint 仍保留在原 summary 中。C 是两功率端点的归一化平均 modeled carbon 中较大者，不是实测碳排。Fixed-Mix 一列是同一 validation 上拟合的 LP 期望值。

| D（小时） | PPO mean TAT（小时） | PPO C | PPO miss | Fixed-Mix C | PPO C 相对 mix |
|---:|---:|---:|---:|---:|---:|
| 15.11 | 22.125 | 1.167 | 14/24 = 58.3% | 不可行 | — |
| 59.79 | 22.125 | 1.167 | 0/24 | 1.122 | 高 4.0% |
| 104.48 | 22.125 | 1.167 | 0/24 | 1.091 | 高 7.0% |
| 193.84 | 23.633 | 1.175 | 0/24 | 1.089 | 高 7.9% |

不能据该 LP 比较声称独立 test 显著劣于 mix；它已经足以说明当前 development 策略没有提供需要扩规模验证的优势。宽松两档下 Fixed-16 本身也是 0/24 miss，C=1.092、mean TAT=52.338h，较 PPO 更符合允许更多时间以降低碳成本的目标。

四轮 validation 的真实序列：

| checkpoint | 96 次评估的序列计数 |
|---|---|
| 16 | 95 次 `[64]`，1 次 `[128]` |
| 32 | 96 次 `[64]` |
| 48 | 96 次 `[64]` |
| 64 | 95 次 `[64]`，1 次 `[16,16]` |

合计 382 次 `[64]`、1 次 `[128]`、1 次 `[16,16]`。383/384 为 single-chunk，384/384 没有规模切换。多一个 chunk 不能视为发生了动态 scale-down。

## 4. 三个需要先解决的问题

### 4.1 最紧预算在当前 validation cohort 上不可达

D=15.1076526944h 取自 train 最快 fixed 的**平均** TAT，不是满足 95% 按时完成的分位数。本案还能给出当前动作集合的有限样本下界：

- 首选 4 或 16 节点，第一个 chunk 的分配时间就约 48h，超过 D；执行中不能切换。
- 首选 64 或 128 节点会在一个 chunk 完成，其等待和总时间与对应 fixed 相同。
- 24 个 validation 到达中，Fixed-128 全部超时；Fixed-64 有 14 个超时。

因此，在这 24 个相同到达、现有立即提交规则与 chunk 定义下，即使事后选择最快首动作也至少 14/24 超时。该结论不依赖 PPO 优化好坏；它不是任意环境的总体不可行性证明。保留该预算为明确的不可达点，或在 development 修改预算定义并给全部方法重建一致协议，均须如实记录，不能隐去旧点。

### 4.2 共享 actor 几乎没有利用预算

训练第 49–64 轮，每档 256 个 rollout 的初始 P(64) 分别为 99.6588%、99.6610%、99.6618%、99.6614%。四档实际输入有不同 D，`policy_inputs.py` 明确传入总预算与剩余预算；观察到的是输出接近不变，尚不能定性为输入遗漏或确定的代码 bug。

初始动作熵由第 1–4 轮均值 1.349 nats 降到第 17–32 轮 0.027，第 49–64 轮为 0.025。critic loss 同时下降不能证明控制目标改善。

完整 train fixed 也排除了“64 对所有预算确实最优”这一解释：归一化 worst C 分别为 Fixed-4=1.000、Fixed-16=1.039、Fixed-64=1.143。Fixed-16 在 59.79h 下仅 1/73 miss，宽松两档 0 miss；Fixed-4 在最宽档仅 3/73 miss，均可按对应预算满足训练经验 5% 门槛。

最终各 budget 的 lambda 约为 1.521/0.912/0.853/0.840。它们没有数值爆炸；现有证据只支持优先排查探索过早集中、共享网络的 budget/action 交互和约束优化尺度，不足以认定其中某项是唯一原因。无需重新引入等待预测器作为主策略的前置条件。

### 4.3 当前动作几何削弱了反馈机会

固定规模对应 chunk 数为 5/2/1/1，64 和 128 节点均一次做完。策略一旦集中到大规模，后续反馈决策自然消失。4 节点的约 192h useful work 加上每 chunk 600 秒开销，需要第五个短 chunk，不能把它写成恰好四个 chunk。

这说明需要区分两件事：现有环境里可执行的多 chunk 序列是否能赢强固定/规划对照，以及 PPO 是否学到这种序列。优先用既定 Plan-once/MPC 与完整 fixed 诊断机会；若确需改变决策粒度，应给出与 checkpoint/resubmit 接口一致的理由，对所有方法统一应用，并新建配置和结果目录。不能强制切换后把切换次数当收益。

## 5. E1 与统计协议的剩余缺口

Frontera validation wait predictor 的 median-MAE=2.583h，mean-RMSE=5.655h，p90 覆盖率=86.84%，中心 80% 区间覆盖率=67.05%；64 节点的 p90 覆盖率仅 75.27%。这些画像应报告给规划对照，不能把预测 p90 当 deadline 保证，也不构成主 PPO 的准确率门槛。来源：[metrics.json](../results/amsp-e1-frontera/validation-diagnostics/metrics.json)。

train/validation 的探针快照平均运行节点比例为 29.81%/53.07%，64 节点平均 probe wait 约 1.31h/6.64h（48h request）。这提示时间划分后的场景差异值得分析；快照均值不是时间积分利用率，也不能单凭此认定 PPO 失败原因。

现有 dependence 报告只有队列 probes，episode spans=0。48h 只是输入历史产生的未完成 span 下界，尚未冻结 block 长度。已有 Fixed-4 任务约 8 天，拟定 7 天 block 需要结合完整 episode 与相关性重新审视，不能直接照搬建议值。还需核验 trace-prefix 敏感性等 E1 承诺，而不能把 `E1 queue stage complete` 等同于整套 E1 已完成。

cohort 为 Frontera 73/24/54、IW 74/24/34（train/validation/test）。仅用于说明样本量限制：即使把 test arrivals 视为完全独立，54 或 34 个样本全为零 miss，单项单侧 95% 上界也约为 5.40% 或 8.43%；实际 calendar-block 和多重比较规则更严格。因此当前设计可以报告经验 tradeoff 和不确定性，但不应计划把“零 miss”直接写成总体 5% 约束已被认证。此计算未读取 test outcomes。

## 6. 向完成 paper 实验推进

| 顺序 | 工作 | 完成判据 |
|---|---|---|
| 1 | 固化本次 development 负结果，明确预算可达性和 chunk 定义 | 保留全部四预算、全部 checkpoint；旧结果可追溯 |
| 2 | 在 Frontera–7B train/validation 上补 Plan-once、Rollout-MPC，并针对 actor 的预算响应做小范围诊断 | 能区分环境缺少收益机会与学习未找到机会；有实际成本/miss/序列证据 |
| 3 | 依据诊断决定最小方法或粒度修改，使用新实验目录 | 与所有对照公平；不靠追加 seeds/panels 掩盖当前退化 |
| 4 | 方法值得继续时完成 E1 与正式协议冻结 | IW E1、前缀/需求画像、episode dependence；训练交互预算、候选 checkpoint、MPC 阈值候选、统计设置、E3/E4 budget 等有明确归档 |
| 5 | E2：两来源×三模型，主 PPO seeds11/23/37，完整 fixed/mix/规划对照 | validation 选择后冻结；test 不反向调参；不再声称 untouched holdout |
| 6 | E3：匹配 Precommitted-RL | 与主方法同完整预算网格、seed、arrival sampling、交互预算；每 panel 一个预设预算评估反馈增益 |
| 7 | E4：端点/功率后处理、单端点训练对照、四类冻结环境重放 | 主方法和 MPC 在两来源 7B 的预设预算下报告成本及 miss；环境改变需重新 replay |
| 8 | 整合 E1–E4 图表和结论 | 用真实导出替换论文占位；无增益、不可达、统计不确定和失败 panel 全部保留 |

主策略和 Precommitted-RL 按 6 panels × 3 seeds × 2 组计算，共 36 条训练轨迹；若单端点对照也覆盖六 panel、全部三 seed，则总计 54 条，另加评估与重放。单端点组训练范围需在最终协议中明确。不能以当前 seed11 的完成状态计为 E2 已完成 1/18 的正式训练：当前仍是 development 配置，是否可复用取决于最终冻结的方法和设置。

本 run 已提交 64 轮合计 rollout 8,197.1 秒、优化 7.75 秒，约 99.9% 的这两项计时花在 rollout。该时间不含初始化、checkpoint I/O 和被中断轮的重复工作。长任务成本主要应从 replay/输入生成/缓存检查；增加 GPU 或单纯优化反向传播不能解决当前瓶颈。这个测量只对应 Frontera–7B，不外推 IW 或规划基线耗时。

## 7. 文档使用注意

`RESEARCH_HANDOFF.md`、`paper/README.md`、`experiment_manifest.yaml` 等保留交接时“55轮、validation未核验”的状态；根 README 还保留更早的 “No paper experiment has run”。后续以本次结果核验为最新进度，以当前正文和方法规范为研究约束。

本文没有把 development 数值填入正式 E2–E4 结果槽，没有改变旧结果，也不宣称已完成真实 Slurm 部署、LLM 跨规模恢复或节点功率测量。

后续准备（同日）：已实现 [冻结策略诊断与规划对照运行入口](MAIN_DIAGNOSIS_RUN.md)，保留本次配置与原结果，支持 PBS batch/交互运行和按到达×预算恢复。轻量检查通过；依据 AGENT.md，真实实验停在提交前，由用户启动。
