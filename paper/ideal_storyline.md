# ScaleDown：当前主线与证据映射（2026-09-21）

此文件与当前main.tex、5algorithm.tex、6design.tex对齐。此前“等待模型预训练→RL”“action-conditioned wait作为主策略输入”的设计已被替代；没有通过改名保留在主方法中。结果仍待验证。

## 一句话 idea

普通用户只能决定下次申请多少节点，不能决定Slurm何时分配。ScaleDown在每次checkpoint后，根据真实剩余工作、已消耗的时间、公开队列历史和因果CI预测，选择下一个allocation的规模；用户只用一个slider选择完成预算D，在可接受的超时率下减少模型估算碳成本。

## 方法是什么

1. 动作只改变下一次提交的节点数，allocation之间可以降规模，也可以再升规模；不修改Slurm、不resize正在运行的作业。
2. 主策略是一阶段budget-conditioned PPO。actor/critic不读等待预测器，没有先训练等待回归器再训练主策略的步骤。E1的GBT仍供Plan-once与Rollout-MPC使用。
3. 输入是剩余updates、实际剩余budget、总budget、可见队列历史、28个未来六小时CI均值，以及候选规模的速率/时长等物理描述。观测具有部分可观测性，不声称恢复Slurm内部状态。
4. 优化两功率端点中较大的期望归一化碳成本，并约束Pr(T>D)<=epsilon。epsilon=.05，rho区间[.25,1]。三critic heads估计两个碳成本与miss；gamma=1，完整episode回报；预算分别有dual，actor共享。
5. slider四档s=[0,.25,.5,1]，从完整train最优fixed平均TAT到Fixed-4 train p95。预算是目标，不是单任务保证，也不能保证学出的曲线单调。

## 为什么可能有动态优势，如何避免空洞论证

固定CI、零等待和零开销下，单凭scaling inefficiency不能证明动态反馈有优势。排队使执行落在不同的CI时段，已实现的等待/进度又会改变下一次决策，才形成反馈可能有价值的条件。

策略类包含关系是fixed/每任务fixed mixture ⊆ 预先完整序列 ⊆ feedback。它只说明理想最优值有改进空间，不保证PPO训练结果胜出。人工构造可以展示4→8→4类动态策略支配fixed的机会，但不是AMSP数据结果。

实际动作使用AMSP有引用的4/16/64/128节点。必须同时比较全部fixed、Fixed-Mix、Plan-once、Rollout-MPC与Precommitted-RL。Precommitted-RL在首次提交前就保存完整规模序列，训练预算/架构与主方法匹配；赢它才更有力地支持后续反馈的价值。只学会为不同任务挑不同的首发规模，不足以支撑任务内动态缩放的核心叙事。

## 未测功率怎么处理

P_n=P0*kappa*[rho+(1-rho)*eta_n]。P0=1kW是参考单位，kappa为未知整节点功率倍率，eta_n来自scaling效率而非实测利用率。执行日志保留各规模的CI暴露量L_n；同一panel内的公共倍率可在比值中抵消，rho的相对规模功率假设仍须敏感性分析。

固定同一policy和比较对象时，碳差对rho是仿射函数，区间结论可由两端点决定；区间外不保证。报告modeled job-attributed operational carbon与node-hours，不写成测得整机功率、绝对碳脚印或避免的电网排放。完整推导见6design.tex。

## 必需证据

| 主张 | 证据 | 结果不支持时 |
|---|---|---|
| 声明场景和输入可解释 | E1：AMSP来源、工作量/开销、连续回放与队列画像 | 修复输入/实现或收缩适用范围 |
| slider产生更好的时间—碳选择 | E2：全部fixed、Fixed-Mix、Plan-once、Rollout-MPC，逐预算报告成本/miss | 不写已优于fixed或RL superiority |
| 任务内反馈有价值 | E3：匹配训练条件的Precommitted-RL，并报告实际换规模比例 | 若主要是单chunk/预计划收益，修改方法或核心主张 |
| 功率和队列条件改变后结论仍成立 | E4：功率端点/独立scale误差及已有冻结环境检查 | 报告有效范围，不能声称任意机器泛化 |

## 当前最重要的未解决问题

2026-09-18已核验日志中，主训练因2h时限停在55/64轮。最后17轮1088个episode全部single-chunk。它没有证明动态反馈收益，也不能仅凭总计数判定原因。应在cluster直接读取当前manifest、validation-summary、training.jsonl和chunk动作分布；先判断学习/目标/动作机制问题，再决定是否需要有根据的设计修正。

## 范围

两份历史负载分开作为声明的128节点资源需求场景，AMSP 2024修订版提供7B/13B/30B公开性能，Texas ERCOT CI作为外加地区情景。不是把旧等待标签转移到任何新硬件。当前不要求Qwen profiling、整节点功率实测、新日志或Slurm改造；也不以新增大规模实验掩盖主方法没有动态优势的问题。
