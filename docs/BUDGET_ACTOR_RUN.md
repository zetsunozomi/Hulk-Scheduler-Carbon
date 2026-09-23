# 下一轮：独立预算 actor 对照

[冻结梯度诊断](../paper/reproducibility/development_findings.md)已完整结束：第1/4轮最宽松预算的局部梯度与共享方向 cosine 为 −0.418/−0.687，它自己的梯度提高 P(4)+P(16)，共享梯度却降低它。这是有记录支持的局部干扰，不是退化唯一根因的证明。

## 这次检验什么

同一个策略入口按**总 budget/Tref**选择四个独立的 product actor 分支，每个分支的 action encoder、context encoder 和 score 参数均独立；critic 仍共享。保留四个原有预算，包括最紧不可达点，不按剩余 budget 切换分支，不对未知预算插值。

四个分支在原 actor/critic 完成初始化后复制同一套初始 actor 权重，不额外消耗 RNG。因此新旧模型在训练开始时具有相同 actor/critic 函数。分支只阻断 actor 参数上的跨预算更新，无法消除共享 critic、整体梯度裁剪和 optimizer 调度带来的所有耦合。

每个 actor 分支有121,985参数；新增三个副本，共增加 **365,955** 参数。这是增加容量的结构对照，不是等参数的严格因果消融。结果之前不把它采用为 paper 正式主方法；单凭更高 TV、entropy 或切换次数不能声称碳与反馈收益。

## 固定比较范围

- Frontera–7B，fresh seed11，不从旧 checkpoint warm start。
- 64轮 × 原四档预算 × 每档16 episodes = **4,096 个总训练 episodes**，不是每个分支都增加4,096个。
- 四个分支合并参加原来的完整 episode minibatch16、epochs4、PPO/dual/entropy 更新；训练代码的更新流程未改。
- 与 shared-product trial 保持同一到达采样流程。训练完成后逐条核验 `(iteration, budget, rollout_sample, episode_id)`；恢复仅计入已提交轮次，若配对不符则拒绝继续评估。
- 第16/32/48/64轮各评价同一24 arrivals × 4 budgets，共384次validation。所有结果保留，不额外选优，不访问test。
- 报告新策略与旧concat/product PPO各checkpoint、fixed、Fixed-Mix、Plan-once/MPC的成本、miss、序列、paired budget TV和端点成本。

## 启动与恢复

登录节点：

```bash
qsub scripts/sophia_budget_actor.sh
```

PBS 1 GPU 交互节点：

```bash
bash scripts/sophia_budget_actor.sh
```

恢复：

```bash
qsub -v CARBON_RESUME=1 scripts/sophia_budget_actor.sh
```

或在交互节点执行 `bash scripts/sophia_budget_actor.sh --resume`。

日志自动保存到 `out/budget-actor.<job>.<UTC>.<随机后缀>.log`，终端仍显示。输出为 `results/amsp-budget-actor-frontera-7b-seed11/`，完成标记为 `budget-actor-summary.md/json` 的 complete 状态。agent 直接读取文件，无需粘贴日志。

沿用 Local-LLM/by-gpu、1 GPU/32 CPU/120GB，CPU单线程，PBS上限4h。新策略可能增加chunk数，当前没有实测耗时；4h不表示保证完成，超时后按原设置恢复。

## 产物与源码

`run-plan.json` 绑定全部新旧输入、源码、脚本、runtime、原product和梯度诊断摘要、参数增量。训练每轮保存checkpoint，恢复写新attempt；完整validation按seal复用，中断stage备份后重跑。

在本次修改核心源码前，已把完成的shared-product源码及相关脚本共41个文件保存到 `results/source-archives/product-shared-5e6c7ddcbb91/`，所有文件与历史run-plan哈希相符，清单在 `archive.json`。这是可供恢复源码的归档，不是自动可运行的独立环境；配置、依赖、数据和根路径约定仍需要匹配。

旧训练/诊断的严格源码绑定保持不变，因此不在当前新源码上恢复这些旧作业。新入口只在 replay、输入、核算、planning 文件不变，旧产物哈希通过时复用旧结果；仅允许四个已声明策略文件变化。原结果和归档保持原样。

## 启动前检查

43 项轻量合成检查通过，包括 actor 参数隔离、Adam 已有动量时未参与预算分支不变、初始函数/RNG/物理 mask 一致、checkpoint 恢复与预算绑定、含多个到达的两轮端到端试验、validation 中断恢复和日志/退出码。另与归档的旧模型在合成张量上逐项比较，默认 shared 和新分支的初始输出、critic、RNG 均一致，实际新增参数数目为365,955。

使用保存的真实 manifest、cohort 和结果文件完成了启动契约校验：56个旧对照点、原4,096条训练到达记录与四轮梯度诊断均可读取，只有声明的四个策略文件改变，除 actor_budget_mode 外所有训练设置一致。这一步没有加载真实trace或运行真实回放/训练。

真实实验由用户在计算节点启动，遵守 [AGENT.md](../AGENT.md)。
