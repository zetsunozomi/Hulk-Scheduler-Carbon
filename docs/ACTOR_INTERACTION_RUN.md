# 已完成：actor 状态与动作交互项

2026-09-21 更新：job `187569` 已完成 64 轮训练和全部 validation。交互项未解决退化，最终 96 次评估全部为 `[64]`，见[真实结果记录](../paper/reproducibility/development_findings.md)。后续梯度诊断也已完成，当前下一步是[独立预算 actor 对照](BUDGET_ACTOR_RUN.md)。以下保留原运行方式；当前核心源码已不同，不重复或恢复本旧训练，对应源码已归档到 `results/source-archives/product-shared-5e6c7ddcbb91/`。

前一轮诊断已完成：192 个 planner episode、384 个冻结 actor 输入/概率检查全部通过，结果见 [paper development 记录](../paper/reproducibility/development_findings.md)。本轮只检验一个学习结构假设，不改环境来制造反馈机会。

## 为什么做这一步

旧 actor 的 score 输入为 `[action_embedding, context_embedding]`。当末层 ReLU 对不同动作具有相同激活模式时，context 对 logits 的公共平移在 softmax 中抵消。这个代数现象在小型控制测试中可复现；实际 checkpoint 的预算 TV 极小，但尚未证明其唯一根因就是该现象。

本次候选增加逐元素乘积，score 输入改为 `[action_embedding, context_embedding, action_embedding*context_embedding]`。action encoder、context encoder、critic、观测、PPO/dual、entropy 等设计保持原样。新增参数来自 score 首层增加 128×128 个权重，不能把这轮比较称为完全等参数量的因果消融。默认通用模型仍为 `concat`，候选必须显式设置 `actor_interaction=product`，并写入 checkpoint 架构元数据。

新 trial 是一个待验证的候选，不承诺会产生预算响应或动态收益，不预先修改 paper 的正式主方法声明。

## 固定实验范围

- Frontera–7B，fresh seed11，无旧 checkpoint warm start。
- 64 轮 × 原四档预算 × 每档 16 episode = 4,096 个训练 episode。
- 第 16/32/48/64 轮，每轮完整 24 到达 × 4 预算，共 384 次 validation；全部候选都报告，不中途选优。
- 原整数工作量、48h 请求上限、600s chunk 开销、CI/queue、参考量和未取整预算不变；最紧不可达预算仍保留。
- 主策略仍无 wait predictor；不蒸馏 planner、不增加监督预训练、不重新训练或运行 planner、不访问 test outcomes。
- 输出新策略与原全部 PPO checkpoint、fixed、Fixed-Mix、Plan-once/MPC 的同 cohort 对比；新增 paired budget TV、动作熵、实际规模序列和双端点成本。

## 启动与恢复

仓库根目录，登录节点：

```bash
qsub scripts/sophia_actor_interaction.sh
```

已经在 PBS 1 GPU 交互节点：

```bash
bash scripts/sophia_actor_interaction.sh
```

登录节点恢复：

```bash
qsub -v CARBON_RESUME=1 scripts/sophia_actor_interaction.sh
```

交互节点恢复：

```bash
bash scripts/sophia_actor_interaction.sh --resume
```

沿用 Local-LLM / by-gpu、1 GPU / 32 CPU / 120 GB、4h 的 PBS 资源形状，实际 CPU 单线程；4h 是申请上限，当前尚无新架构运行耗时测量。脚本要求有效 `PBS_JOBID`，不自行提交作业、不自动安装依赖。

默认输出：`results/amsp-interaction-frontera-7b-seed11/`。

batch 和交互运行都会自动把 stdout/stderr 同时写入终端和 `out/actor-interaction.<PBS_JOBID>.<UTC时间>.<随机后缀>.log`，启动时打印完整日志路径。每次启动或恢复新建日志，不覆盖旧日志，无需额外重定向。运行结束或需要检查进度时，agent 直接读取 `out/` 和 `results/`，用户无需手动粘贴输出。`out/` 已加入 Git 忽略规则。

| 文件 | 含义 |
|---|---|
| `run-plan.json` | 新旧 source 差异、复用文件哈希、固定训练设置、预算和四个评估轮次 |
| `ppo-attempt-*/` | 新模型各轮 checkpoint、episode/chunk/优化日志 |
| `validation-000016/000032/000048/000064/` | 新模型四轮完整配对评估 |
| `interaction-summary.md/json` | 全部旧对照与新模型曲线、miss、序列和预算响应 |

每轮训练保存状态，恢复写新 attempt；已完成 validation 校验后复用，中断 stage 保留备份再重跑。所有输入、代码、runtime 和设置在首次启动后固定。

## 旧结果与源码版本

原 main/diagnosis 绑定修改前的全包源码，当前源码有意保留它们的严格失配拒绝规则。因此不在当前版本恢复旧 main 或重跑旧 checkpoint 推理；旧产物保持原样，训练重新开始于新目录。旧核心源码仍可由 Git 提交 `661c729` 找回。

本 trial 对旧 fixed/planner 采用显式的结果复用检查，只允许 `learning.py`、`policy_runner.py`、`stress.py`、`__main__.py` 四个策略相关文件与旧版本不同。配置、数据、replay、工作/碳核算、特征和 planning 文件必须相同，原输出 seal 必须完整，新旧哈希全部记入 run-plan。没有修改普通 resume/evaluation 的校验规则。

## 结果判据

先看新策略是否确实响应预算，再看成本和 miss；更多切换、较高 entropy 或 TV 本身不是收益。若策略恢复预算响应但仍输强对照，进一步区分优化不足和可利用收益很小。只有主策略值得继续后，才进行同架构、同交互预算的 Precommitted-RL 对照；该对照之前不声称反馈有效。

## 启动前检查

33 项单线程、小型合成测试已通过，覆盖原模型默认行为、交互项的受控预算梯度、checkpoint 架构绑定、训练恢复一致性、评估中断恢复和 PBS 启动入口。还使用现有 manifest/cohort 和结果文件完成真实产物的复用校验：四个源码差异均在声明范围内，96 个 planner stage 完整，40 个旧对照点可读取，除 actor 交互项外所有训练设置与旧运行完全相同。该检查没有加载真实 trace、运行 replay 或进行模型推理，不代替计算节点上的真实新训练。

本轮真实实验由用户启动，遵守 [AGENT.md](../AGENT.md)。
