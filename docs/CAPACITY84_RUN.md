# Frontera–7B：84节点实验

本轮仅纠正模拟容量并重新建立该容量下的实验链。只做 Frontera–7B、seed11，不扩 LS6/IW、模型或 seeds。现有 C128 配置与产物保留为旧场景开发记录，不能当作 C84 结果。

## 本轮冻结的变化

- 新配置：`configs/amsp-frontera-7b-c84.development.json`。
- 总容量84节点，合法动作和吞吐profile仅保留4/16/64；去掉无法在该容量中分配的128。没有插值出84节点吞吐。
- 保留相同后台trace、提交时间/节点数/时长清洗规则、CI、train/validation日期与到达cohort、22993 updates、48h最大请求、600秒开销和调度器。
- 继续使用当前候选：四个独立budget product actor、共享critic、seed11从头初始化、64轮×4预算×每档16 episodes=4096 episodes；其余PPO/dual/entropy设置一致。
- 没有教师、warm start、动作偏置、探索率调整、chunk重设计或baseline降配。主策略不读取等待预测器；新拟合的预测器仅供规划基线。
- 目标吞吐仍来自AMSP的A800公开profile；84是后台资源池容量。本轮仍为明确的trace驱动模拟，不声称在RTX机器实测LLM性能。

新输出目录：`results/amsp-frontera-7b-c84-seed11/`。运行入口拒绝把旧C128目录当作本轮输出。

## 运行步骤（PBS交互式节点）

进入已经分配的1 GPU PBS交互节点，在仓库执行：

```bash
cd /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-latest
bash scripts/sophia_capacity84.sh --stage prepare
```

准备阶段依次完成：

1. C84完整fixed train：73 arrivals×3 scales=219次；重新计算时间/碳参考量和四档预算。
2. C84完整fixed validation：24 arrivals×3 scales=72次；同一批结果按每个新预算计算miss、Best-Fixed与Fixed-Mix。
3. C84 E1：每6小时采一个队列快照，对3 scales×3 walltime探针分别求等待；仅train拟合新的等待模型，validation报告队列占用、等待和误差，执行队列依赖审计。
4. C84 Plan-once和Rollout-MPC：24 arrivals×4 budgets×2 methods=192次validation；保持256 planning paths和5%内部miss阈值。每个arrival/budget单独封存，便于中断恢复。
5. 输出基线表格与`budget-curves.png/pdf/csv`。预算完全由本轮fixed train推导，不沿用C128的15.11/59.79/104.48/193.84小时。

准备完成后训练并验证：

```bash
bash scripts/sophia_capacity84.sh --stage train --resume
```

`--resume`在这里用于接续已经存在的准备目录；首次进入训练仍从随机初始化开始。完整准备阶段只校验复用。PPO到64轮后，依次验证第16/32/48/64轮，每轮24 arrivals×4 budgets，共384次。全部checkpoint报告，不在这一步挑最优。

也可一次执行全部阶段：

```bash
bash scripts/sophia_capacity84.sh
```

任一阶段超时/中断后，在新的交互allocation继续：

```bash
bash scripts/sophia_capacity84.sh --resume
```

恢复会跳过哈希校验通过的完整阶段。探针按已写行恢复；PPO从最后完整轮恢复，未提交轮重跑；fixed和未完成validation保留备份后重跑；规划按小单元恢复。不要同时运行两个写入相同目录的进程。

## qsub及日志

同一脚本包含PBS资源声明：Local-LLM/by-gpu，1 GPU/32 CPU/120GB，**1小时**。该上限是每次allocation的时限，不是整个新实验耗时保证；交互式bash受你已经申请的时限约束。恢复命令不改变预定训练总轮数。

```bash
qsub scripts/sophia_capacity84.sh
qsub -v CARBON_RESUME=1 scripts/sophia_capacity84.sh
```

只提交准备阶段可用`qsub -v CARBON_STAGE=prepare scripts/sophia_capacity84.sh`；准备结束后提交训练用`qsub -v CARBON_STAGE=train,CARBON_RESUME=1 scripts/sophia_capacity84.sh`。

终端输出和错误自动保存在`out/capacity84.<PBS_JOBID>.<UTC>.<unique>.log`。使用已有carbon Python环境，不自动安装或改动依赖。运行完成后直接读取out/results，不需要手动贴日志。

## 结果入口与验收

- `capacity84-summary.md/json`：队列证据、全部基线、全部checkpoint、序列计数、动作概率、耗时和文件绑定。只有表格和图都写成功才发布`complete`。
- `budget-curves.png/pdf/csv`：横轴为四档完成预算，分别画碳成本C、miss、实际平均TAT；包含Fixed-4/16/64、Best-Fixed、Fixed-Mix、Plan-once、MPC和最后完成验证的PPO checkpoint。
- `checkpoint-curves.png/pdf`：保留全部四次验证的成本与任务内换规模曲线。
- `policy-behavior.png/pdf/csv`：四次验证在各预算的首次动作概率、概率范围、实际采样数和argmax数。
- `references.json`、`budget-grid.json`、`pipeline-contract.json`、`run-plan.json`：本容量的新参考值、预算及执行契约。
- 不可行预算保留并标记；未完成任务保留分母，完整成本留空；Fixed-Mix显示validation拟合的期望值，不冒充独立test效果。曲线连线只连接已评价档位，不声称插值预算有效。

本轮允许负结果。先看原先全选64的现象是否改变，再比较同预算下固定/混合固定的碳成本与超时。不会因结果打不赢fixed临时添加优化。

开发核验仅使用六小时合成trace和两轮小型PPO。5项检查已通过，覆盖完整链路与绘图、训练/验证中断恢复、混合基线统计、容量配置边界、旧目录保护及启动脚本日志；日志见`out/capacity84-tests.9WZ8gZ.log`。Python语法、Bash语法和实验manifest解析也已通过。原C128配置、37个核心源码文件和4个旧实验绑定脚本与已完成C128运行的哈希一致。

真实trace回放、E1拟合和64轮训练尚未执行，由用户在计算节点启动，遵守根目录`AGENT.md`。
