# 最少必要实验：AMSP 公开性能 + 两种历史负载（E1–E4）

2026-09-22状态：[总实验进度与两个核心目标](../docs/PROJECT_STATUS_2026-09-22.md)。Frontera7B三版PPO与全部预定validation已完成；独立actor部分恢复多动作行为，尚未建立稳定fixed优势或有效反馈。正式论文结果槽仍留空；AMSP数值是引用输入。下方执行记录保留2026-09-21交接时状态，不用它判断是否需要重跑。
取消 Qwen3/Vista profiling、实机恢复正确性和整节点功耗测量。详见 reproducibility/research_review.md。

## 2026-09-24 新增：广范围扩展效率敏感性分析（待运行）

活动遗留模型仅保留 Medium/XL；16 节点训练锚点分别为 22.6h/110h，XL 已由旧代码和旧表交叉确认。
作者报告两者均为 16×4 A100 40GB、BF16、100000 optimizer steps、global batch 512、seq 1024。
固定 4/8/16/32 节点，通过 `T_n=T16*16*eta16/(n*eta_n)` 生成其他规模的模拟运行时间。
每模型 14 条效率曲线：11 档 eta32 从 10% 到 120%，另加三个中间形状不同的情景；
28 profiles × 3 seeds = 84 次 PPO。核心仍是完整动态碳–TAT曲线相对完整fixed曲线的位置。
有限扫描不保证覆盖所有真实机器；必须报告收益保持、失效和反转区域。
完整方案及锚点核查见 [scaling_sensitivity.md](reproducibility/scaling_sensitivity.md)。
E4 预留此敏感性篇幅；目前仍为 development weighted-PPO 路径，不冒充已完成的 AMSP budget-policy 结果。

## 历史执行记录（2026-09-21交接）

作者回传完整 `E1 queue stage complete: results/amsp-e1-frontera` 日志，末尾为 368 个 validation 快照、4,416 条探针，无删失。详情记录在 reproducibility/cluster_progress.md。证据目前为作者终端输出；尚未读取远端结果文件、拟合误差或完整 artifact。论文正式结果槽仍留空。

Frontera7B完整train固定292次/114.5s、validation固定96次/51.4s已完成，主PPO seed11完成5轮80episode/125chunk。四档预算约15.11/59.79/104.48/193.84h。此次日志没有RL validation收益，不能把125chunk当作真实换规模。

该轮已启动：2026-09-18核验到55/64轮、3520episode后因PBS时限中断；后续是否已完成先检查cluster实际文件。现有训练复用fixed/reference/grid，从seed11初始化的既定上限为64轮×4档×16episode=4096episode；预定16/32/48/64轮各评价24个validation到达×4档，共384次。报告全体checkpoint的实际TAT/碳成本/miss/换规模比例，Fixed-Mix直接复用已有fixed结果解LP。仍属development，不读取test，不自动扩至其他panel/seeds；完整正式训练预算尚未冻结。单层启动/恢复命令见代码仓库docs/MAIN_TRAIN_RUN.md。

主 PPO 与 Precommitted-RL 不依赖等待模型；Plan-once / Rollout-MPC 后续使用 E1 的模型。IW 尚未回传完整 E1，不阻塞本轮 Frontera 主线。没有因中途运行缓慢删掉实验基线或改变原始日期划分。

## E1 — 输入、模拟场景与等待误差画像

- [ ] 两种历史到达流分别使用，不混洗。记录来源、清洗、时区、既往使用历史和时间划分。
- [x] AMSP arXiv:2311.00257v2 图 12，Our Work 系列，LLaMA-7B/13B/30B；PDF hash 与矢量坐标数字化已归档。
- [x] 4/16/64/128 节点 = 32/128/512/1024 张 A800；g=8，B=1024 sequences，seq=4096，microbatch=1，accumulation=32/8/2/1。没有插值 64/256 卡。
- [ ] 名义集群容量为声明的 128 节点场景，不称为原日志容量或真实 AMSP 队列。原 width、arrival、requested duration、occupied duration 为负载定义，runtime 不作硬件换算。
- [ ] 从声明的 trace prefix 起点空状态连续回放；前 28 天不计分，并检查前缀依赖。旧 Start 不能移植作初始 admission；chunk 之间不重置。
- [ ] 按来源报告资源需求、突发性、占用率、等待尾部；容量归一化只是表示，不宣称可移植等待标签。
- [ ] 可独立求解调度案例与工作守恒检查；独立模拟 probes 的预测误差、排序、覆盖率。原 observed-wait 对比不用于构造场景的真实性验证。
- [ ] 冻结 U=ceil(192*q4) updates，约四个四节点的 48h chunk；此为任务长度设计，不是训练收敛目标。
- [ ] 主开销为每 chunk 600 秒，初始化/恢复与保存各 300 秒；注明假设。没有实机 checkpoint 或最终训练质量证据。

## E2 — slider 控制的完整时间–碳曲线

slider 选择完成预算：D(s)=T_ref+s*(D_hi-T_ref)。完整 train 上 T_ref=最快固定策略平均TAT，D_hi=Fixed-4 TAT 的inverse-ECDF第95百分位；四档 s=[0,.25,.5,1]，beta=D/T_ref 自动生成。右端不是可行性保证。所有已冻结档位都展示；不插值冒充新 policy，也不抹掉非单调点。比较每个预算下的完整成本/miss，并并列实际 mean/p95 TAT。

两种历史负载 × 三个 AMSP 模型。每个 panel 保留 Fixed-4/16/64/128、Fixed-Mix、Plan-once、Rollout-MPC、ScaleDown。主策略不加载等待模型，直接用公开队列历史、实际剩余预算/工作、动作物理描述与 28 个未来六小时 CI 均值；budget-conditioned PPO 与三 seeds 设计保留。

- [ ] Texas EIA ERCOT CI 为外加地区情景；记录原始日历到 CI 的映射、时区和可用时刻，不声称 AMSP 原实验位于 Texas。
- [ ] 全部方法使用同一工作量、开销、请求粒度、初始状态、碳强度与完成预算。每种方法独立推进其 target 造成的排队变化。
- [ ] train 固定 normalizers；validation 按 empirical miss<=epsilon 后选最坏端点成本；test 不重选、不改 seed。
- [ ] 强固定/规划基线同时保留，不能只赢 Fixed-128 就声称 adaptive 或 RL 有效。
- [ ] 报告相对 modeled carbon、mean/p95 TAT、miss、node-hours、chunk 数；paired calendar-block uncertainty，完整报告失败和 censoring。
- [ ] 旧日期的使用历史未核实时先 development；正式结果不冒称新的 untouched test。

## E3 — 动态反馈是否比预先混合规模更有用

只训练一个新增对照 Precommitted-RL，替换之前的 Predictor-advised/current-CI 方案，不叠加。它与主策略使用相同 actor/critic、完整 budget grid、arrival sampling、seeds 和 episode 总预算；仅在每个 panel 一个预设 budget 做机制评估。它在第一次提交之前就生成并保存完整 node sequence，输入只有初始公开 queue/CI、计划剩余工作与 D 减计划分配时长，不能看到后来实际排队耗时、新的 queue state 或 CI 发布。真实执行仍逐次排队，完整实际成本/miss 训练同一个目标。

复用 E2 的主策略、MPC、Plan-once；不增加 queue-blind-MPC 或 wait-advice 的正式消融要求。二者的软件开关保留作开发工具。赢 fixed 但不赢 Precommitted-RL，只支持“混合规模有用”，不支持反馈增益。只赢依赖模型的 MPC 也可能只是该基线建模误差。

## E4 — 功率与环境依赖

- [ ] P0=1 kW 只是参考单位，未知 kappa 表示工作负载整节点功率水平。绝对功率/碳排不报告；共同倍率在 panel 相对结果中抵消。
- [ ] rho 区间、两端点、break-even 与独立 scale-power error 沿用既有方法，复用执行日志。并列 node-hours。
- [ ] 保留一个单端点训练对照；范围限预设预算。
- [ ] **必要的环境检查**：仅 7B、两来源、一个预设预算、冻结 full 与 Rollout-MPC；分别改变四个设置：后台 width=min(2*n,128)，strict FCFS，总 overhead=60 秒，总 overhead=1800 秒。
- [ ] width 放大保持到达与时长，披露 capped fraction 与实际资源时需求增幅；它是负载压力情景，不是硬件升级换算。
- [ ] 每个变体新回放，维持相同物理预算、训练参数、参考量、CI 和工作量。不再重训，不做全因子扫描。成本降低但 miss 上升不能判成功。width/FCFS 两项同时检查无显式预测器的策略是否仍过拟合训练期队列规律。

## 停止与结论规则

来源分别报告；对强 planner 无增量则收回 RL 优势主张。环境变体失效则报告范围，不能写任意机器泛化。只对同一执行日志做 rho/功率重算；队列、吞吐或 overhead 改变会改变决策，必须重放。

不增加新模型家族、Qwen/GPU profiling、节点功率、生产部署、跨地区大扫描、混合训练或 zero-shot 泛化主张。

数据历史确认：旧数据主要做过 train/validation 二分。本轮固定 retrospective_temporal 三段划分，test_is_untouched=false；当前 development 配置用于场景/计算预检，随后冻结协议。
