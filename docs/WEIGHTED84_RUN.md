# C84：TAT 与 carbon 加权目标（2026-09-23）

这是新候选目标的独立实验。源 fixed 数据来自已完成的 `results/amsp-frontera-7b-c84-seed11`，新结果默认写入 `results/amsp-frontera-7b-c84-weighted-seed11`。不加载旧 PPO 权重，也不重跑 fixed、wait probes 或等待预测器。

## 目标与观察阶段

对完整任务最小化

`J_alpha = alpha * TAT / Tbar + (1-alpha) * C(rho=1) / Cbar`，

`alpha = [0, 0.2, 0.5, 0.8, 1]`。PPO 最大化负成本。

这里把用户所说的 epoch 定义为**一轮完整 rollout**，不是一次 minibatch 优化遍历。前 5 轮每轮每档 alpha 采集 16 个完整任务，共 400 个任务；从头初始化的随机策略照常与模拟器交互，但 actor、critic 和 Adam 都不更新。五档 alpha 在每轮使用同一组训练到达时间，各自采样动作。对这 400 个任务的完整 TAT 与完整 carbon 分别取算术平均，得到所有 alpha 共用的 Tbar、Cbar。按任务平均，不按 chunk 平均；不使用 validation/test。

第 5 轮完成后将参考量写入 `reward-normalizers.json` 并冻结。第 6 轮开始采新 rollout 做 PPO，不重用观察阶段的数据进行 PPO 更新。总共 64 轮，即 **5 轮观察 + 59 轮更新**，5120 个任务（400 观察、4720 更新）。这里的“动态”是运行时根据观察数据确定参考量，之后不随训练漂移。

每个 chunk 的奖励为

`r_k = -alpha * elapsed_k / Tbar - (1-alpha) * carbon_k(rho=1) / Cbar`。

`elapsed_k` 是提交到 checkpoint 完成的实际经过时间，包含排队、初始化/恢复、训练与保存。gamma=1，因此完整任务奖励之和恰好等于 `-J_alpha`。碳成本包括实际分配期间的各阶段，排队本身不计作此任务的已分配能耗。carbon 是模型的 `gCO2/kappa`，共同未知倍率在归一化中消掉；另一端点 rho=0.25 继续保存在原始结果与汇总中。这里没有 deadline miss、约束乘子或功率端点最坏情况更新。

## 模型与保持一致的物理条件

- Frontera 容量 84，动作 4/16/64，原有 48h 单次请求上限、吞吐、checkpoint 开销、trace、CI 与 train/validation 划分保持一致。
- 每档 alpha 一个独立 product actor，五个 actor 初始参数相同但不共享参数；共享 critic 的两个线性输出分别估计剩余归一化时间成本、碳成本。
- 输入为剩余 updates、已耗时间、alpha、公开队列历史、未来 168h 的 28 个因果 CI 预测均值，以及候选动作物理描述。替换原输入中的两个预算字段。没有等待预测器输入。
- **输入特征缩放与奖励归一化分开**：队列/动作/CI 输入仍用旧 fixed-train 参考量作稳定的数值缩放；本次新估计的 Tbar、Cbar 专用于奖励和评价。这避免观察阶段尚无均值时循环依赖，并使前后输入含义相同。
- PPO 每个 rollout 做 4 次优化遍历，minibatch 16 个完整任务，Adam 3e-4，clip 0.2，entropy 0.01，value coefficient 0.5，gradient clip 0.5；seed=11、CPU 单线程，与旧实验主要优化配置一致。熵项是原有正则化，不是额外性能指标。
- 任务若触及数据覆盖边界而未完成，保留其临时日志并停止，不丢弃、不把截断任务当作完整样本。取消预算约束并不取消模拟数据覆盖边界。

## 启动与恢复

在已有 PBS 交互 allocation（1 GPU）中：

```bash
bash scripts/sophia_weighted84.sh
```

登录节点提交：

```bash
qsub scripts/sophia_weighted84.sh
```

中断后分别使用：

```bash
bash scripts/sophia_weighted84.sh --resume
qsub -v CARBON_RESUME=1 scripts/sophia_weighted84.sh
```

默认执行训练、四个 checkpoint 的 validation 和汇总绘图。可用 `--stage train` 只训练，再用 `--stage validate --resume` 验证。PBS 1h 是可续跑的资源申请上限，不是总运行时长估计。日志每次独立写到 `out/weighted84.<job>.<UTC>.<unique>.log` 并保留终端输出。覆盖路径的环境变量是 `CARBON_SOURCE`、`CARBON_OUTPUT`，Python 为 `CARBON_PYTHON`。

每轮只有在 rollout、优化、checkpoint 和日志全部写完后才原子提交 `iteration-000001/` 等目录；中断的 `.iteration-*` 临时目录保留审计，恢复时重跑该未提交轮，不重复计入观察均值。checkpoint 保存 actor、critic、Adam、动作采样 RNG、到达/优化 shuffle RNG、观察累计量和冻结均值。validation 也按完整 checkpoint 原子提交，中断不重训已完成的 PPO。恢复检查配置、数据、源 fixed 日志、源码和 PyTorch 版本。

## 比较与产物

验证第 16/32/48/64 轮，每轮同一组 24 个到达 × 5 个 alpha，合计 480 个完整任务。旧 fixed 原始 TAT/carbon 按新目标重新计分，与所有 checkpoint 同表报告。Best-Fixed 在当前 validation 上选取，要明确这是 development 比较。固定线性加权成本下，按任务随机选择 fixed 的混合期望不可能低于最便宜的 fixed，因此此目标的最优 Fixed-Mix 与 Best-Fixed 同值；旧 deadline planner 的分数不能直接当作新目标优化结果。

输出包含：

- `run-plan.json`、`manifest.json`、`reward-normalizers.json`：目标、输入绑定、进度和冻结参考量。
- `iteration-*/`：每轮 episodes/chunks、观察或更新阶段、优化步数与可恢复 checkpoint。
- `validation-*/`：所有声明 checkpoint 的五档结果。
- `weighted84-summary.md/json`：新成本、TAT、两端点 carbon、Best-Fixed 改善、逐任务配对胜次、任务内切换、动作频数及初始动作分布。
- `weighted-curves.csv/png/pdf`：加权成本曲线和 TAT–carbon 曲线。

五档包括纯时间、纯碳端点；端点收敛到固定动作本身并不构成实现错误。是否实现“任务内动态决策并胜过 fixed”，仍需看新结果，中间 alpha 也不保证一定出现有益的切换。

实机实验由用户在计算节点启动。实现验证只运行小型 synthetic replay，覆盖五轮无更新、完整任务均值、奖励可加性、分支独立、观察/训练/验证中断恢复、产物篡改拒绝和启动日志。
