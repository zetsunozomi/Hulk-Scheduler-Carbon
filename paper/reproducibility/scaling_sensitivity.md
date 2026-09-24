# 2026-09-24：v2 扩展效率敏感性设计

状态：28 组模拟输入、独立 Slurm 入口和报告已准备；实验未运行。

- Medium T16=22.6h；XL T16=110h。仅保留这两个活动遗留模型。
- 作者确认锚点测量设置：16 节点 × 4 A100 40GB，BF16，100000 optimizer steps，global batch 512，seq 1024。
- 每模型 14 条显式效率向量，eta32 从 10% 到 120%，包含回退、饱和、线性、超线性及不同拐点。
- T_n=T16*16*eta16/(n*eta_n)；eta 相对四节点归一化。其他规模是构造吞吐，不能称为实测。
- 28 profiles × seeds 11/23/37 = 84 次 PPO。rho=1、48h cap、开销、到达和 CI 保持一致，逐组重训。
- 对比完整 fixed/dynamic 曲线；报告切换和 chunk 数。扫描跨度不等于所有真实机器的覆盖保证。
- 当前运行入口是 C84 weighted PPO，区别于主稿 AMSP 的 budget-conditioned 策略；新增敏感性结果需明确报告该方法差异。

[完整实验方案](/Users/shuyuanfan/carbon-latest/docs/SCALING_SENSITIVITY.md)
与[锚点证据和合理性核查](/Users/shuyuanfan/carbon-latest/data/scaling_sensitivity/ANCHOR_AUDIT.md)是权威记录。
旧结果与 AMSP 主矩阵保留；不以模拟检查或硬件峰值核算填补实验结果。
