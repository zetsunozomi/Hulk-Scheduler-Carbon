# 扩展效率敏感性分析：v2 显式效率向量

2026-09-24。状态：28 组输入、运行入口和报告已准备；尚未运行本组 PPO 实验。
设计是在看过旧 Medium development 结果后确定的，不冒称事前预注册。
问题是：**动态碳–TAT 整条曲线相对 fixed 整条曲线的左下移，依赖哪些吞吐假设？**

## 输入与公式

保留节点集合 4/8/16/32。每任务固定 100,000 optimizer steps。
Medium 的 16 节点纯训练时间为 22.6h，XL 为 **110h**。旧代码与旧表两处吻合；
来源与硬件合理性核查见 [ANCHOR_AUDIT.md](../data/scaling_sensitivity/ANCHOR_AUDIT.md)。
作者补充：两者均为 16 节点、每节点 4 张 A100 40GB，BF16、global batch 512、seq 1024。

只需要一个时间锚点和完整的效率向量：

```
eta_n = (q_n/q_4)/(n/4), eta_4 = 1
T_n = T_16 * (16*eta_16)/(n*eta_n)
q_n = (100000/T_16) * (n*eta_n)/(16*eta_16)
```

锚点固定训练时间，不固定排队与开销后的 TAT。其余规模是**模拟估算**，
`profile_status=assumed`；硬件信息是锚点的作者报告，不把模拟吞吐写成实测。
配置的 microbatch=1、accumulation=128/n 只满足模拟 batch 记账，并非恢复出的实测并行方案。

## 范围与形状

每种模型各 14 条效率曲线：

- 11 条幂律曲线：eta32 = 10%、12.5%、15%、20%、25%、35%、50%、65%、80%、100%、120%。
  eta_n=(n/4)^(gamma-1)，gamma=1+log(eta32)/log(8)。
  包括加节点反而变慢、总吞吐饱和、弱/较强扩展、线性和超线性压力情景。
- 3 条非幂律曲线，按 4/8/16/32 顺序：early_knee=(1,.55,.50,.50)，
  late_knee=(1,.95,.90,.50)，dip_recovery=(1,.90,.45,.50)。
  它们与 e050 具有相同 eta32，却有不同中间形状，避免只扫一个端点。

完整数字见 [PROFILES.md](../data/scaling_sensitivity/PROFILES.md)，机器可读设计见
[design.json](../data/scaling_sensitivity/design.json)。v1 的四档配置已移至 archive/，不作为当前运行入口。
遗留模型的活动入口仅保留 Medium/XL；Large 的历史表与配置留档以保持来源哈希。

**有限情景扫描不能证明“必然覆盖任何真实机器”。** 范围再宽也不能穷尽中间形状、
通信、竞争、运行时波动和并行方案。论文应写为：在跨度广、形状不同的声明情景下
检验结论的敏感性，并报告收益保持、消失或反转的区域。情景范围不是实测分布。

## 控制变量与完整运行矩阵

- 2 模型 × 14 profiles × seeds [11,23,37] = **84 次 PPO**；固定基线每个 profile 建一次，共 28 组。
- Frontera 历史需求、84 节点容量、ERCOT CI、73 train/24 validation 到达与日期划分保持一致。
- 固定工作量、节点集合、300s 初始化/恢复、300s checkpoint、48h chunk cap 和调度器。
- 主比较 rho=1，令每节点功率不随效率向量变化。rho=.25 的重评分同时改变了功率与吞吐假设，单独注明。
- 每组独立重建 fixed/reference/normalizer、从头训练；64 轮含 5 轮观察；alpha=[0,.2,.5,.8,1]；16/32/48/64 轮验证。
- 当前 max48 协议：fixed 每次请求 48h，dynamic 请求向上取整的计划时长。
  所以收益同时包含节点选择和请求时长优化；不能单独归因于换规模。
  与旧 0924-2pm 结果比较时，还须匹配此请求协议。
- 训练长度和效率改变会改变 chunk 数与切换机会。XL 的最慢构造点约 497h，可能需要 11 个 chunk。
  必须报告 chunk 数、单 chunk 比例、真实切换比例和完整/删失情况。
  现有日期/CI边界若不足，停止并另立统一更长观察窗的新版本；不得丢弃失败到达后继续比较。

每组保留四个 fixed 点与所有 alpha 点，包含重合点、被支配点和失败区域。
不按每个 alpha 选择一个“最优 fixed”来替代整条曲线。连线仅助读，不是额外执行过的策略。
先固定报告全部 checkpoint；最终选点必须统一冻结规则，不能按收益挑 checkpoint 或 seed。

## 检查、启动与回传

```bash
python scripts/scaling_profiles.py --check
PYTHONPATH=src python -m unittest tests.test_scaling_sensitivity
bash scripts/frontera_scaling_sensitivity.sh --model xl --scenario e050 --check
mkdir -p out
sbatch scripts/frontera_scaling_sensitivity.sh --model xl --scenario e050 --seed 11
# 超时后使用完全相同输入和代码
sbatch scripts/frontera_scaling_sensitivity.sh --model xl --scenario e050 --seed 11 --resume
```

多个 seed 并行前，先 `--stage fixed` 完成对应 model/scenario 的基线，避免共享目录多个 writer。
其他情景名见数值表。每条命令只运行一组模型、效率曲线与 seed，默认目录为：

```
results/scaling-v2-xl-e050-frontera-c84-fixed-max48/
results/scaling-v2-xl-e050-frontera-c84-max48-weighted-seed11/
```

`--check`只读实际配置、trace/CI/cohort和来源哈希，不回放、不训练、不加载 PyTorch。
正式执行仅限计算节点，环境需 PyTorch>=2.6,<3。日志独立保存在 out/，Slurm 邮件已配置。
更改设计、配置或绑定实现时使用新目录；旧输出不可覆盖或换标签继续使用。

回传结果后复用本地画图 workflow：

```bash
/Users/shuyuanfan/miniconda3/envs/writing/bin/python scripts/local_plot_validation_tradeoff.py \
  --fixed-run scaling-v2-xl-e050-frontera-c84-fixed-max48 \
  --dynamic-run scaling-v2-xl-e050-frontera-c84-max48-weighted-seed11 \
  --output local-plots/validation-tradeoff/scaling-v2-xl-e050 \
  --tag scaling-v2-xl-e050 --title 'XL · assumed scaling · eta32=50% · seed 11'
```

`scaling-summary.json/md`保留全部 checkpoint、双指标、chunk 和切换信息，并校验配对到达、工作量和结果哈希。
本地轻量验证覆盖锚点、28 组效率与整数工作守恒、分段、回退/饱和/超线性/不同拐点、绑定与报告。
本机缺少 PyTorch，未做 PPO 端到端运行；没有提交集群作业。

## 文献作用

[Pollux §5.1](https://www.usenix.org/system/files/osdi21-qiao.pdf)说明了实测迭代时间与
构造工作负载的结合，其相对单 GPU 的效率范围不能直接替代这里相对四节点的定义。
[Gavel 官方代码](https://github.com/stanford-futuredata/gavel)和
[Pollux 结果仓库](https://github.com/petuum/pollux-results)是未来引用真实 profile 的入口；
本设计没有从这些来源摘取或调整任何吞吐数值。它新增敏感性证据，不自动替换 AMSP 主实验。

## 本轮检查记录（2026-09-24）

10 项轻量测试通过；全部 28 组配置用实际 trace/CI/cohort 通过输入绑定校验；
Slurm wrapper 的 XL/e050 --check 通过；Python AST/Bash 语法检查通过。
检查未运行真实回放或 PPO。额外的 fixed-max-walltime 测试模块因本地缺少 PyTorch 无法导入，
不计入已通过测试；端到端训练仍需集群计算节点验证。
论文主文件已编译为 7 页正文 + 1 页参考文献；1000 个公式检查与引用/版面检查通过，
最终修改涉及的第 5–8 页已渲染目检。
