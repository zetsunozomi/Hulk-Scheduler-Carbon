# RL-free 动态曲线：opportunity-v1

这轮在 Frontera **计算节点运行 CPU 调度模拟**。后台 trace 是构造的普通作业请求；不会向真实 Slurm 投递这些后台作业，也不做神经网络训练或能耗实测。

## 本轮固定设计

- XL/e050、84 节点、动作 4/8/16/32、100,000 工作单位；沿用现有 scaling 输入与 300s setup/restart/checkpoint 开销。
- 原配置先通过原有完整绑定验证，再建立单独的 synthetic 场景；旧 AMSP/GPT/RL 实验不变。
- **fixed 和 dynamic 每次均请求 48h，包括最后一次；实际完成即释放**。
- 一个 RL-free 控制器；alpha = 0, .1, .2, …, 1。0 更偏向省能耗，1 更偏向完成时间。
- 每个 trace 3 个生成种子（11/23/37），每个种子 12 个随机分层到达；所有方法共享到达与后台请求。
- 每个 trace 共 36 到达 ×（4 fixed + 11 dynamic）= **540 个结果**；五个 trace 共 **2700 个结果**。
- 所有目标任务独立回放，互相不竞争；其对后台作业的影响由各自回放重新计算。
- 7 天背景 warmup；目标在第 7–28 天随机到达；每个目标最多观察 30 天；背景持续生成 60 天。
- 输出目录 `results/opportunity-v1-xl-e050/`。没有训练阶段、训练 checkpoint、wait probes 或预测器依赖。

## 同一个动态框架

每次任务提交或完成一个 checkpoint 后：

1. 读取当前可见的空闲节点、running 节点数及申请剩余时长、pending 节点数/申请时长/已等待时长。
2. running 按申请剩余时长释放容量；pending 按等待年龄排列，保守地放在新请求前面，建立容量日历。
3. 对每个 n，求该公开请求日历中能容纳 n 节点、48h 请求的最早位置，截断到 48h，记为 `D(n)`。
4. 根据固定 scaling profile，计算同一份剩余工作始终用 n 完成需要的执行时间 `T(n,R)` 与 node-hours `H(n,R)`，包含后续所有分段开销。
5. 选择最小分数：

```
score(n) = alpha * [T(n,R) + D(n)] / T_ref
         + (1-alpha) * H(n,R) / H_ref
```

`T_ref` 是完整任务无排队执行时间的最小值；`H_ref` 是完整任务无排队 node-hours 的最小值。
仅由 workload 输入计算；没有根据本轮结果拟合或使用 fixed 回放结果校准。
执行一段后重新观察和选择，允许升回去；重复 alpha 点也完整保留。

**D(n) 是由公开申请量形成的占用压力代理，不是经过校准的等待时间预测。**
运行时间小于申请值、未来到达、私有优先级都会使它失准；48h 截断限制过于悲观的输入。
控制器不读取实际后台 runtime/end、future arrivals、内部 priority、未来 CI，也不克隆模拟器试跑动作。
它在每个决策点用“继续固定规模完成剩余工作”打分，不假装知道以后每段的排队。

真实 Slurm 的 `squeue` 可提供 NumNodes、TimeLimit、TimeUsed、SubmitTime 等字段，`sinfo` 提供节点状态；
这些字段的可见性受站点配置限制。本入口使用模拟器的公开字段等价物，尚不是线上 Slurm 控制器。
参考：[squeue](https://slurm.schedmd.com/squeue.html)、[sinfo](https://slurm.schedmd.com/sinfo.html)。

## 五个 trace

完整生成参数在 `configs/opportunity-v1.json`，所有随机数与目标策略独立。

| 名称 | 生成方式 | 主要检查的问题 |
|---|---|---|
| `wide-long` | 每 72–108h 到达一个 64 节点作业，运行 30–44h，请求 48h | 持续留下 20 节点的机会 |
| `layered-long` | 两股独立 32 节点流，各每 60–84h 到达，运行 30–44h | 多个作业重叠产生机会 |
| `burst-mix` | 每 84–120h 一批 8/16/32 节点作业，批总需求 64 或 80，8h 内陆续到达，运行 24–44h | 忙闲交替与多种资源需求 |
| `loose-walltime` | 64 节点作业实际 12–36h、请求 48h，另有 4/8 节点小作业流 | 申请时长不准确、残余资源受干扰 |
| `short-control` | 与 wide-long 相同的大作业到达过程，但实际 2–8h、请求 12h | 短暂拥挤时是否过度缩小 |

这是几个机制导向的备选环境，**不是负载严格匹配的单因素消融**。输出会保存实际 offered load 和申请/实际 node-hour 比。
生成器不知道目标任务到达、checkpoint、alpha 或任何策略结果。不会为每个方法重抽后台请求。
公开日历排序与引擎实际 age/size priority 并不完全相同，正是所声明的代理误差之一。

## 一次启动全部

现有 carbon 环境已含 matplotlib；无需 PyTorch、额外安装或下载模型。
Slurm 使用已有的 `small`、1 节点、4 CPU、4h；account 留空沿用站点默认，邮件 ALL 发到 sf850@scarletmail.rutgers.edu。
Python 默认 `$HOME/.conda/envs/carbon/bin/python`，可用 `CARBON_PYTHON` 覆盖。

登录节点：

```bash
cd /scratch2/09796/shuyuanfan4814/carbon
git pull --ff-only
mkdir -p out
sbatch scripts/frontera_opportunity.sh
```

可选只读检查（不是必需步骤）：

```bash
bash scripts/frontera_opportunity.sh --check
```

若希望五组独立排队、一起提交：

```bash
for trace in wide-long layered-long burst-mix loose-walltime short-control; do
  sbatch scripts/frontera_opportunity.sh --trace "$trace"
done
```

**整组一次提交与五组分开提交二选一**，避免同一 seed 有两个 writer。
同一个脚本支持计算节点交互式 `bash scripts/frontera_opportunity.sh`；登录节点只有 --check/--report-only 可运行。
默认最多四个 CPU worker，不运行真实历史 trace 扫描；4h 是否充足以集群实跑日志为准。

超时或中断：保持代码和配置不变，在原命令后加 `--resume`。

```bash
sbatch scripts/frontera_opportunity.sh --resume
# 单独重启某组
sbatch scripts/frontera_opportunity.sh --trace wide-long --resume
```

每完成一个方法就原子保存 JSON；中断后只重跑尚未完整保存的那个方法。
恢复会核对输入、代码、生成 trace、到达与保存结果哈希；输入或策略改变应使用新的 --output 目录。

## 看哪些输出

日志：`out/frontera-opportunity-<jobid>.out` 与独立 `.log`。监控：

```bash
squeue -u "$USER"
tail -n 40 out/frontera-opportunity-<jobid>.out
cat results/opportunity-v1-xl-e050/summary.md
```

每个 trace 目录：

- `curves-constant.png/pdf`：横轴模型 carbon、纵轴 TAT；fixed 黑线，dynamic 黄线；CI 固定 400，隔离排队/scaling 机制。
- `curves-ercot.png/pdf`：相同执行轨迹叠加 ERCOT 历史 CI 的第二条评价曲线，未来 CI 不参与控制。
- `curves.csv`、`summary.md/json`：完整 15 个点、每种子结果、切换率、与完整 fixed 下凸前沿的同碳成本时间差。
- `seed-*/trace.json`、`cohort.json`：实际生成输入；`seed-*/episodes/*.json`：每次提交路径、开销、可见状态与全部动作分数。

主功率参数 rho=1；carbon 为 kg/κ，未知绝对 node power 没有被冒称为实测。
表中 **frontier gap < 0** 代表动态在同碳成本处位于完整 fixed 下凸前沿下方。
同时列出高于前沿、重合、超出 fixed 碳范围的点，不以一个有利 alpha 代表整条曲线获胜。
若存在未完成任务，对应均值为空且整组比较标为不完整，不能删除这些到达后宣布胜利。

## 只回传文本

本轮不产生任何模型 checkpoint。当前 `.gitignore` 已允许 JSON/CSV/MD/LOG 文本，PNG/PDF 继续忽略。
远程保持只 pull 和启动；按已有文本回传方式同步 `out/` 和本轮 results 文本即可，无需压缩包。
图可以在本地从文本直接重建：

```bash
cd /Users/shuyuanfan/carbon-latest
CARBON_PYTHON=/Users/shuyuanfan/miniconda3/envs/writing/bin/python \
  bash scripts/frontera_opportunity.sh --report-only
```

本地也可以直接运行 `PYTHONPATH=src python scripts/opportunity_report.py results/opportunity-v1-xl-e050`。
报告读取保存的输入契约，因此无需训练环境或任何二进制模型。

## 准备状态

这一入口用于验证机会能否在随机到达下形成整条曲线。代码验证与微型构造检查不等同于五组正式结果；正式曲线须由计算节点运行后读取。
