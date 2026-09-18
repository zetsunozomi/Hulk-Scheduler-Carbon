# 下一步：主策略训练与 validation 曲线

## 已经完成什么

作者回传的 Sophia 日志确认：Frontera × AMSP 7B 的完整 train fixed 共292次、114.5秒；validation fixed 共96次、51.4秒，全部 completed。由完整 train 得到四档完成预算，终端四舍五入显示为 **15.11 / 59.79 / 104.48 / 193.84 小时**。实际训练读取原始 `budget-grid.json`，不用这些取整显示值重新计算。

主 PPO pilot 已完成5轮、80个episode、125个chunk，checkpoint已保存。5轮rollout合计55.22秒。它验证了主策略无等待预测器也能训练，尚未评价 RL 的 validation 表现。125个chunk不能证明发生了规模变化。

## 本次配置与目的

只推进同一个 Frontera × AMSP 7B development panel、seed=11：

1. 校验并复用 pilot 目录下完整 fixed、references、budget-grid；不重新回放它们。
2. 从 seed 初始化主策略，预定 **64轮 × 4档预算 × 16个完整episode = 4,096 episode**。epochs=4、minibatch=16、CPU单线程，其余主 PPO 设置沿用实现。只采样 train。5轮pilot作为链路检查保留，不改它的设置后延长。
3. 预先指定第16/32/48/64轮，训练结束后依次评价完整 validation：每个checkpoint为24到达×4档=96次，共384次策略回放。保持categorical采样，checkpoint之间使用相同的配对采样seed；不改成argmax。
4. 汇总各fixed和全部四个checkpoint的实际TAT、两功率端点估算碳成本、miss率、单chunk/实际换规模/降规模比例。对已有fixed validation解Fixed-Mix线性规划，不增加回放任务。

这是第一轮较长的开发训练和学习曲线，**不声称4096足以收敛**。此时不做自动checkpoint选优，不读取test，不自动启动另五个panel、多seed或新实验组。后续完整验证仍需规划基线、Precommitted-RL及既定的冻结环境检查。正式论文结果槽继续保留。

## 为什么仍在集群运行

这里只运行CPU历史需求流回放和小型调度策略优化，AMSP已提供LLM训练速度输入，不训练LLM、不测机器功耗。按pilot的平均episode耗时粗算，4096episode的rollout约47分钟；批次从4个增到16个到达时间后缓存命中会改变，另有优化和validation，**不能承诺一小时完成**。现在可在Sophia登录节点直接qsub申请2小时batch，也可继续使用已有interactive allocation；按轮checkpoint支持跨allocation恢复。

## 环境、同步和启动

单层入口 `scripts/sophia_main_train.sh`。**迁移机器重点改**：

- 顶部 `CARBON_PYTHON`，当前默认 `/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python`，已有export可覆盖。
- 顶部PBS资源/account/queue：已按作者提供的成功作业186384及当前队列信息填入Local-LLM、by-gpu、1:ngpus=1:ncpus=32:mem=120gb、place=pack:shared、filesystems=home:eagle；walltime设为02:00:00，当前队列上限24h。这是获批的资源规格，程序仍CPU单线程；未改变训练配置。
- 同一脚本顶部的默认CONFIG/PILOT/OUTPUT支持无参数启动。迁到Slurm时按新站点填写SBATCH参数与module，不照搬Sophia。已有export覆盖Python路径仅适用于直接bash；qsub传入环境覆盖值使用-v，例如-v CARBON_PYTHON=/path/to/python。

现有PyTorch 2.14.0保留，即使装的是CUDA分发，当前模型和张量仍在CPU。以后空环境首次安装改用[PyTorch官方CPU wheel源](https://pytorch.org/get-started/previous-versions/)，不会为了当前运行卸载/重装torch。Fixed-Mix需要此前E1环境已有的SciPy；缺失时在训练开始前明确报错。

本机手动Git三连（助手未自动提交/推送）：

```bash
cd /Users/shuyuanfan/carbon-latest
git add scripts/main_train.py scripts/sophia_main_train.sh scripts/sophia_main_pilot.sh tests/test_main_train.py docs/MAIN_TRAIN_RUN.md docs/MAIN_PILOT_RUN.md docs/CLUSTER_RUNBOOK.md
git commit -m "Train the main policy and compare declared validation checkpoints"
git push
```

### PBS batch：在登录节点提交

务必先进入Sophia的carbon-latest仓库根目录（能看到configs/、src/、scripts/）。然后：

```bash
git pull --ff-only
qsub scripts/sophia_main_train.sh
```

脚本内已写好PBS头与本次配置路径，无须附带位置参数或再加一层wrapper。默认配置为configs/amsp-frontera-7b.development.json，复用results/amsp-main-frontera-7b-pilot，输出results/amsp-main-frontera-7b-seed11。成功提交会打印job ID，可用qstat -u "$USER"查询状态。提交成功不等于训练已经开始。

输出与错误合并，并按[ALCF PBS文档](https://docs.alcf.anl.gov/running-jobs/)使用-k doe写入输出目的地。默认日志名为carbon-main.o<数字jobID>，每次提交各自保留；准确位置可从qstat -f JOBID的Output_Path字段查询。

### Interactive：在已有compute allocation内运行

```bash
bash scripts/sophia_main_train.sh
```

原来的显式参数方式继续支持：

```bash
bash scripts/sophia_main_train.sh \
  configs/amsp-frontera-7b.development.json \
  results/amsp-main-frontera-7b-pilot \
  results/amsp-main-frontera-7b-seed11
```

bash将PBS头视为注释，不会自行申请或提交作业。批处理时从PBS_O_WORKDIR定位仓库，避免PBS的spool副本改变相对路径。

本次未改 `src/carbon/*.py`，因此上次pilot/固定基准的代码哈希继续匹配。两个结果目录都要保留。新流程在 `run-plan.json` 绑定配置、输入、核心代码、编排脚本、torch版本、完整设置和四个评价轮次；不悄悄接受变化。

## 中断与恢复

若因时限中断，在登录节点重新提交并开启恢复：

```bash
qsub -v CARBON_RESUME=1 scripts/sophia_main_train.sh
```

已有interactive allocation内则：

```bash
bash scripts/sophia_main_train.sh --resume
```

也支持在原来的三个显式路径后加--resume。两个入口都传入同一个Python恢复开关；首次启动默认不会自动接管已有输出目录。

训练每轮保存模型、优化器、随机状态和dual；从最后完成轮继续到同一64轮上限，未完成轮重跑。每次训练恢复写入新的 `ppo-attempt-NNN`。已完成的validation checkpoint校验后跳过；若在某个checkpoint的96次评价中断，该评价目录改名保留，仅重跑这个checkpoint的评价。结果没有被覆盖或静默拼接。

## 完成后看什么

```bash
cat results/amsp-main-frontera-7b-seed11/validation-summary.md
```

把这份表带回即可开始判断方法；保留同名JSON以及完整原始目录以供后续审查。

- `validation-summary.md`：全部fixed、四轮PPO和Fixed-Mix。表中C是两端点各自除以train Fixed-4参考量后取较大值，不是实测排放。
- `validation-summary.json`：还包含两端点原始估算成本、node-hours、上下调规模比例和逐轮耗时。
- `ppo-attempt-NNN/training.jsonl`：损失、dual、训练均值与全部64轮checkpoint索引。
- `validation-000016/`等：paired validation的episode/chunk原始数据及哈希。

固定策略的miss按每个新D重新计算，不能沿用旧固定回放的默认deadline标志。有未完成任务时保留全体分母和miss上下界，整体成本/TAT留空，不只平均完成任务。Fixed-Mix是同一validation上拟合的期望值，不作为独立测试结果。24个validation到达的经验miss≤5%也不是统计可行性保证。

判断重点：在相同预算约束下有无成本改善、实际miss是否可接受、收益是否超过Fixed-Mix、动作是否真的在后续chunk改变。单纯选择不同首发规模或者多跑几个chunk，均不足以支持动态反馈的核心贡献。
