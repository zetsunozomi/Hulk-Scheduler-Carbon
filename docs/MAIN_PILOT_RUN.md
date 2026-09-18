# 完整固定基准与主 PPO 开发运行（已完成）

作者已回传整条流程完成日志：train fixed 292次/114.5秒、validation fixed 96次/51.4秒，主PPO 5轮80episode/125chunk。下一步见 [主策略训练与validation曲线](MAIN_TRAIN_RUN.md)。以下保留本次pilot的配置与恢复说明。

## 已完成和这次的目的

Frontera E1 已完成。固定基准 hash 分片 0/16 的 3 个训练到达时间、12 次完整任务全部完成，用时 5.7s。chunk 数 5/2/1/1 与既定 48h 请求上限和每次 600s 开销相符：U=ceil(192*q4) 是约192小时的有用计算，4个含开销的48h chunk容不下全部工作，故Fixed-4有第5个尾块。

这三个样本中 Fixed-64 的 TAT 和两端点估算成本都低于 Fixed-128；Fixed-4 成本最低、时间最长。它们是开发分片结果，不代表完整日期或动态策略效果。

这次只运行 Frontera × AMSP 7B，顺序自动串联：

1. 完整 train 固定基准：73 个到达时间 × 4 种规模 = 292 次回放。
2. 生成完整 train 的 T_ref、成本参考量和四个 slider 档位。
3. 完整 validation 固定基准：24 个到达时间 × 4 种规模 = 96 次回放。
4. 主 PPO：seed=11，5 iterations，每轮4档预算×4个到达样本，共80个episode；epochs=4、minibatch=4，单CPU线程。每轮保存checkpoint。这个短pilot检查训练链路、完整回报、优化与耗时，不作为正式RL效果或最终iteration预算。

原来12个分片结果保留供核对；完整基准采用统一目录重新覆盖完整cohort，包括那12个组合（此前耗时5.7s），不把分片当完整参考量。T_ref只从train取，validation不参与预算或归一化设置。暂不读取test、不跑多seed或另外五个panel。

## 必须修正的 slider 范围

旧CLI通用默认 beta=[1,1.25,1.5,2] 可能过窄：以回传分片的T_ref≈13.62h举例，最大预算≈27.23h，而小规模首个非终端chunk接近48h。这样会把合理的多段缩放压到预算之外。

本次用完整train确定：

- 左端 T_ref：各固定规模平均TAT的最小值。
- 右端 D_hi：训练 Fixed-4 TAT 的第95百分位，使用 inverse empirical CDF，即排序后 ceil(0.95*N)-1 索引。
- 四档 s=[0,.25,.5,1]，D(s)=T_ref+s*(D_hi-T_ref)，beta=D/T_ref。

该标尺只使用train；真实结果中的miss照常报告，右端不保证95% validation/test任务按期完成。不增加档位/实验组，不改请求上限、开销、工作量，不重跑E1。所有正式方法、seeds及Precommitted-RL最终使用同一冻结网格。若 D_hi<=T_ref 则报错要求审查，不偷偷重选百分位。正式运行必须显式传入生成的beta值，不能误用通用CLI默认网格。

## 环境和同步

独立的一层 `scripts/sophia_main_pilot.sh` 使用：
`/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python`。
可用 CARBON_PYTHON 覆盖。脚本检查PyTorch；缺失或版本不兼容时，会按照 requirements-p3.txt 从官方CPU wheel源安装到同一环境；已有兼容torch保持不动。版本记录在checkpoint，恢复时必须一致。计算是CPU队列回放和小型调度策略训练，AMSP提供LLM速度输入。

本机手动同步（本助手不自动commit/push）：

```bash
cd /Users/shuyuanfan/carbon-latest
git add src/carbon/main_pilot.py src/carbon/__main__.py tests/test_main_pilot.py scripts/sophia_main_pilot.sh docs/MAIN_PILOT_RUN.md docs/NEXT_FIXED_RUN.md docs/CLUSTER_RUNBOOK.md
git commit -m "Run full fixed references and a bounded main PPO pilot"
git push
```

Sophia 仓库 `git pull --ff-only` 后，在PBS interactive allocation内运行：

```bash
bash scripts/sophia_main_pilot.sh \
  configs/amsp-frontera-7b.development.json \
  results/amsp-main-frontera-7b-pilot
```

脚本不调用另一层shell，也不调用qsub/sbatch。PBS batch资源/account/queue按本站已确认配置填写；这里不编造这些参数。5.7s分片不能保证全量耗时按比例增长，但已支持从计时分片推进这批完整固定回放和短PPO。

## 中断恢复和输出

若allocation结束，保留目录，在下次allocation内原命令末尾加 `--resume`：

```bash
bash scripts/sophia_main_pilot.sh \
  configs/amsp-frontera-7b.development.json \
  results/amsp-main-frontera-7b-pilot --resume
```

已完成的固定stage校验完整cohort和文件哈希后跳过。固定stage内没有细粒度checkpoint，若中断，原目录改名保留并重跑该stage。PPO从manifest中最后完成的一轮checkpoint继续，保持同一5轮/80episode上限；正在采样或优化的未完成轮会重跑。每次PPO恢复写入新的ppo-attempt-NNN目录。输入、代码或PyTorch版本改变会拒绝恢复，不能为了续跑放松这些检查。

完成后带回终端最后的slider小时数、两行Fixed阶段耗时、PPO各轮日志和 `pilot-summary.json`。保留全目录。关键文件：

- fixed-train/、fixed-validation/：完整固定对照、每chunk日志和stage-seal。
- references.json、budget-grid.json：正式网格的训练来源与小时数；protocol整体冻结前仍为development。
- ppo-attempt-NNN/checkpoint-*.json 与 .pt：网络、优化器、随机状态和dual。
- ppo-attempt-NNN/training.jsonl、episodes.jsonl、chunks.jsonl：训练耗时、损失和实际动作序列。
- pilot-summary.json：固定曲线数据、预算标尺、最后checkpoint与逐轮耗时。没有进行checkpoint选优或test比较。

下一轮依据完整固定曲线、动作是否真的跨chunk变化，以及PPO rollout/update耗时，确定正式训练的iterations和资源。5轮结果不好不能直接否定方法，5轮看起来好也不能当论文结果。
