# 下一步：Sophia 完整 E1

**等待预测的角色（2026-09-17 更新）：** E1 服务于规划基线与场景诊断，输入/探针/模型合同未变。主 ScaleDown 不使用显式等待预测；E1 模型仅供正式规划基线。E1 报告误差与尾部，不设置预测必须准确的门槛，也不把高 coverage 当作 deadline 保证。

两份 preflight 均回传 64 条已完成探针。Frontera 的两天全部等待显示 0；源文件核对发现 2020-01-05/06 没有后台提交，下一条在 01-07。这个窗口可检查链路，但不代表忙碌队列或全段耗时。不更改 split、不删空闲日期、不为得到非零等待放大负载。

## 这次运行什么

每份需求流各运行一套：完整 train 探针 → GBT 等待模型 → validation 探针 → 预测误差/尾部/规模排序 → 日历相关性。三个 AMSP 模型共享同一来源的等待模型，只用 7B 配置作队列入口。不会训练 RL、访问策略 test 结果或测 GPU 吞吐/功率。

| 来源 | train 探针 | validation 探针 |
|---|---:|---:|
| Frontera | 11,520 | 4,416 |
| IW | 11,616 | 4,416 |

数量按当前日历计算：每 6 小时一个快照 × 4 个节点规模 × 12/24/48 小时三种请求长度。完整 E1 的成本不能由空闲窗口的 0.78 秒线性推算。

## 同步与依赖

本机改动可手动 git add / commit / push，然后在 Sophia 仓库 `git pull --ff-only`。

`scripts/sophia_e1.sh` 已内置作者提供的环境路径。新增等待拟合依赖仍安装进同一个环境：

```bash
/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python -m pip install -r requirements-p2.txt
```

## 提交命令

已在 interactive compute 节点时，从仓库根目录依次执行：

```bash
bash scripts/sophia_e1.sh configs/amsp-frontera-7b.development.json results/amsp-e1-frontera
bash scripts/sophia_e1.sh configs/amsp-iw-7b.development.json results/amsp-e1-iw
```

Sophia 的实际作业系统是 PBS（日志为 `qsub ... sophia-pbs-01`）。上面的 `bash` 使用当前 interactive allocation，不会再提交作业。不要在 Sophia 把它换成 `sbatch`。批量提交时，先在脚本顶部填入本站已确认的 `#PBS` 资源/account/queue 指令；这里不臆造站点参数。移植到 Slurm 站点时再采用相应 `#SBATCH` 设置。计算需求为 CPU 回放/拟合，无需 GPU；一小时是 allocation 上限，不是运行完成时间保证。

## 带回什么

保留整个 `results/amsp-e1-*`。先带回完成/报错日志，以及：

- `train-probes/manifest.json` 和 `validation-probes/manifest.json`：新增空闲快照比例、后台提交数、按规模和请求长度的等待摘要；占用率是快照平均，不是时间积分利用率。
- `validation-diagnostics/metrics.json`：误差、coverage、tail/rank。
- `dependence-queue/audit.json`：日期相关性诊断。

原始探针和 model/model.json 保留供复核与下一阶段使用。探针被中断时保留现有目录，按下面的 `--resume` 续跑。若完整训练段也缺少忙碌样本，应报告这个适用范围，不重选日期来保证正向结论。

## 一小时作业中断后续跑

旧版（644e841）没有 probe checkpoint，但 `probes.jsonl` 每行写入后都会 flush。`manifest.json` 可能仍显示 running/0 行；`qsub ... completed` 只表示 allocation 结束，不代表 E1 成功。2020-08-08T00:00 的 10,380 行约占 Frontera **train** 的 90.1%，尚不含后续 4,416 条 validation，也尚未训练 RL。

更新代码后，在下一次 interactive allocation 内，从仓库根目录运行（使用原 CONFIG 和原 OUTPUT）：

```bash
bash scripts/sophia_e1.sh \
  configs/amsp-frontera-7b.development.json \
  results/amsp-e1-frontera --resume
```

这是同一个独立 `.sh` 入口，不再套另一层 shell，也不调用 qsub/sbatch。Sophia Python 路径仍为 `/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python`；可用 `CARBON_PYTHON` 覆盖。原先若指定了探针间隔，续跑时也要保留，例如 `OUTPUT 21600 --resume`。不要把 start 改成最后日期，不要删除原目录。

恢复行为：

- 校验原配置、输入哈希、探针网格、源码兼容性和已有记录的连续顺序。兼容已部署的 644e841；无关的主策略修改不会强制重算探针。
- 复用全部完整行。只有 EOF 的不完整 JSON 尾巴会在备份后移除；完整但损坏的行、缺行、重复行或配置变化会报错。
- 从原始起点重建后台状态，并比对已存 features，随后只计算缺失的探针。**这不是内存快照秒恢复**，重建前缀仍需要 CPU 时间；不会重新计算已保存的等待标签。
- 每个快照更新 manifest，结束时生成完整数据哈希。旧 manifest 与截断尾巴保存在 `resume-history/`，记录每次续跑的软件版本和保留行数。同一目录有写入锁，不要并发启动同一输出。
- 整个 E1 可重复使用 `--resume`：完整阶段校验后跳过；未完成的拟合/诊断目录改名保留后重跑该阶段。GBT 拟合没有中途模型 checkpoint。train/validation 探针均按行恢复。

单独恢复一份探针时，底层入口为 `python -B -m carbon probe-waits --config CONFIG --output OUTPUT/train-probes --split train --resume`（沿用原间隔/请求长度/start/stop）。上述 E1 脚本会自动串起剩余阶段，无需手动拆步骤。
