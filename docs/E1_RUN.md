# 下一步：Sophia 完整 E1

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

若从 login 节点提交，把 `bash` 换为 `sbatch`，并将脚本顶部 account/partition/qos/constraint 按已跑通的 Sophia 设置填写。资源默认仍为 1 CPU 核、16 GB、2 小时；这是初始申请上限，不是完成时间保证。interactive 会使用已有 allocation。

## 带回什么

保留整个 `results/amsp-e1-*`。先带回完成/报错日志，以及：

- `train-probes/manifest.json` 和 `validation-probes/manifest.json`：新增空闲快照比例、后台提交数、按规模和请求长度的等待摘要；占用率是快照平均，不是时间积分利用率。
- `validation-diagnostics/metrics.json`：误差、coverage、tail/rank。
- `dependence-queue/audit.json`：日期相关性诊断。

原始探针和 model/model.json 保留供复核与下一阶段使用。失败保留现有目录，重试使用新输出名。若完整训练段也缺少忙碌样本，应报告这个适用范围，不重选日期来保证正向结论。
