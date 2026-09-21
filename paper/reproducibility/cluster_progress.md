# 集群执行进度

## 最新：2026-09-18，首轮主训练因PBS时限中断，尚无validation结果

证据：本机代码仓库carbon-main.o186777完整65行终端日志，作者确认只拉回日志、尚无results目录。PBS明确报告“walltime 7229 exceeded limit 7200”。本次不是成功跑完64轮。

- 最后打印完成轮为55/64，累计3520个episode、3691个chunk；代码在checkpoint和manifest写入后才打印这一行，因此日志支持“至少已保存第55轮”。具体磁盘完整性须由恢复入口核验，未在本机读取.pt或manifest。
- 还剩9轮、576个预定训练episode；之后仍按原计划评价16/32/48/64轮各96个validation episode。当前流程将评价放在64轮训练之后，日志中没有任何validation结果。
- 第16–55轮合计2560个episode、2562个chunk，只有两轮各多出一个chunk；第39–55轮合计1088个episode、1088个chunk。已完成训练episode至少包含一次分配，因此最后17轮全为single-chunk。它不证明所有任务选择同一规模，但明确未在这些任务内部执行动态scale变化。
- 尚无每档budget的动作分布、成本、miss、entropy/dual等记录，不能推断策略已优于fixed，也不能仅凭控制台计数归因于实现bug、探索不足或环境不存在动态机会。
- 之前从4-arrival短pilot外推的47分钟量级偏乐观。当前批次为16-arrival，replay cache容量仍为8，存在跨预算重复重建前缀的成本；没有分阶段时序文件前，不将其断言为全部耗时来源。按已消耗墙钟/55轮粗算，剩余9轮约20分钟，另加评价，仅用于量级估计。

下一步沿用同一代码/torch/输入/4096episode上限，登录节点提交 `qsub -l walltime=04:00:00 -v CARBON_RESUME=1 scripts/sophia_main_train.sh`。4h只覆盖PBS墙钟上限，不延长训练预算；结束后返回pilot和seed11两目录，核对完整validation、固定对照及真正换规模的情况。结果在.gitignore的/results/内，普通Git同步不会传回，须单独复制。当前不扩展panel/seeds、不改损失或靠缩短chunk制造动态动作；先完成这次已声明的对比。

日志SHA256：`63a529070f8e9e1ba92fa45d0071c9a020e6658cba4a25de66f629e416381dd2`。

## 前一阶段：完整 fixed 与无等待预测器 PPO pilot 完成

证据来源：作者附件终端日志（2026-09-17），尚未读取远端pilot-summary.json或episode/chunk原始文件。原环境装入PyTorch 2.14.0及默认CUDA分发依赖，但本程序在CPU上训练；现有安装保留，后续新环境改从官方CPU wheel源安装。

- 完整Frontera × AMSP 7B train：73到达×4fixed=292次，全部完成，114.5s。
- 完整validation：24到达×4fixed=96次，全部完成，51.4s。
- 每个fixed的chunk数仍为5/2/1/1；完整cohort的平均TAT与碳成本未包含在此次终端输出，不能用之前3样本均值替代。
- 四档预算的显示值为15.11/59.79/104.48/193.84h，来自完整train。实际运行始终读取budget-grid.json未取整值。
- 主PPO seed11完成5轮、80episode、125chunk，每轮checkpoint已保存；rollout各27.00/6.89/2.15/9.41/12.77s，共55.22s。125chunk不是发生规模变化的证据。
- 输出目录results/amsp-main-frontera-7b-pilot，最后checkpoint为ppo-attempt-000/checkpoint-000005.json及匹配.pt。

下一计算批次已准备：同一panel/seed，从seed初始化64轮×4档×16episode=4096episode；随后对预定16/32/48/64轮各跑完整validation 96次，共384次评价。epochs=4、minibatch=16，无wait predictor，不访问test；此为较长development训练，不声称收敛。复用全部fixed/reference/grid；仅对已有fixed validation解Fixed-Mix LP，无额外回放。不会从5轮pilot改设置延长，也不自动扩至多seed和其他panel。

单层scripts/sophia_main_train.sh已准备，按轮恢复训练，按checkpoint恢复评价，输出validation-summary.md/json，保留全部预定checkpoint的miss、TAT、碳成本和真正换规模/降规模比例。核心src/carbon文件未改，旧artifact代码哈希兼容。根据pilot平均耗时估算rollout约47分钟，仅供量级参考；增大批次影响缓存，另外还有优化与评价，不保证1h结束。作者在Sophia compute allocation手动运行，无新增硬件测量。

Sophia batch提交参数已由作者提供的qstat确认：成功作业186384使用Local-LLM/by-gpu、select=1:ngpus=1:ncpus=32:mem=120gb、place=pack:shared、filesystems=home:eagle。启动脚本已写入同一资源规格并将时限设为2h（当前by-gpu上限24h），可在仓库根目录直接qsub scripts/sophia_main_train.sh；恢复使用qsub -v CARBON_RESUME=1 scripts/sophia_main_train.sh。程序仍CPU训练，既定4096episode和四个validation checkpoint不变。保留无参数bash及原三个路径接口。9项离线启动检查通过；没有提交远程作业。

以下均为历史阶段记录；“下一步”只代表当时进度。正式论文结果槽继续为空。

## 前一阶段：Frontera 7B 固定计时分片完成

证据来源：作者回传 Sophia 终端输出，Python 3.11.14。预先确定的三个 train 到达时间、四种固定规模共12个结果全部 completed，无删失；回放耗时5.7s。尚未读取远端逐chunk原始文件。下面是开发分片摘要，不填正式论文结果槽。

| 固定规模 | chunk数（每个到达） | 平均TAT (h) | 平均node-hours | modeled carbon/kappa，rho=.25 / 1 |
|---|---:|---:|---:|---:|
| 4 | 5 | 192.99863 | 771.36047 | 274889.5630 / 274889.5630 |
| 16 | 2 | 50.27733 | 796.96414 | 289848.6300 / 296478.5987 |
| 64 | 1 | 13.61701 | 871.48886 | 297769.8591 / 323961.7182 |
| 128 | 1 | 24.83417 | 983.57423 | 305781.4534 / 360326.2532 |

此分片Fixed-64在平均TAT和两功率端点成本上支配Fixed-128；Fixed-4成本最低。不能推广到完整日期，也未运行动态策略。Fixed-4的第5个chunk来自U=ceil(192*q4)与每次600s开销，四个含开销的48h块不足以装完192h有用工作。

训练前纠正预算跨度：通用默认上限2*T_ref在这个分片仅约27.23h，小规模非终端chunk却接近48h。下一批由完整train确定T_ref和Fixed-4的inverse-ECDF p95，保留s=[0,.25,.5,1]四档。无新增实验组、请求上限/工作/开销/E1输入改动；右端不保证validation/test可行。具体小时数与beta尚待完整train回放，不能使用这3个样本直接冻结。

下一计算批次已准备：完整train固定292次、validation固定96次，自动生成训练参考量与budget-grid.json，再运行主PPO的5轮80episode开发pilot（seed11，无等待预测器）。单层sophia_main_pilot.sh使用原环境，缺失/不兼容torch时按requirements-p3安装；阶段恢复与PPO按轮checkpoint已有合成测试。正式iterations与多seed训练尚未开始。

## 前一阶段：作者回传 Frontera 完整 E1

证据来源：本对话中的终端输出，末尾明确为 `E1 queue stage complete: results/amsp-e1-frontera. This is not RL training.`。尚未读取远端 manifest/模型/诊断文件，不填论文结果图表，不判断等待模型准确率。

末尾摘要对应当前配置的 validation：368 个快照 × 4 种规模 × 12/24/48h 请求长度，共 4,416 条探针，无删失。窗口内后台提交 6,618，空快照 0/368，快照平均运行节点比例 0.5307；后者不是时间积分利用率。

| 节点规模 | 平均等待范围（h，三请求长度） | 最大等待（h） | 有正等待的快照 |
|---|---:|---:|---:|
| 4 | 0.6881–0.7101 | 20.9667 | 47/368 |
| 16 | 1.30625 | 21.1833 | 66/368 |
| 64 | 6.6386–6.6397 | 44.85 | 186/368 |
| 128 | 38.25285 | 63.4833 | 368/368 |

解释范围：在此构造场景，申请规模增大明显提高等待代价。128 节点请求需要整集群可用；平均占用约一半并不意味着这样的请求可以立即开始。表格不代表固定策略完成整个任务的 TAT/碳成本，不支持动态策略已优于 fixed 的结论。

当前下一步：Frontera × AMSP 7B，使用已有 hash 分片 0/16，三到达时间（2020-01-29 04:00、02-13 16:00、06-06 17:00 UTC）× Fixed-4/16/64/128，共 12 次完整任务回放。全部 train 共 73 到达时间。先检查运行耗时和 chunk/成本，再安排完整参考量及 PPO；主策略无需等待模型。独立 Sophia 启动脚本为代码仓库 scripts/sophia_fixed.sh，使用作者已有 Python 环境，不请求 GPU 或新增硬件测量。

以下为此前阶段记录；旧的“尚待完整 E1”仅反映当时状态。IW 当前仍只确认 preflight，完整 E1 尚未回传。

## 作者回传：Sophia 两份 preflight

证据来源：作者粘贴的终端输出；尚未读取集群上的结果文件或 Sophia 本地 preflight 脚本。

- Frontera：2020-01-05 00:00 至 01-06 18:00，每 6 小时一个快照；rows=64、censored=0，打印完成。
- IW：2023-02-01 00:00 至 02-02 18:00，同样 rows=64、censored=0，打印完成。
- Frontera manifest 摘要的探针阶段耗时为 0.7810401250608265 秒；按节点汇总的平均/最大等待都显示 0.000 小时。该耗时不包含之前的完整输入校验，也不代表忙碌窗口成本。
- 作者提供 Sophia 环境目录：`/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon`；下一阶段脚本使用该目录的 bin/python。

## 本地只读源文件检查

按配置中的清洗规则读取源文件，未在本机运行真实 trace 回放或策略实验。

- Frontera 的 2020-01-05/06 没有保留后台提交；训练段第一条提交为 2020-01-07 10:54:12 UTC。零等待有源数据层面的合理解释，不能据此把整个需求流判断为空。
- Frontera 完整训练段：125,507 条保留提交，240 个日历日中 234 日有提交。IW 原 preflight 两天有 220 条保留提交。
- 不修改原时间划分、不删除空闲日期、不提高负载来保证出现等待或策略增益。
- 没有把没有新提交等同于没有后台工作；残留作业仍要由连续回放状态判定。

## 已接好的下一阶段

代码仓库新增 scripts/sophia_e1.sh 与 docs/E1_RUN.md。两个来源分别运行完整 train/validation 等待流水线，三个 AMSP 工作负载共享同来源预测器，不按模型重复采集。当前六个 development 配置保持不变。

probe manifest 新增队列快照摘要和按规模/请求时长的等待摘要；全空闲快照明确提示且保留样本，不按运行结果重选窗口。17 项相关测试通过，包含空队列与无新提交但有残留作业的区别。仅完成软件检查，尚未提交完整 E1。

尚待：完整 E1、忙碌段耗时、拟合诊断、固定与规划基线、PPO、冻结评估及压力测试。新论文结果槽保持空白。
