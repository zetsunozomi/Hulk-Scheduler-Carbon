# 三轮稿件审计（2026-09-15）

## 审计范围

已阅读任务【功率模型】和当前主稿、分章节源文件、原reject.log及功率三层说明。
旧“只看Fixed-4/32”计划与本次提升paper的目标不一致，已归档并替换。
本轮只把论文和实验设计推进为候选稿，未跑E1–E4，也未声称新RL已经实现。

## Round 1 — 重构问题、贡献与证据

| 审稿问题 | 实际修改 |
|---|---|
| 相对Fixed-32的gain主要来自少用节点 | 完整fixed + 可执行Fixed-Mix + MPC，主指标以最强validation feasible non-RL为分母 |
| alpha缺少用户语义 | D小时预算 + 显式miss tolerance，保留不支持的预算 |
| 没有解释adaptive何时可能赢 | fixed-hull null proposition；把queue/CI/chunk边界作为机会来源 |
| node power无测量但原文暗示measured/calibrated | 保留三层估算；Pspec/kappa/s区分；没有现场减碳结论 |
| RL像装饰，未对齐实际决策 | 加remaining work/budget/action forecast，gamma=1，terminal miss且继续完成任务 |
| 新增实验无限膨胀 | 收敛到E1–E4，所有图表和数值为空槽 |

## Round 2 — 方法反例与公平性

逐条检查而不是只润色：

- 初始b/D总等于1，无法区分预算：改为b/Tref和D/Tref两个输入。
- final chunk duration和requested walltime必须进入predictor；work按整数updates推进。
- checkpoint必须在walltime前落盘；不能由外层wrapper在任务死后“保存”。
- replay history window不等于完整scheduler state；连续chunk不重置队列。
- probe逐action独立fork，actual policy不使用probe或未来真实CI。
- 最大endpoint目标是expected episode cost；不在每个chunk单独选择不同的adversarial rho。
- chance constraint是population目标，PPO不承诺SLA；argmax会改变策略，保持categorical evaluation/deployment。
- MPC必须计入过去已产生的endpoint cost，并使用相同reference归一化。
- Fixed-Mix可行性按miss probability混合；不能按mean TAT连线推SLA。
- 剩余work可从partial action特征推断：不再把仅删除raw U当成删除work信息。最终用Plan-once替代该较弱消融。
- 报告single-/multi-chunk比例；不能靠任意拉长U制造RL机会。
- 旧test已参与修改，必须新的未触碰holdout，不能重命名旧test。

## Round 3 — 反例、功率稳健性、复现与表达

- 发现chance deadline下，即便常CI零wait，预先within-run mixing也可能胜per-run Fixed-Mix；修正引言对null命题的过强推论。
- 加入复用MPC的Plan-once基线，替换一次需要重训的state消融；online replanning增量与单纯混合规模分开。

- 增加constant-CI energy的仿射性质：rho不会自动翻转原node-hour能耗排序。
- 端点比较必须固定policy和comparator；ratio-of-means需positive affine reference。
- 增加独立scale系数误差的box worst-case表达式，仍只有三层模型；无需节点测量或重新replay。
- endpoint统计推断用联合单侧区间；不能把点估计的代数结论称为实证保证。
- 明确固定日志constant-CI重算与重新跑policy的区别。
- 数据来源、代码、配置、seeds、正常/失败episode、censor和发表时可支持claim统一进入manifest。
- 主稿入口与分章节同步；历史图片不再进入新主图；Abstract/Intro/Conclusion共享E1–E4槽位。
- 检查全部正文引用、LaTeX标签、数学恒等式、编译日志与渲染版面；详见tools/check_manuscript.py和最终QA记录。

## 原拒稿意见的闭环

| 原问题 | 状态 | 剩余内容 |
|---|---|---|
| fixed/heuristic baselines缺失 | 设计已修正 | E2结果待填 |
| RL与预测器必要性 | 主张已收敛，便宜predictor，强planner | E1/E3结果待填 |
| 静态power不可信 | 模型边界+exact conditional sensitivity已写 | E4结果待填；不需要实测节点power |
| gradient accumulation等同quality | 删除过强claim、细化执行合同 | E1短correctness结果 |
| multi-user / facility收益过度 | 删除对应claim | 当前不需要adoption实验 |
| California主结果选择性 | Texas主场景，年份/发布时序明确 | CI manifest待填 |
| 随机性与选择偏差 | paired blocks + seeds + validation锁定 | 实施E2 |
| 新模型泛化 | 收窄到现有profiles | 当前不增加现代LLM |
| 普通用户部署未经证实 | 明确intended interface/simulation design | 不冒称生产prototype |
| trace replay真实性 | recorded replay与simulated probe分开 | E1验证，保留model-conditional范围 |

## 为什么在这里停止继续改稿

在不拿到新实证结果的前提下，继续添加模块、oracle、workload或地区主要增加成本，
不会解决剩余的不确定性。现在的主张、方法、指标和验证路径已经形成闭环；
纯写作与算法定义上的主要硬伤已修正。后续最有价值的提升是填E1–E4，而不是再改名称和措辞。

这不是“已达到录用水平”的保证。最强剩余风险是：
1. 新策略可能仍输给Fixed-Mix或MPC；
2. 实际long-run/queue/CI机会可能很小；
3. replay或旧profile不足以支持生产层面的推论；
4. 功率结论可能只在窄区间成立。

## 拿到结果后的分支

- **RL同时胜fixed与MPC且feasible**：保留当前学习主线。
- **MPC胜fixed但RL无增益**：将Rollout-MPC升为ScaleDown主控制器，移除RL必要性主张。
- **只有某些power/queue regime有效**：以明确有效范围作为结论，不再扩大通用性。
- **adaptive不胜fixed**：取消adaptive-superiority主张，重写成机会边界研究；当前不能仅靠写作变成顶会实证论文。

## 最终验证记录

- 主PDF：7页，References从第7页开始。
- 22个LaTeX labels与17个引用键全部解析。
- 无undefined citations/references、无overfull box；两个underfull hbox断行提示已目视检查。
- 已检查最终7页PNG，未见裁切、重叠或表格溢出；没有压缩字号或更改AAAI模板来凑页。
- 1000组人工生成输入核查fixed hull、exposure仿射、endpoint ratio、独立scale误差box与常CI能耗恒等式。它们是数学QA，不是论文实验。
- 具体记录：reproducibility/final_qa.json。E1–E4结果仍为空。


## 2026-09-17：AMSP／历史需求场景收尾

另见 `reproducibility/research_review.md` 四轮自审。AMSP v2 公开曲线替代 Qwen；严格使用 4/16/64/128 个八卡节点的已发表点。历史流分别定义后台需求，新的 128 节点场景从固定前缀空状态演化，旧 admission 不作为新平台真值。只增加一个预算、7B、两来源、full/MPC 的四项环境变体，冻结已有模型与真实小时预算。功率参考改为明确的归一化单位。

作者确认旧数据曾做 train/validation 二分。本轮三段划分明确标为 retrospective_temporal，保留 test_is_untouched=false；新日志不是启动前提。六个 development 配置已完成真实文件只读校验；尚未运行真实规模回放与训练。135 项程序测试、14 个公开柱形输入提取核验、shell 语法与 dry-run 均通过。依赖清单原被 *.txt 规则忽略，已修正以支持手动 Git 同步。

PDF 为 8 页，参考文献从第 7 页开始；全页缩略图与关键页视觉检查通过。E1–E4 设计/结果仍为明确占位，不能将输入提取和程序测试当成论文结果。下一步是目标集群短窗口 CPU preflight。按作者后续偏好，Python 和 Slurm 参数已内置到各 scripts/cluster_*.sh 顶部，同一脚本支持 bash 交互执行与 sbatch 提交；外层提交脚本和环境模板已移除。7 个脚本语法检查与模拟 Slurm 副本路径的启动检查通过，未实际提交作业。


## 2026-09-17：准确等待预测不再是主方法前提

作者指出 user-level 只改变请求，无法掌控隐藏 Slurm admission。复审确认：原 RL 的 realized reward 不能证明它不依赖输入中的 wait mean/p90/predicted exposure；residual distribution 也不自动覆盖模型失配。

- 主 actor/critic 删除三项显式等待派生输入，保留公开队列、实际剩余工作/预算、动作物理描述，新增相对提交时刻的 28 个六小时 CI 均值。
- 代码默认 wait_features=none，不加载 predictor artifact；训练和 run-policy 不要求该路径。特征 schema 升为 v2，旧 checkpoint 不混用。
- 等待模型保留给 MPC/Plan-once/辅助对照。E1 合同未动，正在执行的任务继续；不设 accuracy gate。
- E3 用 Predictor-advised 替换 Current-CI，不叠加消融。训练匹配主策略完整预算网格、arrival sampling、seeds 和总交互预算；机制评估限每 panel 一个预设预算。
- critic 仍学习 queue→cost 关系，不能据去掉 predictor 宣称任意 queue robustness。已有 frozen width/FCFS 检查这个剩余依赖，任何用户侧方案都不能保证 admission 被无限拖延时仍按期完成。
- 仅 advised 有收益时收缩无显式预测依赖的性能主张；所有结果继续留槽，不用实现性质代替实验。

140 项合成软件测试通过，15 个代码/文档文件经哈希核对安装到本机 carbon-latest；未操作 Git 提交/推送，未启动真实日志实验。论文结构/引用/1000 组代数 QA 与最终渲染检查通过：8 页，References 从第 7 页开始。下一步等作者当前 Sophia E1 输出，再固定开发阶段的计算预算；本轮不新增硬件测量。


## 2026-09-17：以 user-level slider 曲线为唯一主线

作者明确要求自洽的动态 scale-down + 单 slider + 不修改 Slurm。主稿改为完成预算 slider，一阶段端到端的反馈策略；原等待回归训练段从主方法移除，只在评估中说明规划基线。动作在 allocation 之间可上可下，不能运行中 resize。

补齐两层论证：fixed/preplanned/feedback 的理想策略类嵌套仅保证机会，不保证有限网络训练找得到；人工有理数构造显示两条动态点可严格支配固定 4/8 的时间与碳，但不写入任何实验槽。现有 AMSP 规模仍为 4/16/64/128。

E3 唯一新增训练对照改为 Precommitted-RL，替换 Predictor-advised/current-CI。同架构、同预算网格、同训练交互，首次提交前采样并保存全部动作，不读后续 queue/CI/实际 elapsed。已实现训练、评估、slider 档位、plan 日志与 test 文件封存；145 项合成测试通过。18 个文件哈希核对后同步到本机代码仓库，保留此前未提交改动；未提交/推送 Git，未运行真实规模实验。

PDF 共 8 页：正文 7 页，参考文献单独第 8 页。修正 QA 原先将“References 必须开始于第 7 页”误当 7 页正文限制的检查；现检查实际正文页数。未调整模板、字号或边距。全页渲染与关键页检查通过，22 labels/12 引用/1000 组代数 QA 通过。所有正式结果仍留占位。


## 2026-09-17：固定计时结果后的预算范围与执行流程

作者回传Frontera7B三到达时间的12个固定结果，全部完成，用时5.7秒。开发数据记录在cluster_progress.md，正式结果槽不填。4/16节点首个非终端chunk接近48h，原1–2倍T_ref网格在该分片只到约27h，会不必要地限制多段动作的可用预算。保留四档与单slider，改用完整train上最快fixed均值至Fixed-4 inverse-ECDF p95的跨度；所有方法共同使用，不能据此保证收益或miss。未改work、48h上限、开销、队列或E1。

新增run-main-pilot与Sophia单层脚本：完整train/validation fixed → train references与budget grid → 5轮80episode无等待模型PPO开发pilot。完整cohort/文件哈希检查拒绝把计时分片作正式参考量；固定stage中断保留后重跑，PPO按完整轮checkpoint恢复且保留原attempt。恢复的合成模型与连续运行最终tensor一致。158项软件测试通过，启动脚本合成完整/恢复检查通过；未在本机运行真实trace实验。

主稿E2写明预算标尺、192h有用工作与chunk数量的区别，精简重复的limitation表述以保持7页正文；未改模板/字号/边距。更新PDF共8页，References从第8页开始，22labels/12cite keys/1000组代数QA通过。全8页联系表与第5、7页完整图检查，无重叠裁切。


## 2026-09-17：pilot成功后的可恢复训练与validation汇总

作者回传完整Frontera7B fixed train/validation和5轮80episode PPO的成功日志；完整fixed平均成本/TAT和策略validation效果未包含在该日志中，未臆造数值。更新集群进度、实验计划及development manifest，正式结果槽维持空白，主稿PDF未修改。

代码仓库新增scripts/main_train.py与单层sophia_main_train.sh，复用已封存fixed/reference/grid；seed11预定64×4×16=4096训练episode，固定16/32/48/64轮各评价完整validation。全部候选均报告，不选优、不读取test。Fixed-Mix只对已有fixed结果解LP；汇总按新预算重新计算miss，区别相同规模多chunk与真正换规模，删失保留分母且不输出伪完整成本。训练按轮恢复，部分validation归档后只重跑该checkpoint，完成stage验证文件哈希后跳过。

7个脚本/测试/说明文件经安装前哈希核对后写入代码仓库并保留旧文件备份；全部37个src/carbon Python文件哈希未变，兼容现有pilot。162项合成测试通过（26.746s），两个bash脚本语法与新入口参数解析检查通过；无真实trace本机回放、无自动Git提交/推送、无远程作业提交。现有torch保持不动，空环境自动安装改用官方CPU wheel源。


## 2026-09-17：Sophia PBS单层提交入口

作者提供当前qstat队列和成功作业186384的资源信息。将确认的Local-LLM、by-gpu、1GPU/32CPU/120GB、pack:shared、home:eagle写入sophia_main_train.sh，运行时限设为队列允许的2h。加入本次配置/旧pilot/新结果路径的默认值、qsub环境变量恢复开关及无参数bash支持；原显式路径接口保留。PBS头不复制调度器内部记账/权限字段。

仅修改一个shell脚本和两份代码仓库说明，哈希核对安装前备份并保留前轮未提交文件。9项离线命令路由检查及bash语法检查通过，包括PBS spool目录定位、默认路径、环境覆盖、两种恢复入口和非法参数拒绝；37个核心Python文件及main_train.py哈希未变，无需改变checkpoint合同。没有真实trace回放或远程qsub提交，论文结果/PDF不变。


## 2026-09-18：首次主训练回传日志审核

完整检查carbon-main.o186777，识别为PBS 7200s时限中断，最后完成55/64轮（3520episodes、3691chunks），没有validation阶段。作者确认只带回日志。逐轮计数显示第39–55轮1088个完整episode全部single-chunk，不能支持任务内动态缩放收益；暂不归因于某种规模、损失错误或环境无机会，须读取动作/成本/miss记录。

记录进度与development manifest，安排同一训练合同的恢复和既定四checkpoint评价；通过qsub命令行给4h上限，无需改变脚本或Git同步。代码、训练预算、torch和artifact合同均未改，无新软件测试需要运行。results被Git忽略，后续应单独传回两个结果目录；论文结果继续留空。
