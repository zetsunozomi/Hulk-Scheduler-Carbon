# 候选方法实现规范

状态：**设计规范，部分已有软件实现**。回放、预测、基线、PPO、validation 选择、冻结 test 执行、逐 seed/跨 seed 报告与功率后处理已有合成检查；E1 回放误差、预测尾部/排序与日期相关性诊断也已有合成检查；图表导出也已接通，正式结果仍未取得。以下设置不是已经运行的正式实验配置。
正文定义了决定论文主张的算法；此文件补齐工程和复现实验细节，避免下一轮实施时重新猜测。

工作负载更新（2026-09-17，取代 Qwen/Vista 计划）：用户确定使用 AMSP 2024 修订版图 12。三个 LLaMA 模型，动作 4/16/64/128 节点，对应 32/128/512/1024 张 A800；每节点 8 卡、GBS=1024 sequences、seq=4096、microbatch=1、accumulation=32/8/2/1。公开 TGS 已从 PDF 矢量柱提取，记录见 amsp/profiles.json。原论文没有 64/256 卡结果，不插值补点。

名义模拟是 128 节点的声明场景，后台来自两份历史流各自的 arrival/width/request/runtime；不称为原平台 admission 重建或任意机器泛化。新模式从固定 trace-prefix 起点空状态重新生成 admission，至少 28 天不计分，不借用旧平台正在运行的状态。旧 source capacity 仅用于原平台审计，不再决定构造场景的容量。Texas CI 是外加地区情景；原始时间解释不确定时明确记录假设。

每个模型 U=ceil(192*q4)；主开销每 chunk 为 600 秒（初始化/恢复 300、保存 300）。这些是模拟任务与开销设计，非实测。取消 Qwen profiling 和真实 restore correctness 要求，只验证模拟工作守恒，不声称 AMSP 弹性训练实现或收敛结果。功率参考 P0=1 kW 为单位归一化，未知 kappa 未校准，仅报告相对 carbon 和 node-hours。

## 1. 什么改变了

| 原稿设计 | 候选设计 | 原因 |
|---|---|---|
| 模糊 alpha / scale-down intensity | 支持的 D 小时预算 + 预设 miss tolerance | 用户目标可以核验 |
| 不输入 remaining work | U/U_total、剩余预算、总预算、q_n 与 partial chunk | 完成边界影响动作 |
| submission-time CI scalar | 主策略读取未来 168h 的 28 个六小时 CI 均值 | 保留碳强度的时间结构，不猜 admission 时刻 |
| 四套大型 Transformer wait 模型 | 主策略不使用等待预测；GBT 只供规划/辅助对照 | 准确预测不能成为 user-level 方法的前提 |
| PG/AC 双路线、终点 weighted sum、gamma=.99 | PPO + 三 cost heads + episodic dual、gamma=1 | 目标与最终成本一致 |
| 固定功率或含糊“校准” | 三层假设 + exposure + endpoint/scale-error 分析 | 不需要测不到的整节点功率 |
| Fixed-4/32 为主要对照 | 所有 fixed、Fixed-Mix、Plan-once、Rollout-MPC | 真正检验增量价值 |

## 2. 仿真执行合同

所有方法共享以下规则。

- 初始提交时间与背景 arrivals 配对；同一 initial state 深拷贝。不同方法分别推进自己的 scheduler，不能直接复用同一串固定 wait。
- 一次只注入一个 target run；target 占用会影响其 episode 中后续 background jobs。
- 持续推进状态，不能每个 chunk 从48h历史重新构造一台“空”集群。
- U 是整数 optimizer updates。q_n 来自同一 prescribed work 的 profile，单位 update/hour。
- v=min(U,floor(q_n*(H-r_n-h_n)))；v<=0 的动作全部屏蔽。d=r_n+v/q_n+h_n。
- H 为配置中允许的上限，当前设计取48h。requested walltime=向上取整(d, Slurm resolution)，且不超过H；实际 allocation 到d结束即释放。H必须是该granularity的整数倍。
- partition 只允许更短 walltime 或少于128nodes时，缩小所有方法的同一动作集合，不伪造支持4/16/64/128。
- initial initialization 与后续 restart 可有不同 profile；正文 r_n 是当前 chunk 对应的已知开销。模拟器需标注 phase，不能把同一开销扣两次。
- final chunk 也保留输出保存时间；完整checkpoint在allocation到期前完成。
- 工作直到U=0才结束，deadline miss 不结束工作。失败重试、超时、未完成要显式记录。
- 模拟器记录 global next-sample index 并验证工作守恒；假设底层支持重分片 checkpoint。该假设不代表已实现真实恢复。
- 固定 B 和 updates 不证明 bitwise equality 或最终 quality；本轮不要求实机 correctness 实验，也不提出这类结论。

## 3. 主策略的可见特征与辅助等待分布

建议滞后点：当前、0.5、2、8、24、48小时。对每个点，用同一公开字段计算：

- pending job count、总requested nodes、requested walltime统计、elapsed wait统计；
- running job count、allocated nodes、elapsed runtime统计、requested remaining walltime统计；
- partition capacity与节点可用性（能从普通用户接口得到时）；
- hour-of-day/day-of-week 的sin/cos；每项missing mask。

**禁止**使用未来实际runtime、真实剩余运行时间、未来完成时间或admin-only priority。
“requested remaining walltime”仅是 max(requested limit-elapsed,0)，不是实际remaining time。

以下 GBT 仅供正式规划基线 MPC、Plan-once；wait-advice 软件开关只是开发功能，不列正式消融。主策略不读取 predictor 文件；E1 的误差不是主方法启动或可信度的门槛。

辅助训练：
1. 从训练期保存snapshot，分别插入(n,requested_walltime) probe；probe长度包含full与partial requests。
2. 只保留label在训练边界内已完成的probe；不能让未来validation/test状态产生训练label。
3. 用chronological expanding folds生成out-of-fold residual，再用全部训练数据拟合最终shared regressor。
4. target=log1p(wait_hours)，squared loss。建议256 trees、max_depth=4、learning_rate=.05；数据量不足时在训练内部调小，不看test。
5. 对每个n，取其training residual CDF的32个mid-quantiles，tau_j=(j+0.5)/32, j=0..31。
6. w_j=max(0,exp(predicted_log_wait+residual_quantile_j)-1)，32个atom等权。mean和p90均由此分布计算。
7. 该近似不处理所有conditional heteroscedasticity；E1按queue regime与scale报告coverage。不为达到某个准确率而反复加模型；若修复代码或输入错误，只在 development 进行并重新冻结。

当前实现使用4个expanding folds；同一arrival的请求放在同一fold，并从每个fold的训练侧剔除label到fold起点仍未可见的行。probe长度默认取walltime上限的1/4、1/2、全部（按申请粒度向下取整），吞吐数据不参与这一步。独立validation probes报告median预测MAE、mean预测RMSE/bias、CRPS、p90覆盖率、中心80%区间覆盖率/宽度，按节点数、申请时长与公开队列状态分组。分组为queued（有pending nodes）、busy（无pending且可用节点不足一半）、light（其余）。这些是描述性诊断；未完成label单独计数，不伪造长尾wait。新增p50/p90 absolute error针对median预测的逐条绝对误差；same-snapshot、相同申请时长下，按mean predicted wait评价不同node count的pairwise rank，真实tie单列、预测tie计半分、censored pair未知。

原平台工具仍提供连续background-only回放对历史admission的检查（构造场景禁止使用该结果作为真实性验证）：按submission预选全部作业，初始running/pending仅恢复状态，不把已观测的初始start作为预测；记录双方等待censoring与paired wait的median/tail差异。它复用cleaned历史runtime，不重建隐藏Slurm状态。日期相关性诊断只读train/validation probes和可选的固定策略评估日志，保留真实时间缺口；constant/不足样本的相关系数为null，不视作独立。完整目标 episode尚未接入时，不冻结block长度。

这是一种廉价的工作分布，不是校准概率保证；不能把p90直接当deadline承诺。

## 4. CI输入与exposure

已整理的主场景输入是 EIA ERCOT consumption-based operational CO2（`data/texas_eia`，代码仓库）。源单位 lbCO2/kWh 乘453.59237 转成代码的 gCO2/kWh；论文按 kgCO2/kWh 展示。它不包含非CO2温室气体，不标为完整CO2e。2019-01-01至2025-01-04共52,680小时，其中11个缺失小时仅向前填充、连续不超过2小时，逐行标记。24小时发布延迟是明确的可用性情景，档案含修订，不是历史发布记录。最终queue时区/日期配对仍待冻结。

默认CI forecast不需要历史“预报档案”：
- 在decision t，仅使用release timestamp<=t的CI observations。
- 滚动过去8周的同hour-of-week平均；不足历史时退回训练期同hour-of-week均值，再退回训练期总体均值。训练期间仅用当前已发布的训练前缀；训练结束后，该fallback固定为训练截止前已发布的数据。
- 最早训练决策尚无可用训练记录时，使用已发布warmup历史的时长加权均值。连warmup观测也没有时停止并要求更早输入，不能借用未来CI。
- 初始训练参考不能使用test历史的未来部分。历史archive的补录/修订如无发布延迟记录，应明确采用固定lag可用性情景。
- UTC记录绝对时间，本地时间只用于calendar feature；DST必须明确。
- 全部forecast horizon都有上述causal forecast；不把远期realized CI混入fallback。
- realized CI用于结果积分。若与queue年份不匹配，明确counterfactual paired calendar。
- 固定stepwise CI采样上做精确分段积分，不用submission CI乘整段时长。
- 主策略读取 g_j=(1/(6*c_ref))*integral_{t+6j}^{t+6(j+1)} forecast(s) ds，j=0..27。这个固定 168h 窗口相对提交时刻，不暗示开始时间；超窗等待仍真实发生。
- 仅 MPC/辅助对照使用 L_kn=n*mean_j integral_{t+w_j}^{t+w_j+d_kn} forecast(s) ds。

## 5. Actor、critic、budget与objective

每个machine–workload panel共享训练一个budget-conditioned actor；每个budget可从该共享训练轨迹选一个validation checkpoint，部署记录budget-to-checkpoint映射。每个选择仍在整个rho区间固定，不能随rho切换。

建议：
- T_ref=min_n E_train[T_Fixed-n]，只用于单位归一化；不宣称最快的生产配置。
- D(s)=T_ref+s*(D_hi-T_ref)，s=[0,.25,.5,1]。T_ref 为完整 train 上最快固定策略的平均 TAT；D_hi 为完整 train Fixed-4 TAT 的 inverse-ECDF 第95百分位（sorted[ceil(.95*N)-1]）。beta=D/T_ref 自动生成并冻结；右端是预算标尺，不是 validation/test 可行性保证。epsilon=0.05 为声明的目标。
- 不看test剔除“不好看”的budget；不支持的budget照样报告为infeasible/unsupported。
- 若用户需要未训练D或新epsilon，需验证/重训；当前不承诺任意连续预算泛化。

state：
- U/U_total、(D-elapsed)/T_ref、D/T_ref、queue/calendar summary、28-bin CI forecast；
- 每个 action 的 6 项：[n/capacity、q_n*T_ref/U_total、v/U_total、d/T_ref、requested_limit/T_ref、eta_n]；
- 主 actor/critic 均无 wait mean/p90、predicted exposure、predicted start；advised 变体才追加前两项与 exposure 三项。
- 每种输入按training normalizer变换，保留missing/feasibility mask。

当前 v2 schema 固定除数，不增加一轮 normalizer 采样实验：CI 用 c_ref=L_ref/(4*T_ref)，L_ref=训练 Fixed-4 在 rho+ 下的平均 modeled carbon/P0；它仅是正的训练参考量，不是测得的平均 CI。wait、duration、budget 用训练 T_ref；rate 用 T_ref/U_total；Lhat 用训练 Fixed-4 的平均 exposure；queue count 的 log1p 值用 log1p(capacity)，queue duration 的 log1p 值用 log1p(T_ref)，history age 用声明的最长滞后（至少1小时）。容量、比例、calendar 与 missing mask 的除数随 schema 保存。validation/test 不重新拟合，也不按在线数据更新这些量。

网络：
- action encoder: 2层128 ReLU，参数在n之间共享；
- 对可行动作的 embedding 做mean pool，再与global state拼接；对每个action打分，屏蔽不可行动作；
- critic同宽独立网络，有endpoint_minus、endpoint_plus、violation三head；
- 禁止使用full simulator state作为critic隐形特权输入；
- rollout和test用同一categorical policy，不偷偷改成argmax。

cost：
- Cref_e = E_train[C_Fixed4(rho_e)]，固定并为正。
- rho interval建议[.25,1]；rho=0单列stress，不说是实测范围。
- J_e=E[C_e]/Cref_e；J_V=Pr(T>D)。
- actor目标 min max_e J_e, s.t. J_V<=epsilon。
- 对每个budget保留独立p和lambda；p初始(.5,.5)，lambda初始1。
- 每个iteration，先冻结p/lambda，再收集完整episode和做PPO更新；最后用pre-update rollout的episode均值做dual更新。
- p按指数梯度更新；lambda=max(0,lambda+eta_lambda*(J_V-epsilon))。
- 建议eta_p=.05、eta_lambda=.05；数值不稳定在development处理，不用静默clip lambda掩盖infeasibility。
- 每个chunk reward是负的加权归一化endpoint carbon；只有terminal chunk加lambda*1[T>D]。
- gamma=1，MC cost-to-go。每个episode的梯度按chunk求和，再在episode间平均。
- critic通过各自MC return做regression；actor advantage为weighted negative-cost advantage。使用一致的cost sign。

建议PPO起点（待manifest冻结）：
Adam lr=3e-4、clip=.2、4epochs/rollout、32complete episodes/budget/rollout、
entropy=.01、gradient norm=.5、value coefficient=.5；
三个随机种子[11,23,37]；先规定每panel的环境interaction预算再训练。
policy/predictor training compute和inference overhead从运行日志报告，无额外测碳要求。

当前实现用按完整episode组成的minibatch（默认32个），每个minibatch内先对chunk求和、再除以episode数；不做chunk平均或advantage重新标准化。每轮对所有budget采样同一批training arrivals，行为仍按各自预算和categorical policy产生。iterations必须显式给定，manifest记录总episode预算及实际chunk数；不能把一次合成检查的迭代数当正式训练预算。每轮checkpoint保存网络、Adam、两个随机状态与dual。恢复必须使用相同输入、代码、PyTorch版本和完整预定设置，写入新目录；不补选seed或延长既定训练预算。未完成episode先写入日志，然后停止训练，不能按已观测的局部cost进行MC更新。对checkpoint的原始评估尚不等于通过validation可行性选择。

重要：
- 这里minmax的是endpoint的**expected full-episode** cost；不能逐chunk取max，否则改了问题。
- 虽然actor只需要当前的work/budget/history，而无须观测dual参数，但一轮收集与优化过程中dual必须固定。
- 这是数值求解，不继承CPO的保证；partial observation、函数近似和nonconvexity都保留。
- miss population与validation可行性针对实验定义的initial-arrival分布，不保证每个job，也不保证漂移后仍可行。

## 6. Fixed-Mix的精确定义

从validation得到每个fixed n的两个normalized mean carbon c_ne和miss probability v_n。
对于每个预算解LP：

min z
s.t. sum_n x_n*c_ne <= z, e in {-,+}
     sum_n x_n*v_n <= epsilon
     sum_n x_n = 1, x_n >= 0.

冻结x后，每个新episode开始按x采一个n并维持到完成。
test上的mean carbon、miss和mean TAT可以按冻结x对paired fixed outcomes精确加权；
不必用少量随机抽样增加统计噪声。p95必须由mixture distribution求出，不能平均各fixed的p95。

验证时采用与其他方法一致的经验miss标准，并单列置信可行性检查；经验LP可行不自动代表统计可行。
经验目标未达或置信界不确定分别标明，不把mean TAT低于D当可行。
hull绘图与deadline可行性是不同对象。

## 7. Rollout-MPC

- 枚举(a0,a1,tail_n)最多64个plan；若只剩1/2chunk，忽略多余动作，不重复增加开销。
- 对每plan，模拟work和时间直到U=0；tail后每次都固定tail_n。
- future queue与wait模型calendar特征固定为当前可见值，候选request长度随remaining work更新。CI积分窗口仍随模拟执行时间前进。
- wait从同一predictor分布采样；同一MPC invocation内各plan用配对随机数，建议256paths。
- 每个路径的CI forecast始终使用当前t发布的信息，不能在imagined future time查询真正future observations。
- 目标 max_e (Cpast_e+E[future C_e])/Cref_e；deadline比较elapsed+future time与D。
- Cpast是决策时可得的估计：已发布区间用观测CI，尚未发布的过去区间沿用该chunk提交时冻结的forecast。事后真实CI只用于结果记账，不能从日志泄漏到controller。
- 使用同一miss tolerance；如无plan可行，选minimum estimated miss plan，按worst carbon与node count打破并列。
- 执行首动作后等待真实完成并重新规划。
- 必须显示MPC计算预算和latency；不能故意限成弱greedy，又称强MPC。
- 估计wait跨chunk独立、future queue固定都是近似；replay actual queue依旧完整演化。
- queue-blind ablation只在规划内部令wait=0，actual execution仍排队。
- Plan-once只在initial submission运行同一MPC，保存完整(a0,a1,tail_n)计划；每chunk按该顺序立即重投，不读取新的queue/CI来改计划。它能利用初始queue/forecast且允许within-run mixing，排除将预先混合规模误称online adaptation。
- Plan-once与MPC分别在validation上调整内部risk threshold，最终用同一个test epsilon判断实际feasibility。
- E3 仅重训 Precommitted-RL，替换等待辅助/current-CI 组。匹配主策略完整预算网格、到达采样、seeds 与 episode 总预算，仅在每 panel 一个预设预算评价。它在任何 target admission 前采样完整序列，保存到 plans.jsonl；只用初始 history/CI 与各动作已知工作/分配时长，真实后续 elapsed/queue/CI 不进入 actor/critic。训练回报仍使用完整真实回放的成本与 miss。

MPC在每个state的条件chance约束与RL的population约束不是相同求解器，但**最终相同E2指标**决定优劣。
为避免对MPC施加额外保守性，在development/validation上允许小范围调整内部miss阈值，
最终比较仍用同一个真实test miss tolerance；冻结阈值，不看test调。

## 8. Validation、test和统计

- 作者确认旧数据主要做过 train/validation 二分。本轮三段时间划分是 retrospective_temporal benchmark，保持 test_is_untouched=false；先冻结方法/协议，再报告新 test，不能据此继续调参。无需把全新日志作为运行前提。
- predictor训练label、RL episodes以及CI forecast拟合不能穿越split边界。
- 正式运行前冻结selection_rule=empirical_miss：每个budget/seed在完整validation cohort上选observed miss<=epsilon的checkpoint，再比max_e J_e；没有则保留最少miss的完整候选并标记为fallback。one-sided upper miss bound单列，不作为这条经验选模规则的筛选门槛。
- MPC内部阈值、Fixed-Mix权重、baseline选择也只用validation，接受同一实际miss标准。
- test报告每个seed，不能选择最优seed。图展示跨seed均值与variation，原始值随结果数据保留。
- replication：paired initial arrivals；按calendar blocks整体重采样所有方法。建议起点7天，但block必须覆盖development依赖长度与典型episode span，需要在test前锁定。
- 以week blocks而非所有chunk/10min snapshots为独立样本。少block时只能报告描述性uncertainty，不造精确显著性。
- budget-selection与comparison-selection在validation完成，test bootstrap不重新选择最强baseline。
- within-panel compare ratio of means；跨panel只聚合dimensionless ratio，禁止假设不同workload的kappa相同。
- power interval结论用两端点联合one-sided置信界（可用Bonferroni-adjusted paired block intervals）；
  positivity of denominator 必须核查。
- censoring：预定initial-arrival cohort，留足背景trace至完成。发现censoring就报告比例，
  miss已超过D可判定，但完整carbon未知，不能宣称全cohort碳占优；不能按哪个方法完成来筛episode。

当前 selection 实现保留全部声明的budget/seed以及Fixed、Fixed-Mix、MPC、Plan-once；不齐全或cohort不配对时停止。正式计划采用empirical_miss规则：经验达标候选中按最坏端点成本选，未达标才按观测miss和成本排序保留fallback。Fixed-Mix也接受同一标准。独立的confidence_upper_miss模式保留用于显式预声明的认证型选模；旧配置省略字段时仍沿用该旧行为，research配置则必须明确写出选择规则。

经验达标只描述冻结cohort的结果，不证明总体可行。主稿的research不确定性仍采用calendar blocks，保留独立/可交换假设与development依赖性审计。miss上界对全部candidate-budget做Bonferroni后用bounded-variable Chernoff反演；零观测miss仍有正上界，不等长block的保守有效数为1/max(weight)。Fixed-Mix使用同时成立的fixed-component上界加权。块数不足或没有审计时，置信可行性标记保持false，但这不等于已证明违反目标。具体block长度、alpha与正式设置仍待冻结；research入口仍拒绝用fixed-cohort条件采样诊断代替日期推广范围。

当前 test 入口按 validation 封存的 checkpoint hash、planner 设置和 mixture 权重执行，核对原 cohort hash；不会在 test 重选策略。报告保留所有 seed，跨 seed 的 p95 来自等权合并分布。日期块和 seed 分别重采样后交叉组合，作为跨 seed variation；少量 seed 不提供对未来任意训练 seed 的校准保证。逐 seed 的 observed_budgeted_carbon_reduction 仅表示双方完整validation/test cohort的观测miss达标且两端点平均成本比小于1。更强的 budgeted-improvement 标记仍同时要求双方 validation/test 置信可行与两个成本端点上界小于 1，并在当前 panel 的全部预算/比较中分配 alpha。bootstrap 调整后的每个尾部须至少有 10 个期望重采样值，否则不标为支持。selection 另输出零 miss 时的上界下限和所需独立等长 block 数，便于区分样本上的tradeoff与现有数据无法认证的更强论断；它不是经验选模或运行主矩阵的最小样本数。具体正式配置仍未冻结。

结果导出从封存的report与raw outcomes核对后生成：每个预算保留所有fixed、逐seed信息、censoring与unsupported标记。E2使用当前panel的training Fixed-4归一化，E4在整个rho区间保持同一个comparator并单列rho=0 stress；E3缺少的消融明确列为not in plan。图示区间是paired calendar与crossed seed的marginal variability，不能替代report的joint cost/miss判据。表格数据与绘图分离后，可在不重跑仿真的情况下重画；全部合成图仅作临时软件/排版检查，不填论文结果槽。正式profile数量确定后再组装跨panel主图。

## 9. 日志schema（E1–E4共用）

每条chunk记录：
episode_id, panel, initial_arrival_utc, method, seed, budget_hours, epsilon,
chunk_id, remaining_updates_before, selected_nodes, requested_walltime_hours,
submit_utc, start_utc, end_utc, completed_updates,
restart_intervals, training_intervals, checkpoint_intervals,
forecast_issue_utc, forecast_input_id, predictor_version, policy_checkpoint,
fallback_reason, final_status。

每条CI记录保留timestamp、value、unit、publication/assumed availability time、region、source版本。
每个episode汇总Ucompleted、TAT、miss、nodehours、4-dimensional L、A、B、
两个C endpoint及censor flag。

不要存一个“carbon total”后再给整个policy curve乘系数，必须按scale与执行interval重算。

## 10. 两项廉价数学敏感性

1. 对同一fixed policy和comparator：DeltaC/(P0*kappa)=DeltaB+rho*(DeltaA-DeltaB)。
   若斜率非零，rho*= -DeltaB/(DeltaA-DeltaB)；检查是否在声明区间。
2. 在任意声明nominal scale coefficient s0附近，s4固定1，其他独立relative error∈[-d,d]：
   worst difference = sum(s0*DeltaL)+d*sum_{n≠4}(s0*abs(DeltaL))。
   当nominal difference<0且sensitivity>0，最大允许半径为
   min(1, -nominal_difference/sensitivity)，严格正功率要求d<1。
   使用均值DeltaL得到expected-cost边界，不能误用mean of per-job radii。

两者都是条件代数，不是新的物理测量。敏感性分析不等于验证真实rho或s_n。

## 汇总的一个注意点

端点定理按machine–workload panel成立。不同panel的baseline归一化分母可能不同；不能只看宏平均ratio的两个端点，就自动宣称宏平均在整个rho区间成立。应逐panel求worst-endpoint bound再汇总，或直接计算声明的宏平均函数的完整极值。

## 2026-09-17 环境敏感性执行协议

见 research_review.md 与 plan_exp_list.md。仅 7B、一个预设 budget、两来源、full/MPC，四个独立环境改变：后台宽度乘 2 后封顶 128；strict FCFS；总 overhead 60/1800 秒。保留来源/训练引用/模型哈希，使用 source 的固定归一化，禁止重拟合 target 等待模型。每个场景从它自身的同一前缀重建 background 状态；同场景 methods 配对。此为环境压力测试，不能复用旧 output 做后处理，也不绕过一般 artifact 匹配检查来偷偷混用配置。

环境敏感性入口已实现为 carbon run-stress，输出单独的 source/derived manifest、chunk/episode 日志。合成检查覆盖冻结输入与配置匹配边界；正式回放与跨来源汇总图仍待运行。

## 2026-09-17：等待预测依赖的界限

主策略的 CLI 默认 `--wait-features none`，`--predictor` 可省略。即使旧启动命令提供该路径，主策略也不加载文件，checkpoint 的 predictor hash 为 null。`--wait-features advice` 明确要求有效模型并核对来源/hash，记为 Predictor-advised。v2 特征维度变化，旧 v1 checkpoint 不能直接加载或作为 v2 结果。E1 probe 与模型合同未变，正在运行的 E1 可复用。

实际 wait 经由完成时刻更新 remaining budget，完整 realized returns 训练 actor/critic；预测不会生成动作 mask。没有显式等待回归不等于无需学习 queue→outcome 关系，也没有漂移或任意 hidden Slurm 状态的保证。若所有可行动作的 admission 都晚于 D，任何只改用户请求的策略都会 miss；不能靠 calibration、p90 或 RL 名称消掉这个边界。

E4 的 width/FCFS 冻结回放才是环境依赖检查，不增加无意义的“扰动未读取输入”实验。主策略和 Precommitted-RL 不拟合 GBT；只有规划基线需要，其计算成本单独计入。

## 2026-09-17：最终主线与 slider / 预先排完整序列对照

本节取代上轮把 Predictor-advised 作为 E3 的设计。主方法只保留一阶段、端到端的 budget-conditioned PPO；旧“等待回归预训练 + policy”结构不保留，也不把回归输出换名继续输入。队列历史是公开观测，不是一项必须校准准确的中间任务。

- s∈[0,1]，D(s)=T_ref*(beta_min+s*(beta_max-beta_min))。s=0 紧预算，s=1 更宽松；epsilon 与 rho 不是用户另两个旋钮。当前是预设验证档位，不承诺未训练档位的性能。
- `run-policy --slider-position S` 将已支持档位映射到训练 budget，不能与 `--budgets` 同时指定；不支持位置明确报错。checkpoint/evaluation/export 保存同一 slider 映射，实际曲线仍画 measured replay TAT 与 modeled carbon。
- `--decision-mode feedback` 为主方法默认；`--decision-mode precommitted` 为唯一新增学习对照，且禁止 wait advice。其全部动作在 env.step 第一次调用前产生并写入 plans.jsonl，hash 写入 chunk/episode/manifest，test 封存包含该文件。
- 计划状态只更新 U、候选实际/申请时长，以及 D 减累计计划 allocated time；history timestamp 和 CI issue 均固定在首次提交。这个量不是实际剩余预算，也不假设真实排队为零；真实 queue wait 仍进入最终训练目标。
- 形式上 fixed（包括 per-run mixture）⊂ initial-only preplanned ⊂ feedback。理想受约束最优值随集合扩大不会升高，随 D 放宽不会升高；这是策略空间的性质，不是 PPO 找到最优或实际曲线单调的保证。
- scale-down 指相对最大规模减配，后续可再次增配，全部发生在不同 allocations 之间；不对运行中 Slurm 作业 resize。现有 H=48h 与 U 可能使大规模动作很快完成，因此不强迫生成 3 段序列；single-chunk 占比必须报告。

例子 `4→8→4` 在独立人工构造中可严格支配两种固定策略，核验见 adaptive_opportunity.md 与 tools/check_adaptive_opportunity.py。该构造不等于 AMSP 结果，也不证明反馈或 RL 必要；正式来源与规模保持不变。

## 2026-09-17：固定分片后的预算范围修正

作者回传的 3 个 train 到达时间中，Fixed-4/16/64/128 的 chunk 数为 5/2/1/1，固定64平均TAT约13.62h。若沿用通用 CLI 的默认 beta=[1,1.25,1.5,2]，该例最高预算约27.23h，而小规模首个非终端chunk接近48h，会把合理的多段动作压到预算之外。这是训练前的设计检查，不能据这3个样本估计完整参考量。

正式场景改为上述从完整 train 固定结果确定的四档跨度；不增加档位/实验组，不改48h请求上限、工作量或开销，不重跑E1。各方法、各seed及Precommitted-RL使用相同的预算网格。实际动态使用率仍需报告，不能靠范围扩大就宣称动态缩放一定有益。CLI的旧默认网格仅保留兼容；实际运行必须传入 budget-grid.json 中的生成值。

下一计算批次：完整Frontera7B train/validation固定结果与参考量，随后seed11、5iterations、4episodes/budget/iteration的80-episode开发pilot。epochs=4、minibatch=4、wait_features=none；不是正式interaction预算或已选择策略，正式训练不从该短pilot改设置延长。恢复维持同一80-episode上限与完整设置，输出另一个attempt目录。


## 2026-09-17：成功 pilot 后的首轮学习曲线

作者已回传完整fixed与5轮80episode主PPO全部完成，无等待预测器。下一轮development预定seed11、64iterations、16complete episodes/budget/iteration、minibatch16、epochs4，共4096episode，从该seed重新初始化。四档读取完整train生成的原始grid。评价轮次在运行前固定为16/32/48/64，各评价同一24个validation到达×4档；全部结果报告，不自动选优、不读取test。训练结束后按这些轮次评价，重复启动只恢复缺失进度。

这是将上文建议超参数具体化的一轮开发配置，正式跨panel/multiseed及Precommitted-RL预算仍待开发结束后冻结；不将4096episode称为已收敛。该批次不改环境、数据、奖励或PPO实现；scripts/main_train.py仅编排现有接口并封存运行设置，避免破坏已完成pilot的package hash。

汇总按当前每个D重新计算fixed的deadline miss，记录单chunk比例和chunk之间实际规模变化；连续相同规模的多个chunk不算动态缩放。若发生删失，全体分母保留，miss给已知下界/含未知上界，全任务平均TAT和碳成本缺失，不丢弃失败任务。Fixed-Mix复用fixed validation解LP，明确这是同一validation集上拟合的期望值，不能当独立测试表现。实际时间/碳曲线、统计可行性、反馈贡献仍待证据。
