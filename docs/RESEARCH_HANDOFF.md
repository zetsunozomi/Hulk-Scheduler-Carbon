# ScaleDown：研究与集群交接（2026-09-21）

这是继续本项目的总入口。**后续直接在cluster读结果、诊断和推进，不要求把实验压缩包传回Mac。** 目标是把user-level动态scale-down做成有证据支持的顶会候选研究；当前已有候选稿和模拟实现，核心动态收益尚未证实。

## 1. 不要丢失的目标和限制

- 用户控制一个slider，选择完成预算D；希望对应的实际时间—碳曲线优于固定规模。4→8→4是机制例子，意图是任务内规模切换有真实收益，不是硬编码这条序列。
- 只能在checkpoint完成后提交下一次资源请求；不改Slurm调度器、不修改排队优先级、不resize正在运行的allocation。任务之间/各chunk之间可降规模，也可重新增配。
- 主方法不能依赖准确预测queue wait。现行设计已取消“等待回归预训练→RL”的两阶段主流程。
- 不要求Qwen/GPU throughput重测、整节点功率或碳脚印实测，也不要求获得新机器的历史作业日志。
- 尽量少加实验。先解决核心机制和最强对照，再完成既定E1–E4；不要靠扩panel、seed或消融数量掩盖主方法缺乏增益。
- 初始写作允许把所需实验作为条件推演，但真实结果不能编造。正式结果槽仍留空；负结果应引导方法修改或收缩主张。

## 2. 当前draft idea

### 2.1 请求决策与单slider

Slurm决定何时开跑，用户决定下一次请求多少节点。规模同时改变吞吐、node-hours、等待以及执行遇到的碳强度。每个allocation结束并保存训练进度后，策略根据真实已耗时间和剩余工作重新选择规模。

主策略读取剩余updates、真实剩余budget、总budget、公开可见的队列历史、未来168小时的28个六小时CI预测均值，以及候选动作的规模/速率/分配时长等描述。CI预测仅用此前可用数据。actor与critic都不读等待预测器；未来真实作业结束时间不是可见输入。

四档slider位置为s=[0,.25,.5,1]，D(s)=T_ref+s*(D_hi-T_ref)。T_ref是完整train上最快fixed的平均TAT，D_hi是train Fixed-4 TAT的inverse-ECDF p95。当前Frontera7B的显示值约15.11/59.79/104.48/193.84h；运行必须读取budget-grid.json未取整值。D是目标，不是单任务SLA，也不保证PPO实际曲线单调。

### 2.2 学习目标

一个budget-conditioned PPO actor对应每个panel。优化两种功率端点下较大的期望归一化碳成本，约束Pr(T>D)<=.05；budget各自有dual变量。三个critic heads分别学习两个碳成本与miss。gamma=1，使用完整episode的真实模拟回报；超过deadline仍完成工作并计入后续成本。没有等待模型预训练作为主策略的前置步骤。

这是数值求解，尚无找到最优、满足总体约束或跨任意队列泛化的保证。观测仍部分可见，策略仍可能学到只适合训练场景的规律。当前论文描述的真实Slurm checkpoint/resubmit接口是设计边界，线上部署效果未验证。

### 2.3 未测功率的处理

每节点功率情景为P_n=P0*kappa*[rho+(1-rho)*eta_n]，P0=1kW是单位参考，kappa为未知共同倍率，eta_n来自scaling效率。rho区间为[.25,1]，不是测量置信区间。

执行日志保存各规模的CI暴露量L_n。共同正倍率在同一panel的碳比值/排序中抵消；相对规模功率假设仍不能消掉。固定同一policy和比较对象时，碳差对rho为仿射函数，故两端点决定整个声明区间内的比较。必须报告相对modeled job-attributed operational carbon和node-hours，不能改称实测绝对碳排、全设施碳脚印或避免的电网排放。

### 2.4 为什么旧日志与公开profile可以组合

AMSP 2024修订版图12的Our Work系列提供LLaMA-7B/13B/30B吞吐；四个合法动作是4/16/64/128节点，每节点8张A800，即32/128/512/1024卡。没有插值出8节点等未引用点。

Frontera和iw两份历史流分别提供arrival/width/requested duration/occupied duration的资源需求场景。名义模拟容量128节点，新的admission由模拟器重建；旧Start不作为新机器的等待标签，不按FLOPS换算未知后台作业，也不把两份日志混洗后声称任意机器泛化。至少28天前缀回放不计分，chunk之间状态持续推进。

Texas ERCOT CI是附加的地区情景，不能声称AMSP机器、旧日志或Sophia实际都在Texas。Sophia只是运行本研究CPU模拟的机器。

### 2.5 贡献成立的门槛

全部Fixed-4/16/64/128、Fixed-Mix、Plan-once、Rollout-MPC是正式对照。E3的Precommitted-RL在第一次提交前生成完整规模序列，与主方法匹配架构/预算/到达采样/seed/交互数，后续不看真实反馈。

只赢一个较差fixed不能支撑主贡献；只做任务间规模选择不能当任务内动态scale-down；赢fixed但不赢预先序列不能证明反馈有价值。策略类fixed⊆预先序列⊆feedback只说明理想最优存在改进空间，不保证PPO胜出。人工4→8→4构造是机会说明，不能填入AMSP结果槽。

## 3. 已到哪里：证据与未知分开

下表是本次交接已核验的最新证据；cluster当前文件可能更新，接手时先查文件，不据此重复启动已有工作。

| 阶段 | 已确认进度 | 尚不能声称 |
|---|---|---|
| 稿件 | 已重写当前问题、功率分析、无等待预测器的PPO和E1–E4协议；PDF 8页，正文7页 | 已有正式收益或可直接投稿 |
| 软件 | 需求流回放、工作/碳核算、fixed/规划基线、PPO、Precommitted、按轮恢复、validation和冻结环境检查已实现；最近全套162项合成测试通过 | 真实Slurm部署、实际LLM跨规模恢复正确性 |
| Frontera E1 | 已回传成功日志；末尾validation 368快照×12探针=4416条，无删失 | 预测器准确、旧等待可移植或RL有效 |
| Frontera7B fixed | train 73到达×4规模=292次，114.5s；validation 24×4=96次，51.4s；全部完成 | 用早期3样本均值代替完整cohort均值 |
| 短PPO pilot | seed11，5轮×4预算×4episode=80episode，125chunk，完成 | 125chunk说明真正换过规模 |
| 首轮较长PPO | 作业186777最后核验55/64轮、3520episode、3691chunk；PBS因7229s超过7200s终止 | 64轮完成或validation已运行 |
| 后续恢复 | 已提供4h上限的恢复命令，但本次交接未核验cluster恢复后的文件 | 默认重跑，或默认已完成 |
| IW/其他panel | IW仅确认preflight；其他正式组尚无完成证据 | 六panel/三seed正式对比已完成 |

**最需要处理的研究信号：** 第39–55轮1088个episode全部single-chunk。这些训练任务没有执行任务内规模切换；不能据总计数判断全都选相同规模，也不能据此归因于实现bug/探索不足/环境无机会。先读逐budget的动作概率、成本、miss、entropy、dual和实际chunk序列。

当前工作量U=ceil(192*q4)，请求上限48h，每chunk开销600秒。已回传fixed的chunk数为5/2/1/1；因此较大规模一次完成是现有动作几何的真实情况。应区分设计是否给反馈留有合理决策机会、学习是否找到这种机会，不能强迫多次切换后把它当收益。

## 4. 文件地图：先读哪些

代码仓库根目录下的相对路径可直接用于cluster。

| 内容 | 当前入口 |
|---|---|
| 本文：目标、状态、下一步 | `docs/RESEARCH_HANDOFF.md` |
| 当前一句话idea、贡献和失败判据 | [paper/ideal_storyline.md](../paper/ideal_storyline.md) |
| 当前PDF | [paper/main.pdf](../paper/main.pdf) |
| 稿件主入口、摘要 | [paper/main.tex](../paper/main.tex) |
| 问题/动作/工作与时间定义 | [paper/4Jobtrace.tex](../paper/4Jobtrace.tex) |
| 功率、exposure与端点推导 | [paper/6design.tex](../paper/6design.tex) |
| 主PPO、观测、目标与选择 | [paper/5algorithm.tex](../paper/5algorithm.tex) |
| E1–E4与结果槽 | [paper/7exp.tex](../paper/7exp.tex)、[paper/plan_exp_list.md](../paper/plan_exp_list.md) |
| 精确实现规范 | [paper/reproducibility/method_spec.md](../paper/reproducibility/method_spec.md) |
| 声明的配置、冻结项、待填结果 | [paper/reproducibility/experiment_manifest.yaml](../paper/reproducibility/experiment_manifest.yaml) |
| 场景合理性与审稿挑战 | [paper/reproducibility/research_review.md](../paper/reproducibility/research_review.md)，尤其第六轮 |
| 完整执行历史与数据来源 | [paper/reproducibility/cluster_progress.md](../paper/reproducibility/cluster_progress.md) |
| 本批次运行/恢复细节 | [docs/MAIN_TRAIN_RUN.md](MAIN_TRAIN_RUN.md) |
| 核心实现 | `src/carbon/{environment,workload,carbon,policy_inputs,policy_runner,learning,baselines,planning,precommit}.py` |
| 编排与PBS入口 | `scripts/main_train.py`、`scripts/sophia_main_train.sh` |

旧`plan*.md`、`power_model_layers.md`下半部以及各审查文件的早期小节含已替代设计，不能压过本交接和当前正文。主方法无wait predictor；GBT仅供规划基线。当前Precommitted-RL已替代早期wait-advice/current-CI正式消融。

### 本机与cluster位置

- 本机原稿：`/Users/shuyuanfan/local-papers/carbon-rewrite/newest_writing/`。
- 本机原PDF：`/Users/shuyuanfan/local-papers/carbon-rewrite/newest_writing/main.pdf`。
- 本机代码仓库：`/Users/shuyuanfan/carbon-latest/`；此次加入可Git同步的`paper/`普通文件副本。
- 已由作业日志确认的Sophia仓库：`/lus/eagle/projects/Local-LLM/shuyuanfan/carbon-latest/`。
- 本次Git同步后，cluster PDF：`/lus/eagle/projects/Local-LLM/shuyuanfan/carbon-latest/paper/main.pdf`。
- Python：`/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python`。

PDF内容本次未改，SHA256为`dfdcfe9c55fa1de3e5beb41485ef65ba44e7b650609ee9b7fc615761ef706860`。`paper/SNAPSHOT.json`记录导入文件哈希。之后在cluster修改`paper/`并正常Git提交；Mac原稿不自动双向同步，不能再用旧副本覆盖cluster进展。已有AAAI2027样式仅是写作模板，不代表已确认投稿会议/时限。

## 5. 在cluster直接开始

### 5.1 先查真实进度：只读，无训练

```bash
cd /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-latest
/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python - <<'PY'
from pathlib import Path
import json
root = Path('results/amsp-main-frontera-7b-seed11')
summary = root/'validation-summary.json'
if summary.exists():
    data = json.loads(summary.read_text())
    print('validation summary:', data.get('status'))
    print('evaluated iterations:', [s['iteration'] for s in data.get('checkpoints', [])])
else:
    print('No validation summary yet')
for path in sorted(root.glob('ppo-attempt-*/manifest.json')):
    data = json.loads(path.read_text())
    print(path, {k:data.get(k) for k in ('status','completed_iteration','last_checkpoint','total_episodes','total_chunks')})
for path in sorted(root.glob('validation-*/manifest.json')):
    data = json.loads(path.read_text())
    print(path, data.get('status'), data.get('completed_episode_methods'))
PY
```

若summary complete，先读`results/amsp-main-frontera-7b-seed11/validation-summary.md`，并核对4个预定评价轮次和每轮96个结果。若未完成且没有同一输出目录的作业正在运行，在登录节点恢复：

```bash
qstat -u "$USER"
qsub -l walltime=04:00:00 -v CARBON_RESUME=1 scripts/sophia_main_train.sh
```

Python路径、Local-LLM/by-gpu、1GPU/32CPU/120GB、pack:shared、home:eagle已在脚本顶部；程序仍CPU执行。4h只覆盖墙钟上限，训练仍固定64×4×16=4096episode。每轮保存checkpoint，未完成轮重跑；旧attempt保留，完整validation stage校验后跳过。

不要用旧5轮pilot checkpoint修改设置直接延长，也不要在恢复前改`src/carbon/*.py`或`scripts/main_train.py`：run-plan/checkpoint绑定这些哈希、torch、数据和完整设置。若确需算法修改，保留原结果、明确新实验配置和输出目录。

### 5.2 直接在cluster读的结果位置

| 文件 | 用途 |
|---|---|
| `results/amsp-main-frontera-7b-pilot/fixed-train/`、`fixed-validation/` | 完整固定对照与chunk日志 |
| pilot目录的`references.json`、`budget-grid.json` | train参考量和未取整预算 |
| `results/amsp-main-frontera-7b-seed11/run-plan.json` | 本轮固定设置和输入/代码绑定 |
| seed11目录的`ppo-attempt-*/training.jsonl` | 各budget成本/miss、dual、损失与rollout/update耗时 |
| 同attempt的`episodes.jsonl`、`chunks.jsonl` | 实际动作、概率、进度和成本；恢复时先按已提交checkpoint链去重 |
| 同attempt的`checkpoint-*.json/.pt` | 网络/优化器/随机状态；以manifest最后完整轮为准 |
| `validation-000016/000032/000048/000064/` | 四轮完整paired validation日志 |
| `validation-summary.md/json` | 全部候选、fixed和Fixed-Mix的对比表 |

JSONL尾部可能有中断的半行；未完成训练轮不能混入已提交轮的均值。每个D必须重新计算fixed的miss；有删失时保留分母，不把部分成本当总成本。Fixed-Mix的validation LP期望值不是独立test表现。主策略仍categorical sampling，不临时切argmax。

### 5.3 接手后的研究顺序

1. 先确认本批次实际完成状态，完整核对4档预算的成本/miss与动作，不再重复pilot或E1。
2. 首要诊断单chunk集中现象：按budget看初始动作分布、四种动作概率、entropy、duals、更新损失；区分共享actor不使用budget、探索/优化问题和当前环境确实偏好一次完成。对照完整fixed与Fixed-Mix，而非单看训练loss。
3. 若动态策略未体现优势，不直接扩六panel/三seed；先提出与真实用户接口一致、能解释收益来源的最小方法修改。允许有根据地重设决策粒度/训练设计，但须公平应用于所有方法、记录旧结果，不能为保证正结果暗改环境或强迫动作切换。
4. 当前方法在本panel值得继续后，再按既定E1–E4推进：两来源×三profile、seeds11/23/37、强规划比较、一个匹配Precommitted训练对照，以及已规定的功率和冻结环境检查。完整正式训练预算/统计选择仍须冻结，当前4096只是development配置。
5. 将核验结果、方法修正和适用范围直接同步到`paper/`。正式表图没有证据就继续占位；顶会candidate是研究目标，不是已经达到的状态。

## 6. 同步方式

代码、稿件源码、PDF和交接文档通过Git走；大型results留在cluster并就地分析。`.gitignore`中的`/results/`保持不变，不再把“先传结果压缩包到Mac”设为继续工作的前提。

本次本机准备后可手动提交：

```bash
cd /Users/shuyuanfan/carbon-latest
git add docs/RESEARCH_HANDOFF.md docs/CLUSTER_RUNBOOK.md README.md paper
git commit -m "Add cluster research handoff and current paper"
git push
```

cluster仓库执行`git pull --ff-only`后从本文开始。没有自动提交、推送、训练或qsub。正式结果保留在cluster，接手者可直接用同一个Python环境读取。
