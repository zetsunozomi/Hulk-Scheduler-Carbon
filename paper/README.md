# Cluster working draft

从2026-09-21起可在本目录直接继续写稿，以本代码仓库的提交为准。总入口：[研究交接](../docs/RESEARCH_HANDOFF.md)。导入时文件记录在SNAPSHOT.json；后续正常修改不会自动更新旧快照哈希。

# ScaleDown candidate draft

当前主稿入口是 [main.tex](main.tex)，编译结果为 [main.pdf](main.pdf)。
正文从 1intro、3related、4Jobtrace、6design、5algorithm、7exp、8conclusion 依次载入。
main-input.tex 只是兼容入口；AnonymousSubmission2027.tex 是原模板，不是主稿。

2026-09-21 交接更新：正式速度输入改用 AMSP 2024 修订版图 12，LLaMA-7B/13B/30B；取消 Qwen3/Vista 自测。采用 4/16/64/128 个八卡 A800 节点，旧日志分别作为声明的资源需求场景，全部新结果留空。

## 本轮交付的范围

- 已直接重写论文主线、问题定义、功率分析、RL 方法、比较协议和结论。
- 正式结果图表和性能结论仍是E1–E4占位。集群已完成Frontera E1、完整fixed与PPO pilot；较长主训练最后核验到55/64轮超时，最新状态须从cluster直接读。
- 新模拟器、功率/工作量核算、无显式等待预测器的PPO、对照策略、checkpoint恢复与validation流程已在carbon-latest实现并做合成检查。真实Slurm弹性训练部署未完成验证；软件实现不等于论文效果成立。
- “顶会 candidate”指值得按这套主张和证据投入的候选稿；并不代表已有实证支持或可直接投稿。
- 旧稿和旧计划保存在 ../revision_history/2026-09-15-before-candidate/；历史数据和图片保留，但不进入新主稿。

## 核心材料

- [本轮研究自审](reproducibility/research_review.md)：为什么日志可作场景输入、不能预测任意新机器，以及必要验证。

- [实验占位清单](plan_exp_list.md)：四组必需证据、复用方式、成本限制。
- [方法实现规范](reproducibility/method_spec.md)：精确算法、基线、建议初始配置。
- [实验配置与冻结清单](reproducibility/experiment_manifest.yaml)：当前development设置、待冻结项与正式结果槽。
- [迭代审稿记录](revision_audit.md)：逐轮问题、修复、尚依赖结果的结论。
- 功率当前定义以[6design.tex](6design.tex)和method_spec为准；[power_model_layers.md](power_model_layers.md)下半部含已替代的旧规格/Vista历史内容。
- [主线与证据映射](ideal_storyline.md)。

## 编译与核查

在本目录运行：

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

PDF 检查和数学核查使用：

    /Users/shuyuanfan/miniconda3/envs/writing/bin/python tools/check_manuscript.py

核查工具不运行论文实验，不生成拟造数据，不把占位符当作完成的结果。
沿用 AAAI 2027 模板仅作紧凑的 7+2 页写作约束；没有决定新投稿会议或声称其投稿窗口仍开放。

## Cluster交接

代码仓库docs/RESEARCH_HANDOFF.md是交接总入口。2026-09-21将当前正文源码、PDF和上述规范以普通文件导入代码仓库paper/，可通过Git同步到cluster，不要求回传结果压缩包。之后若在cluster继续写稿，以代码仓库paper/中的提交为准；本机原newest_writing目录保留交接时版本，不会自动双向同步。main.pdf此次内容未改，8页（7页正文+1页参考文献）。

历史research_review/revision_audit/method_spec的早期小节保留设计演化；发生冲突时，以交接总入口、当前正文及method_spec最新更新为准。
