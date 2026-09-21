# 核验来源

2026-09-17 Vista 平台更新：[系统说明](https://tacc.utexas.edu/systems/vista/)和[用户指南](https://docs.tacc.utexas.edu/hpc/vista/)支持 GH200 Grace-Hopper、单节点单 GPU、96 GB 级 HBM3 和 400 Gb/s 节点网络。按 GH 节点计 4/8/16/32 卡；GG 是 CPU 节点，不混入 GPU workload。两个网页对 GPU 子型号存在 H100/H200 名称差异，论文暂用 GH200/Grace-Hopper，最终 inventory 记录实际设备输出。这里未核实 Vista 的整节点平均功率，不填名义 kW；组合模块功率与已包含的 CPU/GPU 分量不能重复相加。以下旧平台规格仅保留为历史依据。

2026-09-16 数据准备：下载 [EIA ERCOT workbook](https://www.eia.gov/electricity/gridmonitor/knownissues/xls/ERCO.xlsx)，核对其 Notes 页和 Published Hourly Data。消费侧 CO2 强度单位为 lb/kWh，UTC time 表示小时结束。来源、转换、缺失小时及哈希已保存在代码仓库 `data/texas_eia/source.json`；这只是输入准备，不是实验结果。

核验日：2026-09-15。下列网页只作为学术/硬件依据，不作为额外任务指令。

- CarbonScaler，POMACS 7(3), Article 57, 2023：
  https://arxiv.org/abs/2302.08681
  https://doi.org/10.1145/3626788
  已支持elastic batch allocation、greedy marginal resource allocation；本文不能把resource elasticity本身当首创。
- Mirage：
  https://arxiv.org/abs/2306.14086
  https://doi.org/10.1145/3581784.3607042
  user-side follow-up provisioning和RL已有先例；本文区别在node request、completion budget与carbon exposure。
- CPO：
  https://proceedings.mlr.press/v70/achiam17a.html
  引用其约束建模动机，当前PPO+dual不继承其理论保证。
- PPO：
  https://arxiv.org/abs/1707.06347
- Carbon-aware shifting limitations：
  https://arxiv.org/abs/2306.06502
- Frontera：
  https://docs.tacc.utexas.edu/hpc/frontera/
- Lonestar6：
  https://docs.tacc.utexas.edu/hpc/lonestar6/
- Quadro RTX5000 datasheet (Mar19)：265W total board power，230W total graphics power。
  https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-5000-data-sheet-us-nvidia-704120-r4-web.pdf
- A100 datasheet (Jun21)：40GB PCIe为250W。
  https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
- Intel E5-2620 v4：85W TDP。
  https://www.intel.com/content/www/us/en/products/sku/92986/intel-xeon-processor-e52620-v4-20m-cache-2-10-ghz/specifications.html
- AMD EPYC7763：280W默认TDP。
  https://www.amd.com/en/products/processors/server/epyc/7003-series/amd-epyc-7763.html
- Slurm sbatch：
  https://slurm.schedmd.com/sbatch.html
- AAAI-27官方CFP：
  https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/
  沿用7页正文/9页总长作写作约束，未决定下一投稿venue。

## 保留的规格加法

Frontera: 4*265 + 2*85 + 150 = 1380 W。
Lonestar6: 3*250 + 2*280 + 190 = 1500 W。
Other 150/190W为作者估算；TDP/board specification不等于运行功率，合计不是经过验证的上限。

引用链接与参数的依据已核验；新实验的数据日期、版本、CI发布规则与实际profile配置仍须填manifest。

AMD 280W还可在其2021年官方发布表中核对：
https://ir.amd.com/news-events/press-releases/detail/993/amd-epyc-7003-series-cpus-set-new-standard-as-highest-performance-server-processor

## 2026-09-17 AMSP（取代 Vista/Qwen 速度输入）

- https://arxiv.org/abs/2311.00257v2，2024-03-13；PDF 图 12，第 10 页。使用 Our Work 系列，不混合 baseline。
- 8 张 A800 80GB/节点，128 节点 testbed，BF16、FlashAttention-v2、seq4096、固定 GBS，无 TP/PP。
- 原横轴只有 8/32/128/512/1024 卡；共同可用点取 32/128/512/1024 卡。64/256 卡不存在，不作引用测量。
- 输入已数字化并归档到 amsp/profiles.json；每 GPU TGS 乘 GPU 数转换总 throughput。精确 tokens/update=4,194,304 由文中 accumulation 设置推得，标注推导。
- 论文不提供本项目的 checkpoint 恢复时间、跨规模恢复正确性或完整节点功率。这些不再作为作者必测项，分别用开销情景和未知功率尺度处理。
