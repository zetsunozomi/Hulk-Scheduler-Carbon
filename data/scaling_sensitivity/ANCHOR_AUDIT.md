# Medium/XL 的 16 节点运行时锚点核查

2026-09-24。区分旧文件证据、作者补充和计算推论；本记录不产生新实测。

## 已确定的锚点

| 模型 | 旧代码运行时 | 旧表 16-node node-hours | 除以 16 后 | 采用 T16 |
|---|---:|---:|---:|---:|
| Medium（legacy 345M 标签） | 22.6h | 361.60 | 22.6h | 22.6h |
| XL（约 1.5B） | 110h | 1760.00 | 110h | **110h** |

- `src/sim/application.py` 的构造函数明确赋值 `hrs_at_16=22.6` / `110`；对应注释也重复了这两个数。
- `data/old_gpt/source-table.tex` 是旧稿原表副本；表的单位是完成 100000 次训练迭代的 node-hours。
- 旧代码的其他规模效率与旧表不完全相同。本设计仅取二者一致的 16 节点时间，不混用其他点。
- 两个来源的 SHA-256 在 `design.json` 中绑定。Large 退出活动实验；原表完整保留。

## 作者确认的测量设置

2026-09-24 对话确认，两种模型均为：100000 **optimizer steps**；16 节点；
每节点四张 NVIDIA A100 **40GB**；**BF16**；**global batch=512 sequences**；
**sequence length=1024 tokens**。合计 64 GPUs、每 optimizer step 524288 tokens；
全部工作量为 52.4288 billion processed tokens（不是独立数据集大小）。

这是作者报告的设置，不是从本地旧日志独立恢复的证据。原始训练日志、软件版本、
microbatch/accumulation、数据/张量/流水并行方式、网络互联、计时边界和重复测量仍未找到。
因此可以在稿件中记录作者报告的测量条件，但不能声称本轮重现实测或完成原始日志审计。
本模拟将锚点解释为训练时间；若原计时包含数据预处理、评估或 checkpoint，需先核对，
避免与每 chunk 额外 300s+300s 模拟开销重复计费。

## 算术与常识检查

| 指标 | Medium | XL |
|---|---:|---:|
| 总训练时间 | 22.6h | 110h |
| 秒/optimizer step | 0.8136 | 3.9600 |
| 集群 tokens/s | 644405 | 132396 |
| 每 GPU tokens/s（均摊） | 10068.8 | 2068.7 |
| GPU-hours | 1446.4 | 7040.0 |
| 近似模型 TFLOPS/GPU | 20.84 | 18.62 |
| 相对 312 TFLOPS 理论峰值 | 6.68% | 5.97% |

最后两行仅为数量级检查：以 legacy 参数近似值 N=345M / 1.5B，
使用 `6*N*global_batch*sequence_length / (step_seconds*64)` 估算。
[Kaplan et al. §2.1](https://arxiv.org/pdf/2001.08361)用约 6N FLOPs/token 估计非 embedding 训练计算。
这里使用 legacy 总参数标签近似 N，并忽略精确 attention/embedding、重计算、优化器等差异，
因此这些百分比**不是测得的 SM 利用率，也不是严谨测量的 MFU**。
[NVIDIA A100 官方数据表](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf)
给出稠密 BF16 Tensor Core 峰值 312 TFLOPS；未使用结构化稀疏的 624 TFLOPS。
[NVIDIA 性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
说明实际吞吐还受内存、算子大小和延迟等影响。

结论：**没有超出硬件常识，但在 BF16 的 64 张 A100 上显得利用率偏低。**
时间比例 110/22.6≈4.87，参数比例约 1.5/.345≈4.35，没有明显数量级冲突。
如果使用纯数据并行，每 GPU 每 step 分到 8 条序列；这是有效 batch，并不证明实际 microbatch=8。
小 microbatch、通信、实现和输入流水线都可能拉低有效吞吐，但没有原始 profile 不能指定原因。
Reviewer 仍可能要求训练实现和计时日志；模拟消融能检验调度结论对吞吐假设的依赖，
不能代替测量来源证明，也不能用“范围大所以必然覆盖现实”来回答来源问题。

## 实现关系

固定工作量 U 后，只需 T16 和各规模 eta_n：
`T_n=T16*(16*eta16)/(n*eta_n)`，`q_n=(U/T16)*(n*eta_n)/(16*eta16)`。
eta 相对四节点归一化；若给的是相对十六节点的效率，先统一定义再输入。
只给 eta32 不足以确定 eta8/eta16；v2 因而存四个显式效率值，并加入相同端点但不同拐点的情景。
