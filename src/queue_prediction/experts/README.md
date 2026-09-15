70 维特征的详细构成：
1. Pending Jobs (排队作业) - 共 24 维

1 维: pending_count (排队作业数量)
1 维: pending_nodes (排队作业所需的总节点数)
11 维: pending_time_limit_stats (排队作业的时间限制分布，11个分位点)
11 维: pending_queue_stats (排队作业的等待时间分布，11个分位点)
2. Running Jobs (运行作业) - 共 46 维

1 维: running_count (运行作业数量)
1 维: running_nodes (运行作业占用的总节点数)
11 维: running_time_limit_stats (运行作业的时间限制分布，11个分位点)
11 维: running_queue_stats (运行作业的排队等待时间分布，11个分位点)
11 维: running_runtime_stats (运行作业已运行时间分布，11个分位点)
11 维: running_time_left_stats (运行作业剩余时间分布，11个分位点)
总计验证： $1 + 1 + 11 + 11 + 1 + 1 + 11 + 11 + 11 + 11 = 24 + 46 = 70$

这里的“11个分位点”是指：
在 
get_stats函数中（第 202 行）：

对于 time_left，取的百分位点是 [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] (均匀分布)。
对于其他属性（如 time_limit, queue_wait_sec 等），取的百分位点是 [0, 5, 10, 15, 20, 50, 80, 85, 90, 95, 100] (更关注两端分布)。