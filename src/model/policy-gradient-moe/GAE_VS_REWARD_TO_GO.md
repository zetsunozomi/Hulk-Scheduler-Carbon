# GAE vs Reward-to-Go 对比

## 核心算法对比

### Reward-to-Go (旧方法)
```
┌─────────────────────────────────────────┐
│  1. 计算Cost-to-Go                      │
│     R_t = r_t + γr_{t+1} + γ²r_{t+2}... │
│                                          │
│  2. Advantage = R_t - V(s_t)            │
│     (简单差值)                           │
│                                          │
│  3. Policy Loss = log π * Advantage     │
│                                          │
│  4. Critic Loss = MSE(V, R_t)           │
└─────────────────────────────────────────┘

问题:
❌ 高方差 (high variance)
❌ Critic难以学习
❌ 训练不稳定
❌ 容易过早收敛
```

### GAE (新方法)
```
┌─────────────────────────────────────────┐
│  1. 计算TD Error                        │
│     δ_t = r_t + γV(s_{t+1}) - V(s_t)   │
│                                          │
│  2. GAE Advantage                       │
│     A_t = δ_t + (γλ)δ_{t+1} + ...      │
│                                          │
│  3. Normalize Advantage                 │
│     A_norm = (A - μ) / σ                │
│                                          │
│  4. Policy Loss = log π * A_norm        │
│                                          │
│  5. Returns = A + V                     │
│                                          │
│  6. Critic Loss = Huber(V, Returns)     │
└─────────────────────────────────────────┘

优势:
✅ 低方差 (lower variance)
✅ Critic学习更快
✅ 训练稳定
✅ 保持探索
✅ 业界标准 (PPO, SAC都用)
```

---

## 训练流程对比

### 旧流程 (单一优化器)
```
Episode Loop:
  ├─ play_one_episode()
  │   └─ 收集 (states, actions, rewards, values)
  │
  ├─ 计算 cost-to-go
  ├─ 计算 advantage = cost-to-go - values
  ├─ policy_loss = log_prob * advantage
  ├─ critic_loss = MSE(values, cost-to-go)
  │
  └─ total_loss = policy + 0.5*critic + entropy
      └─ optimizer.step()  # 同时更新policy和critic
```

### 新流程 (分离优化器 + GAE)
```
Episode Loop:
  ├─ play_one_episode()
  │   └─ 收集 (states, actions, rewards, values)
  │
  ├─ 计算 GAE advantages
  ├─ 归一化 advantages
  ├─ 计算 returns = advantages + values
  │
  ├─ policy_loss = log_prob * advantages_norm
  ├─ critic_loss = Huber(values, returns)
  │
  ├─ critic_loss.backward(retain_graph=True)
  ├─ clip_grad_norm_(critic_params, 0.5)
  ├─ critic_optimizer.step()  # 先更新critic
  │
  ├─ policy_loss.backward()
  ├─ clip_grad_norm_(policy_params, 0.5)
  └─ policy_optimizer.step()  # 再更新policy
```

---

## 超参数对比

| 参数                | 旧值         | 新值  | 说明              |
| ------------------- | ------------ | ----- | ----------------- |
| **算法**            | Reward-to-Go | GAE   | 核心改变          |
| **gamma (γ)**       | 0.99         | 0.95  | 降低长期依赖      |
| **lambda (λ)**      | N/A          | 0.95  | GAE参数           |
| **优化器**          | 1个          | 2个   | 分离policy/critic |
| **critic_lr**       | 1x           | 3x    | Critic学习更快    |
| **policy_lr**       | 1x           | 1x    | 保持不变          |
| **梯度裁剪**        | 无           | 0.5   | 防止爆炸          |
| **Advantage归一化** | 无           | 有    | 稳定训练          |
| **Critic Loss**     | MSE          | Huber | 更鲁棒            |

---

## 期望效果对比

### 训练稳定性

#### 旧方法
```
Critic Loss:  12.6 → 6.3 → 13.4 → 8.2 → ...  (波动大)
Policy Loss:  42.1 → 27.8 → 42.7 → ...       (不稳定)
Entropy:      1.38 → 0.97 → 0.67 → ...       (快速下降)
```

#### 新方法 (期望)
```
Critic Loss:  8.0 → 5.2 → 3.8 → 2.5 → ...   (稳定下降)
Policy Loss:  20.0 → 15.0 → 12.0 → 10.0 ... (平稳收敛)
Entropy:      1.38 → 1.20 → 1.05 → 0.90 ... (缓慢下降)
```

### 学习效果

#### 旧方法
```
Wait Probability:
  ToD 3 (夜间):  0.50 → 0.78 → 0.95  (过度等待)
  ToD 12 (白天): 0.50 → 0.64 → 0.89  (没有区分)
  
问题: 学会了"总是等待"，但没学会"何时等待"
```

#### 新方法 (期望)
```
Wait Probability:
  ToD 3 (夜间):  0.50 → 0.65 → 0.80  (合理等待)
  ToD 12 (白天): 0.50 → 0.35 → 0.20  (积极提交)
  
目标: 学会"在高碳时段等待，低碳时段提交"
```

---

## 代码改动统计

### 修改的文件
1. ✅ `policy_gradient.py` (主要改动)
   - 添加 ~50 行 (GAE实现)
   - 修改 ~30 行 (优化器分离)
   - 删除 ~20 行 (旧reward-to-go)

2. ✅ `validation.py` (兼容性修改)
   - 修改 ~5 行 (返回值处理)

### 新增的文件
1. 📄 `GAE_IMPLEMENTATION_SUMMARY.md` (详细说明)
2. 📄 `FINAL_CHECKLIST.md` (检查清单)
3. 📄 `validate_gae.sh` (验证脚本)
4. 📄 `GAE_VS_REWARD_TO_GO.md` (本文档)

---

## 性能预测

### Critic Loss
```
旧方法: 平均 ~8-12, 波动大
新方法: 平均 ~2-4,  稳定
改善:   50-70% ↓
```

### 训练速度
```
旧方法: 100 episodes 收敛 (但收敛到错误策略)
新方法: 50-80 episodes 收敛 (到正确策略)
改善:   20-50% 更快
```

### 最终性能
```
Carbon Emission:
  旧方法: 学会等待，但不分时段 → 次优
  新方法: 学会在高碳时段等待 → 最优
  
TAT (Turnaround Time):
  旧方法: 过度等待 → TAT过长
  新方法: 平衡等待 → TAT合理
```

---

## 理论基础

### 为什么GAE更好？

#### 1. Bias-Variance Tradeoff
```
Reward-to-Go:
  Bias:     低 (unbiased)
  Variance: 高 (high variance)
  结果:     训练不稳定

GAE (λ=0.95):
  Bias:     略高 (slightly biased)
  Variance: 低 (low variance)
  结果:     训练稳定
```

#### 2. Credit Assignment
```
Reward-to-Go:
  问题: 长期回报难以归因到具体action
  
GAE:
  优势: TD error提供即时反馈
       λ参数控制长期vs短期权重
```

#### 3. Value Function Learning
```
Reward-to-Go:
  Critic学习目标: 完整episode的累积回报
  问题: 方差大，学习慢
  
GAE:
  Critic学习目标: Bootstrapped returns
  优势: 方差小，学习快
```

---

## 实验建议

### 对比实验
```bash
# 1. 用旧方法训练 (如果还有旧代码)
python policy_gradient_old.py ... > log_old.txt

# 2. 用新方法训练
python policy_gradient.py ... > log_new.txt

# 3. 对比指标
./compare_logs.sh log_old.txt log_new.txt
```

### A/B测试
```bash
# 测试不同的lambda值
for lambda in 0.9 0.95 0.98; do
    python policy_gradient.py ... \
        --model_output gpt345M_lambda${lambda}
done
```

---

## 总结

### 关键改进
1. ✅ **算法**: Reward-to-Go → GAE
2. ✅ **优化**: 单一优化器 → 分离优化器
3. ✅ **稳定性**: 无归一化 → Advantage归一化
4. ✅ **鲁棒性**: MSE → Huber Loss
5. ✅ **安全性**: 无裁剪 → 梯度裁剪

### 期望结果
- 🎯 Critic Loss下降 50-70%
- 🎯 训练速度提升 20-50%
- 🎯 学会时间依赖的wait策略
- 🎯 Carbon emission降低
- 🎯 训练过程稳定

### 下一步
1. 运行 `./validate_gae.sh` 验证代码
2. 启动训练并监控指标
3. 对比新旧方法的性能

**状态**: ✅ 准备就绪，可以开始训练！
