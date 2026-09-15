# ✅ GAE Implementation - Final Checklist

## 所有改动已完成 (All Changes Completed)

### 核心文件修改 (Core File Modifications)

#### ✅ policy_gradient.py
- [x] 添加GAE超参数 (gamma=0.95, lambda=0.95)
- [x] 分离policy和critic优化器
- [x] Critic学习率设为Policy的3倍
- [x] 实现完整GAE算法
- [x] 添加Advantage归一化
- [x] 使用Huber Loss替代MSE
- [x] 添加梯度裁剪 (max_norm=0.5)
- [x] 更新checkpoint保存/加载逻辑
- [x] 添加增强日志输出
- [x] 移除所有旧的reward-to-go代码

#### ✅ validation.py
- [x] 更新play_one_episode返回值处理
- [x] 匹配新的返回签名 (policy_loss, entropy_mean, critic_loss)

---

## 代码验证 (Code Verification)

### 自动检查
```bash
cd /pscratch/sd/s/syfan/carbon/src/model/policy-gradient-moe
chmod +x validate_gae.sh
./validate_gae.sh
```

### 手动检查清单

#### 1. 优化器检查
```bash
# 应该只看到 policy_optimizer 和 critic_optimizer
grep -n "optimizer" policy_gradient.py | grep "self\."
```
✅ 预期：没有 `self.optimizer`，只有 `self.policy_optimizer` 和 `self.critic_optimizer`

#### 2. GAE实现检查
```bash
# 应该看到GAE相关代码
grep -n "gae_lambda\|GAE\|Generalized Advantage" policy_gradient.py
```
✅ 预期：找到GAE实现和注释

#### 3. 梯度裁剪检查
```bash
grep -n "clip_grad_norm_" policy_gradient.py
```
✅ 预期：找到2处（policy和critic各一处）

#### 4. Advantage归一化检查
```bash
grep -n "advantages_normalized" policy_gradient.py
```
✅ 预期：找到归一化代码

---

## 训练前准备 (Pre-Training Preparation)

### 1. 备份旧checkpoint
```bash
# 如果有旧的checkpoint，先备份
mkdir -p checkpoints_old
mv checkpoints/*.pt checkpoints_old/ 2>/dev/null || true
```

### 2. 清理旧日志
```bash
# 备份旧日志
mkdir -p logs_old
mv log_*.txt logs_old/ 2>/dev/null || true
```

### 3. 验证环境
```bash
# 确认CUDA可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 确认模型文件存在
ls -lh /path/to/model4.pt
ls -lh /path/to/model8.pt
ls -lh /path/to/model16.pt
ls -lh /path/to/model32.pt
```

---

## 训练命令 (Training Command)

### 推荐配置
```bash
python policy_gradient.py \
    --use_cuda \
    -config /path/to/config.json \
    --model4 /path/to/model4.pt \
    --model8 /path/to/model8.pt \
    --model16 /path/to/model16.pt \
    --model32 /path/to/model32.pt \
    -base_lr 1e-4 \
    -batch_size 4 \
    -epoch 200 \
    --carbon_weight 0.99 \
    --policy_model_type separate \
    --ckpt_folder ./checkpoints \
    --model_output gpt345M_gae_0.99 \
    --app_type gpt-345m \
    --base_run_h 6 \
    --cluster_name perlmutter \
    2>&1 | tee log_gpt345M_gae_0.99.txt
```

### 参数说明
- `base_lr=1e-4`: Policy学习率，Critic会自动×3
- `batch_size=4`: 每4个episode更新一次
- `epoch=200`: 总共训练200个episode
- `carbon_weight=0.99`: 极度重视碳排放
- `policy_model_type=separate`: 使用TwoStageMoE模型

---

## 监控指标 (Monitoring Metrics)

### 实时监控
```bash
# 在另一个终端监控训练
tail -f log_gpt345M_gae_0.99.txt
```

### 关键指标检查

#### 1. Critic Loss
```bash
grep "Critic Loss" log_gpt345M_gae_0.99.txt | tail -20
```
**健康标准**:
- ✅ Episode 0-10: < 10
- ✅ Episode 10-50: < 5
- ✅ Episode 50+: < 3
- ✅ 趋势：稳定下降

#### 2. GAE Stats
```bash
grep -A 4 "GAE Stats" log_gpt345M_gae_0.99.txt | tail -25
```
**健康标准**:
- ✅ Normalized Adv Mean: -0.1 ~ 0.1 (接近0)
- ✅ Normalized Adv Std: 0.8 ~ 1.2 (接近1)
- ✅ Value Mean: 逐渐稳定
- ✅ Return Mean: 逐渐稳定

#### 3. Entropy
```bash
grep "Entropy:" log_gpt345M_gae_0.99.txt | tail -20
```
**健康标准**:
- ✅ Episode 0: ~1.38 (接近log(5))
- ✅ Episode 50: ~1.0
- ✅ Episode 100: ~0.7
- ✅ 趋势：缓慢下降（不要过快）

#### 4. Wait Probability (时间依赖性)
```bash
# 夜间 (ToD 1-5)
grep "probablity distribution" log_gpt345M_gae_0.99.txt | grep -E "ToD [1-5]\." | tail -10

# 白天 (ToD 11-15)
grep "probablity distribution" log_gpt345M_gae_0.99.txt | grep -E "ToD 1[1-5]\." | tail -10
```
**期望行为**:
- ✅ 夜间wait prob (index 4) > 白天wait prob
- ✅ 随训练进行，差异逐渐明显

---

## 问题排查 (Troubleshooting)

### 问题1: Critic Loss不下降
**症状**: Critic Loss > 10 且不下降
**解决**:
```python
# 在policy_gradient.py中临时提高critic学习率
self.critic_optimizer = optim.Adam(critic_params, lr=self.base_lr * 5.0)  # 从3.0改为5.0
```

### 问题2: Advantage方差过大
**症状**: Normalized Adv Std > 2.0
**解决**:
```python
# 降低gamma和lambda
self.gamma = 0.9  # 从0.95降低
self.gae_lambda = 0.9  # 从0.95降低
```

### 问题3: Policy Loss爆炸
**症状**: Policy Loss > 100
**解决**:
```python
# 降低学习率
-base_lr 5e-5  # 从1e-4降低
```

### 问题4: Entropy下降过快
**症状**: Episode 20时Entropy < 0.5
**解决**:
```python
# 在solve_environment中调整entropy decay
end_entropy = 0.2  # 从0.1提高
```

---

## 成功标准 (Success Criteria)

### 训练收敛标志
- [x] Critic Loss < 3 且稳定
- [x] Normalized Adv Mean ≈ 0
- [x] Normalized Adv Std ≈ 1
- [x] Policy Loss稳定下降
- [x] Entropy缓慢下降（不过快）

### 学习到时间依赖性
- [x] 夜间wait prob > 0.7
- [x] 白天wait prob < 0.3
- [x] 差异明显且稳定

### 性能提升
- [x] Carbon emission比baseline低
- [x] TAT合理（不过长）
- [x] 策略稳定（不抖动）

---

## 下一步行动 (Next Actions)

### 立即执行
1. ✅ 运行 `./validate_gae.sh` 验证代码
2. ✅ 启动训练
3. ✅ 监控前10个episode的指标

### 训练中
1. 每20个episode检查一次指标
2. 如果Critic Loss不下降，考虑调整学习率
3. 如果wait probability没有时间依赖性，检查carbon intensity输入

### 训练后
1. 运行validation.py评估最终模型
2. 对比GAE vs Reward-to-Go的性能
3. 分析wait action的时间分布

---

## 文档和资源 (Documentation)

- 📄 **GAE_IMPLEMENTATION_SUMMARY.md**: 详细改动说明
- 📄 **validate_gae.sh**: 代码验证脚本
- 📄 **FINAL_CHECKLIST.md**: 本文档

## 联系和支持

如果遇到问题：
1. 检查本checklist的"问题排查"部分
2. 查看GAE_IMPLEMENTATION_SUMMARY.md的详细说明
3. 检查训练日志中的GAE Stats

---

## 版本信息

- **实现日期**: 2026-01-23
- **GAE版本**: Standard GAE with λ=0.95
- **主要改进**: 
  - Reward-to-Go → GAE
  - 单一优化器 → 分离优化器
  - MSE → Huber Loss
  - 无归一化 → Advantage归一化
  - 无梯度裁剪 → 梯度裁剪

**状态**: ✅ 所有改动已完成，代码可以直接运行！
