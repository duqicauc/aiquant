# 模型升级计划：v2.7.0-ensemble → v2.9.1 修复与升级

> 制定日期: 2026-04-25
> 目标: 在保留 v2.7.0-ensemble 成功经验的基础上，解决 v2.8.0/v2.9.1 的退化问题，训练出指标优于 v2.7.0 的新模型。

---

## 一、Phase 0: 数据质量排查（方案C）✅ 已完成

### 0.1 排查结果

| 检查项 | 结果 | 严重程度 |
|--------|------|----------|
| enhanced/ vs features/ 特征列一致性 | ✅ 179个特征完全一致 | 低风险 |
| breakout 核心特征在数据中存在性 | ✅ 全部存在 | 低风险 |
| v291 硬负样本多出28个市场指数特征 | ⚠️ hs300_*, sh_* 特征在 enhanced 硬负中缺失 | **中风险** |
| v2.9.1 hard_negative_ratio | ❌ 31.4%，远超合理范围15-20% | **高风险** |
| v2.8.0/v2.9.1 训练脚本排除特征 | ❌ 排除了6个核心 breakout 特征 | **致命** |

### 0.2 关键发现

**发现1：v2.7.0-ensemble 实际包含29个 breakout/volume 特征**
```
breakout_high_55d, breakout_ma5, breakout_ma20, breakout_volume_ratio,
breakout_strength_10d/20d/55d/avg/max, breakout_confirmed_10d/20d,
high_volume_breakout（被v2.9.1排除但v2.7.0保留）, ...
```

**发现2：v2.8.0/v2.9.1 排除了6个关键特征**
```python
"breakout_high_10d",      # 10日高点突破（起爆信号）
"breakout_high_20d",      # 20日高点突破（起爆信号）
"breakout_ma10",          # 站上MA10（趋势确认）
"breakout_ma55",          # 站上MA55（中期趋势）
"high_volume_breakout",   # 放量突破（量能确认）
"volume_price_match",     # 量价匹配（资金确认）
```

**发现3：v291 硬负样本多了28个市场指数特征**
这些特征（hs300_*, sh_*）在 enhanced 硬负中不存在，可能导致特征不一致。

---

## 二、Phase 1: 保守升级（方案A）🔴 P0 - 最高优先级

### 目标
基于 v2.7.0-ensemble 的成功经验，用最新数据重新训练，恢复所有退化项。

### 1.1 修复版训练脚本 `scripts/train_v271_conservative.py`

**改进点（对比 v2.7.0-ensemble）：**

| 维度 | v2.7.0-ensemble | v2.9.1 | v2.7.1-conservative（新） |
|------|-----------------|--------|---------------------------|
| 数据来源 | enhanced/ v5 | 混合（enhanced+features）| **enhanced/ v5 统一** |
| 排除特征 | 15个 | 21个（含6个breakout）| **15个（恢复breakout）** |
| scale_pos_weight | 1.5 | 动态 | **1.5（固定）** |
| 概率校准 | ✅ Isotonic | ❌ 无 | **✅ Isotonic** |
| 集成权重 | 三等分 | AUC动态 | **三等分（验证）** |
| hard negative 比例 | ~11% | 31.4% | **≤18%** |

**脚本基于**: `scripts/train_ensemble_model.py`（v2.7.0 同款）

### 1.2 执行步骤

```bash
# Step 1: 运行修复版训练脚本
python scripts/train_v271_conservative.py

# Step 2: 记录训练指标
# 目标: AUC ≥ 0.980, Precision ≥ 0.860, Recall ≥ 0.880

# Step 3: 回测验证
python scripts/backtest_v271_strategy.py
```

### 1.3 成功标准
- 训练指标 **≥ v2.7.0-ensemble**（AUC≥0.982, F1≥0.879）
- 回测胜率 **≥ 50%**
- 最大回撤 **≤ 10%**

---

## 三、Phase 2: 修复 v2.9.1（方案B）🟡 P1 - 中等优先级

### 目标
在保守升级成功的基础上，引入 v2.9.1 的 hard negative 增强思路，但修复其问题。

### 2.1 修复清单

| # | 修复项 | 具体操作 | 风险 |
|---|--------|----------|------|
| 1 | 恢复 breakout 特征 | 从 exclude_cols 中删除6个特征 | 低 |
| 2 | 统一数据来源 | 硬负样本改用 enhanced/ 目录 | 低 |
| 3 | 加回概率校准 | 集成预测后加入 IsotonicRegression | 低 |
| 4 | 收紧 hard negative 阈值 | min_return 15% → 20%, samples_per_date 30 → 15 | 中 |
| 5 | 控制 hard negative 比例 | 下采样至 ≤18% | 中 |
| 6 | 权重策略对比 | A) 固定三等分 B) AUC动态 C) 差异阈值法 | 中 |

### 2.2 修复版脚本 `scripts/train_v291_fixed.py`

基于 `scripts/train_v291_model.py`，做以下修改：

```python
# 修改1: 恢复 breakout 特征
exclude_cols = [
    # ... 保留原有排除项
    # 删除: breakout_high_10d, breakout_high_20d, breakout_ma10,
    #       breakout_ma55, high_volume_breakout, volume_price_match
]

# 修改2: 统一数据来源
hard_new_file = PROJECT_ROOT / "data" / "training" / "enhanced" / "hard_negative_feature_data_34d_v5_enhanced.csv"

# 修改3: 加回概率校准（集成预测后）
from sklearn.isotonic import IsotonicRegression
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(ensemble_pred, y_cal)

# 修改4: 权重策略优化
if max(weights.values()) - min(weights.values()) < 0.02:
    weights = {"xgboost": 1/3, "lightgbm": 1/3, "catboost": 1/3}
```

### 2.3 成功标准
- 训练指标 **≥ v2.7.0-ensemble**
- hard negative 比例控制在 **15-20%**
- 回测表现稳定，不同季度波动 < 5%

---

## 四、Phase 3: 评估与对比 🔵 P2 - 验证阶段

### 3.1 对比维度

| 维度 | v2.7.0-ensemble | v2.7.1-conservative | v2.9.1-fixed |
|------|-----------------|---------------------|--------------|
| 训练指标（AUC/P/R/F1）| 基准 | 目标: ≥基准 | 目标: ≥基准 |
| 纯模型 Top10 次日胜率 | 50.2% | 目标: ≥50% | 目标: ≥50% |
| 策略回测收益率 | 基准 | 目标: ≥基准 | 目标: ≥基准 |
| 策略回测胜率 | 基准 | 目标: ≥基准 | 目标: ≥基准 |
| 最大回撤 | 基准 | 目标: ≤基准 | 目标: ≤基准 |
| 模型稳定性（Top50重叠率）| 24% | 目标: ≥30% | 目标: ≥30% |

### 3.2 自动化评估脚本

```bash
# 生成对比报告
python scripts/evaluate_model_comparison.py \
  --models v2.7.0-ensemble,v2.7.1-conservative,v2.9.1-fixed \
  --period 2025q1,2025q2,2026q1
```

---

## 五、实施优先级与风险矩阵

```
P0 (立即执行) ──────────────────────────────>
  ├─ Phase 0: 数据质量排查 ✅ 已完成
  └─ Phase 1: 保守升级（方案A）
       ├─ 编写 train_v271_conservative.py
       ├─ 训练模型（预计 10-30 分钟）
       └─ 回测验证

P1 (Phase 1 成功后执行) ─────────────────────>
  └─ Phase 2: 修复 v2.9.1（方案B）
       ├─ 编写 train_v291_fixed.py
       ├─ 多组对比实验（权重策略、hard negative 比例）
       └─ 选择最佳配置

P2 (Phase 2 成功后执行) ─────────────────────>
  └─ Phase 3: 评估对比
       ├─ 自动生成对比报告
       └─ 生产环境切流决策
```

---

## 六、Rollback 策略

如果新模型训练失败或指标不达标：
1. **保留 v2.7.0-ensemble 作为 fallback**
2. `current.json` 不修改，直到新模型通过全部评估
3. 模型文件保存到独立版本目录，不覆盖现有版本

---

## 七、预期时间线

| 阶段 | 预计耗时 | 依赖 |
|------|----------|------|
| Phase 0 | 30分钟 | ✅ 已完成 |
| Phase 1 | 1-2小时 | Phase 0 |
| Phase 2 | 2-4小时 | Phase 1 |
| Phase 3 | 1-2小时 | Phase 2 |
| **总计** | **4-8小时** | - |
