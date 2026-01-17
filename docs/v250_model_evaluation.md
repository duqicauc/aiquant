# v2.5.0 模型全面评估报告

**评估时间**: 2026-01-13  
**评估范围**: 训练样本、特征工程、验证过程、模型参数

---

## 📊 一、训练样本评估

### 1.1 样本来源与质量 ✅

**正样本**:
- ✅ 使用 `min_listing_days=300`，过滤不稳定的次新股
- ✅ 合并v3（最新扫描）和v4（历史数据），样本更全面
- ✅ 正样本条件：连续3周阳线 + 总涨幅≥50% + 最高涨幅≥70%
- ✅ 时间跨度：2000年至今，覆盖多个市场周期

**负样本**:
- ✅ 普通负样本：同期其他股票（1:1比例）
- ✅ 硬负样本：相似条件但未成功的股票（提升模型区分度）
- ✅ 负样本数量充足，正负比例合理

**评估结论**: ✅ **样本质量优秀**
- 样本筛选标准严格，能确保正样本质量
- 300天上市要求确保能计算所有长周期技术指标
- 正负样本比例合理，有助于模型学习

### 1.2 样本分布 ⚠️

**潜在问题**:
- ⚠️ **未使用时间序列划分**：`train_test_split` 使用随机划分，存在**未来函数风险**
- ⚠️ 训练集可能包含未来数据，导致过拟合和泛化能力差

**建议**:
```python
# 应该使用时间序列划分
def time_series_split(df, test_size=0.2):
    df = df.sort_values('trade_date')
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test
```

**评估结论**: ⚠️ **需要改进** - 必须使用时间序列划分避免未来函数

---

## 🔧 二、特征工程评估

### 2.1 特征完整性 ✅

**特征数量**:
- 正样本：106个特征
- 负样本：97个特征
- 共同特征：约90+个

**新增233日均线特征**:
- ✅ `ma_233d` - 233日均线（长期趋势）
- ✅ `return_233d` - 233日收益率
- ✅ `volatility_233d` - 233日波动率
- ✅ `breakout_high_233d` - 突破233日新高
- ✅ `max_drawdown_233d` - 233日最大回撤
- ✅ 支撑/阻力、趋势强度等相关特征

**特征体系**:
- ✅ 多周期体系完整：5/8/10/20/34/55/233天
- ✅ 涵盖动量、量价、趋势、风险等维度
- ✅ 233天≈11个月，接近完整年度周期

**评估结论**: ✅ **特征工程优秀**
- 特征数量充足，覆盖全面
- 233日均线是重要的长期趋势指标
- 斐波那契周期体系完整

### 2.2 特征质量 ⚠️

**潜在问题**:
- ⚠️ 正负样本特征数量不一致（106 vs 97），可能导致特征对齐问题
- ⚠️ 需要检查零方差特征、高相关特征对
- ⚠️ 需要检查缺失值和异常值处理

**建议**:
1. 确保正负样本使用相同的特征集
2. 运行 `evaluate_training_data_quality.py` 检查特征质量
3. 移除零方差特征和高相关特征对

**评估结论**: ⚠️ **需要验证** - 建议运行质量评估脚本

---

## 🔍 三、验证过程评估

### 3.1 数据划分 ⚠️ **严重问题**

**当前实现**:
```python
# train_v250_model.py 第220-225行
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)
```

**问题**:
1. ❌ **使用随机划分而非时间序列划分** - 存在未来函数风险
2. ❌ 训练集可能包含测试集之后的数据
3. ❌ 这会导致模型在测试集上表现虚高，但实际应用时表现差

**正确做法**:
```python
# 应该按 trade_date 或 t1_date 排序后划分
df = df.sort_values('trade_date')
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]
```

**评估结论**: ❌ **验证过程不合理** - 必须修复

### 3.2 概率校准 ✅

**实现**:
- ✅ 使用 `IsotonicRegression` 进行概率校准
- ✅ 使用独立的校准集（从训练集中分出25%）
- ✅ 校准前后概率分布对比

**评估结论**: ✅ **概率校准合理**

### 3.3 评估指标 ⚠️

**当前实现**:
```python
# 只评估了不同阈值下的准确率
for thresh in [0.9, 0.8, 0.7, 0.6, 0.5]:
    cal_high = cal_probs >= thresh
    acc = y_test[cal_high].mean()
```

**问题**:
- ⚠️ 缺少AUC、精确率、召回率、F1等完整指标
- ⚠️ 缺少特征重要性分析
- ⚠️ 缺少混淆矩阵

**建议**:
```python
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report

# 计算AUC
auc = roc_auc_score(y_test, cal_probs)

# 计算精确率-召回率曲线
precision, recall, thresholds = precision_recall_curve(y_test, cal_probs)

# 分类报告
report = classification_report(y_test, (cal_probs >= 0.5).astype(int))
```

**评估结论**: ⚠️ **评估指标不完整** - 建议补充

---

## ⚙️ 四、模型参数评估

### 4.1 XGBoost参数 ✅

**当前参数**:
```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 3,              # ✅ 较浅，防过拟合
    'learning_rate': 0.03,       # ✅ 较小，需要更多轮次
    'subsample': 0.6,            # ✅ 行采样，防过拟合
    'colsample_bytree': 0.5,     # ✅ 列采样，防过拟合
    'min_child_weight': 10,      # ✅ 较大，防过拟合
    'gamma': 0.3,                # ✅ 正则化
    'reg_alpha': 1.0,            # ✅ L1正则化
    'reg_lambda': 3.0,           # ✅ L2正则化
    'scale_pos_weight': 1.5,     # ✅ 处理类别不平衡
    'random_state': 42,
    'n_jobs': -1
}
num_boost_round=300,
early_stopping_rounds=30
```

**评估**:
- ✅ 参数设置**保守且合理**，充分考虑了防过拟合
- ✅ 使用早停机制，避免过度训练
- ✅ 使用类别权重处理样本不平衡

**评估结论**: ✅ **模型参数合理**

### 4.2 训练配置 ⚠️

**潜在问题**:
- ⚠️ `learning_rate=0.03` 较小，配合 `num_boost_round=300` 可能不够
- ⚠️ 建议使用学习率衰减或增加轮次

**建议**:
```python
# 方案1: 增加轮次
num_boost_round=500

# 方案2: 使用学习率衰减
# 或使用回调函数动态调整学习率
```

**评估结论**: ⚠️ **训练配置可优化**

---

## 📋 五、综合评估总结

### 5.1 优点 ✅

1. ✅ **样本质量高**：300天上市要求，严格筛选标准
2. ✅ **特征工程优秀**：233日均线特征，多周期体系完整
3. ✅ **模型参数合理**：防过拟合措施充分
4. ✅ **概率校准**：使用IsotonicRegression校准概率
5. ✅ **硬负样本**：提升模型区分度

### 5.2 严重问题 ❌

1. ❌ **验证过程不合理**：使用随机划分而非时间序列划分
   - **影响**：存在未来函数风险，测试集表现虚高
   - **优先级**：🔴 **必须修复**

2. ⚠️ **评估指标不完整**：缺少AUC、精确率、召回率等
   - **影响**：无法全面评估模型性能
   - **优先级**：🟡 **建议补充**

3. ⚠️ **特征对齐问题**：正负样本特征数量不一致
   - **影响**：可能导致特征对齐错误
   - **优先级**：🟡 **需要验证**

### 5.3 改进建议

#### 🔴 高优先级（必须修复）

1. **修复数据划分方式**
   ```python
   # 修改 train_v250_model.py
   def time_series_split(df, test_size=0.2, cal_size=0.15):
       df = df.sort_values('trade_date').reset_index(drop=True)
       n = len(df)
       test_start = int(n * (1 - test_size))
       cal_start = int(n * (1 - test_size - cal_size))
       
       train = df.iloc[:cal_start]
       cal = df.iloc[cal_start:test_start]
       test = df.iloc[test_start:]
       return train, cal, test
   ```

2. **确保特征对齐**
   ```python
   # 在加载数据后，确保正负样本使用相同的特征集
   common_features = set(df_pos.columns) & set(df_neg.columns)
   exclude_cols = ['ts_code', 'name', 't1_date', 't2_date', 'sample_id', 'label', 'trade_date']
   feature_cols = [f for f in common_features if f not in exclude_cols]
   ```

#### 🟡 中优先级（建议改进）

3. **补充评估指标**
   ```python
   from sklearn.metrics import (
       roc_auc_score, precision_recall_curve, 
       classification_report, confusion_matrix
   )
   
   # 计算完整指标
   auc = roc_auc_score(y_test, cal_probs)
   report = classification_report(y_test, (cal_probs >= 0.5).astype(int))
   ```

4. **优化训练配置**
   - 增加 `num_boost_round` 到 500
   - 或使用学习率衰减策略

5. **运行数据质量评估**
   ```bash
   python scripts/evaluate_training_data_quality.py
   ```

#### 🟢 低优先级（可选优化）

6. **特征重要性分析**
   ```python
   feature_importance = booster.get_score(importance_type='gain')
   # 可视化特征重要性
   ```

7. **交叉验证**
   - 考虑使用时间序列交叉验证（TimeSeriesSplit）

---

## 🎯 六、最终评分

| 评估项 | 评分 | 说明 |
|--------|------|------|
| 训练样本 | ⭐⭐⭐⭐⭐ | 样本质量高，筛选标准严格 |
| 特征工程 | ⭐⭐⭐⭐⭐ | 特征全面，233日均线特征优秀 |
| 验证过程 | ⭐⭐ | **严重问题：随机划分存在未来函数风险** |
| 模型参数 | ⭐⭐⭐⭐ | 参数合理，防过拟合措施充分 |
| **综合评分** | **⭐⭐⭐** | **需要修复验证过程后才能使用** |

---

## 📝 七、行动建议

### 立即行动（修复后才能训练）

1. ✅ 修改 `train_v250_model.py`，使用时间序列划分
2. ✅ 确保正负样本特征对齐
3. ✅ 运行数据质量评估脚本

### 训练后验证

4. ✅ 补充完整的评估指标（AUC、精确率、召回率等）
5. ✅ 分析特征重要性
6. ✅ 在真实数据上验证模型表现

---

**评估结论**: v2.5.0模型在样本质量和特征工程方面表现优秀，但**验证过程存在严重问题**，必须修复后才能用于实际训练。修复后，预期模型性能会显著提升。
