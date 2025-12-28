# 机器学习模型选择指南 🤖

针对股票选股模型的完整模型对比与实验设计

---

## 📊 问题定义

### 数据特点
```
任务类型: 二分类（正样本/负样本）
特征维度: 34天 × N个指标（价格、涨跌幅、MACD、RSI、MA、量比等）
样本量: ~2,300个（1,145正 + 1,145负）
数据类型: 时间序列 → 表格化特征
```

### 目标
在T1时刻，根据前34天的数据，预测该股票是否为潜力股（未来3周涨幅>50%）

---

## 🏆 模型推荐排行榜

### 🥇 第一名：树模型（XGBoost / LightGBM）⭐⭐⭐⭐⭐

**最强烈推荐！**

#### 推荐理由

| 维度 | 评分 | 说明 |
|------|------|------|
| **效果** | ⭐⭐⭐⭐⭐ | 在金融表格数据上表现卓越 |
| **速度** | ⭐⭐⭐⭐⭐ | 训练快（几分钟） |
| **易用性** | ⭐⭐⭐⭐⭐ | 调参简单，默认参数就不错 |
| **鲁棒性** | ⭐⭐⭐⭐⭐ | 对缺失值、异常值不敏感 |
| **可解释性** | ⭐⭐⭐⭐⭐ | 可查看特征重要性 |
| **样本需求** | ⭐⭐⭐⭐⭐ | 2000+样本已足够 |

#### 优点 ✅
- **效果好**：在Kaggle金融竞赛中常胜军
- **训练快**：XGBoost训练几分钟，LightGBM更快
- **不过拟合**：有正则化，样本少也不怕
- **特征工程友好**：自动处理特征交互
- **可解释**：知道哪些指标最重要（如：MACD权重0.23，量比0.18...）
- **生产部署简单**：模型小，推理快

#### 缺点 ⚠️
- 不能直接处理原始时间序列（需要特征工程）
- 难以捕捉复杂的时序依赖

#### 适用场景
✅ **你的数据已经是表格型特征** → 完美匹配！  
✅ **样本量2000+** → 足够了！  
✅ **需要快速验证** → 最佳选择！  
✅ **想知道哪些指标重要** → 可解释性强！

#### 实现示例

```python
"""
XGBoost/LightGBM 训练脚本
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# 1. 加载数据
df_pos = pd.read_csv('data/processed/feature_data_34d.csv')
df_pos['label'] = 1

df_neg = pd.read_csv('data/processed/negative_feature_data_v2_34d.csv')
# label=0已包含

df = pd.concat([df_pos, df_neg])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. 特征工程（展平时间序列）
# 每个样本34天 → 需要转换为一行多列
# 方法1：聚合统计特征
features = []
for sample_id in df['sample_id'].unique():
    sample_data = df[df['sample_id'] == sample_id].sort_values('days_to_t1')
    
    feature_dict = {
        'sample_id': sample_id,
        'label': sample_data['label'].iloc[0],
        
        # 价格相关
        'close_mean': sample_data['close'].mean(),
        'close_std': sample_data['close'].std(),
        'close_max': sample_data['close'].max(),
        'close_min': sample_data['close'].min(),
        'close_trend': (sample_data['close'].iloc[-1] - sample_data['close'].iloc[0]) / sample_data['close'].iloc[0],
        
        # 涨跌幅
        'pct_chg_mean': sample_data['pct_chg'].mean(),
        'pct_chg_std': sample_data['pct_chg'].std(),
        'pct_chg_sum': sample_data['pct_chg'].sum(),
        'positive_days': (sample_data['pct_chg'] > 0).sum(),
        'negative_days': (sample_data['pct_chg'] < 0).sum(),
        
        # 量比
        'volume_ratio_mean': sample_data['volume_ratio'].mean(),
        'volume_ratio_max': sample_data['volume_ratio'].max(),
        'volume_ratio_gt_2': (sample_data['volume_ratio'] > 2).sum(),
        
        # MACD（如果有）
        'macd_mean': sample_data['macd'].mean() if 'macd' in sample_data.columns else np.nan,
        'macd_positive_days': (sample_data['macd'] > 0).sum() if 'macd' in sample_data.columns else np.nan,
        
        # MA
        'ma5_mean': sample_data['ma5'].mean() if 'ma5' in sample_data.columns else np.nan,
        'ma10_mean': sample_data['ma10'].mean() if 'ma10' in sample_data.columns else np.nan,
        'price_above_ma5': (sample_data['close'] > sample_data['ma5']).sum() if 'ma5' in sample_data.columns else np.nan,
        
        # 市值
        'total_mv_mean': sample_data['total_mv'].mean() if 'total_mv' in sample_data.columns else np.nan,
    }
    
    features.append(feature_dict)

df_features = pd.DataFrame(features)

# 3. 准备训练数据
X = df_features.drop(['sample_id', 'label'], axis=1)
y = df_features['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. XGBoost训练
print("="*80)
print("训练 XGBoost 模型...")
print("="*80)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# 预测
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# 评估
print("\nXGBoost 性能:")
print(classification_report(y_test, y_pred_xgb, target_names=['负样本', '正样本']))
print(f"AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}")

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特征重要性 Top 10:")
print(feature_importance.head(10))

# 5. LightGBM训练
print("\n" + "="*80)
print("训练 LightGBM 模型...")
print("="*80)

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

print("\nLightGBM 性能:")
print(classification_report(y_test, y_pred_lgb, target_names=['负样本', '正样本']))
print(f"AUC: {roc_auc_score(y_test, y_prob_lgb):.4f}")

# 6. 保存模型
import joblib
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
joblib.dump(lgb_model, 'models/lightgbm_model.pkl')

print("\n✅ 模型训练完成并已保存！")
```

---

### 🥈 第二名：随机森林（Random Forest）⭐⭐⭐⭐

**稳定可靠的baseline**

#### 推荐理由

| 维度 | 评分 | 说明 |
|------|------|------|
| **效果** | ⭐⭐⭐⭐ | 稳定，不过通常不如XGBoost |
| **速度** | ⭐⭐⭐⭐ | 较快，可并行 |
| **易用性** | ⭐⭐⭐⭐⭐ | 超简单，几乎不需要调参 |
| **鲁棒性** | ⭐⭐⭐⭐⭐ | 非常鲁棒 |
| **可解释性** | ⭐⭐⭐⭐ | 有特征重要性 |

#### 优点 ✅
- **极简单**：基本不需要调参
- **鲁棒**：对噪声不敏感
- **可解释**：特征重要性直观

#### 缺点 ⚠️
- 效果通常不如XGBoost
- 模型文件大

#### 实现示例

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1  # 并行训练
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(classification_report(y_test, y_pred))
```

---

### 🥉 第三名：LSTM（长短期记忆网络）⭐⭐⭐

**适合复杂时序模式，但不是必需**

#### 推荐理由

| 维度 | 评分 | 说明 |
|------|------|------|
| **效果** | ⭐⭐⭐ | 可能更好，但需要大量调参 |
| **速度** | ⭐⭐ | 训练慢（需要GPU） |
| **易用性** | ⭐⭐ | 复杂，调参困难 |
| **鲁棒性** | ⭐⭐ | 容易过拟合 |
| **可解释性** | ⭐ | 黑盒 |
| **样本需求** | ⭐⭐ | 需要更多数据（5000+更好） |

#### 优点 ✅
- **捕捉时序依赖**：能学习复杂的时间模式
- **自动特征学习**：不需要手动特征工程
- **潜力大**：数据多时效果好

#### 缺点 ⚠️
- **样本量需求高**：2000+样本可能不够，容易过拟合
- **训练慢**：需要GPU，调参耗时
- **不可解释**：黑盒模型
- **调参复杂**：层数、单元数、dropout等

#### 何时使用LSTM？

✅ **适合的场景**：
- 样本量 > 5000
- 特征间有复杂的时序依赖
- 有GPU资源
- 有时间调参
- 追求极致效果

❌ **不适合的场景**（你的情况）：
- 样本量 ~2300（偏少）
- 特征已经很好（MACD、RSI等都是成熟指标）
- 需要快速验证
- 需要可解释性

#### 实现示例

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 准备3D数据：(samples, timesteps, features)
# 需要将数据reshape为 (样本数, 34天, 特征数)

# 假设每个样本34天，每天10个特征
X_train_lstm = X_train.values.reshape(-1, 34, 10)
X_test_lstm = X_test.values.reshape(-1, 34, 10)

# 构建LSTM模型
model = Sequential([
    LSTM(64, input_shape=(34, 10), return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)

# 训练
history = model.fit(
    X_train_lstm, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# 预测
y_pred_prob = model.predict(X_test_lstm)
y_pred = (y_pred_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred))
```

---

### 🏅 第四名：传统神经网络（MLP）⭐⭐⭐

**中规中矩的深度学习方案**

#### 推荐理由

| 维度 | 评分 | 说明 |
|------|------|------|
| **效果** | ⭐⭐⭐ | 中等 |
| **速度** | ⭐⭐⭐ | 一般 |
| **易用性** | ⭐⭐⭐ | 需要调参 |
| **鲁棒性** | ⭐⭐ | 容易过拟合 |
| **可解释性** | ⭐ | 黑盒 |

#### 适合场景
- 特征已经是表格型
- 想尝试深度学习但不想太复杂
- 数据量适中

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)
```

---

## 🎯 终极推荐方案

### 第一阶段：快速验证（1-2天）

```
1️⃣ XGBoost/LightGBM（必做）
   ↓
   效果好 → 直接用
   效果不好 → 下一步
```

### 第二阶段：模型优化（3-5天）

```
2️⃣ 随机森林（ensemble baseline）
   +
   更多特征工程
   +
   超参数调优
   ↓
   效果提升 → 使用优化后的模型
   效果卡住 → 下一步
```

### 第三阶段：深度学习尝试（1-2周）

```
3️⃣ LSTM / Transformer
   （如果有更多数据、GPU资源）
   ↓
   效果显著提升 → 考虑使用
   提升不明显 → 回到树模型
```

---

## 📊 实验对比设计

### 完整实验矩阵

| 模型 | 负样本方案 | 预期准确率 | 训练时间 | 复杂度 |
|------|-----------|-----------|---------|--------|
| XGBoost | V1 | 85-92% | 5分钟 | 简单 |
| XGBoost | V2 | 83-90% | 5分钟 | 简单 |
| LightGBM | V1 | 85-92% | 3分钟 | 简单 |
| LightGBM | V2 | 83-90% | 3分钟 | 简单 |
| Random Forest | V2 | 80-88% | 10分钟 | 简单 |
| LSTM | V2 | 80-90% | 1小时 | 复杂 |
| MLP | V2 | 78-86% | 20分钟 | 中等 |

### 推荐实验流程

```bash
# 实验1: XGBoost + V2负样本（最快baseline）
python scripts/train_model.py --model xgboost --neg_version v2

# 实验2: XGBoost + V1负样本（更难负样本）
python scripts/train_model.py --model xgboost --neg_version v1

# 实验3: LightGBM + V2负样本（对比XGBoost）
python scripts/train_model.py --model lightgbm --neg_version v2

# 实验4: LSTM（如果前面效果不理想）
python scripts/train_model.py --model lstm --neg_version v2
```

---

## 💡 特征工程建议

### 时间序列特征转换

对于34天的时序数据，可以提取以下统计特征：

#### 1. 价格相关
```python
- close_mean, close_std, close_max, close_min
- close_trend (首尾差/首值)
- close_volatility (标准差/均值)
```

#### 2. 涨跌幅
```python
- pct_chg_mean, pct_chg_std, pct_chg_sum
- positive_days_count, negative_days_count
- max_gain, max_loss
- consecutive_positive_days (最长连续上涨天数)
```

#### 3. 技术指标
```python
- macd_mean, macd_positive_days
- rsi_mean, rsi_overbought_days (>70)
- volume_ratio_mean, volume_ratio_gt_2_count
- price_above_ma5_days, price_above_ma10_days
```

#### 4. 动量特征
```python
- return_1w (最后1周收益)
- return_2w (最后2周收益)
- return_4w (全部4周收益)
- acceleration (收益率的变化率)
```

---

## 🔧 调参建议

### XGBoost 关键参数

```python
xgb.XGBClassifier(
    n_estimators=100,        # 树的数量，先从100开始
    max_depth=5,             # 树深度，5-7比较好
    learning_rate=0.1,       # 学习率，0.01-0.1
    subsample=0.8,           # 样本采样比例
    colsample_bytree=0.8,    # 特征采样比例
    min_child_weight=3,      # 最小子节点权重
    gamma=0.1,               # 分裂最小损失减少
    reg_alpha=0.1,           # L1正则
    reg_lambda=1.0,          # L2正则
    scale_pos_weight=1,      # 正负样本权重（如果不平衡）
)
```

### LSTM 关键参数

```python
- 层数: 2-3层
- 单元数: 64 → 32（递减）
- Dropout: 0.2-0.5
- Batch size: 32-64
- Epochs: 50-100 (用early stopping)
- Optimizer: Adam (lr=0.001)
```

---

## ✅ 评估指标

### 必看指标

| 指标 | 公式 | 重要性 | 说明 |
|------|------|--------|------|
| **Accuracy** | (TP+TN)/(P+N) | ⭐⭐⭐ | 整体准确率 |
| **Precision** | TP/(TP+FP) | ⭐⭐⭐⭐⭐ | 预测为牛股的准确率 |
| **Recall** | TP/(TP+FN) | ⭐⭐⭐⭐⭐ | 真牛股被找出的比例 |
| **F1-Score** | 2×P×R/(P+R) | ⭐⭐⭐⭐⭐ | 综合指标 |
| **AUC-ROC** | - | ⭐⭐⭐⭐ | 分类能力 |

### 业务指标

- **Top-K准确率**: 模型预测概率最高的K只股票中，实际牛股的占比
- **回测收益**: 基于模型选股的实际收益
- **夏普比率**: 收益/波动率

---

## 📝 最终建议

### 🎯 针对你的项目

**强烈推荐：XGBoost/LightGBM** ✅

理由：
1. ✅ 你的数据已经是表格型特征（MACD、RSI等）
2. ✅ 样本量适中（2300个）
3. ✅ 训练快速（5分钟）
4. ✅ 效果好（金融数据上proven）
5. ✅ 可解释（知道哪些指标重要）
6. ✅ 生产部署简单

**不推荐：LSTM** ❌（至少暂时不推荐）

理由：
1. ❌ 样本量偏少（LSTM更适合5000+）
2. ❌ 你的特征已经很好（手工特征工程质量高）
3. ❌ 训练慢，调参复杂
4. ❌ 容易过拟合
5. ❌ 黑盒，不可解释

### 🚀 行动计划

```
第1天: 
  - 准备数据（特征展平）
  - 训练XGBoost baseline
  - 查看效果和特征重要性
  
第2天:
  - 优化特征工程
  - 调整超参数
  - 对比V1和V2负样本
  
第3-5天:
  - 尝试LightGBM、Random Forest
  - Ensemble多个模型
  - 回测验证
  
如果效果卡住（1-2周后）:
  - 再考虑LSTM
  - 或者收集更多数据
```

---

## 📚 相关资源

### 学习资料
- [XGBoost官方文档](https://xgboost.readthedocs.io/)
- [LightGBM官方文档](https://lightgbm.readthedocs.io/)
- [Scikit-learn用户指南](https://scikit-learn.org/stable/user_guide.html)

### 竞赛案例
- Kaggle金融竞赛：几乎都用XGBoost/LightGBM
- 量化交易：树模型 > 深度学习（在表格数据上）

---

**文档版本**: v1.0  
**创建时间**: 2024-12-23  
**最后更新**: 2024-12-23

