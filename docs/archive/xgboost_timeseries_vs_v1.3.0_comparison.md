# xgboost_timeseries模型 vs v1.3.0模型构建逻辑对比

**生成时间**: 2025-12-29  
**对比对象**: 
- 旧模型: `xgboost_timeseries_v2_20251225_205905.json` (2025-12-25 20:59:05)
- 新模型: `breakout_launch_scorer v1.3.0` (正在训练中)

---

## 📊 一、最新xgboost_timeseries模型信息

### 1.1 模型文件
- **文件名**: `xgboost_timeseries_v2_20251225_205905.json`
- **创建时间**: 2025-12-25 20:59:05
- **训练脚本**: `scripts/train_xgboost_timeseries.py`
- **负样本版本**: v2

### 1.2 训练方式
```bash
python scripts/train_xgboost_timeseries.py
```

---

## 🔄 二、训练流程对比

### 2.1 流程步骤

| 步骤 | xgboost_timeseries | v1.3.0 (新框架) | 状态 |
|------|-------------------|-----------------|------|
| 1. 数据加载 | `load_and_prepare_data()` | `_load_and_prepare_data()` | ✅ 一致 |
| 2. 特征提取 | `extract_features_with_time()` | `_extract_features()` | ✅ 一致 |
| 3. 时间划分 | `timeseries_split()` | `_timeseries_split()` | ✅ 一致 |
| 4. 模型训练 | `train_model()` | `_train_model()` | ✅ 一致 |
| 5. 模型保存 | `save_model()` | `_save_model()` | ✅ 一致 |

---

## 📁 三、数据加载逻辑对比

### 3.1 数据文件路径

| 数据类型 | xgboost_timeseries | v1.3.0 | 状态 |
|---------|-------------------|--------|------|
| 正样本 | `data/training/features/feature_data_34d.csv` | `data/training/features/feature_data_34d.csv` | ✅ 相同 |
| 负样本 | `data/training/features/negative_feature_data_v2_34d.csv` | `data/training/features/negative_feature_data_v2_34d.csv` | ✅ 相同 |

### 3.2 数据加载代码

**xgboost_timeseries** (`scripts/train_xgboost_timeseries.py:35-70`):
```python
def load_and_prepare_data(neg_version='v2'):
    # 加载正样本
    df_pos = pd.read_csv('data/training/features/feature_data_34d.csv')
    df_pos['label'] = 1
    
    # 加载负样本
    if neg_version == 'v2':
        neg_file = 'data/training/features/negative_feature_data_v2_34d.csv'
    else:
        neg_file = 'data/training/features/negative_feature_data_34d.csv'
    
    df_neg = pd.read_csv(neg_file)
    
    # 合并
    df = pd.concat([df_pos, df_neg])
    return df
```

**v1.3.0** (`src/models/lifecycle/trainer.py:156-197`):
```python
def _load_and_prepare_data(self, neg_version='v2'):
    # 加载正样本（使用与旧模型完全相同的路径）
    df_pos = pd.read_csv('data/training/features/feature_data_34d.csv')
    df_pos['label'] = 1
    
    # 加载负样本（使用与旧模型完全相同的路径和逻辑）
    if neg_version == 'v2':
        neg_file = 'data/training/features/negative_feature_data_v2_34d.csv'
    else:
        neg_file = 'data/training/features/negative_feature_data_34d.csv'
    
    df_neg = pd.read_csv(neg_file)
    
    # 合并
    df = pd.concat([df_pos, df_neg])
    return df
```

**结论**: ✅ **完全一致**

---

## 🔧 四、特征提取逻辑对比

### 4.1 特征列表

| 特征类别 | 特征名称 | xgboost_timeseries | v1.3.0 | 状态 |
|---------|---------|-------------------|--------|------|
| 价格特征 | close_mean, close_std, close_max, close_min, close_trend | ✅ | ✅ | ✅ 一致 |
| 涨跌幅特征 | pct_chg_mean, pct_chg_std, pct_chg_sum, positive_days, negative_days, max_gain, max_loss | ✅ | ✅ | ✅ 一致 |
| 量比特征 | volume_ratio_mean, volume_ratio_max, volume_ratio_gt_2, volume_ratio_gt_4 | ✅ | ✅ | ✅ 一致 |
| MACD特征 | macd_mean, macd_positive_days, macd_max | ✅ | ✅ | ✅ 一致 |
| MA特征 | ma5_mean, price_above_ma5, ma10_mean, price_above_ma10 | ✅ | ✅ | ✅ 一致 |
| 市值特征 | total_mv_mean, circ_mv_mean | ✅ | ✅ | ✅ 一致 |
| 动量特征 | return_1w, return_2w | ✅ | ✅ | ✅ 一致 |

### 4.2 特征提取代码对比

**关键逻辑对比**:

1. **量比特征** (完全一致):
```python
# 旧模型和新模型都使用相同的逻辑
if 'volume_ratio' in sample_data.columns:
    feature_dict['volume_ratio_mean'] = sample_data['volume_ratio'].mean()
    feature_dict['volume_ratio_max'] = sample_data['volume_ratio'].max()
    feature_dict['volume_ratio_gt_2'] = (sample_data['volume_ratio'] > 2).sum()
    feature_dict['volume_ratio_gt_4'] = (sample_data['volume_ratio'] > 4).sum()
```

2. **MACD特征** (完全一致):
```python
# 旧模型和新模型都使用相同的逻辑
if 'macd' in sample_data.columns:
    macd_data = sample_data['macd'].dropna()
    if len(macd_data) > 0:
        feature_dict['macd_mean'] = macd_data.mean()
        feature_dict['macd_positive_days'] = (macd_data > 0).sum()
        feature_dict['macd_max'] = macd_data.max()
```

3. **缺失值处理** (完全一致):
```python
# 旧模型和新模型都在特征提取后，在时间划分时处理缺失值
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
```

**结论**: ✅ **特征提取逻辑完全一致**

---

## ⏰ 五、时间序列划分逻辑对比

### 5.1 划分方式

| 项目 | xgboost_timeseries | v1.3.0 | 状态 |
|------|-------------------|--------|------|
| 划分比例 | 80% 训练，20% 测试 | 80% 训练，20% 测试 | ✅ 一致 |
| 划分依据 | t1_date（T1日期） | t1_date（T1日期） | ✅ 一致 |
| 数据泄露检查 | ✅ 检查训练集和测试集时间重叠 | ✅ 检查训练集和测试集时间重叠 | ✅ 一致 |

### 5.2 划分代码对比

**xgboost_timeseries** (`scripts/train_xgboost_timeseries.py:216-301`):
```python
def timeseries_split(df_features, train_end_date=None, test_start_date=None):
    # 确保t1_date是datetime类型
    df_features['t1_date'] = pd.to_datetime(df_features['t1_date'])
    
    # 按时间排序
    df_features = df_features.sort_values('t1_date').reset_index(drop=True)
    
    # 如果未指定划分点，使用80%作为训练集
    if train_end_date is None:
        n_train = int(len(df_features) * 0.8)
        train_end_date = df_features.iloc[n_train]['t1_date']
        test_start_date = df_features.iloc[n_train + 1]['t1_date']
    
    # 划分训练集和测试集
    train_mask = df_features['t1_date'] <= train_end_date
    test_mask = df_features['t1_date'] >= test_start_date
    
    # 处理缺失值
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
```

**v1.3.0** (`src/models/lifecycle/trainer.py:331-411`):
```python
def _timeseries_split(self, df_features):
    # 确保t1_date是datetime类型
    df_features['t1_date'] = pd.to_datetime(df_features['t1_date'])
    
    # 按时间排序
    df_features = df_features.sort_values('t1_date').reset_index(drop=True)
    
    # 使用配置中的划分方式（如果未指定，使用80%作为训练集，与旧模型一致）
    train_end_date = self.config.get('training', {}).get('train_end_date')
    test_start_date = self.config.get('training', {}).get('test_start_date')
    
    if train_end_date is None:
        n_train = int(len(df_features) * 0.8)
        train_end_date = df_features.iloc[n_train]['t1_date']
        test_start_date = df_features.iloc[n_train + 1]['t1_date']
    
    # 划分训练集和测试集
    train_mask = df_features['t1_date'] <= train_end_date
    test_mask = df_features['t1_date'] >= test_start_date
    
    # 处理缺失值（与旧模型完全一致）
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
```

**结论**: ✅ **时间序列划分逻辑完全一致**

---

## 🤖 六、模型训练逻辑对比

### 6.1 XGBoost参数

| 参数 | xgboost_timeseries | v1.3.0 | 状态 |
|------|-------------------|--------|------|
| n_estimators | 100 | 100 | ✅ 一致 |
| max_depth | 5 | 5 | ✅ 一致 |
| learning_rate | 0.1 | 0.1 | ✅ 一致 |
| subsample | 0.8 | 0.8 | ✅ 一致 |
| colsample_bytree | 0.8 | 0.8 | ✅ 一致 |
| min_child_weight | 3 | 3 | ✅ 一致 |
| gamma | 0.1 | 0.1 | ✅ 一致 |
| reg_alpha | 0.1 | 0.1 | ✅ 一致 |
| reg_lambda | 1.0 | 1.0 | ✅ 一致 |
| random_state | 42 | 42 | ✅ 一致 |
| eval_metric | logloss | logloss | ✅ 一致 |

### 6.2 训练代码对比

**xgboost_timeseries** (`scripts/train_xgboost_timeseries.py:304-398`):
```python
def train_model(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
```

**v1.3.0** (`src/models/lifecycle/trainer.py:413-507`):
```python
def _train_model(self, X_train, y_train, X_test, y_test):
    # 从配置读取参数，但确保与旧模型完全一致
    model_params = self.config.get('model_params', {})
    
    # 如果配置中没有参数，使用旧模型的默认参数
    if not model_params:
        model_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    
    model = xgb.XGBClassifier(**model_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
```

**结论**: ✅ **模型训练参数完全一致**

---

## 📝 七、总结

### 7.1 构建逻辑一致性

| 环节 | 一致性 | 说明 |
|------|--------|------|
| 数据加载 | ✅ 100% | 使用相同的数据文件和加载逻辑 |
| 特征提取 | ✅ 100% | 特征列表、提取逻辑、缺失值处理完全一致 |
| 时间划分 | ✅ 100% | 80/20划分、时间排序、数据泄露检查完全一致 |
| 模型训练 | ✅ 100% | XGBoost参数完全一致 |
| 模型评估 | ✅ 100% | 评估指标和计算方法完全一致 |

### 7.2 关键差异

| 差异项 | xgboost_timeseries | v1.3.0 | 影响 |
|--------|-------------------|--------|------|
| 代码组织 | 独立脚本 | 类方法（新框架） | 无影响 |
| 版本管理 | 文件名时间戳 | 版本号系统 | 无影响 |
| 配置管理 | 硬编码参数 | YAML配置文件 | 无影响（参数值相同） |
| 数据范围 | 使用现有数据文件 | 可重新生成数据（从2000-01-01开始） | **有影响** |

### 7.3 结论

**✅ v1.3.0模型的构建逻辑与xgboost_timeseries模型完全一致**

唯一区别是：
- **xgboost_timeseries**: 使用预先准备好的数据文件（可能不包含2000年数据）
- **v1.3.0**: 可以重新生成数据，确保从2000-01-01开始

**当前正在训练的v1.3.0模型**:
- ✅ 使用相同的特征提取逻辑
- ✅ 使用相同的训练参数
- ✅ 使用相同的时间序列划分方式
- ✅ **使用从2000-01-01开始重新准备的数据**（这是关键改进）

---

**文档版本**: v1.0  
**创建日期**: 2025-12-29

