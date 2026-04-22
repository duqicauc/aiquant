# v2.5.0 模型修复方案

## 🔴 问题1: 数据划分方式错误（必须修复）

### 当前问题
```python
# train_v250_model.py 第220-225行
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
**问题**: 使用随机划分，存在未来函数风险

### 修复方案

在 `train_v250_model.py` 中添加时间序列划分函数：

```python
def time_series_split(df, test_size=0.2, cal_size=0.15):
    """
    时间序列划分（避免未来函数）

    Args:
        df: 包含 trade_date 或 t1_date 的DataFrame
        test_size: 测试集比例
        cal_size: 校准集比例（从训练集中分出）

    Returns:
        train, cal, test: 三个DataFrame
    """
    # 确定日期列
    date_col = 'trade_date' if 'trade_date' in df.columns else 't1_date'

    # 转换为日期类型
    if df[date_col].dtype != 'datetime64[ns]':
        df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d', errors='coerce')

    # 按日期排序
    df = df.sort_values(date_col).reset_index(drop=True)

    # 计算划分点
    n = len(df)
    test_start = int(n * (1 - test_size))
    cal_start = int(n * (1 - test_size - cal_size))

    # 划分
    train = df.iloc[:cal_start].copy()
    cal = df.iloc[cal_start:test_start].copy()
    test = df.iloc[test_start:].copy()

    log.info(f"\n时间序列划分:")
    log.info(f"  训练集: {train[date_col].min().date()} ~ {train[date_col].max().date()} ({len(train)}条)")
    log.info(f"  校准集: {cal[date_col].min().date()} ~ {cal[date_col].max().date()} ({len(cal)}条)")
    log.info(f"  测试集: {test[date_col].min().date()} ~ {test[date_col].max().date()} ({len(test)}条)")

    return train, cal, test
```

### 修改 main() 函数

将原来的随机划分改为时间序列划分：

```python
def main():
    # ... 前面的代码保持不变 ...

    # 特征
    feature_cols = get_feature_columns(df)
    log.info(f"特征数: {len(feature_cols)}")

    # 显示新增的风险特征
    risk_features = [f for f in feature_cols if any(k in f for k in ['drawdown', 'atr', 'days_from_high', 'recovery'])]
    log.info(f"风险特征: {risk_features}")

    # ⚠️ 修改这里：使用时间序列划分
    train_df, cal_df, test_df = time_series_split(df, test_size=0.2, cal_size=0.15)

    # 准备特征和标签
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_cal = cal_df[feature_cols].values
    y_cal = cal_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values

    # 处理NaN和无穷值
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_cal = np.nan_to_num(X_cal, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    log.info(f"\n数据集: 训练{len(X_train)}, 校准{len(X_cal)}, 测试{len(X_test)}")

    # ... 后续代码保持不变 ...
```

---

## 🟡 问题2: 特征对齐问题

### 当前问题
- 正样本有106个特征
- 负样本有97个特征
- 可能导致特征对齐错误

### 修复方案

在 `load_training_data()` 函数中确保特征对齐：

```python
def load_training_data():
    """加载带风险特征的训练数据"""
    log.info("加载训练数据...")

    # ... 加载数据的代码 ...

    # ⚠️ 添加：确保特征对齐
    pos_feature_cols = set(df_pos.columns)
    neg_feature_cols = set(df_neg.columns)
    common_cols = pos_feature_cols & neg_feature_cols

    # 找出差异
    pos_only = pos_feature_cols - neg_feature_cols
    neg_only = neg_feature_cols - pos_feature_cols

    if pos_only:
        log.warning(f"  正样本独有特征: {len(pos_only)} 个")
        log.info(f"    {list(pos_only)[:5]}")
    if neg_only:
        log.warning(f"  负样本独有特征: {len(neg_only)} 个")
        log.info(f"    {list(neg_only)[:5]}")

    # 只保留共同特征（排除非特征列）
    exclude_cols = {'ts_code', 'name', 't1_date', 't2_date', 'sample_id', 'label',
                    'trade_date', 'weekly_return_1', 'weekly_return_2', 'weekly_return_3',
                    'total_return_34d', 'weekly_volume_1', 'weekly_volume_2', 'weekly_volume_3'}

    feature_cols = [col for col in common_cols if col not in exclude_cols]

    # 只保留共同特征列
    df_pos = df_pos[list(common_cols)]
    df_neg = df_neg[list(common_cols)]

    df = pd.concat([df_pos, df_neg], ignore_index=True)
    log.success(f"✓ 数据加载完成: {len(df)} 条，共同特征: {len(feature_cols)} 个")

    return df
```

---

## 🟡 问题3: 评估指标不完整

### 修复方案

在 `evaluate()` 函数中补充完整指标：

```python
def evaluate(booster, calibrator, X_test, y_test, feature_names):
    """评估模型"""
    log.info("评估模型...")

    from sklearn.metrics import (
        roc_auc_score, precision_recall_curve,
        classification_report, confusion_matrix,
        precision_score, recall_score, f1_score
    )

    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    raw_probs = booster.predict(dtest)
    cal_probs = calibrator.predict(raw_probs)

    # 计算AUC
    auc = roc_auc_score(y_test, cal_probs)
    log.info(f"  AUC: {auc:.4f}")

    # 不同阈值下的指标
    log.info(f"\n不同阈值下的性能:")
    log.info(f"{'阈值':<8} {'样本数':<10} {'精确率':<10} {'召回率':<10} {'F1':<10} {'准确率':<10}")
    log.info("-" * 60)

    for thresh in [0.9, 0.8, 0.7, 0.6, 0.5]:
        y_pred = (cal_probs >= thresh).astype(int)
        if y_pred.sum() > 0:
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            acc = (y_test[y_pred == 1] == 1).mean() if y_pred.sum() > 0 else 0
            log.info(f"{thresh:<8.1f} {y_pred.sum():<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {acc:<10.4f}")
        else:
            log.info(f"{thresh:<8.1f} {0:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

    # 混淆矩阵（阈值0.5）
    y_pred_05 = (cal_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_05)
    log.info(f"\n混淆矩阵（阈值=0.5）:")
    log.info(f"              预测负  预测正")
    log.info(f"  实际负      {cm[0,0]:<8} {cm[0,1]:<8}")
    log.info(f"  实际正      {cm[1,0]:<8} {cm[1,1]:<8}")

    # 分类报告
    log.info(f"\n分类报告（阈值=0.5）:")
    report = classification_report(y_test, y_pred_05, target_names=['负样本', '正样本'])
    log.info(f"\n{report}")

    return {
        'test_samples': len(X_test),
        'positive_samples': y_test.sum(),
        'auc': auc,
        'precision': precision_score(y_test, y_pred_05),
        'recall': recall_score(y_test, y_pred_05),
        'f1': f1_score(y_test, y_pred_05)
    }
```

---

## 📋 完整修复步骤

1. **备份原文件**
   ```bash
   cp scripts/train_v250_model.py scripts/train_v250_model.py.backup
   ```

2. **应用修复**
   - 添加 `time_series_split()` 函数
   - 修改 `main()` 函数中的数据划分逻辑
   - 修改 `load_training_data()` 确保特征对齐
   - 增强 `evaluate()` 函数添加完整指标

3. **测试修复**
   ```bash
   python scripts/train_v250_model.py
   ```

4. **验证结果**
   - 检查日志中的时间序列划分信息
   - 检查评估指标是否完整
   - 确认特征对齐正确

---

## ✅ 修复后的预期效果

1. ✅ **消除未来函数风险**：测试集只包含训练集之后的数据
2. ✅ **特征对齐正确**：正负样本使用相同的特征集
3. ✅ **评估指标完整**：AUC、精确率、召回率、F1等
4. ✅ **模型更可靠**：真实应用时表现与测试集表现更接近
