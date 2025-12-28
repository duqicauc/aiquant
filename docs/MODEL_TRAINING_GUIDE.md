# 模型训练快速指南 🚀

3步开始训练你的选股模型

---

## 🎯 目标

基于正负样本数据，训练一个二分类模型，预测哪些股票是潜力股（未来3周涨幅>50%）

---

## 📋 前置条件

确保已完成数据准备：

```bash
# ✓ 正样本数据
data/processed/positive_samples.csv
data/processed/feature_data_34d.csv

# ✓ 负样本数据（至少有一个）
data/processed/negative_samples_v2.csv              # V2（推荐）
data/processed/negative_feature_data_v2_34d.csv
```

---

## 🚀 快速开始（3步）

### 第1步：安装依赖

```bash
pip install xgboost lightgbm scikit-learn
```

### 第2步：训练模型

```bash
python scripts/train_xgboost.py
```

### 第3步：查看结果

训练完成后，会生成：

```
models/xgboost_v2_20241223_120000.json      # 模型文件
data/results/xgboost_v2_metrics.json        # 评估报告
```

**预计耗时**：5-10分钟

---

## 📊 输出示例

### 训练日志

```
================================================================================
第一步：加载数据
================================================================================
✓ 正样本加载完成: 38930 条
✓ 负样本加载完成: 38930 条 (版本: v2)
✓ 数据合并完成: 77860 条

================================================================================
第二步：特征工程
================================================================================
将34天时序数据转换为统计特征...
进度: 500/2290
进度: 1000/2290
...
✓ 特征提取完成: 2290 个样本
✓ 特征维度: 22 个特征

================================================================================
第三步：训练XGBoost模型
================================================================================
特征矩阵: (2290, 22)
标签分布: {0: 1145, 1: 1145}

训练集: 1832 个样本
测试集: 458 个样本

开始训练...
✓ 模型训练完成！

================================================================================
第四步：模型评估
================================================================================

分类报告:
              precision    recall  f1-score   support

        负样本       0.85      0.88      0.86       229
        正样本       0.87      0.84      0.86       229

    accuracy                           0.86       458
   macro avg       0.86      0.86      0.86       458
weighted avg       0.86      0.86      0.86       458

AUC-ROC: 0.9234

混淆矩阵:
  真负例(TN):  201  |  假正例(FP):   28
  假负例(FN):   37  |  真正例(TP):  192

================================================================================
特征重要性 Top 10:
================================================================================
  pct_chg_sum              : 0.1423
  close_trend              : 0.1289
  volume_ratio_gt_2        : 0.0987
  positive_days            : 0.0876
  macd_positive_days       : 0.0754
  return_2w                : 0.0698
  max_gain                 : 0.0621
  pct_chg_std              : 0.0543
  volume_ratio_mean        : 0.0498
  price_above_ma5          : 0.0432

================================================================================
✅ 模型训练完成！
================================================================================

📊 模型性能总结:
  准确率 (Accuracy):  86.03%
  精确率 (Precision): 87.27%
  召回率 (Recall):    83.84%
  F1分数 (F1-Score):  85.52%
  AUC-ROC:            0.9234

🎯 下一步:
  1. 查看特征重要性，优化特征工程
  2. 尝试不同的负样本版本（v1 vs v2）
  3. 调整超参数提升性能
  4. 使用模型进行回测验证
```

---

## 🎨 自定义配置

### 修改负样本版本

编辑 `scripts/train_xgboost.py`：

```python
# 第52行
NEG_VERSION = 'v2'  # 改为 'v1' 使用V1负样本
```

### 调整超参数

编辑 `scripts/train_xgboost.py` 第105-116行：

```python
model = xgb.XGBClassifier(
    n_estimators=100,      # 树的数量：100-500
    max_depth=5,           # 树深度：3-10
    learning_rate=0.1,     # 学习率：0.01-0.3
    subsample=0.8,         # 样本采样：0.6-1.0
    colsample_bytree=0.8,  # 特征采样：0.6-1.0
    min_child_weight=3,    # 最小子节点权重：1-10
    gamma=0.1,             # 分裂最小增益：0-1
    reg_alpha=0.1,         # L1正则：0-1
    reg_lambda=1.0,        # L2正则：0-10
)
```

### 修改测试集比例

编辑 `scripts/train_xgboost.py` 第216行：

```python
model, metrics, X_test, y_test, y_prob = train_model(
    df_features, 
    test_size=0.2  # 改为 0.3 使用30%测试集
)
```

---

## 📈 评估指标说明

| 指标 | 含义 | 重要性 |
|------|------|--------|
| **Accuracy（准确率）** | 预测正确的比例 | ⭐⭐⭐ |
| **Precision（精确率）** | 预测为牛股中，真是牛股的比例 | ⭐⭐⭐⭐⭐ |
| **Recall（召回率）** | 所有牛股中，被找出的比例 | ⭐⭐⭐⭐⭐ |
| **F1-Score** | Precision和Recall的调和平均 | ⭐⭐⭐⭐⭐ |
| **AUC-ROC** | 分类器整体性能 | ⭐⭐⭐⭐ |

### 业务解读

**Precision高（如87%）= 预测准确**
- 模型推荐100只股票，有87只真的是牛股
- **适合保守投资者**：宁可少推荐，也要准确

**Recall高（如84%）= 覆盖全面**
- 所有牛股中，84%被模型找出来了
- **适合激进投资者**：不能漏掉潜力股

**F1-Score = 平衡指标**
- 综合考虑准确性和覆盖面
- **一般追求F1最大化**

---

## 🔧 进阶使用

### 1. 对比不同负样本版本

```bash
# 训练V1模型
# 编辑 train_xgboost.py: NEG_VERSION = 'v1'
python scripts/train_xgboost.py

# 训练V2模型
# 编辑 train_xgboost.py: NEG_VERSION = 'v2'
python scripts/train_xgboost.py

# 对比两个模型的 data/results/xgboost_*_metrics.json
```

### 2. 特征工程优化

根据特征重要性，可以：
- **删除不重要的特征**（importance < 0.01）
- **创造新的组合特征**
  - 如：`price_momentum = return_1w * volume_ratio_mean`
  - 如：`ma_cross = (close_mean - ma10_mean) / ma10_mean`

### 3. 超参数调优

使用网格搜索：

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 500],
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(),
    param_grid,
    cv=5,
    scoring='f1'
)

grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
```

### 4. 模型融合（Ensemble）

训练多个模型，投票决策：

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', rf_model)
    ],
    voting='soft'  # 软投票（基于概率）
)

ensemble.fit(X_train, y_train)
```

---

## ❓ 常见问题

### Q1: 模型性能不理想怎么办？

**A**: 按以下顺序排查：
1. **检查数据质量** - 运行 `python scripts/check_sample_quality.py`
2. **尝试V1负样本** - 更有挑战性
3. **特征工程** - 创造更多有意义的特征
4. **调整超参数** - 使用网格搜索
5. **增加数据量** - 扩大时间范围

### Q2: 训练时间太长？

**A**:
- XGBoost已经很快了（5分钟）
- 如果还嫌慢，试试LightGBM（更快）
- 减少 `n_estimators` 参数

### Q3: 过拟合怎么办？

**A**: 症状：训练集准确率很高，测试集很低

解决方法：
- 增加正则化：`reg_alpha`, `reg_lambda`
- 减少模型复杂度：`max_depth`, `min_child_weight`
- 增加数据：扩大时间范围

### Q4: 欠拟合怎么办？

**A**: 症状：训练集和测试集准确率都低

解决方法：
- 增加模型复杂度：`max_depth`, `n_estimators`
- 创造更多特征
- 减少正则化

### Q5: 能否使用其他模型？

**A**: 可以！参考 [模型对比文档](MODEL_COMPARISON.md)
- LightGBM：更快
- Random Forest：更稳定
- MLP/LSTM：需要更多数据和调参

---

## 🎯 下一步

训练完模型后：

1. **特征分析** - 查看哪些指标最重要
2. **错误分析** - 分析被误判的样本
3. **回测验证** - 在历史数据上测试策略
4. **模拟交易** - 纸面交易验证
5. **实盘部署** - 小资金试运行

---

## 📚 相关文档

- [模型选择对比](MODEL_COMPARISON.md) - XGBoost vs LSTM等
- [负样本方案对比](NEGATIVE_SAMPLE_COMPARISON.md) - V1 vs V2
- [样本准备指南](SAMPLE_PREPARATION_GUIDE.md) - 数据准备流程

---

**文档版本**: v1.0  
**创建时间**: 2024-12-23  
**最后更新**: 2024-12-23


