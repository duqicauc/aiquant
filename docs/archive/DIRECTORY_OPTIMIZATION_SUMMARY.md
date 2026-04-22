# 目录结构优化总结

## 📋 优化目标

1. ✅ 分离训练数据和预测数据
2. ✅ 删除空目录和废弃目录
3. ✅ 统一数据存储位置
4. ✅ 简化目录结构

## 🔄 优化内容

### 1. 数据目录重组

#### 训练数据 (`data/training/`)
- **samples/**: 正负样本数据
- **features/**: 特征数据
- **models/**: 训练好的模型
- **metrics/**: 模型评估指标
- **charts/**: 训练过程可视化图表

#### 预测数据 (`data/prediction/`)
- **results/**: 预测结果（CSV、TXT报告）
- **metadata/**: 预测元数据（JSON，用于准确率分析）
- **analysis/**: 准确率分析结果
- **history/**: 历史预测归档

#### 缓存数据 (`data/cache/`)
- **quant_data.db**: SQLite缓存数据库

### 2. 删除的目录

以下目录已删除（数据已迁移）：
- ❌ `data/processed/` → 迁移到 `data/training/`
- ❌ `data/results/` → 迁移到 `data/prediction/results/`
- ❌ `data/models/` → 迁移到 `data/training/models/`
- ❌ `data/charts/` → 迁移到 `data/training/charts/`
- ❌ `data/backtest/` → 删除（未使用）
- ❌ `data/backup/` → 删除（未使用）
- ❌ `data/database/` → 删除（未使用）
- ❌ `models/` → 删除（已迁移）
- ❌ `tests/` 下的空子目录 → 删除

### 3. 更新的脚本路径

以下脚本已更新路径引用：
- ✅ `scripts/score_current_stocks.py` - 预测结果保存路径
- ✅ `scripts/train_xgboost_timeseries.py` - 模型和指标保存路径
- ✅ `scripts/prepare_positive_samples.py` - 样本保存路径
- ✅ `scripts/prepare_negative_samples_v2.py` - 负样本保存路径
- ✅ `scripts/analyze_prediction_accuracy.py` - 分析结果保存路径

## 📁 最终目录结构

```
data/
├── training/              # 模型训练相关
│   ├── samples/          # 训练样本
│   ├── features/         # 特征数据
│   ├── models/           # 训练好的模型
│   ├── metrics/          # 评估指标
│   └── charts/           # 可视化图表
│
├── prediction/            # 实际预测相关
│   ├── results/         # 预测结果
│   ├── metadata/        # 预测元数据
│   ├── analysis/        # 准确率分析
│   └── history/         # 历史归档
│
└── cache/                # 数据缓存
    └── quant_data.db
```

## ✅ 优化效果

1. **清晰分离**: 训练数据和预测数据完全分开
2. **结构简洁**: 删除所有空目录和废弃目录
3. **易于管理**: 按功能分类，便于查找和维护
4. **便于分析**: 预测元数据独立存储，方便准确率分析

## 📝 注意事项

- 所有脚本已更新为新路径
- 旧路径已废弃，数据已迁移
- 空目录保留 `.gitkeep` 文件以便git跟踪
