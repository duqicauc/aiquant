# 左侧潜力牛股模型工作流程

## 📋 概述

左侧潜力牛股模型采用**分步数据准备**的工作流程，避免训练时实时下载数据，提高效率和稳定性。

## 🚀 推荐工作流程

### 步骤1: 数据准备（一次性）
```bash
python scripts/prepare_left_breakout_data.py
```

**功能:**
- 筛选正样本（涨幅≥50%）
- 筛选负样本（涨幅<15%）
- 提取50个技术指标特征
- 保存到 `data/training/features/left_breakout_features_latest.csv`
- 生成元信息文件

**耗时:** 约5-10分钟（一次性）
**输出:** 完整的训练数据集

### 步骤2: 模型训练（多次可重复）
```bash
python scripts/train_left_breakout_model.py --load-prepared-data
```

**功能:**
- 直接加载已准备的数据
- 训练XGBoost模型
- 执行模型验证
- 保存模型文件

**耗时:** 约2-5分钟
**输出:** 训练好的模型

### 步骤3: 模型预测（日常使用）
```bash
python scripts/predict_left_breakout.py
```

## 📊 数据文件结构

```
data/training/
├── samples/
│   ├── left_positive_samples.csv    # 正样本
│   └── left_negative_samples.csv    # 负样本
└── features/
    ├── left_breakout_features_latest.csv      # 最新特征数据
    ├── left_breakout_features_20251227.csv    # 历史版本
    └── left_breakout_metadata_20251227.json   # 元信息
```

## 🔄 传统流程 vs 新流程对比

### 传统流程（耦合式）
```bash
# 每次训练都要重新下载数据
python scripts/train_left_breakout_model.py
# 内部执行: 下载数据 → 特征提取 → 训练
```

**问题:**
- 训练时间长（包含数据下载）
- 网络不稳定时容易失败
- 无法复用数据
- 调试困难

### 新流程（分离式）
```bash
# 一次性准备数据
python scripts/prepare_left_breakout_data.py

# 多次训练（只加载本地数据）
python scripts/train_left_breakout_model.py --load-prepared-data
```

**优势:**
- ✅ 数据准备与训练分离
- ✅ 训练速度快（无需下载）
- ✅ 数据可复用和版本控制
- ✅ 网络问题不影响训练
- ✅ 便于调试和优化

## 🎯 使用建议

### 初次使用
1. 运行数据准备: `python scripts/prepare_left_breakout_data.py`
2. 检查数据质量
3. 运行训练: `python scripts/train_left_breakout_model.py --load-prepared-data`

### 日常使用
- 直接使用 `--load-prepared-data` 参数训练
- 数据已准备好，无需重复下载

### 数据更新
- 当需要新数据时，重新运行准备脚本
- 旧数据自动备份，可回溯历史版本

## 📈 性能对比

| 指标 | 传统流程 | 新流程 |
|------|----------|--------|
| 首次运行时间 | 15-20分钟 | 10-15分钟 |
| 后续训练时间 | 15-20分钟 | 2-5分钟 |
| 网络依赖 | 高 | 低（仅数据准备时） |
| 稳定性 | 中等 | 高 |
| 调试便利性 | 差 | 优秀 |

## 🔧 技术实现

### 数据准备脚本
- 复用现有的 `LeftBreakoutModel.prepare_samples()`
- 复用现有的特征工程 `LeftBreakoutFeatureEngineering`
- 新增数据持久化功能

### 训练脚本优化
- 新增 `--load-prepared-data` 参数
- 智能检测数据文件存在性
- 保持向后兼容性

### 文件版本管理
- 自动生成时间戳版本
- 软链接指向最新版本
- 元信息记录数据统计

## 📝 注意事项

1. **数据版本**: 使用软链接确保总是加载最新数据
2. **向后兼容**: 原有训练方式仍然可用
3. **磁盘空间**: 特征数据约占用几MB空间
4. **更新频率**: 根据需要定期更新基础数据

## 🎉 总结

新工作流程显著提高了开发和使用效率：

- **开发效率**: 分离关注点，便于调试优化
- **运行效率**: 训练速度提升3-5倍
- **稳定性**: 减少网络依赖，降低失败率
- **可维护性**: 数据版本控制，易于回溯

**强烈推荐使用新的分步工作流程！** 🚀
