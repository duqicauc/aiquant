# AIQuant 文档索引

本文档是 AIQuant 量化交易系统的完整文档索引。

---

## 📁 文档结构

```
docs/
├── README.md           # 本文档（索引）
├── guides/             # 用户指南（如何使用）
├── reference/          # 技术参考（原理和规范）
└── archive/            # 历史文档（归档，供参考）
```

---

## 🚀 快速入门 (guides/)

适合新用户快速上手的文档：

| 文档 | 说明 |
|------|------|
| [快速开始指南](guides/QUICK_START_GUIDE.md) | 5分钟快速上手 |
| [使用指南](guides/USAGE_GUIDE.md) | 系统使用说明 |
| [完整工作流程](guides/COMPLETE_WORKFLOW.md) | 从数据准备到模型训练 |

---

## 📋 工作流程指南 (guides/)

详细的工作流程文档：

| 文档 | 说明 |
|------|------|
| [样本准备指南](guides/SAMPLE_PREPARATION_GUIDE.md) | 正负样本数据准备 |
| [模型训练指南](guides/MODEL_TRAINING_GUIDE.md) | 模型训练流程 |
| [质量检查指南](guides/QUALITY_CHECK_GUIDE.md) | 数据质量检查 |
| [样本监控指南](guides/SAMPLE_MONITOR_GUIDE.md) | 自动监控样本准备 |
| [训练进度监控](guides/TRAINING_PROGRESS_MONITORING.md) | 监控长时间训练任务 |

---

## 🎯 功能指南 (guides/)

各功能模块的使用说明：

| 文档 | 说明 |
|------|------|
| [股票体检指南](guides/STOCK_HEALTH_CHECK_GUIDE.md) | 单股票健康检查 |
| [可视化指南](guides/VISUALIZATION_GUIDE.md) | 数据可视化和图表生成 |
| [训练可视化指南](guides/TRAINING_VISUALIZATION_GUIDE.md) | 训练过程可视化 |
| [测试指南](guides/TESTING_GUIDE.md) | 测试流程和用例 |

---

## 🔧 技术参考 (reference/)

### 核心概念

| 文档 | 说明 |
|------|------|
| [选股模型原理](reference/STOCK_SELECTION_MODEL.md) | 正负样本选股模型详解 |
| [避免未来函数](reference/AVOID_FUTURE_FUNCTION.md) | 时间序列划分原理 |
| [模型对比](reference/MODEL_COMPARISON.md) | XGBoost vs LSTM |
| [特征提取指南](reference/FEATURE_EXTRACTION_GUIDE.md) | 技术指标特征提取 |

### 模型管理

| 文档 | 说明 |
|------|------|
| [模型版本管理](reference/MODEL_VERSION_MANAGEMENT.md) | 版本管理方案 |
| [模型生命周期标准](reference/MODEL_LIFECYCLE_STANDARD.md) | 生命周期规范 |
| [模型生命周期快速参考](reference/MODEL_LIFECYCLE_QUICK_REFERENCE.md) | 快速参考卡 |
| [多模型架构](reference/ARCHITECTURE_MULTI_MODEL.md) | 多模型并行架构 |

### API 和数据

| 文档 | 说明 |
|------|------|
| [API参考文档](reference/API_REFERENCE.md) | 完整API接口说明 |
| [Tushare Pro功能](reference/TUSHARE_PRO_FEATURES.md) | Tushare高级功能 |
| [Tushare优化](reference/TUSHARE_OPTIMIZATION.md) | API优化方案 |
| [缓存与限流](reference/CACHE_AND_RATE_LIMIT.md) | 数据缓存机制 |
| [原始数据字段](reference/RAW_DATA_FIELDS.md) | 数据字段说明 |

### 项目结构

| 文档 | 说明 |
|------|------|
| [目录结构](reference/DIRECTORY_STRUCTURE.md) | 项目目录结构 |
| [项目结构说明](reference/PROJECT_STRUCTURE_CLARIFICATION.md) | 结构澄清 |
| [预测目录关系](reference/PREDICTION_DIRECTORY_RELATIONSHIP.md) | 预测目录关系 |

---

## 📦 归档文档 (archive/)

历史变更记录、优化笔记、对比分析等，仅供参考：

- 缓存/限流优化记录
- 模型版本迁移记录
- 参数对比分析
- 诊断和优化结果

---

## 🔍 按使用场景查找

### 我是新用户，想快速开始
1. [快速开始指南](guides/QUICK_START_GUIDE.md)
2. [使用指南](guides/USAGE_GUIDE.md)
3. [完整工作流程](guides/COMPLETE_WORKFLOW.md)

### 我想训练模型
1. [样本准备指南](guides/SAMPLE_PREPARATION_GUIDE.md)
2. [模型训练指南](guides/MODEL_TRAINING_GUIDE.md)
3. [质量检查指南](guides/QUALITY_CHECK_GUIDE.md)

### 我想了解技术细节
1. [API参考文档](reference/API_REFERENCE.md)
2. [选股模型原理](reference/STOCK_SELECTION_MODEL.md)
3. [避免未来函数](reference/AVOID_FUTURE_FUNCTION.md)

### 我想管理模型版本
1. [模型版本管理](reference/MODEL_VERSION_MANAGEMENT.md)
2. [模型生命周期标准](reference/MODEL_LIFECYCLE_STANDARD.md)

---

## 🛠️ 常用命令

```bash
# 查看模型版本状态
python scripts/model_version_manager.py status

# 比较两个版本
python scripts/model_version_manager.py compare v1.3.0 v1.4.0

# 股票评分（新框架）
python scripts/score_stocks.py

# 股票评分（旧框架，兼容）
python scripts/score_current_stocks.py
```

---

**最后更新**: 2025-12-30
