# AIQuant 文档索引

本文档是 AIQuant 量化交易系统的完整文档索引，所有文档按功能分类整理。

---

## 📚 文档分类

### 🚀 快速入门

适合新用户快速上手的文档：

- **[快速开始指南](QUICK_START_GUIDE.md)** - 5分钟快速上手，完成第一个预测
- **[使用指南](USAGE_GUIDE.md)** - 系统使用说明和常用操作

---

### 📋 工作流程

完整的工作流程文档，从数据准备到模型训练：

- **[完整工作流程](COMPLETE_WORKFLOW.md)** - 从数据准备到模型训练的完整步骤
- **[样本准备指南](SAMPLE_PREPARATION_GUIDE.md)** - 正负样本数据准备详细说明
- **[模型训练指南](MODEL_TRAINING_GUIDE.md)** - 模型训练流程和参数配置
- **[训练进度监控](TRAINING_PROGRESS_MONITORING.md)** - 如何监控长时间训练任务
- **[样本监控指南](SAMPLE_MONITOR_GUIDE.md)** - 自动监控样本准备状态

---

### 🎯 功能指南

各功能模块的详细使用说明：

- **[质量检查指南](QUALITY_CHECK_GUIDE.md)** - 数据质量检查流程
- **[股票体检指南](STOCK_HEALTH_CHECK_GUIDE.md)** - 单股票全方位健康检查
- **[可视化指南](VISUALIZATION_GUIDE.md)** - 数据可视化和图表生成
- **[测试指南](TESTING_GUIDE.md)** - 测试流程和测试用例

---

### 🔧 技术参考

技术实现细节和API文档：

- **[API参考文档](API_REFERENCE.md)** - 完整API接口说明
- **[Tushare Pro功能](TUSHARE_PRO_FEATURES.md)** - Tushare Pro高级功能使用
- **[缓存与限流](CACHE_AND_RATE_LIMIT.md)** - 数据缓存和API限流机制
- **[Tushare优化](TUSHARE_OPTIMIZATION.md)** - Tushare API优化方案
- **[Tushare恢复指南](TUSHARE_RECOVERY_GUIDE.md)** - Tushare服务恢复方法
- **[SSL权限修复](SSL_PERMISSION_FIX.md)** - SSL证书权限问题解决

---

### 📊 模型与策略

模型原理和策略说明：

- **[模型版本管理](MODEL_VERSION_MANAGEMENT.md)** - 同一模型不同版本的完整管理方案 🆕
- **[选股模型原理](STOCK_SELECTION_MODEL.md)** - 正负样本选股模型详解
- **[模型对比](MODEL_COMPARISON.md)** - XGBoost vs LSTM 模型对比
- **[负样本对比](NEGATIVE_SAMPLE_COMPARISON.md)** - 不同负样本方案对比
- **[避免未来函数](AVOID_FUTURE_FUNCTION.md)** - 时间序列划分避免数据泄露
- **[特征提取指南](FEATURE_EXTRACTION_GUIDE.md)** - 技术指标特征提取说明
- **[MACD参数对比](MACD_PARAMETER_COMPARISON.md)** - MACD指标参数优化

---

### 📈 优化与分析

性能优化和结果分析：

- **[准确率分析](ACCURACY_ANALYSIS_AND_OPTIMIZATION.md)** - 模型准确率分析和优化
- **[诊断结果](DIAGNOSIS_RESULTS.md)** - 系统诊断和问题分析
- **[优化完成报告](OPTIMIZATION_COMPLETED.md)** - 系统优化完成总结
- **[最终优化建议](FINAL_OPTIMIZATION_RECOMMENDATION.md)** - 系统优化最终建议
- **[优化计划](OPTIMIZATION_PLAN_FINAL.md)** - 系统优化计划文档
- **[缓存优化修复](CACHE_OPTIMIZATION_FIX.md)** - 缓存系统优化修复

---

### 🏗️ 项目结构

项目架构和目录结构说明：

- **[多模型并行架构](ARCHITECTURE_MULTI_MODEL.md)** - 支持多模型并行的完整架构设计 🆕
- **[多模型架构实施计划](IMPLEMENTATION_PLAN_MULTI_MODEL.md)** - 架构迁移的详细实施计划 🆕
- **[目录结构](DIRECTORY_STRUCTURE.md)** - 项目目录结构详细说明
- **[项目结构说明](PROJECT_STRUCTURE_CLARIFICATION.md)** - 项目结构澄清文档
- **[目录优化总结](DIRECTORY_OPTIMIZATION_SUMMARY.md)** - 目录结构优化总结
- **[预测目录关系](PREDICTION_DIRECTORY_RELATIONSHIP.md)** - 预测相关目录关系
- **[预测结果目录](PREDICTION_RESULT_DIRECTORY.md)** - 预测结果存储说明

---

### 📝 其他文档

其他重要文档：

- **[模型优化笔记 2025-12-28](MODEL_OPTIMIZATION_NOTES_20251228.md)** - 模型优化思路和技术因子改进建议
- **[样本目标分析](SAMPLE_TARGET_ANALYSIS.md)** - 样本目标值分析
- **[严格标准恢复](STRICT_CRITERIA_RESTORED.md)** - 严格筛选标准恢复说明
- **[可视化工具评估](VISUALIZATION_TOOLS_EVALUATION.md)** - 可视化工具对比评估
- **[可视化脚本评估](VISUALIZATION_SCRIPTS_EVALUATION.md)** - 可视化脚本功能评估

---

## 🔍 按使用场景查找

### 我是新用户，想快速开始
1. [快速开始指南](QUICK_START_GUIDE.md)
2. [使用指南](USAGE_GUIDE.md)
3. [完整工作流程](COMPLETE_WORKFLOW.md)

### 我想训练模型
1. [样本准备指南](SAMPLE_PREPARATION_GUIDE.md)
2. [模型训练指南](MODEL_TRAINING_GUIDE.md)
3. [质量检查指南](QUALITY_CHECK_GUIDE.md)

### 我想使用预测功能
1. [使用指南](USAGE_GUIDE.md) - 预测部分
2. [股票体检指南](STOCK_HEALTH_CHECK_GUIDE.md)

### 我想了解技术细节
1. [API参考文档](API_REFERENCE.md)
2. [选股模型原理](STOCK_SELECTION_MODEL.md)
3. [避免未来函数](AVOID_FUTURE_FUNCTION.md)

### 我想优化系统性能
1. [Tushare优化](TUSHARE_OPTIMIZATION.md)
2. [缓存与限流](CACHE_AND_RATE_LIMIT.md)
3. [准确率分析](ACCURACY_ANALYSIS_AND_OPTIMIZATION.md)

---

## 📖 文档更新记录

- **2025-12-28**: 整理文档结构，创建索引文档
- **2025-12-24**: v3.0版本文档更新

---

## 💡 文档贡献

如果发现文档有误或需要补充，欢迎：
1. 提交 Issue
2. 创建 Pull Request
3. 联系项目维护者

---

**最后更新**: 2025-12-28

