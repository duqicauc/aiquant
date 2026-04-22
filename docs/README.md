# AIQuant 文档索引

本文档是 AIQuant 量化交易系统的完整文档索引。

---

## 📁 文档结构

```
docs/
├── README.md           # 本文档（索引）
├── guides/             # 用户指南（如何使用）
├── reference/          # 技术参考（原理和规范）
├── archive/            # 历史文档（归档，供参考）
├── plans/              # 功能计划与方案
└── external/           # 第三方外部文档
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

## 🔄 核心策略文档 (guides/)

当前生产策略相关的核心文档：

| 文档 | 说明 |
|------|------|
| [v232+v270 互补策略](guides/V232_V270_COMPLEMENTARY_STRATEGY.md) | v2.3.2 与 v2.7.0 互补策略逻辑 |
| [v232+v270 交易落地](guides/V232_V270_TRADING_IMPLEMENTATION.md) | 回测与实盘落地说明 |
| [组合脚本使用](guides/COMBINE_V232_V270_USAGE.md) | combine_v232_v270.py 使用指南 |
| [A股实盘准入清单](guides/ASHARE_GO_NO_GO_CHECKLIST.md) | 模拟盘与实盘 Go/No-Go 检查 |

---

## 🔧 技术参考 (reference/)

### 核心概念

| 文档 | 说明 |
|------|------|
| [选股模型原理](reference/STOCK_SELECTION_MODEL.md) | 正负样本选股模型详解 |
| [避免未来函数](reference/AVOID_FUTURE_FUNCTION.md) | 时间序列划分原理 |
| [模型对比](reference/MODEL_COMPARISON.md) | XGBoost vs LSTM |
| [特征提取指南](reference/FEATURE_EXTRACTION_GUIDE.md) | 技术指标特征提取 |
| [v232 预测逻辑与特征重要性](reference/v232_prediction_logic_and_v230_feature_importance.md) | v2.3.2 核心逻辑 |
| [v270 底部放量突破公式](reference/V270_BOTTOM_VOLUME_BREAKOUT_FORMULA.md) | v2.7.0 底部放量计算 |
| [v270 特征与底部放量](reference/V270_FEATURES_AND_BOTTOM_VOLUME_BREAKOUT.md) | v2.7.0 特征说明 |

### 策略优化与评估

| 文档 | 说明 |
|------|------|
| [v232/v270 优化与评估](reference/V232_V270_OPTIMIZATION_AND_EVALUATION.md) | 互补策略优化思路 |
| [v232/v270 重训决策](reference/V232_V270_RETRAIN_DECISION.md) | 是否重训的评估指南 |
| [v270 预测状态](reference/v270_prediction_status.md) | v2.7.0 预测生成状态 |
| [v270 稳定性评估](reference/v270_stability_evaluation_guide.md) | v2.7.0 稳定性评估指南 |
| [准确率分析与优化](reference/ACCURACY_ANALYSIS_AND_OPTIMIZATION.md) | 模型准确率优化思路 |

### 模型管理

| 文档 | 说明 |
|------|------|
| [模型版本管理](reference/MODEL_VERSION_MANAGEMENT.md) | 版本管理方案 |
| [模型生命周期标准](reference/MODEL_LIFECYCLE_STANDARD.md) | 生命周期规范 |
| [模型生命周期快速参考](reference/MODEL_LIFECYCLE_QUICK_REFERENCE.md) | 快速参考卡 |
| [模型生命周期人工介入](reference/MODEL_LIFECYCLE_HUMAN_INTERVENTION.md) | 人工介入点说明 |
| [多模型架构](reference/ARCHITECTURE_MULTI_MODEL.md) | 多模型并行架构 |

### API 和数据

| 文档 | 说明 |
|------|------|
| [API参考文档](reference/API_REFERENCE.md) | 完整API接口说明 |
| [Tushare Pro功能](reference/TUSHARE_PRO_FEATURES.md) | Tushare高级功能 |
| [Tushare优化](reference/TUSHARE_OPTIMIZATION.md) | API优化方案 |
| [缓存与限流](reference/CACHE_AND_RATE_LIMIT.md) | 数据缓存机制 |
| [指标数据缓存](reference/INDICATOR_DATA_CACHE.md) | 指标数据缓存 |
| [原始数据字段](reference/RAW_DATA_FIELDS.md) | 数据字段说明 |

### 项目与测试

| 文档 | 说明 |
|------|------|
| [目录结构](reference/DIRECTORY_STRUCTURE.md) | 项目目录结构 |
| [预测目录关系](reference/PREDICTION_DIRECTORY_RELATIONSHIP.md) | 预测目录关系 |
| [预测时间估算](reference/PREDICTION_TIME_ESTIMATE.md) | 脚本执行时间估算 |
| [项目优化总结](reference/PROJECT_OPTIMIZATION_SUMMARY.md) | 项目优化总结 |
| [测试覆盖率分析](reference/TEST_COVERAGE_ANALYSIS.md) | 测试覆盖率报告 |
| [测试框架指南](reference/TESTING_FRAMEWORK_GUIDE.md) | 测试框架使用 |

### 基本面与样本

| 文档 | 说明 |
|------|------|
| [基本面筛选](reference/FUNDAMENTAL_SCREENING.md) | 基本面筛选功能 |
| [基本面筛选问题](reference/FUNDAMENTAL_SCREENING_ISSUES.md) | 筛选逻辑问题分析 |
| [基本面筛选策略](reference/FUNDAMENTAL_SCREENING_STRATEGY.md) | 筛选策略分析 |
| [基本面筛选阈值](reference/FUNDAMENTAL_SCREENING_THRESHOLDS.md) | 阈值推荐方案 |
| [样本目标分析](reference/SAMPLE_TARGET_ANALYSIS.md) | 样本目标分析 |
| [Top100基本面综合结果](reference/TOP100_FUNDAMENTAL_COMBINED_RESULTS.md) | 综合结果说明 |
| [交易计划模板](reference/TRADING_PLAN_TEMPLATE.md) | 固定目录结构模板 |

---

## 📦 归档文档 (archive/)

历史变更记录、优化笔记、对比分析等，仅供参考：

| 文档 | 说明 |
|------|------|
| [缓存优化修复](archive/CACHE_OPTIMIZATION_FIX.md) | stk_factor 缓存优化 |
| [特征对比](archive/FEATURE_COMPARISON.md) | 模型特征对比 |
| [人工介入提醒](archive/HUMAN_INTERVENTION_REMINDERS.md) | 人工介入机制 |
| [MACD参数对比](archive/MACD_PARAMETER_COMPARISON.md) | MACD参数选择分析 |
| [模型迁移 v1.0.0-legacy](archive/MODEL_MIGRATION_v1.0.0-legacy.md) | 旧模型迁移记录 |
| [模型版本迁移](archive/MODEL_VERSION_MIGRATION.md) | 版本迁移说明 |
| [负样本对比](archive/NEGATIVE_SAMPLE_COMPARISON.md) | 负样本筛选方案对比 |
| [负样本优化v3](archive/negative_sample_optimization_v3.md) | 负样本策略优化v3 |
| [正样本标准对比](archive/POSITIVE_SAMPLE_CRITERIA_COMPARISON.md) | 正负样本筛选逻辑 |
| [预测差异分析](archive/PREDICTION_DIFFERENCE_ANALYSIS.md) | 新旧框架预测差异 |
| [预测性能分析](archive/PREDICTION_PERFORMANCE_ANALYSIS.md) | 预测脚本性能分析 |
| [Tushare限流分析](archive/TUSHARE_RATE_LIMIT_ANALYSIS.md) | 限流规则分析 |
| [XGBoost vs v1.3.0对比](archive/xgboost_timeseries_vs_v1.3.0_comparison.md) | 模型构建逻辑对比 |

---

## 📐 计划文档 (plans/)

| 文档 | 说明 |
|------|------|
| [MA233特征计划](plans/ma233_feature_plan.md) | MA233均线特征开发计划 |

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

### 我想了解当前策略
1. [v232+v270 互补策略](guides/V232_V270_COMPLEMENTARY_STRATEGY.md)
2. [v232+v270 交易落地](guides/V232_V270_TRADING_IMPLEMENTATION.md)
3. [v232 预测逻辑](reference/v232_prediction_logic_and_v230_feature_importance.md)
4. [v270 特征说明](reference/V270_FEATURES_AND_BOTTOM_VOLUME_BREAKOUT.md)

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

# v270 预测
python scripts/predict_v270_ensemble_top50.py YYYYMMDD

# v232 预测
python scripts/predict_v232_top10.py --date YYYYMMDD

# 组合策略
python scripts/combine_v232_v270.py --date YYYYMMDD --strategy complementary --top 10

# 回测
python scripts/backtest_v232_v270_complementary.py \
  --start-date 20260105 --end-date 20260421 \
  --stop-loss-mode close --initial-cash 10000000
```

---

**最后更新**: 2026-04-22
