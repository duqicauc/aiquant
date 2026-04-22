# 多模型并行架构设计

## 📋 概述

本文档描述支持多个模型并行的完整架构设计，涵盖从数据获取到模型迭代的全生命周期管理。

---

## 🎯 设计目标

1. **多模型并行支持** - 同时管理多个模型，每个模型独立运行
2. **模块化设计** - 各环节解耦，可独立扩展
3. **数据共享** - 基础数据共享，模型数据隔离
4. **版本管理** - 完整的模型版本和迭代管理
5. **自动化流程** - 支持自动化训练、评测、预测流程

---

## 🏗️ 架构设计

### 1. 目录结构优化

```
aiquant/
├── config/
│   ├── settings.yaml              # 全局配置
│   └── models/                    # 模型配置目录
│       ├── xgboost_timeseries.yaml
│       ├── lstm_momentum.yaml
│       ├── ensemble_v1.yaml
│       └── ...
│
├── data/
│   ├── raw/                       # 原始数据（共享）
│   │   ├── daily/                 # 日线数据
│   │   ├── weekly/                # 周线数据
│   │   ├── financial/             # 财务数据
│   │   └── indicators/            # 技术指标
│   │
│   ├── processed/                 # 处理后数据（共享）
│   │   ├── features/             # 基础特征
│   │   └── annotations/          # 人工标注
│   │
│   ├── models/                    # 模型数据（按模型隔离）
│   │   ├── {model_name}/          # 模型名称
│   │   │   ├── config.yaml        # 模型配置
│   │   │   ├── data/              # 模型专用数据
│   │   │   │   ├── samples/      # 训练样本
│   │   │   │   ├── features/     # 模型特征
│   │   │   │   └── annotations/  # 模型标注
│   │   │   ├── training/          # 训练相关
│   │   │   │   ├── models/       # 模型文件
│   │   │   │   ├── metrics/      # 训练指标
│   │   │   │   ├── logs/         # 训练日志
│   │   │   │   └── checkpoints/  # 检查点
│   │   │   ├── evaluation/       # 评测相关
│   │   │   │   ├── backtest/     # 回测结果
│   │   │   │   ├── validation/   # 验证结果
│   │   │   │   └── reports/      # 评测报告
│   │   │   ├── prediction/       # 预测相关
│   │   │   │   ├── results/      # 预测结果
│   │   │   │   ├── metadata/     # 预测元数据
│   │   │   │   └── history/      # 历史预测
│   │   │   └── iteration/       # 迭代管理
│   │   │       ├── versions/     # 版本历史
│   │   │       └── experiments/  # 实验记录
│   │   │
│   │   └── ensemble/              # 集成模型
│   │       ├── config.yaml
│   │       ├── models/           # 子模型列表
│   │       └── weights/          # 权重配置
│   │
│   └── cache/                     # 缓存（共享）
│       └── quant_data.db
│
├── src/
│   ├── data/                      # 数据管理
│   │   ├── fetcher/              # 数据获取
│   │   │   ├── base.py          # 基础获取器
│   │   │   ├── tushare_fetcher.py
│   │   │   └── parallel_fetcher.py  # 并行获取
│   │   ├── storage/              # 数据存储
│   │   │   ├── cache_manager.py
│   │   │   └── data_organizer.py
│   │   └── annotation/          # 人工标注
│   │       ├── annotation_manager.py
│   │       ├── annotation_loader.py
│   │       └── annotation_validator.py
│   │
│   ├── features/                 # 特征工程
│   │   ├── extractors/           # 特征提取器
│   │   │   ├── base.py          # 基础提取器
│   │   │   ├── technical/       # 技术指标
│   │   │   │   ├── ma.py
│   │   │   │   ├── macd.py
│   │   │   │   ├── rsi.py
│   │   │   │   └── ...
│   │   │   ├── financial/       # 财务指标
│   │   │   │   ├── profitability.py
│   │   │   │   ├── liquidity.py
│   │   │   │   └── ...
│   │   │   └── market/          # 市场指标
│   │   │       ├── volume.py
│   │   │       └── volatility.py
│   │   ├── pipeline.py          # 特征流水线
│   │   └── selector.py          # 特征选择
│   │
│   ├── models/                    # 模型管理
│   │   ├── registry.py          # 模型注册表（增强）
│   │   ├── lifecycle/           # 生命周期管理
│   │   │   ├── manager.py      # 模型管理器
│   │   │   ├── trainer.py      # 训练器
│   │   │   ├── evaluator.py    # 评测器
│   │   │   ├── predictor.py    # 预测器
│   │   │   └── iterator.py     # 迭代器
│   │   ├── ensemble/            # 模型集成
│   │   │   ├── base.py         # 基础集成器
│   │   │   ├── voting.py       # 投票集成
│   │   │   ├── stacking.py     # 堆叠集成
│   │   │   └── blending.py     # 混合集成
│   │   └── types/               # 模型类型
│   │       ├── xgboost_model.py
│   │       ├── lstm_model.py
│   │       └── ...
│   │
│   └── pipeline/                 # 流水线
│       ├── data_pipeline.py     # 数据流水线
│       ├── training_pipeline.py # 训练流水线
│       ├── evaluation_pipeline.py # 评测流水线
│       └── prediction_pipeline.py # 预测流水线
│
└── scripts/
    ├── models/                   # 模型管理脚本
    │   ├── create_model.py     # 创建新模型
    │   ├── list_models.py       # 列出所有模型
    │   └── delete_model.py      # 删除模型
    │
    ├── data/                     # 数据管理脚本
    │   ├── fetch_data.py        # 数据获取
    │   ├── annotate_data.py     # 人工标注
    │   └── extract_features.py # 特征提取
    │
    ├── training/                 # 训练脚本
    │   ├── train_model.py       # 训练单个模型
    │   ├── train_all.py         # 训练所有模型
    │   └── train_parallel.py    # 并行训练
    │
    ├── evaluation/               # 评测脚本
    │   ├── evaluate_model.py    # 评测单个模型
    │   ├── compare_models.py    # 模型对比
    │   └── backtest_model.py    # 回测
    │
    ├── prediction/               # 预测脚本
    │   ├── predict_model.py     # 单个模型预测
    │   ├── predict_ensemble.py  # 集成模型预测
    │   └── predict_all.py       # 所有模型预测
    │
    └── iteration/                # 迭代脚本
        ├── create_version.py    # 创建新版本
        ├── rollback_version.py  # 回滚版本
        └── compare_versions.py  # 版本对比
```

---

## 🔄 核心流程设计

### 1. 数据获取流程

```python
# 并行数据获取
from src.data.fetcher.parallel_fetcher import ParallelDataFetcher

fetcher = ParallelDataFetcher(
    workers=4,  # 并行工作线程数
    cache_enabled=True
)

# 获取多个股票的数据
data = fetcher.fetch_batch(
    stock_codes=['000001.SZ', '600000.SH', ...],
    start_date='20200101',
    end_date='20241231',
    data_types=['daily', 'weekly', 'financial']
)
```

**特点**：
- 支持并行获取多个股票数据
- 自动缓存，避免重复请求
- 支持增量更新
- 数据统一存储到 `data/raw/`

### 2. 人工标注流程

```python
# 人工标注管理
from src.data.annotation.annotation_manager import AnnotationManager

manager = AnnotationManager()

# 创建标注任务
task = manager.create_annotation_task(
    model_name='xgboost_timeseries',
    prediction_date='20241228',
    samples=100  # 需要标注的样本数
)

# 加载标注数据
annotations = manager.load_annotations(
    model_name='xgboost_timeseries',
    date='20241228'
)

# 验证标注质量
quality = manager.validate_annotations(annotations)
```

**特点**：
- 支持为不同模型创建独立的标注任务
- 标注数据存储在 `data/models/{model_name}/data/annotations/`
- 支持标注质量验证
- 支持标注历史追踪

### 3. 特征提取流程

```python
# 特征提取流水线
from src.features.pipeline import FeaturePipeline
from src.features.extractors.technical import MAExtractor, MACDExtractor
from src.features.extractors.financial import ProfitabilityExtractor

# 创建特征流水线
pipeline = FeaturePipeline()

# 添加特征提取器
pipeline.add_extractor(MAExtractor(periods=[5, 10, 20, 60]))
pipeline.add_extractor(MACDExtractor())
pipeline.add_extractor(ProfitabilityExtractor())

# 为特定模型提取特征
features = pipeline.extract(
    model_name='xgboost_timeseries',
    stock_data=data,
    lookback_days=34
)
```

**特点**：
- 模块化特征提取器
- 支持为不同模型配置不同特征
- 特征缓存机制
- 特征选择支持

### 4. 模型训练流程

```python
# 模型训练
from src.models.lifecycle.trainer import ModelTrainer
from src.models.registry import ModelRegistry

# 获取模型配置
model_config = ModelRegistry.get('xgboost_timeseries')

# 创建训练器
trainer = ModelTrainer(model_config)

# 训练模型
result = trainer.train(
    samples_path='data/models/xgboost_timeseries/data/samples/train.csv',
    validation_split=0.2,
    time_series_split=True
)

# 保存模型
trainer.save_model(
    version='v1.0',
    metrics=result.metrics
)
```

**特点**：
- 支持并行训练多个模型
- 自动版本管理
- 训练过程可中断和恢复
- 完整的训练日志和指标记录

### 5. 模型集成流程

```python
# 模型集成
from src.models.ensemble.voting import VotingEnsemble

# 创建集成模型
ensemble = VotingEnsemble(
    name='ensemble_v1',
    models=[
        ('xgboost_timeseries', 'v1.0', 0.4),
        ('lstm_momentum', 'v1.0', 0.3),
        ('xgboost_breakout', 'v1.0', 0.3)
    ],
    method='weighted'  # 加权投票
)

# 训练集成模型（可选）
ensemble.fit(validation_data)

# 预测
predictions = ensemble.predict(stock_data)
```

**特点**：
- 支持多种集成方法（投票、堆叠、混合）
- 支持动态权重调整
- 集成模型独立管理

### 6. 模型评测流程

```python
# 模型评测
from src.models.lifecycle.evaluator import ModelEvaluator

evaluator = ModelEvaluator('xgboost_timeseries')

# 评测模型
results = evaluator.evaluate(
    model_version='v1.0',
    test_data='data/models/xgboost_timeseries/data/samples/test.csv',
    metrics=['accuracy', 'precision', 'recall', 'f1', 'auc']
)

# 回测
backtest_results = evaluator.backtest(
    model_version='v1.0',
    start_date='20230101',
    end_date='20241231'
)

# 生成评测报告
evaluator.generate_report(results, backtest_results)
```

**特点**：
- 多维度评测（准确率、回测、风险指标）
- 支持模型对比
- 自动生成评测报告

### 7. 模型预测流程

```python
# 模型预测
from src.models.lifecycle.predictor import ModelPredictor

predictor = ModelPredictor('xgboost_timeseries')

# 预测
predictions = predictor.predict(
    model_version='v1.0',
    stock_data=current_stock_data,
    top_n=50
)

# 保存预测结果
predictor.save_predictions(
    predictions=predictions,
    prediction_date='20241228',
    metadata={'market_state': 'bull'}
)
```

**特点**：
- 统一预测接口
- 支持批量预测
- 自动保存预测结果和元数据
- 支持预测历史追踪

### 8. 模型迭代流程

```python
# 模型迭代
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('xgboost_timeseries')

# 创建新版本
new_version = iterator.create_version(
    base_version='v1.0',
    changes={
        'features': ['added_obv', 'added_kdj'],
        'parameters': {'n_estimators': 150}
    }
)

# 训练新版本
iterator.train_version(new_version)

# 对比版本
comparison = iterator.compare_versions('v1.0', 'v1.1')

# 如果新版本更好，升级
if comparison['v1.1']['score'] > comparison['v1.0']['score']:
    iterator.promote_version('v1.1', 'production')
```

**特点**：
- 完整的版本管理
- 支持A/B测试
- 版本对比和回滚
- 实验记录追踪

---

## 📊 模型配置示例

### 模型配置文件：`config/models/xgboost_timeseries.yaml`

```yaml
# 模型基本信息
model:
  name: xgboost_timeseries
  display_name: XGBoost时间序列模型
  description: 基于XGBoost的时间序列选股模型
  type: xgboost
  version: v1.0

# 数据配置
data:
  sample_preparation:
    positive_criteria:
      consecutive_weeks: 3
      total_return_threshold: 50
    negative_criteria:
      method: same_period_other_stocks
      sample_ratio: 1.0

  feature_extraction:
    lookback_days: 34
    extractors:
      - technical.ma
      - technical.macd
      - technical.rsi
      - financial.profitability
      - market.volume

# 模型参数
model_params:
  objective: binary:logistic
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 5
  subsample: 0.8
  colsample_bytree: 0.8

# 训练配置
training:
  validation_split: 0.2
  time_series_split: true
  early_stopping: true
  n_splits: 5

# 预测配置
prediction:
  top_n: 50
  min_probability: 0.0
  exclusion_rules:
    exclude_st: true
    exclude_new_listed: true
    min_listing_days: 180

# 评测配置
evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1
    - auc
  backtest:
    enabled: true
    start_date: 20230101
    end_date: 20241231
```

---

## 🔧 实现步骤

### 阶段1：基础架构（1-2周）

1. ✅ 优化目录结构
2. ✅ 增强模型注册表
3. ✅ 实现模型生命周期管理器
4. ✅ 创建模型配置系统

### 阶段2：数据管理（1-2周）

1. ✅ 实现并行数据获取
2. ✅ 实现人工标注系统
3. ✅ 优化特征提取模块
4. ✅ 实现数据组织器

### 阶段3：模型训练（1-2周）

1. ✅ 实现模型训练器
2. ✅ 支持并行训练
3. ✅ 实现版本管理
4. ✅ 实现训练监控

### 阶段4：模型集成（1周）

1. ✅ 实现集成框架
2. ✅ 支持多种集成方法
3. ✅ 实现权重优化

### 阶段5：评测和预测（1周）

1. ✅ 实现评测器
2. ✅ 实现预测器
3. ✅ 实现结果管理

### 阶段6：迭代管理（1周）

1. ✅ 实现迭代器
2. ✅ 实现版本对比
3. ✅ 实现实验追踪

---

## 📈 优势

1. **模块化** - 各环节独立，易于维护和扩展
2. **并行化** - 支持数据获取和模型训练的并行
3. **可扩展** - 易于添加新模型类型和特征提取器
4. **可追踪** - 完整的版本和实验记录
5. **自动化** - 支持自动化训练、评测、预测流程
6. **隔离性** - 不同模型数据隔离，互不影响

---

## 🎯 使用示例

### 创建新模型

```bash
python scripts/models/create_model.py \
    --name lstm_momentum \
    --type lstm \
    --config config/models/lstm_momentum.yaml
```

### 训练模型

```bash
python scripts/training/train_model.py \
    --model xgboost_timeseries \
    --version v1.0
```

### 并行训练多个模型

```bash
python scripts/training/train_parallel.py \
    --models xgboost_timeseries,lstm_momentum \
    --workers 2
```

### 评测模型

```bash
python scripts/evaluation/evaluate_model.py \
    --model xgboost_timeseries \
    --version v1.0
```

### 集成模型预测

```bash
python scripts/prediction/predict_ensemble.py \
    --ensemble ensemble_v1 \
    --date 20241228
```

---

**文档版本**: v1.0
**创建日期**: 2025-12-28
**最后更新**: 2025-12-28
