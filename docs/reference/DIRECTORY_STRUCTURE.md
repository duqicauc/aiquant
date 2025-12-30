# 项目目录结构说明

## 📁 目录结构

```
aiquant/
├── config/                 # 配置文件
│   ├── settings.yaml       # 主配置文件
│   └── ...
│
├── data/                   # 数据目录
│   ├── training/          # 模型训练相关数据
│   │   ├── samples/       # 训练样本
│   │   │   ├── positive_samples.csv
│   │   │   ├── negative_samples_v2.csv
│   │   │   └── *.json     # 样本统计信息
│   │   ├── features/      # 特征数据
│   │   │   ├── feature_data_34d.csv
│   │   │   └── negative_feature_data_v2_34d.csv
│   │   ├── models/        # 训练好的模型
│   │   │   └── xgboost_timeseries_*.json
│   │   ├── metrics/       # 模型评估指标
│   │   │   ├── xgboost_timeseries_v2_metrics.json
│   │   │   └── walk_forward_validation_results.json
│   │   └── charts/        # 训练过程可视化图表
│   │       └── *.png, *.html
│   │
│   ├── prediction/        # 实际预测相关数据
│   │   ├── results/       # 预测结果
│   │   │   ├── stock_scores_*.csv
│   │   │   ├── top_50_stocks_*.csv
│   │   │   └── prediction_report_*.txt
│   │   ├── metadata/      # 预测元数据（用于准确率分析）
│   │   │   └── prediction_metadata_*.json
│   │   ├── annotations/   # 人工标注数据
│   │   │   ├── YYYYMMDD_人工标注.xlsx
│   │   │   └── README.md
│   │   ├── analysis/      # 准确率分析结果
│   │   │   ├── accuracy_*.csv
│   │   │   ├── accuracy_report_*.txt
│   │   │   └── accuracy_*.json
│   │   └── history/       # 历史预测归档
│   │       └── YYYYMMDD/
│   │
│   └── cache/             # 数据缓存
│       └── quant_data.db  # SQLite缓存数据库
│
├── models/                # 模型目录（已废弃，使用 data/training/models）
│
├── src/                   # 核心源代码（39个Python文件）
│   ├── data/              # 数据管理模块
│   │   ├── data_manager.py
│   │   ├── fetcher/      # 数据获取
│   │   └── storage/      # 数据存储
│   ├── strategy/         # 策略模块
│   │   ├── screening/   # 筛选器
│   │   ├── portfolio/   # 组合管理
│   │   └── timing/       # 择时策略
│   ├── models/           # 模型模块
│   ├── utils/            # 工具函数
│   ├── analysis/         # 分析模块
│   ├── backtest/         # 回测模块
│   └── visualization/    # 可视化模块
│
├── scripts/               # 可执行脚本（27个Python脚本）
│   ├── prepare_positive_samples.py      # 导入 src 模块
│   ├── prepare_negative_samples_v2.py   # 导入 src 模块
│   ├── train_xgboost_timeseries.py      # 导入 src 模块
│   ├── score_current_stocks.py          # 导入 src 模块
│   ├── analyze_prediction_accuracy.py   # 导入 src 模块
│   └── ...
│
├── tests/                # 测试代码（待补充）
│   └── __init__.py
│
└── docs/                 # 文档目录
    ├── COMPLETE_WORKFLOW.md
    ├── MODEL_TRAINING_GUIDE.md
    └── ...
```

## 💻 代码组织说明

### 源代码 (`src/`)

**作用**: 核心业务逻辑，可复用的模块

- **data/**: 数据管理（DataManager、Fetcher、Cache）
- **strategy/**: 策略模块（筛选器、财务过滤）
- **models/**: 模型相关（评估、预测）
- **utils/**: 工具函数（日志、日期、限流）
- **analysis/**: 分析模块（市场分析、健康检查）
- **backtest/**: 回测模块
- **visualization/**: 可视化模块

**使用方式**: 被 `scripts/` 中的脚本导入使用

### 可执行脚本 (`scripts/`)

**作用**: 项目入口，完成具体任务

- **训练脚本**: 样本准备、模型训练
- **预测脚本**: 股票评分、准确率分析
- **工具脚本**: 质量检查、可视化

**特点**: 导入 `src/` 模块，处理业务逻辑，读写 `data/` 数据

### 测试代码 (`tests/`)

**当前状态**: 基本为空，待补充

**应该包含**: 单元测试、集成测试、数据质量测试

---

## 📊 数据分类说明

### 训练数据 (`data/training/`)

**用途**: 模型训练和评估

- **samples/**: 正负样本数据
- **features/**: 提取的特征数据
- **models/**: 训练好的模型文件
- **metrics/**: 模型评估指标和验证结果

### 预测数据 (`data/prediction/`)

**用途**: 实际预测和准确率分析

- **results/**: 每次预测的详细结果（原始输出，带时间戳）
- **metadata/**: 预测元数据（推荐股票列表，用于后续准确率分析）
- **annotations/**: 人工标注数据（用于评估模型准确率）
- **analysis/**: 准确率分析结果（基于 metadata 和 annotations 分析实际表现）
- **history/**: 历史预测归档（按日期组织，从 results/ 复制）

**目录关系**:
```
预测脚本 → results/ + metadata/
                ↓
         history/（归档）
                ↓
         annotations/（人工标注）
                ↓
         analysis/（准确率分析，基于 metadata + annotations）
```

详细说明请参考：[预测目录关系说明](PREDICTION_DIRECTORY_RELATIONSHIP.md)

### 缓存数据 (`data/cache/`)

**用途**: 数据缓存，避免重复下载

- SQLite数据库，存储从API获取的数据

## 🔄 数据流转

### 训练流程
```
原始数据 → 样本准备 → 特征提取 → 模型训练 → 模型评估
  ↓           ↓          ↓          ↓          ↓
data/raw  training/   training/  training/  training/
         samples/    features/  models/    metrics/
```

### 预测流程
```
模型 → 股票评分 → 预测结果 → 元数据保存 → 准确率分析
 ↓        ↓         ↓          ↓            ↓
training/ 预测脚本  prediction/ prediction/ prediction/
models/            results/    metadata/    analysis/
```

## 📝 文件命名规范

### 训练数据
- 样本: `positive_samples.csv`, `negative_samples_v2.csv`
- 特征: `feature_data_34d.csv`, `negative_feature_data_v2_34d.csv`
- 模型: `xgboost_timeseries_v2_YYYYMMDD_HHMMSS.json`
- 指标: `xgboost_timeseries_v2_metrics.json`

### 预测数据
- 评分结果: `stock_scores_YYYYMMDD_HHMMSS.csv`
- Top推荐: `top_50_stocks_YYYYMMDD_HHMMSS.csv`
- 预测报告: `prediction_report_YYYYMMDD_HHMMSS.txt`
- 元数据: `prediction_metadata_YYYYMMDD_HHMMSS.json`
- 分析结果: `accuracy_YYYYMMDD_Nw.csv`

## 🗑️ 已废弃目录

以下目录已废弃并删除，数据已迁移：
- `data/processed/` → `data/training/` ✅ 已删除
- `data/results/` → `data/prediction/results/` ✅ 已删除
- `models/` → `data/training/models/` ✅ 已删除
- `data/predictions/` → `data/prediction/history/` ✅ 已删除
- `data/charts/` → `data/training/charts/` ✅ 已删除
- `data/models/` → `data/training/models/` ✅ 已删除
- `data/backtest/` → 已删除（未使用）
- `data/backup/` → 已删除（未使用）
- `data/database/` → 已删除（未使用）

## 📚 相关文档

- [完整工作流程](COMPLETE_WORKFLOW.md)
- [模型训练指南](MODEL_TRAINING_GUIDE.md)
- [预测准确率分析](analyze_prediction_accuracy.py)

