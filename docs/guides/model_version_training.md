# 模型版本训练指南

本文档介绍如何使用标准流程训练新版本模型，包括数据准备、模型训练、验证和备份。

## 快速开始

### 训练新版本（完整流程）

```bash
# 训练 v1.5.0 版本（完整流程）
python scripts/train_model_version.py --version v1.5.0
```

这个命令会自动执行：
1. 准备正样本数据
2. 准备负样本数据
3. 添加高级技术因子
4. 训练模型（带版本管理）
5. Walk-forward 验证
6. 备份训练数据到版本目录
7. 生成版本报告

## 常用命令

### 完整训练流程

```bash
# 从零开始训练新版本
python scripts/train_model_version.py --version v1.5.0
```

### 使用现有数据训练

```bash
# 跳过数据准备，直接使用现有数据训练
python scripts/train_model_version.py --version v1.5.0 --skip-data-prep
```

### 快速训练（跳过验证）

```bash
# 跳过数据准备和验证
python scripts/train_model_version.py --version v1.5.0 --skip-data-prep --skip-validation
```

### 只备份数据

```bash
# 为已有版本补充备份数据
python scripts/train_model_version.py --version v1.4.0 --backup-only
```

### 使用基础特征

```bash
# 不使用高级技术因子
python scripts/train_model_version.py --version v1.5.0 --no-advanced-factors
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--version` | 模型版本号（必填） | - |
| `--model-name` | 模型名称 | `breakout_launch_scorer` |
| `--neg-version` | 负样本版本 | `v2` |
| `--skip-data-prep` | 跳过数据准备 | False |
| `--skip-training` | 跳过训练 | False |
| `--skip-validation` | 跳过验证 | False |
| `--backup-only` | 只执行备份 | False |
| `--no-advanced-factors` | 不使用高级因子 | False |

## 流程详解

### Step 1: 数据准备

#### 1a. 准备正样本
```bash
python scripts/prepare_positive_samples.py
```

正样本筛选条件（配置在 `config/settings.yaml`）：
- 连续 3 周上涨
- 总涨幅 ≥ 50%
- 最高涨幅 ≤ 70%
- 上市天数 ≥ 180 天

输出文件：
- `data/training/samples/positive_samples.csv`

#### 1b. 准备负样本
```bash
python scripts/prepare_negative_samples_v2.py
```

负样本筛选方法：同期其他股票法

输出文件：
- `data/training/samples/negative_samples_v2.csv`

#### 1c. 添加高级技术因子
```bash
python scripts/add_advanced_factors.py --sample-type positive
python scripts/add_advanced_factors.py --sample-type negative
```

添加的因子包括：
- 动量因子（5日、10日、20日）
- 量价配合度
- 多时间框架特征（8日、55日）
- 突破形态
- 支撑阻力位
- 高级成交量指标

输出文件：
- `data/training/processed/feature_data_34d_advanced.csv`
- `data/training/features/negative_feature_data_v2_34d_advanced.csv`

### Step 2: 模型训练

```bash
python scripts/train_xgboost_timeseries.py \
  --neg-version v2 \
  --use-advanced-factors \
  --version v1.5.0
```

训练特点：
- 时间序列划分（80% 训练，20% 测试）
- 避免未来函数
- 自动保存到版本目录

输出文件：
- `data/models/breakout_launch_scorer/versions/v1.5.0/model/model.json`
- `data/models/breakout_launch_scorer/versions/v1.5.0/training/metrics.json`

### Step 3: Walk-forward 验证

```bash
# 基础用法（结果保存到 data/results/）
python scripts/walk_forward_validation.py \
  --neg-version v2 \
  --use-advanced-factors

# 带版本管理（结果同时保存到版本目录）
python scripts/walk_forward_validation.py \
  --neg-version v2 \
  --use-advanced-factors \
  --version v1.5.0
```

验证方法：
- 5 个时间窗口
- 每个窗口 60% 训练，40% 测试
- 计算各窗口的性能指标和稳定性

输出文件：
- `data/results/walk_forward_validation_results.json`
- （指定版本时）`data/models/.../versions/v1.5.0/evaluation/walk_forward_validation.json`
- （指定版本时）`data/models/.../versions/v1.5.0/evaluation/validation_report.md`

### Step 4: 备份训练数据

自动备份以下内容到版本目录：

```
data/models/breakout_launch_scorer/versions/v1.5.0/training_data/
├── samples/                    # 样本数据
│   ├── positive_samples.csv
│   ├── negative_samples_v2.csv
│   └── negative_sample_statistics_v2.json
├── positive_features/          # 正样本特征
│   ├── feature_data_34d.csv
│   └── feature_data_34d_advanced.csv
├── negative_features/          # 负样本特征
│   ├── negative_feature_data_v2_34d.csv
│   └── negative_feature_data_v2_34d_advanced.csv
└── BACKUP_README.md            # 备份说明
```

### Step 5: 生成版本报告

输出文件：
- `data/models/breakout_launch_scorer/versions/v1.5.0/version_report.json`

## 版本目录结构

训练完成后，版本目录结构如下：

```
data/models/breakout_launch_scorer/
├── current.json                    # 版本指针
└── versions/
    └── v1.5.0/
        ├── metadata.json           # 版本元数据
        ├── training_config.yaml    # 训练配置
        ├── version_report.json     # 版本报告
        ├── model/
        │   ├── model.json          # 模型文件
        │   └── feature_names.json  # 特征名称
        ├── training/
        │   └── metrics.json        # 训练指标
        ├── charts/                 # 可视化图表
        ├── evaluation/             # 评估结果
        ├── experiments/            # 实验记录
        └── training_data/          # 训练数据备份
            ├── samples/
            ├── positive_features/
            ├── negative_features/
            └── BACKUP_README.md
```

## 版本管理

### 查看版本状态

```bash
python -c "
from src.models.lifecycle import ModelIterator
mi = ModelIterator('breakout_launch_scorer')
mi.print_status()
"
```

### 提升版本到生产环境

```bash
python -c "
from src.models.lifecycle import ModelIterator
mi = ModelIterator('breakout_launch_scorer')
mi.set_current_version('v1.5.0', 'production')
"
```

### 比较两个版本

```bash
python -c "
from src.models.lifecycle import ModelIterator
mi = ModelIterator('breakout_launch_scorer')
comparison = mi.compare_versions('v1.4.0', 'v1.5.0')
mi.print_comparison(comparison)
"
```

## 从备份恢复

如果需要使用备份数据重新训练：

```bash
# 1. 复制备份数据到训练目录
VERSION=v1.4.0
cp -r data/models/breakout_launch_scorer/versions/$VERSION/training_data/samples/* data/training/samples/
cp -r data/models/breakout_launch_scorer/versions/$VERSION/training_data/positive_features/* data/training/processed/
cp -r data/models/breakout_launch_scorer/versions/$VERSION/training_data/negative_features/* data/training/features/

# 2. 重新训练
python scripts/train_xgboost_timeseries.py \
  --neg-version v2 \
  --use-advanced-factors \
  --version ${VERSION}_retrain
```

## 最佳实践

### 1. 版本命名规范

- 主版本号：重大改动（如新模型架构）
- 次版本号：功能改进（如新特征）
- 修订号：Bug 修复

示例：
- `v1.0.0` - 初始版本
- `v1.1.0` - 添加市场因子
- `v1.2.0` - 添加技术因子
- `v1.2.1` - 修复数据问题

### 2. 训练前检查

- 确保数据源可用（Tushare API）
- 检查磁盘空间（备份需要 ~1GB）
- 确认配置文件正确

### 3. 训练后验证

- 检查 AUC ≥ 0.7
- 检查 F1-Score ≥ 0.7
- 检查 Walk-forward 稳定性

### 4. 定期备份

每次训练新版本都会自动备份，无需手动操作。

## 故障排除

### 数据准备失败

```bash
# 检查 Tushare API
python -c "import tushare as ts; print(ts.__version__)"

# 检查网络连接
python scripts/recover_tushare.py
```

### 训练失败

```bash
# 检查数据文件
ls -la data/training/samples/
ls -la data/training/processed/
ls -la data/training/features/

# 检查配置文件
cat config/settings.yaml
```

### 备份失败

```bash
# 检查磁盘空间
df -h

# 手动备份
python scripts/train_model_version.py --version v1.4.0 --backup-only
```

## 相关脚本

| 脚本 | 功能 |
|------|------|
| `train_model_version.py` | 完整训练流程 |
| `train_xgboost_timeseries.py` | 模型训练 |
| `walk_forward_validation.py` | Walk-forward 验证 |
| `prepare_positive_samples.py` | 准备正样本 |
| `prepare_negative_samples_v2.py` | 准备负样本 |
| `add_advanced_factors.py` | 添加高级因子 |
| `score_current_stocks.py` | 股票评分预测 |

## 参考文档

- [模型配置说明](../reference/model_config.md)
- [特征工程说明](../reference/feature_engineering.md)
- [版本管理说明](../reference/version_management.md)

