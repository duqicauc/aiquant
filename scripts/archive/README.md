# 历史脚本与模型归档说明

> 归档日期: 2026-05-09
> 原因: 中期 ML 模型训练框架重构，消除版本化脚本爆炸

---

## 一、归档内容清单

### 1.1 训练脚本 (`train_scripts/`)

共 **18 个**版本化训练脚本，总计约 **9,699 行代码**。

| 脚本 | 版本 | 架构 | 说明 |
|------|------|------|------|
| `train_v270_model.py` | v2.7.0 | 单 XGBoost | 基础模型 |
| `train_v271_conservative.py` | v2.7.1 | 单 XGBoost | 保守版 |
| `train_v280_model.py` | v2.8.0 | 单 XGBoost | 特征扩展 |
| `train_v281_model.py` | v2.8.1 | 单 XGBoost | — |
| `train_v291_model.py` | v2.9.1 | 单 XGBoost | 策略优化版 |
| `train_v292_catboost.py` | v2.9.2 | CatBoost | CatBoost 实验 |
| `train_v292_catboost_conservative.py` | v2.9.2 | CatBoost | 保守 CatBoost |
| `train_v292_catboost_conservative_highhn.py` | v2.9.2 | CatBoost | 高HN CatBoost |
| `train_v292_catboost_v291_params.py` | v2.9.2 | CatBoost | v291参数CatBoost |
| `train_v292_ensemble.py` | v2.9.2 | 集成 | 早期集成 |
| `train_v292_ensemble_v2.py` | v2.9.2 | 集成 | 集成v2 |
| `train_v293_ensemble_calibrated.py` | v2.9.3 | 校准集成 | Isotonic校准 |
| `train_v294_ensemble_professional.py` | v2.9.4 | 专业集成 | 温度缩放 |
| `train_v295_ensemble.py` | v2.9.5 | 集成 | — |
| `train_v296_ensemble.py` | v2.9.6 | 集成 | — |
| `train_v296b_ensemble.py` | v2.9.6b | 集成 | — |
| `train_v297_ensemble.py` | v2.9.7 | 集成 | 动态类别权重，放宽Gates |
| `train_v298_model.py` | v2.9.8 | 单 XGBoost | v2.7.0原始特征回归 |

### 1.2 最新代码快照 (`latest_snapshot/`)

保留最新的 **5 个**训练脚本作为重构基线参考：

- `train_v295_ensemble.py`
- `train_v296_ensemble.py`
- `train_v296b_ensemble.py`
- `train_v297_ensemble.py` ⭐ **主要重构基线**
- `train_v298_model.py` ⭐ **对比参考（单模型路线）**

同时保存了提取的配置文件：
- `v297_config_extracted.yaml` — v297 所有硬编码参数
- `v298_config_extracted.yaml` — v298 所有硬编码参数

### 1.3 特征提取脚本 (`extractors/`)

| 脚本 | 说明 |
|------|------|
| `extract_hard_negative_features_v291.py` | v291 硬负特征提取 |
| `extract_hard_negative_v5_base.py` | v5 基础硬负提取 |

### 1.4 样本生成脚本 (`generators/`)

| 脚本 | 说明 |
|------|------|
| `generate_hard_negatives_v291.py` | v291 硬负生成 |
| `generate_hard_negatives_v291_fast.py` | v291 快速硬负生成 |

### 1.5 预测脚本 (`predictors/`)

共 **17 个**历史预测脚本，从 v232 到 v295。

**当前生产预测脚本**（未归档，仍在 `scripts/` 根目录）：
- `predict_v296_with_stk_factor.py`

### 1.6 回测脚本 (`backtesters/`)

| 脚本 | 说明 |
|------|------|
| `backtest_v232_only.py` | v232 回测 |
| `backtest_v232_v270_complementary.py` | v232+v270 互补回测 |
| `backtest_v270_comparison.py` | v270 对比回测 |
| `backtest_v280_strategy.py` | v280 策略回测 |
| `backtest_v281_realistic.py` | v281 真实回测 |

**当前生产回测脚本**（未归档，仍在 `scripts/` 根目录）：
- `backtest_v291_realistic.py`

### 1.7 评分脚本 (`scorers/`)

| 脚本 | 说明 |
|------|------|
| `score_stocks_v292.py` | v292 评分 |

### 1.8 旧训练入口 (`old_training_entries/`)

| 脚本 | 说明 |
|------|------|
| `train_xgboost.py` | 基础XGBoost训练（README文档化的旧入口） |
| `train_xgboost_timeseries.py` | 时间序列XGBoost训练（README文档化的旧入口） |
| `train_ensemble_model.py` | 早期集成模型入口 |
| `train_calibrated_model.py` | 校准模型入口 |
| `train_optimized_model.py` | 优化模型入口 |
| `train_optimized_model_advanced.py` | 高级优化模型入口 |
| `train_model_version.py` | 模型版本管理入口 |

### 1.9 快速测试脚本 (`quick_tests/`)

| 脚本 | 说明 |
|------|------|
| `quick_test_conservative.py` | 保守策略快速测试 |
| `quick_test_conservative_local.py` | 本地保守测试 |
| `quick_test_conservative_full.py` | 完整保守测试 |
| `quick_test_highhn.py` | 高HN快速测试 |
| `quick_test_ensemble_v2.py` | 集成v2快速测试 |

### 1.10 旧样本筛选 (`old_screeners/`)

| 脚本 | 说明 |
|------|------|
| `screen_positive_samples.py` | 旧正样本筛选 |
| `screen_negative_samples_v2.py` | 旧负样本筛选 |

### 1.11 诊断脚本 (`diagnostics/`)

| 脚本 | 说明 |
|------|------|
| `diagnose_hard_neg_perf.py` | 硬负样本性能诊断 |
| `diagnose_hard_negative.py` | 硬负样本诊断 |
| `diagnose_mv_batch.py` | 市值批次诊断 |
| `diagnose_feature_extractor.py` | 特征提取器诊断 |

### 1.12 评估脚本 (`evaluations/`)

| 脚本 | 说明 |
|------|------|
| `evaluate_v270_top10_winrate.py` | v270 Top10胜率评估 |
| `evaluate_v280_trend.py` | v280趋势评估 |
| `evaluate_v280_wfv.py` | v280 WFV评估 |
| `evaluate_v270_stability.py` | v270稳定性评估 |
| `evaluate_v23x_complete.py` | v23x完整评估 |
| `evaluate_v232_v270_from_backtest_csv.py` | v232 vs v270 CSV评估 |
| `evaluate_v232_v270_20260105_20260303.py` | v232 vs v270特定区间评估 |
| `evaluate_v231_top10.py` | v231 Top10评估 |

### 1.13 分析脚本 (`analysis/`)

| 脚本 | 说明 |
|------|------|
| `analyze_v280_backtest.py` | v280回测分析 |
| `analyze_v251_optimization.py` | v251优化分析 |
| `analyze_v250_features.py` | v250特征分析 |

### 1.14 模型版本 (`data/models/archive/`)

已归档 **31 个**旧版本模型目录（v1.0.0 ~ v2.9.2 系列）。
已压缩旧备份：`data/models_backup_v270_20260422.tar.gz` (713M)

**保留的活跃版本**（仍在 `data/models/breakout_launch_scorer/versions/`）：
- `v2.9.1-ensemble` — testing 环境
- `v2.9.3-ensemble` — 曾在 production
- `v2.9.4-ensemble` — 曾在 production
- `v2.9.5-ensemble` — 曾在 production
- `v2.9.6-ensemble` — **当前 production / staging / development**
- `v2.9.8` — 最新单模型实验

---

## 二、迁移路径

### 旧方式（已废弃）
```bash
# 复制脚本、改参数、训练
python scripts/train_v297_ensemble.py
```

### 新方式（配置驱动）
```bash
# 统一入口 + YAML 配置
python scripts/train_midterm_model.py
python scripts/train_midterm_model.py --config config/models/midterm_experiments/v300_baseline.yaml
```

---

## 三、如需恢复旧脚本

所有归档文件仅被移动，未被删除。如需恢复：

```bash
# 恢复单个脚本
cp scripts/archive/train_scripts/train_v297_ensemble.py scripts/

# 恢复全部训练脚本
cp scripts/archive/train_scripts/*.py scripts/
```

---

## 四、关联文档

- [中期模型训练指南](../../docs/guides/MIDTERM_TRAINING_GUIDE.md) — 新框架完整文档
- [3L 评分系统规格书](../../docs/archive/3l_scoring_spec.md) — 3L 架构说明（已归档）
- [AGENTS.md](../../AGENTS.md) — 项目开发原则
