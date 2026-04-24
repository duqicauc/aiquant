# AIQuant v2.8.0 模型重训练与参数调优计划

> **状态**: 计划阶段，待用户确认后执行
> **创建日期**: 2026-04-22
> **触发条件**: 策略层优化已完成（止损4%/MA5_cd2/无跟踪止盈/无板块限制为最优），模型样本已117天未更新
> **预计总耗时**: 4-8 小时（可分阶段执行）
> **Tushare积分预算**: ~500-1000 积分

---

## 一、项目现状与问题诊断

### 1.1 数据时效性（核心瓶颈）

| 数据源 | 最新日期 | 距今天数 | 状态 |
|--------|---------|---------|------|
| 正样本 `positive_samples.csv` | 2025-12-26 | **117天** | 🔴 严重滞后 |
| 负样本 `negative_samples_v2.csv` | 2025-12-26 | **117天** | 🔴 严重滞后 |
| Cache DB `daily_data` | 2025-12-26 | **117天** | 🔴 严重滞后 |
| v6 特征数据 | 2025-12-24 | **119天** | 🔴 严重滞后 |
| 回测预测数据 | 2026-04-21 | 0天 | 🟢 实时拉取 |

**关键发现**：2026-01-05 至 2026-04-21 的回测数据是通过 DataManager 实时从 Tushare 拉取的，但**未写回 cache**。模型训练依赖的 cache 数据全面停留在 2025-12-26。

### 1.2 模型漂移证据

| 指标 | v2.7.0 训练时 | WFV 最新窗口 | 变化 |
|------|-------------|-------------|------|
| AUC | 0.9782 | 0.791 | -0.187 |
| Accuracy | 88% | 71% | -17pp |
| F1 | 0.86 | 0.761 | -0.099 |

**结论**：模型对近期市场环境（2025-2026）的适应性显著下降，重训练是必要且紧迫的。

### 1.3 当前超参数问题

| 参数 | 当前值 | 问题 |
|------|--------|------|
| `scale_pos_weight` | 1.5（硬编码） | 未根据实际正负比例动态调整 |
| 超参数搜索 | 仅6组手动组合 | 6维参数空间搜索过于稀疏 |
| 时序划分 | 按样本数65%/15%/20% | 样本密度不均时时间边界模糊，存在轻微泄露风险 |
| 特征集 | 167列 | 低重要性特征（<0.3%）未系统性剔除 |

---

## 二、重训练目标

### 2.1 核心目标

1. **数据新鲜度**：正样本覆盖至 2026-04-21，消除117天数据断层
2. **模型性能**：测试集 AUC ≥ 0.85，Accuracy ≥ 80%（WFV 最新窗口基准）
3. **参数优化**：超参数搜索覆盖 ≥30 组组合，找到全局更优配置
4. **策略验证**：新模型 Top10 胜率 ≥ 48%（vs 当前纯模型 50.21% 的基准不降低）

### 2.2 版本命名

- 重训练后版本：**v2.8.0**（数据更新 + 参数调优 + 特征优化）
- 若仅数据更新无结构变化：**v2.7.1**（保守命名）

---

## 三、实施阶段详表

### 阶段 0：前置检查与环境准备（10分钟）

**目标**：确认环境状态，避免中途失败

| 步骤 | 操作 | 检查点 | 预期结果 |
|------|------|--------|---------|
| 0.1 | 检查 Tushare 积分余额 | `python3 -c "from src.data.fetcher.tushare_fetcher import TushareFetcher; print(TushareFetcher().pro.query('user'))"` | 积分 ≥ 4000（当前5120） |
| 0.2 | 检查磁盘空间 | `df -h` | 可用空间 ≥ 10GB（当前106GB） |
| 0.3 | 检查 Python 环境 | `python3 --version && pip list \| grep -E "xgboost\|lightgbm\|catboost\|sklearn"` | xgboost≥2.0, sklearn≥1.3 |
| 0.4 | 备份当前模型 | `cp -r data/models data/models_backup_v270_$(date +%Y%m%d)` | 备份完成，大小 ~2.1GB |
| 0.5 | 备份当前样本 | `cp data/training/samples/positive_samples.csv data/training/samples/positive_samples_backup_$(date +%Y%m%d).csv` | 备份完成 |

**可中断性**：✅ 安全，可随时重启
**失败处理**：修复环境问题后重新执行

---

### 阶段 1：数据补全（预计 2-4 小时）

**目标**：将 cache DB 从 2025-12-26 补全至 2026-04-21

#### 1.1 拉取 daily_data（预计 1.5-2.5 小时）

**背景**：需要为 ~3,000 只股票拉取 2025-12-27 至 2026-04-21（约 80 个交易日）的日线数据。

**执行方式**：

```bash
# 方案A：使用 DataManager 批量拉取（推荐，自动缓存）
python3 scripts/batch/fetch_missing_daily_data.py \
    --start-date 20251227 \
    --end-date 20260421 \
    --batch-size 100
```

> **注**：当前项目暂无 `fetch_missing_daily_data.py` 脚本，需先创建或使用 DataManager 的批量拉取能力。

**替代方案**（如果无批量脚本）：

```bash
# 使用现有 cache 补全机制
python3 -c "
from src.data.data_manager import DataManager
from src.utils.logger import log
import pandas as pd

dm = DataManager()

# 获取所有股票列表
df_stocks = dm.get_stock_list()
stock_codes = df_stocks['ts_code'].tolist()

# 分批拉取（每批50只，避免超时）
batch_size = 50
total = len(stock_codes)
start_date = '20251227'
end_date = '20260421'

for i in range(0, total, batch_size):
    batch = stock_codes[i:i+batch_size]
    log.info(f'拉取批次 {i//batch_size + 1}/{(total-1)//batch_size + 1}: {len(batch)} 只股票')
    for code in batch:
        try:
            dm.get_daily_data(code, start_date, end_date)
        except Exception as e:
            log.warning(f'{code} 拉取失败: {e}')

    # 每批完成后保存缓存（避免内存溢出）
    if dm.cache:
        dm.cache.commit()

log.success('数据补全完成')
"
```

**检查点**：
- 每 10 分钟检查一次 cache DB 最新日期：`SELECT MAX(trade_date) FROM daily_data`
- 预期进度：~50 只股票/分钟（含限流等待）

**失败处理**：
- Tushare 限流 → 自动等待后重试（DataManager 内置 rate limiter）
- 网络中断 → 记录已完成的股票，断点续传
- 个别股票缺失 → 记录到日志，跳过不影响整体

#### 1.2 拉取 daily_basic 和 stk_factor（预计 0.5-1 小时）

**执行方式**：同上，替换表名为 `daily_basic` 和 `stk_factor`

```bash
python3 -c "
from src.data.data_manager import DataManager
dm = DataManager()
# 批量拉取 daily_basic 和 stk_factor...
"
```

**检查点**：
- `daily_basic` 最新日期 ≥ 2026-04-21
- `stk_factor` 最新日期 ≥ 2026-04-21

#### 1.3 验证数据完整性（10分钟）

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data/cache/quant_data.db')
cursor = conn.cursor()

for table in ['daily_data', 'daily_basic', 'stk_factor']:
    cursor.execute(f'SELECT MAX(trade_date), COUNT(*) FROM {table} WHERE trade_date >= \"20251227\"')
    max_date, count = cursor.fetchone()
    print(f'{table}: 最新={max_date}, 新增记录数={count}')

conn.close()
"
```

**预期结果**：
- daily_data 新增 ≥ 200,000 条
- daily_basic 新增 ≥ 200,000 条
- stk_factor 新增 ≥ 200,000 条

**可中断性**：✅ 支持断点续传（cache 会自动跳过已存在的数据）
**阶段产出**：cache DB 数据覆盖至 2026-04-21

---

### 阶段 2：样本生成（预计 1-2 小时）

**目标**：基于补全后的 cache 数据，生成 2025-12-27 至 2026-04-21 的新正负样本

#### 2.1 正样本扫描（预计 30-60 分钟）

**背景**：正样本基于 T1 条件（未来1周有显著上涨），需要扫描 cache 中 2025-12-27 至 2026-04-14 的数据（因为需要预留1周看未来涨幅）。

**执行方式**：

```bash
python3 scripts/screen_positive_samples.py \
    --start-date 20251227 \
    --end-date 20260414 \
    --output data/training/samples/positive_samples_new.csv
```

> **注**：需确认 `screen_positive_samples.py` 的参数接口。若不支持日期范围，需修改或创建新脚本。

**检查点**：
- 新正样本数量预期：~200-400 个（4个月 × 每周一批，考虑春节休市）
- 与现有正样本合并后总数预期：~3,450-3,650 个

#### 2.2 负样本生成（预计 20-40 分钟）

**执行方式**：

```bash
python3 scripts/screen_negative_samples_v2.py \
    --start-date 20251227 \
    --end-date 20260421 \
    --output data/training/samples/negative_samples_new.csv
```

**检查点**：
- 新负样本数量预期：~400-800 个
- 正负比例保持 ~1:2

#### 2.3 样本合并与去重（10分钟）

```bash
python3 -c "
import pandas as pd

# 合并正样本
df_pos_old = pd.read_csv('data/training/samples/positive_samples.csv')
df_pos_new = pd.read_csv('data/training/samples/positive_samples_new.csv')
df_pos = pd.concat([df_pos_old, df_pos_new]).drop_duplicates(subset=['ts_code', 't1_date'])
df_pos.to_csv('data/training/samples/positive_samples_v280.csv', index=False)
print(f'正样本合并: {len(df_pos_old)} + {len(df_pos_new)} -> {len(df_pos)} (去重后)')

# 合并负样本
df_neg_old = pd.read_csv('data/training/samples/negative_samples_v2.csv')
df_neg_new = pd.read_csv('data/training/samples/negative_samples_new.csv')
df_neg = pd.concat([df_neg_old, df_neg_new]).drop_duplicates(subset=['ts_code', 't1_date'])
df_neg.to_csv('data/training/samples/negative_samples_v280.csv', index=False)
print(f'负样本合并: {len(df_neg_old)} + {len(df_neg_new)} -> {len(df_neg)} (去重后)')
"
```

**可中断性**：⚠️ 部分可中断。样本扫描中断后需重新开始，但合并步骤可安全重启
**阶段产出**：
- `positive_samples_v280.csv`
- `negative_samples_v280.csv`

---

### 阶段 3：特征工程（预计 1-2 小时）

**目标**：为新增样本提取 v6 版本特征（165 列）

#### 3.1 正样本特征提取（预计 30-60 分钟）

**执行方式**：

```bash
# 检查现有 v6 特征提取脚本
python3 scripts/enrich_features_v6.py \
    --input data/training/samples/positive_samples_v280.csv \
    --output data/training/processed/feature_data_34d_v6_positive.csv
```

> **注**：`enrich_features_v6.py` 可能不支持直接输入样本文件。实际执行时可能需要通过 `add_advanced_factors_v3.py` 或其他特征提取脚本。

**更可能的方式**（基于 v250_training_status.md）：

```bash
python3 scripts/add_advanced_factors_v3.py \
    --input data/training/processed/feature_data_34d_v3.csv \
    --output data/training/processed/feature_data_34d_v6_positive.csv
```

**检查点**：
- 特征列数 = 165（与 v6 一致）
- 无缺失值、无无穷值
- 特征分布与历史数据一致（通过 `evaluate_training_data_quality.py` 检查）

#### 3.2 负样本特征提取（预计 20-40 分钟）

同上，替换输入为负样本。

#### 3.3 硬负样本生成（可选，20分钟）

如果历史训练使用了硬负样本，需同步更新。

#### 3.4 数据质量评估（10分钟）

```bash
python3 scripts/evaluate_training_data_quality.py \
    --positive data/training/processed/feature_data_34d_v6_positive.csv \
    --negative data/training/processed/feature_data_34d_v6_negative.csv \
    --output data/training/quality_reports/v280_quality_report.md
```

**可中断性**：✅ 支持断点续传（`add_advanced_factors_v3.py` 内置 checkpoint 机制）
**阶段产出**：
- `feature_data_34d_v6_positive.csv`
- `feature_data_34d_v6_negative.csv`
- 质量评估报告

---

### 阶段 4：模型训练与超参数调优（预计 30-60 分钟）

**目标**：基于新数据训练 v2.8.0 模型，并进行系统性超参数搜索

#### 4.1 超参数搜索空间设计

当前 v2.7.0 超参数（6组手动搜索后的结果）：

| 参数 | 当前值 | 搜索范围 | 步长 |
|------|--------|---------|------|
| `max_depth` | 6 | 4-8 | 1 |
| `learning_rate` | 0.1 | 0.05-0.2 | 0.05 |
| `subsample` | 0.9 | 0.7-1.0 | 0.1 |
| `colsample_bytree` | 0.8 | 0.6-1.0 | 0.1 |
| `reg_alpha` | 0.1 | 0-0.5 | 0.1 |
| `reg_lambda` | 0.5 | 0.1-1.0 | 0.3 |
| `scale_pos_weight` | 1.5（硬编码） | 动态计算: `neg/pos * [0.8, 1.0, 1.2, 1.5]` | - |

**搜索策略**：

**方案 A：网格搜索（推荐，确定性高）**
- 总组合数：5 × 4 × 4 × 5 × 6 × 4 × 4 = **38,400 组**（太多，不可行）
- 精简网格（聚焦关键参数）：
  - max_depth: [5, 6, 7]
  - learning_rate: [0.05, 0.1, 0.15]
  - subsample: [0.8, 0.9, 1.0]
  - colsample_bytree: [0.7, 0.8, 0.9]
  - scale_pos_weight: [动态, 1.0, 1.5, 2.0]
  - 总组合：**3 × 3 × 3 × 3 × 4 = 324 组**（仍太多）

**方案 B：随机搜索 + 时间序列 CV（推荐，平衡效率与覆盖）**
- 随机采样 30-50 组组合
- 每组使用 3-fold 时间序列交叉验证
- 评估指标：AUC + F1 + Precision（多目标排序）
- 预计时间：30-50 组 × 3-fold × 30秒 ≈ **45-75 分钟**

**方案 C：Optuna 贝叶斯优化（最智能，但需额外依赖）**
- 自动探索高价值区域
- 30-50 轮迭代
- 需安装 `optuna`：若未安装则回退到方案 B

**推荐采用方案 B（随机搜索 + 时序 CV）**，原因：
1. 无需额外依赖
2. 时间可控（45-75 分钟）
3. 覆盖度足够（30-50 组随机组合在 6 维空间中分布较均匀）

#### 4.2 超参数搜索脚本

新建 `scripts/hyperparameter_search_v280.py`：

```python
# 伪代码
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

param_distributions = {
    "max_depth": [5, 6, 7, 8],
    "learning_rate": [0.05, 0.08, 0.1, 0.15, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.05, 0.1, 0.3, 0.5],
    "reg_lambda": [0.1, 0.3, 0.5, 1.0],
    "scale_pos_weight": ["auto", 1.0, 1.5, 2.0],
}

# 随机采样 40 组
n_trials = 40
best_score = 0
best_params = None

tscv = TimeSeriesSplit(n_splits=3)

for trial in range(n_trials):
    params = {k: np.random.choice(v) for k, v in param_distributions.items()}
    if params["scale_pos_weight"] == "auto":
        params["scale_pos_weight"] = len(neg) / len(pos)

    scores = []
    for train_idx, val_idx in tscv.split(X):
        # 训练并评估...
        score = evaluate(X_train, y_train, X_val, y_val)
        scores.append(score)

    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_params = params
        print(f"Trial {trial}: New best! AUC={avg_score:.4f}, params={params}")
```

#### 4.3 最终模型训练（10-15 分钟）

使用搜索得到的最优参数，训练最终模型：

```bash
python3 scripts/train_v280_model.py \
    --positive data/training/processed/feature_data_34d_v6_positive.csv \
    --negative data/training/processed/feature_data_34d_v6_negative.csv \
    --hard-negative data/training/features/hard_negative_feature_data_34d_v6.csv \
    --output data/models/breakout_launch_scorer/versions/v2.8.0/
```

**模型架构**：XGBoost + LightGBM + CatBoost ensemble（与 v2.7.0 保持一致）

**检查点**：
- 训练集 AUC ≥ 0.95
- 测试集 AUC ≥ 0.85（WFV 基准）
- 校准后 Brier Score ≤ 0.15

#### 4.4 特征重要性分析与筛选（10分钟）

```bash
python3 scripts/analyze_feature_importance.py \
    --model data/models/breakout_launch_scorer/versions/v2.8.0/ \
    --output data/training/quality_reports/v280_feature_importance.md
```

**可中断性**：✅ 超参数搜索可随时中断并恢复（记录已评估的组合）
**阶段产出**：
- `v2.8.0` 模型文件（XGBoost + LightGBM + CatBoost + 校准器）
- 超参数搜索报告
- 特征重要性分析报告

---

### 阶段 5：评估与验证（预计 20-30 分钟）

**目标**：全面评估新模型，与 v2.7.0 对比

#### 5.1 WFV（Walk Forward Validation）

```bash
python3 scripts/evaluate_wfv.py \
    --model data/models/breakout_launch_scorer/versions/v2.8.0/ \
    --output data/training/quality_reports/v280_wfv_report.md
```

**检查点**：
- 最新窗口 AUC ≥ 0.85（vs v2.7.0 的 0.791）
- AUC 衰减趋势减缓

#### 5.2 纯模型 Top10 胜率评估

```bash
python3 scripts/evaluate_v270_top10_winrate.py \
    --model-version v2.8.0 \
    --start-date 20260105 \
    --end-date 20260421 \
    --top-n 10
```

**检查点**：
- Top10 次日胜率 ≥ 48%（vs v2.7.0 的 50.21%，允许小幅下降）

#### 5.3 策略回测对比

```bash
# 使用新模型生成预测
python3 scripts/predict_v280.py --date 20260105 --end-date 20260421

# 运行回测
python3 scripts/backtest_v232_v270_complementary.py \
    --start-date 20260105 \
    --end-date 20260421 \
    --stop-loss-pct 4.0 \
    --stop-loss-mode close \
    --ma-window 5 \
    --ma-consecutive-days 2
```

**检查点**：
- 策略胜率 ≥ 32%（vs 当前 32.98%）
- 总收益 ≥ +5%（vs 当前 +5.54%）

#### 5.4 生成对比报告

```bash
python3 scripts/compare_model_versions.py \
    --old v2.7.0 \
    --new v2.8.0 \
    --output docs/reports/v280_vs_v270_comparison.md
```

**可中断性**：✅ 每个评估步骤独立，可单独重跑
**阶段产出**：
- WFV 报告
- Top10 胜率报告
- 策略回测报告
- 版本对比报告

---

### 阶段 6：部署与归档（10分钟）

| 步骤 | 操作 |
|------|------|
| 6.1 | 更新 `config/models.yaml` 指向 v2.8.0 |
| 6.2 | 更新 `data/models/current.json` |
| 6.3 | 生成 `CHANGELOG.md` 条目 |
| 6.4 | Git 提交：`feat: retrain v2.8.0 with data through 2026-04-21` |
| 6.5 | 清理临时文件（checkpoint、中间 CSV） |

---

## 四、超参数调优专项说明

### 4.1 调优重点

基于 v2.7.0 的问题诊断，本次调优聚焦以下 4 个维度：

1. **scale_pos_weight 动态化**（最高优先级）
   - 当前硬编码 1.5，未考虑实际正负比例变化
   - 新数据正负比例可能不同（新增样本可能改变比例）
   - 搜索范围：`auto`（动态计算）、0.8、1.0、1.5、2.0

2. **正则化参数精细化**（高优先级）
   - v2.7.0 的 WFV 显示 AUC 衰减，可能是过拟合信号
   - 增大 `reg_alpha` 和 `reg_lambda` 可能提升泛化能力
   - 搜索范围：reg_alpha [0, 0.05, 0.1, 0.3, 0.5], reg_lambda [0.1, 0.3, 0.5, 1.0]

3. **学习率与树的深度平衡**（中优先级）
   - v2.7.0 使用 lr=0.1, max_depth=6
   - 尝试 lr=0.05 + max_depth=7（更多树，每棵树更浅）
   - 尝试 lr=0.15 + max_depth=5（更少树，每棵树更深）

4. **采样比例优化**（中优先级）
   - subsample 和 colsample_bytree 影响模型的泛化能力
   - 当前 0.9 和 0.8，尝试更低值增加随机性

### 4.2 评估指标权重

超参数搜索的排序规则（多目标优化）：

```
综合评分 = 0.4 × 测试集AUC + 0.3 × 测试集F1 + 0.2 × 校准后BrierScore(反向) + 0.1 × 训练/测试AUC差距(反向)
```

- **AUC 权重最高**：模型区分能力是核心
- **F1 次之**：避免 Precision/Recall 严重失衡
- **Brier Score**：评估概率校准质量
- **AUC 差距**：惩罚过拟合（训练AUC >> 测试AUC）

### 4.3 时序 CV 设计

为避免未来函数，采用时间序列交叉验证：

```
Fold 1: 训练[2000-2015] → 验证[2016-2018]
Fold 2: 训练[2000-2018] → 验证[2019-2021]
Fold 3: 训练[2000-2021] → 验证[2022-2024]
```

---

## 五、风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| Tushare 限流导致数据补全超时 | 中 | 阶段1延迟 | 分多天执行，每晚补一批；或降低 batch_size |
| 新增样本质量差（市场风格突变） | 中 | 模型性能不升反降 | 严格质量评估，若 AUC < 0.80 则回退到 v2.7.0 |
| 超参数搜索过拟合 | 低 | 测试集表现虚高 | 时序 CV + AUC 差距惩罚 + WFV 验证 |
| 特征工程脚本不兼容新数据 | 低 | 阶段3失败 | 提前测试特征提取脚本，准备 fallback 方案 |
| 磁盘空间不足 | 低 | 阶段中断 | 阶段0已检查，106GB 足够 |

---

## 六、执行确认清单

请在确认执行前勾选以下事项：

- [ ] **Tushare 积分 ≥ 4000**（当前约 5120，数据补全预计消耗 500-1000）
- [ ] **今晚/明天有 4-8 小时不间断运行时间**
- [ ] **接受风险：新模型可能性能不如 v2.7.0**（若发生则回退）
- [ ] **确认超参数搜索方案**（默认方案 B：随机搜索40组 + 时序CV）
- [ ] **是否需要保留 v2.7.0 并行运行**（建议保留2周，对比实盘表现）

---

## 七、里程碑时间线

```
T+0h  ~ T+0.5h    阶段0: 环境检查与备份
T+0.5h~ T+4.5h    阶段1: 数据补全（可夜间挂机）
T+4.5h~ T+6.5h    阶段2+3: 样本生成 + 特征工程
T+6.5h~ T+7.5h    阶段4: 超参数搜索 + 模型训练
T+7.5h~ T+8.0h    阶段5+6: 评估验证 + 部署归档
```

**最快路径**：若数据补全通过夜间挂机完成，白天只需执行阶段 2-6（约 3-4 小时）。

---

*计划撰写完成，等待用户确认后执行。*
