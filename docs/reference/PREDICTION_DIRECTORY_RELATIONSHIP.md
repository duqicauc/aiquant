# 预测目录关系说明

## 📁 目录结构

```
data/prediction/
├── results/          # 预测结果（原始输出）
├── metadata/         # 预测元数据（用于准确率分析）
├── analysis/         # 准确率分析结果
└── history/          # 历史预测归档
```

## 🔄 目录关系和数据流

### 1. **results/** - 预测结果（原始输出）

**作用**: 每次运行预测脚本时，直接保存的原始结果文件

**文件类型**:
- `stock_scores_YYYYMMDD_HHMMSS.csv` - 所有股票的完整评分
- `top_50_stocks_YYYYMMDD_HHMMSS.csv` - Top 50 推荐股票
- `prediction_report_YYYYMMDD_HHMMSS.txt` - 预测报告文本

**特点**:
- ✅ 包含时间戳（精确到秒），支持同一天多次预测
- ✅ 每次预测都会生成新文件，不会覆盖
- ✅ 是预测脚本的直接输出

**生成时机**: `scripts/score_current_stocks.py` 执行时

**示例**:
```
results/
├── stock_scores_20251224_213533.csv
├── top_50_stocks_20251224_213533.csv
└── prediction_report_20251224_213533.txt
```

---

### 2. **metadata/** - 预测元数据

**作用**: 保存预测的关键信息，用于后续准确率分析

**文件类型**:
- `prediction_metadata_YYYYMMDD_HHMMSS.json` - 预测元数据

**内容**:
```json
{
  "prediction_date": "20251224",
  "prediction_timestamp": "2025-12-24 21:35:33",
  "is_backtest": false,
  "model_path": "xgboost_timeseries_v2_20251220_120000.json",
  "total_scored": 5000,
  "top_n": 50,
  "top_stocks": [
    {
      "rank": 1,
      "code": "000510.SZ",
      "name": "新金路",
      "probability": 0.9665,
      "price": 10.93
    },
    ...
  ],
  "scores_file": "data/prediction/results/stock_scores_20251224_213533.csv",
  "top_file": "data/prediction/results/top_50_stocks_20251224_213533.csv",
  "report_file": "data/prediction/results/prediction_report_20251224_213533.txt"
}
```

**特点**:
- ✅ 包含推荐股票列表（用于后续验证）
- ✅ 包含文件路径引用（指向 results/ 中的文件）
- ✅ 轻量级，只保存关键信息
- ✅ 用于 `analyze_prediction_accuracy.py` 读取

**生成时机**: `scripts/score_current_stocks.py` 执行时（与 results/ 同时生成）

**使用场景**: 
- `scripts/analyze_prediction_accuracy.py` 读取 metadata，获取推荐股票列表
- 然后获取这些股票的实际表现，计算准确率

---

### 3. **analysis/** - 准确率分析结果

**作用**: 基于 metadata 分析推荐股票的实际表现，生成准确率报告

**文件类型**:
- `accuracy_YYYYMMDD_4w.csv` - 详细分析结果（每只股票的表现）
- `accuracy_report_YYYYMMDD_4w.txt` - 准确率分析报告
- `accuracy_YYYYMMDD_4w.json` - 分析结果元数据

**内容示例**:
```csv
股票代码,股票名称,预测概率,预测价格,最大涨幅,最终涨幅,是否牛股
000510.SZ,新金路,0.9665,10.93,45.2%,38.5%,否
000036.SZ,华联控股,0.9360,6.74,52.3%,48.1%,是
...
```

**特点**:
- ✅ 基于 metadata 中的推荐股票列表
- ✅ 获取实际市场数据，计算收益率
- ✅ 判断是否达到"牛股"标准（50%涨幅）
- ✅ 生成准确率统计（命中率、平均收益率等）

**生成时机**: `scripts/analyze_prediction_accuracy.py` 执行时

**数据来源**: 
- 输入: `metadata/` 中的预测元数据
- 数据: 从 API 获取实际股价数据
- 输出: `analysis/` 中的分析结果

---

### 4. **history/** - 历史预测归档

**作用**: 按日期组织的历史预测归档，便于查看和管理

**目录结构**:
```
history/
├── 20251224/
│   ├── stock_scores_20251224_213533.csv
│   ├── top_50_stocks_20251224_213533.csv
│   └── prediction_report_20251224_213533.txt
├── 20251225/
│   └── ...
└── index.json          # 预测索引（记录每次预测的摘要）
```

**特点**:
- ✅ 按日期（YYYYMMDD）组织，每天一个目录
- ✅ 从 `results/` 复制文件过来（不删除原文件）
- ✅ 包含 `index.json` 记录每次预测的摘要信息
- ✅ 便于按日期查找历史预测

**生成时机**: `scripts/weekly_prediction.py` 执行时（通过 `organize_prediction_results` 函数）

**与 results/ 的关系**:
- `results/` 保留所有原始文件（带时间戳）
- `history/` 按日期归档，便于查看（通常只保留每天最新的）

---

## 📊 完整数据流

### 预测流程

```
1. 运行 score_current_stocks.py
   ↓
2. 生成 results/ 文件（原始结果）
   ├── stock_scores_*.csv
   ├── top_50_stocks_*.csv
   └── prediction_report_*.txt
   ↓
3. 同时生成 metadata/ 文件（元数据）
   └── prediction_metadata_*.json
   ↓
4. （可选）运行 weekly_prediction.py
   ↓
5. 整理到 history/ 目录（按日期归档）
   └── YYYYMMDD/
       └── （从 results/ 复制文件）
```

### 准确率分析流程

```
1. 运行 analyze_prediction_accuracy.py
   ↓
2. 读取 metadata/ 文件
   └── prediction_metadata_YYYYMMDD_*.json
   ↓
3. 获取推荐股票列表
   ↓
4. 从 API 获取实际股价数据
   ↓
5. 计算每只股票的实际表现
   ├── 最大涨幅
   ├── 最终涨幅
   └── 是否达到牛股标准（50%）
   ↓
6. 生成 analysis/ 文件（分析结果）
   ├── accuracy_YYYYMMDD_4w.csv
   ├── accuracy_report_YYYYMMDD_4w.txt
   └── accuracy_YYYYMMDD_4w.json
```

---

## 🔗 目录关系总结

| 目录 | 作用 | 数据来源 | 数据去向 | 特点 |
|------|------|---------|---------|------|
| **results/** | 原始预测结果 | `score_current_stocks.py` | `history/`（复制） | 带时间戳，不覆盖 |
| **metadata/** | 预测元数据 | `score_current_stocks.py` | `analyze_prediction_accuracy.py`（读取） | 轻量级，关键信息 |
| **analysis/** | 准确率分析 | `analyze_prediction_accuracy.py` | - | 基于 metadata 分析 |
| **history/** | 历史归档 | `results/`（复制） | - | 按日期组织，便于查看 |

---

## 💡 使用建议

### 日常预测
1. 运行 `score_current_stocks.py` → 生成 `results/` 和 `metadata/`
2. 查看 `results/prediction_report_*.txt` 获取预测报告

### 定期归档
1. 运行 `weekly_prediction.py` → 整理到 `history/` 目录
2. 查看 `history/index.json` 获取预测摘要

### 准确率分析
1. 等待一段时间（如4周）
2. 运行 `analyze_prediction_accuracy.py` → 生成 `analysis/` 文件
3. 查看 `analysis/accuracy_report_*.txt` 获取准确率报告

---

## ⚠️ 注意事项

1. **results/ 和 history/ 的关系**
   - `results/` 保留所有原始文件（带时间戳）
   - `history/` 是归档目录，通常只保留每天最新的
   - 两者可以共存，`history/` 是从 `results/` 复制过来的

2. **metadata/ 的重要性**
   - 必须保存，否则无法进行准确率分析
   - 包含推荐股票列表，是分析的基础

3. **analysis/ 的生成时机**
   - 需要等待一段时间（如4周）才能获取实际表现
   - 基于 metadata 中的推荐股票列表进行分析

4. **文件命名规则**
   - `results/` 和 `metadata/`: 使用时间戳（精确到秒）
   - `analysis/`: 使用日期 + 观察周期（如 `_4w`）
   - `history/`: 按日期（YYYYMMDD）组织目录

