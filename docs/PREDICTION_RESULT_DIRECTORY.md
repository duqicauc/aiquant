# 预测结果目录结构说明

## 📁 目录结构

```
data/
├── result/                    # 最新预测结果（当前活跃结果）
│   └── {model_name}/          # 按模型分类
│       ├── {model}_predictions_YYYYMMDD_HHMMSS.csv
│       └── {model}_prediction_report_YYYYMMDD_HHMMSS.txt
│
└── prediction/
    ├── history/               # 历史预测归档
    │   └── {model_name}/      # 按模型分类
    │       └── YYYYMMDD/      # 按日期归档
    │           ├── index.json
    │           └── {model}_predictions_*.csv
    │
    └── analysis/              # 预测效果分析
        ├── accuracy_YYYYMMDD_Nw.csv
        ├── accuracy_report_YYYYMMDD_Nw.txt
        └── accuracy_YYYYMMDD_Nw.json
```

## 🔄 目录说明

### 1. **data/result/** - 最新预测结果

**作用**: 存放当前最新的预测结果，每次预测都会更新

**特点**:
- ✅ 最新结果，随时可查看
- ✅ 按模型分类（如 `left_breakout`、`momentum` 等）
- ✅ 文件带时间戳，支持同一天多次预测
- ✅ 定期清理旧文件（默认保留7天）

**目录结构**:
```
data/result/
├── left_breakout/
│   ├── left_breakout_predictions_20251228_081953.csv
│   └── left_breakout_prediction_report_20251228_081953.txt
└── momentum/
    └── ...
```

**使用场景**:
- 查看最新预测结果
- 实时监控模型表现
- 快速获取推荐股票

---

### 2. **data/prediction/history/** - 历史预测归档

**作用**: 归档历史预测结果，便于回溯和分析

**特点**:
- ✅ 按模型和日期组织
- ✅ 包含索引文件（`index.json`）
- ✅ 永久保存，不自动清理
- ✅ 便于按日期查找历史预测

**目录结构**:
```
data/prediction/history/
└── left_breakout/
    ├── 20251225/
    │   ├── index.json
    │   ├── left_breakout_predictions_20251225_081121.csv
    │   └── left_breakout_prediction_report_20251225_081121.txt
    └── 20251228/
        ├── index.json
        └── ...
```

**归档方式**:
- 手动归档: `python scripts/archive_predictions.py --model left_breakout --date 20251225`
- 自动归档: `python scripts/archive_predictions.py --auto`

**索引文件格式** (`index.json`):
```json
{
  "model_name": "left_breakout",
  "prediction_date": "20251225",
  "archived_at": "2025-12-28 08:30:00",
  "last_updated": "2025-12-28 08:30:00",
  "files": [
    "left_breakout_predictions_20251225_081121.csv",
    "left_breakout_prediction_report_20251225_081121.txt"
  ]
}
```

---

### 3. **data/prediction/analysis/** - 预测效果分析

**作用**: 存放预测准确率分析结果

**特点**:
- ✅ 基于历史预测和实际表现计算
- ✅ 包含详细分析报告
- ✅ 支持不同观察周期（如4周、8周）

**文件类型**:
- `accuracy_YYYYMMDD_Nw.csv` - 详细分析结果（CSV格式）
- `accuracy_report_YYYYMMDD_Nw.txt` - 分析报告（文本格式）
- `accuracy_YYYYMMDD_Nw.json` - 分析元数据（JSON格式）

**生成方式**:
```bash
python scripts/analyze_prediction_accuracy.py --date 20251225 --weeks 4
```

---

## 📊 数据流

### 预测流程

```
1. 运行预测脚本
   python scripts/predict_left_breakout.py --date 20251225
   ↓
2. 生成最新结果
   data/result/left_breakout/
   ├── left_breakout_predictions_20251225_081121.csv
   └── left_breakout_prediction_report_20251225_081121.txt
   ↓
3. （可选）归档到历史目录
   python scripts/archive_predictions.py --model left_breakout --date 20251225
   ↓
4. 历史归档
   data/prediction/history/left_breakout/20251225/
```

### 分析流程

```
1. 等待一段时间（如4周）
   ↓
2. 运行准确率分析
   python scripts/analyze_prediction_accuracy.py --date 20251225 --weeks 4
   ↓
3. 生成分析结果
   data/prediction/analysis/
   ├── accuracy_20251225_4w.csv
   ├── accuracy_report_20251225_4w.txt
   └── accuracy_20251225_4w.json
```

---

## 🛠️ 工具脚本

### 归档脚本

```bash
# 归档指定模型的预测结果
python scripts/archive_predictions.py --model left_breakout --date 20251225

# 自动归档所有模型的最新结果
python scripts/archive_predictions.py --auto

# 清理7天前的旧文件
python scripts/archive_predictions.py --clean --keep-days 7

# 清理指定模型的旧文件
python scripts/archive_predictions.py --clean --model left_breakout --keep-days 7
```

---

## 🔗 目录关系总结

| 目录 | 作用 | 数据来源 | 数据去向 | 特点 |
|------|------|---------|---------|------|
| **result/** | 最新预测结果 | 预测脚本 | `history/`（归档） | 最新结果，定期清理 |
| **history/** | 历史归档 | `result/`（归档） | - | 永久保存，按日期组织 |
| **analysis/** | 效果分析 | `history/`（分析） | - | 基于历史预测分析 |

---

## 💡 使用建议

### 日常预测
1. 运行预测脚本 → 生成 `data/result/{model_name}/` 文件
2. 查看最新结果 → 直接查看 `data/result/` 目录

### 定期归档
1. 运行归档脚本 → 将结果移动到 `data/prediction/history/`
2. 清理旧文件 → 使用 `--clean` 选项清理 `data/result/` 中的旧文件

### 效果分析
1. 等待一段时间（如4周）
2. 运行分析脚本 → 生成 `data/prediction/analysis/` 文件
3. 查看分析报告 → 了解模型准确率

---

## ⚠️ 注意事项

1. **result/ 和 history/ 的关系**
   - `result/` 存放最新结果，定期清理
   - `history/` 是归档目录，永久保存
   - 归档时从 `result/` 复制到 `history/`（不删除原文件）

2. **文件命名规则**
   - `result/`: 使用完整时间戳（`YYYYMMDD_HHMMSS`）
   - `history/`: 按日期组织目录，文件保留原时间戳
   - `analysis/`: 使用日期 + 观察周期（如 `_4w`）

3. **清理策略**
   - `result/` 目录默认保留7天
   - `history/` 目录不自动清理
   - 使用 `--clean` 选项手动清理

4. **多模型支持**
   - 每个模型有独立的目录
   - 支持同时管理多个模型的预测结果
   - 归档和分析都支持按模型分类

