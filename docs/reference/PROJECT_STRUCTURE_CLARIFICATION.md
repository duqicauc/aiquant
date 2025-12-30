# 项目结构说明

## 📁 目录职责划分

### ✅ 正确的结构

```
aiquant/
├── src/              # 核心源代码（39个Python文件）
│   ├── data/        # 数据管理模块
│   ├── strategy/    # 策略模块
│   ├── models/      # 模型模块
│   ├── utils/       # 工具函数
│   └── ...
│
├── scripts/          # 可执行脚本（27个Python脚本）
│   ├── score_current_stocks.py      # 导入 src 模块
│   ├── train_xgboost_timeseries.py  # 导入 src 模块
│   └── ...
│
├── data/             # 数据文件（0个Python文件，只有数据）
│   ├── training/    # 训练数据（CSV、JSON）
│   ├── prediction/  # 预测数据（CSV、JSON、TXT）
│   └── cache/       # 缓存数据（DB）
│
└── tests/            # 测试代码（待补充）
```

## 🔍 详细说明

### 1. `src/` 目录 - 核心源代码

**作用**: 存放可复用的核心模块，被 `scripts/` 中的脚本导入使用

**包含模块**:
- `src/data/` - 数据管理（DataManager、Fetcher、Cache等）
- `src/strategy/` - 策略模块（筛选器、财务过滤等）
- `src/models/` - 模型相关（评估、预测等）
- `src/utils/` - 工具函数（日志、日期、限流等）
- `src/analysis/` - 分析模块（市场分析、健康检查等）
- `src/backtest/` - 回测模块
- `src/visualization/` - 可视化模块

**使用方式**:
```python
# scripts/score_current_stocks.py
from src.data.data_manager import DataManager
from src.utils.logger import log
```

**文件数量**: 39个Python文件

### 2. `scripts/` 目录 - 可执行脚本

**作用**: 存放可执行的入口脚本，导入 `src/` 模块完成具体任务

**脚本类型**:
- **训练脚本**: `prepare_positive_samples.py`, `train_xgboost_timeseries.py`
- **预测脚本**: `score_current_stocks.py`, `analyze_prediction_accuracy.py`
- **工具脚本**: `check_sample_quality.py`, `visualize_sample_comparison.py`

**特点**:
- 可以直接运行: `python scripts/score_current_stocks.py`
- 导入src模块: `from src.data.data_manager import DataManager`
- 处理具体业务逻辑

**文件数量**: 27个Python脚本

### 3. `data/` 目录 - 数据文件

**作用**: 只存放数据文件，不包含任何代码

**数据类型**:
- **训练数据**: CSV样本、JSON统计、模型文件
- **预测数据**: CSV结果、TXT报告、JSON元数据
- **缓存数据**: SQLite数据库

**文件类型**: CSV、JSON、TXT、DB、PNG、HTML

**代码文件**: 0个（正确！）

### 4. `tests/` 目录 - 测试代码

**当前状态**: 基本为空，只有 `__init__.py`

**应该包含**:
- 单元测试（测试src模块）
- 集成测试（测试scripts脚本）
- 数据质量测试

**待补充**: 需要添加测试代码

## 📊 代码分布统计

| 目录 | Python文件数 | 说明 |
|------|-------------|------|
| `src/` | 39 | 核心源代码模块 |
| `scripts/` | 27 | 可执行脚本 |
| `data/` | 0 | 只有数据文件 ✅ |
| `tests/` | 1 | 待补充测试代码 |

## ✅ 结论

**当前结构是合理的**：

1. ✅ `data/` 目录下没有代码，只有数据文件
2. ✅ `src/` 目录包含核心源代码，被scripts导入使用
3. ✅ `scripts/` 目录包含可执行脚本，是项目入口
4. ⚠️ `tests/` 目录基本为空，需要补充测试代码

## 🔄 代码调用关系

```
scripts/score_current_stocks.py
    ↓ 导入
src/data/data_manager.py
src/utils/logger.py
    ↓ 使用
data/training/models/  (读取模型)
data/cache/quant_data.db  (读取缓存)
    ↓ 输出
data/prediction/results/  (保存结果)
```

## 💡 建议

1. **保持当前结构** - 代码和数据分离是正确的
2. **补充测试代码** - 在 `tests/` 目录添加单元测试和集成测试
3. **文档说明** - 在README中明确说明目录职责

