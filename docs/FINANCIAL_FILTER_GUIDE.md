# 财务筛选功能使用指南

## 📊 功能概述

在模型预测的基础上，增加**基本面财务筛选**，过滤掉财务状况不佳的股票，降低投资风险。

### 筛选逻辑

```
模型预测 Top 150 → 财务筛选 → 最终 Top 50 推荐
```

### 筛选条件（财务退市指标）

1. **营收 > 3亿元** - 确保公司有一定规模
2. **连续3年净利润 > 0** - 确保持续盈利能力
3. **净资产 > 0** - 确保资不抵债

## 🎯 使用方式

### 方式1: 自动筛选（推荐）

**配置文件** `config/settings.yaml`:

```yaml
prediction:
  financial_filter:
    enabled: true              # 启用财务筛选
    revenue_threshold: 3.0     # 营收阈值（亿元）
    profit_years: 3            # 连续盈利年数
    initial_candidates: 150    # 初选候选数量
```

**运行预测**:
```bash
python scripts/score_current_stocks.py
# 或
python scripts/weekly_prediction.py
```

**自动执行流程**:
1. 模型预测所有股票
2. 按概率排序，取Top 150
3. 对Top 150进行财务筛选
4. 输出通过筛选的Top 50

### 方式2: 手动调用

```python
from src.data.data_manager import DataManager
from src.strategy.screening.financial_filter import FinancialFilter
import pandas as pd

# 初始化
dm = DataManager()
financial_filter = FinancialFilter(dm)

# 准备股票列表
df_stocks = pd.DataFrame({
    '股票代码': ['000001.SZ', '600000.SH', '000002.SZ'],
    '股票名称': ['平安银行', '浦发银行', '万科A']
})

# 执行筛选
df_filtered = financial_filter.filter_stocks(
    df_stocks,
    revenue_threshold=3.0,  # 营收 > 3亿
    profit_years=3          # 连续3年盈利
)

print(df_filtered)
```

## 📋 筛选结果说明

### 输出列

原有列 + 新增列：
- `财务状况`: '良好' 或筛选原因
- `营收(亿)`: 最新年报营收（亿元）
- `连续盈利年数`: 连续盈利的年数
- `净资产(亿)`: 净资产（亿元）

### 示例输出

```
==================================================================================
🔍 开始财务指标筛选
==================================================================================

筛选条件：
  1. 营收 > 3.0亿元
  2. 连续3年净利润 > 0
  3. 净资产 > 0

进度: 10/150 (6.7%)
  ✓ 平安银行 通过筛选
  ✓ 浦发银行 通过筛选
  ✗ XX公司 未通过: 营收2.1亿 <= 3.0亿
  ✗ YY公司 未通过: 连续盈利2年 < 3年
  ✓ 万科A 通过筛选
  ...

==================================================================================
📊 筛选结果
==================================================================================
原始数量: 150
通过筛选: 68
剔除数量: 82
通过率: 45.3%
```

## ⚙️ 配置参数说明

### enabled (boolean)
- **说明**: 是否启用财务筛选
- **默认值**: `true`
- **建议**: 
  - 启用：降低风险，推荐
  - 禁用：不进行财务筛选，仅用模型结果

### revenue_threshold (float)
- **说明**: 营收阈值（亿元）
- **默认值**: `3.0`
- **建议**:
  - 大盘股：5.0 - 10.0
  - 中小盘：3.0
  - 小盘股：1.0 - 2.0

### profit_years (integer)
- **说明**: 连续盈利年数
- **默认值**: `3`
- **建议**:
  - 保守：3 - 5年
  - 中性：2 - 3年
  - 激进：1 - 2年

### initial_candidates (integer)
- **说明**: 初选候选数量（筛选前）
- **默认值**: `150`
- **建议**:
  - Top 50需要：至少100-150
  - Top 30需要：至少60-90
  - 根据历史通过率调整

## 📊 数据来源

### Tushare API接口

1. **利润表** (`income`)
   - `revenue`: 营业收入
   - `n_income`: 净利润

2. **资产负债表** (`balancesheet`)
   - `total_assets`: 总资产
   - `total_liab`: 总负债
   - 净资产 = 总资产 - 总负债

### 数据要求

- 使用年报数据（报告期以1231结尾）
- 至少需要3年年报数据
- 数据自动按报告期排序

## 🎯 实战案例

### 案例1: 标准配置

```yaml
financial_filter:
  enabled: true
  revenue_threshold: 3.0
  profit_years: 3
  initial_candidates: 150
```

**效果**: 
- 150只候选 → 68只通过 → Top 50推荐
- 通过率: 45%
- 风险: 低

### 案例2: 放宽条件

```yaml
financial_filter:
  enabled: true
  revenue_threshold: 1.0      # 放宽到1亿
  profit_years: 2             # 放宽到2年
  initial_candidates: 100
```

**效果**:
- 100只候选 → 75只通过 → Top 50推荐
- 通过率: 75%
- 风险: 中

### 案例3: 严格筛选

```yaml
financial_filter:
  enabled: true
  revenue_threshold: 10.0     # 提高到10亿
  profit_years: 5             # 提高到5年
  initial_candidates: 200
```

**效果**:
- 200只候选 → 35只通过 → Top 35推荐
- 通过率: 17.5%
- 风险: 极低（但可能错过潜力股）

## ⚠️ 注意事项

### 1. API调用限制

**问题**: 财务数据查询会消耗Tushare积分

**解决**:
- 已集成限流器，自动控制调用频率
- 5000积分：60次/分钟
- 2000积分：20次/分钟

**建议**:
- 减少`initial_candidates`数量
- 使用缓存（未来功能）

### 2. 数据缺失

**问题**: 部分股票可能无财务数据

**处理**:
- 自动跳过无数据的股票
- 记录到日志中
- 不影响其他股票筛选

### 3. 通过率过低

**问题**: 筛选后剩余股票少于Top 50

**解决方案**:
```yaml
# 方案1: 增加候选数量
initial_candidates: 200

# 方案2: 放宽条件
revenue_threshold: 2.0
profit_years: 2

# 方案3: 暂时禁用
enabled: false
```

### 4. 时效性

**问题**: 年报数据更新频率

**说明**:
- 年报：每年4月30日前发布
- 半年报：每年8月31日前发布
- 季报：较为及时

**建议**: 使用最新年报数据，定期更新

## 📈 效果评估

### 预期效果

**风险降低**:
- ✅ 避免财务造假风险
- ✅ 避免退市风险
- ✅ 避免业绩暴雷

**收益影响**:
- 短期：可能错过部分高风险高收益股票
- 中长期：提高稳定性，降低回撤

### 历史数据（示例）

| 配置 | 通过率 | 1月收益 | 最大回撤 | 夏普比率 |
|------|--------|---------|----------|----------|
| 不筛选 | 100% | +8.5% | -18.2% | 1.20 |
| 标准筛选 | 45% | +7.2% | -12.5% | 1.45 |
| 严格筛选 | 18% | +6.1% | -8.3% | 1.68 |

## 🔧 调试技巧

### 查看筛选详情

```bash
# 运行评分脚本时会自动显示
python scripts/score_current_stocks.py

# 查看日志
tail -f logs/aiquant.log | grep "财务"
```

### 测试单只股票

```python
from src.data.data_manager import DataManager
from src.strategy.screening.financial_filter import FinancialFilter

dm = DataManager()
filter_obj = FinancialFilter(dm)

# 测试
result = filter_obj.check_financial_indicators('000001.SZ')
print(result)
```

### 分析筛选结果

```python
import pandas as pd

# 读取筛选后的结果
df = pd.read_csv('data/predictions/20251224/top_50_stocks_*.csv')

# 分析财务指标分布
print(df['营收(亿)'].describe())
print(df['连续盈利年数'].describe())
print(df['净资产(亿)'].describe())
```

## 💡 最佳实践

1. **默认启用**: 降低投资风险
2. **定期调整**: 根据市场环境调整阈值
3. **结合使用**: 技术面（模型）+ 基本面（财务）
4. **持续跟踪**: 关注推荐股票的后续表现
5. **动态优化**: 根据回测结果优化参数

## 📚 相关文档

- `config/settings.yaml` - 配置文件
- `src/strategy/screening/financial_filter.py` - 源代码
- `scripts/score_current_stocks.py` - 集成应用

## ❓ 常见问题

**Q: 为什么有的股票没有财务数据？**
A: 可能是新上市、退市、或数据源暂缺，系统会自动跳过。

**Q: 筛选会影响预测速度吗？**
A: 会略有影响，每只股票需要2次API调用，150只约需3-5分钟。

**Q: 能否只用财务筛选，不用模型？**
A: 可以，但建议结合使用。单独财务筛选无法预测短期走势。

**Q: 如何验证筛选效果？**
A: 使用`review_predictions.py`回顾脚本，对比有无筛选的效果差异。

## ✨ 总结

财务筛选功能：
- ✅ 基于财务退市指标
- ✅ 降低投资风险
- ✅ 提高推荐质量
- ✅ 灵活可配置
- ✅ 自动集成到预测流程

**开始使用**:
```bash
# 1. 确认配置已启用
cat config/settings.yaml | grep financial_filter -A 5

# 2. 运行预测（自动应用筛选）
python scripts/weekly_prediction.py

# 3. 查看结果
cat data/predictions/$(date +%Y%m%d)/prediction_report_*.txt
```

祝投资顺利！📈

