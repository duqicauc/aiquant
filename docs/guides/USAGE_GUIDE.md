# 正样本数据准备 - 使用指南 🚀

## 快速开始

### 步骤1: 环境准备

确保已安装依赖并配置好Tushare Token：

```bash
# 安装依赖
pip install -r requirements.txt

# 配置Token（在.env文件中）
TUSHARE_TOKEN=你的token
```

**⚠️ Tushare积分要求**：
- **基础功能**：0积分（免费）
- **完整功能**：2000积分（推荐）
- **高级功能**：5000积分（技术因子API，强烈推荐！）

💡 查看详情：[Tushare Pro功能说明](TUSHARE_PRO_FEATURES.md)

### 步骤2: 快速测试（推荐）

先运行测试脚本验证逻辑：

```bash
cd /Users/javaadu/Documents/GitHub/aiquant
python scripts/test_positive_samples.py
```

选择测试模式：
- `1`: 测试单只股票（贵州茅台）- 约1分钟
- `2`: 测试5只股票 - 约3-5分钟

### 步骤3: 运行完整流程

确认测试通过后，运行完整脚本：

```bash
python scripts/prepare_positive_samples.py
```

**⚠️ 注意**：
- 首次运行可能需要**几小时**（取决于日期范围和股票数量）
- 建议先修改脚本中的日期范围进行小规模测试
- 确保有足够的Tushare积分和API调用额度

---

## 配置说明

### 修改日期范围

编辑 `scripts/prepare_positive_samples.py`：

```python
# 测试模式（快速）
START_DATE = '20220101'  # 最近2年
END_DATE = None          # 到今天

# 完整模式（慢）
START_DATE = '20000101'  # 从2000年开始
END_DATE = None
```

### 修改回看天数

默认提取T1前34天数据，可修改：

```python
df_features = screener.extract_features(
    df_samples,
    lookback_days=34  # 改为其他值，如50、60等
)
```

---

## 输出文件

运行完成后，会生成以下文件：

### 1. 正样本列表
**文件**: `data/processed/positive_samples.csv`

| 字段 | 说明 | 示例 |
|-----|------|------|
| ts_code | 股票代码 | 600519.SH |
| name | 股票名称 | 贵州茅台 |
| t1_date | T1日期 | 20150601 |
| week1_open | 第1周开盘价 | 180.50 |
| week3_close | 第3周收盘价 | 285.60 |
| total_return | 总涨幅(%) | 58.23 |
| max_return | 最高涨幅(%) | 73.45 |

### 2. 特征数据集
**文件**: `data/processed/feature_data_34d.csv`

| 字段 | 说明 | 示例 |
|-----|------|------|
| sample_id | 样本ID | 0 |
| ts_code | 股票代码 | 600519.SH |
| name | 股票名称 | 贵州茅台 |
| trade_date | 交易日期 | 2015-05-01 |
| close | 收盘价 | 175.20 |
| pct_chg | 涨跌幅(%) | 2.35 |
| total_mv | 总市值(万元) | 3500000 |
| circ_mv | 流通市值(万元) | 2800000 |
| ma5 | 5日均线 | 172.5 |
| ma10 | 10日均线 | 168.3 |
| volume_ratio | 量比 | 1.25 |
| days_to_t1 | 距T1天数 | -34 |

### 3. 统计报告
**文件**: `data/processed/sample_statistics.json`

```json
{
  "generation_time": "2024-12-22 18:30:00",
  "date_range": "20220101 - today",
  "total_samples": 156,
  "unique_stocks": 148,
  "avg_total_return": 67.8,
  "avg_max_return": 89.3,
  "feature_records": 5304,
  "lookback_days": 34
}
```

---

## 性能优化建议

### 1. 分批处理

修改脚本，每次处理部分股票：

```python
# 获取股票列表
stock_list = self._get_valid_stock_list()

# 只处理前100只
stock_list = stock_list.head(100)
```

### 2. 使用缓存

将下载的数据保存到本地数据库，避免重复下载。

### 3. 并行处理

使用多进程加速（需要修改代码）。

---

## 常见问题

### Q1: 报错"请设置TUSHARE_TOKEN"
**A**: 在项目根目录创建`.env`文件，添加你的Token。

### Q2: 运行很慢怎么办？
**A**:
1. 先改为小日期范围测试（如最近1年）
2. 减少股票数量
3. 检查网络连接

### Q3: 未找到任何样本？
**A**: 可能原因：
1. 日期范围太小
2. 筛选条件太严格
3. 数据质量问题

可以适当放宽条件进行测试。

### Q4: 积分不足？
**A**:
1. Tushare需要至少2000积分才能获取市值数据
2. 完善个人资料、每日签到获取积分
3. 或使用捐赠方式快速获得积分

### Q5: 数据缺失怎么办？
**A**:
- T1前34天数据不足的样本会有警告
- 可以忽略这些样本或减少lookback_days

---

## 数据验证

运行完成后，建议进行数据验证：

```python
import pandas as pd

# 1. 检查样本质量
df_samples = pd.read_csv('data/processed/positive_samples.csv')
print(df_samples.describe())

# 2. 检查特征完整性
df_features = pd.read_csv('data/processed/feature_data_34d.csv')
print(df_features.isnull().sum())

# 3. 可视化分析
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(df_samples['total_return'], bins=30)
plt.title('总涨幅分布')

plt.subplot(132)
plt.hist(df_samples['max_return'], bins=30)
plt.title('最高涨幅分布')

plt.subplot(133)
df_samples['ts_code'].value_counts().head(20).plot(kind='bar')
plt.title('Top20高频股票')

plt.tight_layout()
plt.savefig('data/processed/sample_analysis.png')
print("分析图表已保存")
```

---

## 下一步

完成正样本准备后：

1. **数据分析**
   - 查看样本分布
   - 检查数据质量
   - 可视化分析

2. **准备负样本**
   - 未来下跌或横盘的股票
   - 随机采样正常股票

3. **特征工程**
   - 添加更多技术指标
   - 特征标准化
   - 特征选择

4. **模型训练**
   - 使用XGBoost/LightGBM
   - 交叉验证
   - 超参数调优

5. **回测验证**
   - 历史回测
   - 绩效评估
   - 策略优化

---

**祝数据准备顺利！** 📊🚀

如有问题，请查看：
- [项目设计文档](PROJECT_DESIGN.md)
- [API参考](API_REFERENCE.md)
- [快速开始](QUICKSTART.md)
