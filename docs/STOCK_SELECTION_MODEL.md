# 选股模型开发文档 📈

## 项目概述

开发一个基于技术指标的选股模型，通过历史数据筛选强势上涨股票作为正样本，用于模型训练。

## 模型逻辑

### 正样本筛选条件

从历史所有股票数据中，筛选满足以下条件的股票：

1. **周K连续三周收阳线**
   - 连续3周收盘价 > 开盘价
   
2. **总涨幅超50%**
   - (第3周收盘价 - 第1周开盘价) / 第1周开盘价 > 50%
   
3. **最高涨幅超70%**
   - (3周内最高价 - 第1周开盘价) / 第1周开盘价 > 70%
   
4. **剔除特殊股票**
   - 排除ST、*ST、SST等特殊处理股票
   - 排除停牌（HALT）股票
   - 排除退市（DELISTING）股票
   
5. **剔除北交所**
   - 排除股票代码以.BJ结尾的股票
   
6. **上市超过半年**
   - 上市日期距离T1至少180天
   
7. **T1定义**
   - T1 = 第一周的第一个交易日
   
8. **去重规则**
   - 同一支股票如果有多个符合条件的时间段，只保留最早的一个

### 负样本筛选条件 🆕

基于正样本的特征统计，筛选具有相似特征但不属于正样本的数据：

1. **特征统计**
   - 统计正样本T1前34天内的特征分布：
     * 量比>2的次数（n次）
     * MACD负转正的次数（x次）
     * 34天累计涨跌幅范围（a% ~ b%）

2. **负样本筛选**
   - 在全部历史数据中搜索连续34个交易日，满足：
     * 量比>2的次数在[Q25, Q75]范围内
     * MACD负转正次数在[Q25, Q75]范围内
     * 累计涨跌幅在[Q25, Q75]范围内
   - 排除已在正样本中的股票和时间段
   - 负样本数量与正样本数量相同

3. **负样本标签**
   - 所有负样本标记为 `label=0`
   - 正样本标记为 `label=1`

### 特征数据

对于每个正样本，提取**T1之前34天**的交易数据，包含以下字段：

| 字段 | 说明 | 来源 |
|-----|------|------|
| ts_code | 股票代码 | Tushare |
| name | 股票名称 | Tushare |
| trade_date | 交易日期 | Tushare |
| close | 收盘价 | Tushare |
| pct_chg | 当日涨跌幅(%) | Tushare |
| total_mv | 总市值(万元) | Tushare |
| circ_mv | 流通市值(万元) | Tushare |
| ma5 | 5日移动平均线 | 计算 |
| ma10 | 10日移动平均线 | 计算 |
| volume_ratio | 量比 | Tushare |

## 实施计划

### Phase 1: 数据准备（1-2天）

#### 步骤1.1: 获取股票列表
```python
# 获取所有A股列表
# 剔除ST股票
# 筛选上市时间符合要求的股票
```

#### 步骤1.2: 下载历史数据
```python
# 下载2000年至今的日线数据
# 下载周线数据
# 下载每日指标（市值、量比）
```

#### 步骤1.3: 数据存储
```python
# 保存到本地数据库/HDF5
# 建立索引加速查询
```

### Phase 2: 正样本筛选（2-3天）

#### 步骤2.1: 周K数据处理
```python
# 将日线数据聚合为周线
# 计算每周开盘价、收盘价、最高价、最低价
```

#### 步骤2.2: 三连阳筛选
```python
# 滑动窗口遍历每3周
# 检查是否连续收阳线
# 计算涨幅是否满足条件
```

#### 步骤2.3: 去重和筛选
```python
# 应用上市时间、ST等过滤条件
# 同一股票去重，保留最早的
```

### Phase 3: 特征提取（1-2天）

#### 步骤3.1: 计算技术指标
```python
# MA5、MA10移动平均线
# 确保有足够的历史数据
```

#### 步骤3.2: 提取T1前34天数据
```python
# 根据T1日期向前推34个交易日
# 整理成训练样本格式
```

#### 步骤3.3: 数据验证
```python
# 检查数据完整性
# 处理缺失值
# 异常值检测
```

### Phase 4: 数据导出（1天）

```python
# 导出为CSV格式
# 生成数据统计报告
# 可视化分析
```

## 数据格式

### 正样本列表格式
```csv
ts_code,name,t1_date,week1_open,week3_close,total_return,max_return
600519.SH,贵州茅台,20150601,180.50,285.60,58.23%,73.45%
```

### 特征数据格式
```csv
sample_id,ts_code,name,trade_date,close,pct_chg,total_mv,circ_mv,ma5,ma10,volume_ratio
1,600519.SH,贵州茅台,20150501,175.20,2.35,350000,280000,172.5,168.3,1.25
1,600519.SH,贵州茅台,20150502,176.80,0.91,352800,282240,173.2,169.1,0.95
...
```

## 预期输出

1. **正样本列表**: `data/processed/positive_samples.csv`
2. **特征数据集**: `data/processed/feature_data_34d.csv`
3. **统计报告**: `data/processed/sample_statistics.json`

## 技术要点

### 1. 周K线计算
```python
# 按周聚合
weekly_data = daily_data.resample('W-FRI', on='trade_date').agg({
    'open': 'first',    # 一周第一个交易日的开盘价
    'close': 'last',    # 一周最后一个交易日的收盘价
    'high': 'max',      # 一周最高价
    'low': 'min'        # 一周最低价
})
```

### 2. 涨跌幅计算
```python
# 总涨幅
total_return = (week3_close - week1_open) / week1_open * 100

# 最高涨幅
max_return = (three_week_high - week1_open) / week1_open * 100
```

### 3. 移动平均线
```python
# MA5
data['ma5'] = data['close'].rolling(window=5).mean()

# MA10
data['ma10'] = data['close'].rolling(window=10).mean()
```

## 注意事项

1. **数据完整性**
   - 确保34天交易日数据完整
   - 缺失数据需要特殊处理

2. **交易日历**
   - 使用实际交易日历，排除节假日
   - T1前34天是指34个交易日，非自然日

3. **复权处理**
   - 使用前复权数据，避免除权影响
   - 保持价格连续性

4. **ST股票**
   - 动态判断是否为ST
   - 包括ST、*ST、SST、S*ST等

5. **性能优化**
   - 批量处理减少API调用
   - 使用数据库索引加速查询
   - 考虑并行处理

## 使用脚本

### 准备正样本

```bash
python scripts/prepare_positive_samples.py
```

输出文件：
- `data/processed/positive_samples.csv` - 正样本列表
- `data/processed/feature_data_34d.csv` - 正样本特征数据
- `data/processed/sample_statistics.json` - 统计报告

### 准备负样本 🆕

```bash
python scripts/prepare_negative_samples.py
```

输出文件：
- `data/processed/negative_samples.csv` - 负样本列表
- `data/processed/negative_feature_data_34d.csv` - 负样本特征数据
- `data/processed/negative_sample_statistics.json` - 统计报告

### 检查数据质量

```bash
# 检查正样本质量
python scripts/check_sample_quality.py

# 可视化分析
python scripts/visualize_sample_quality.py
```

## 下一步

完成正负样本准备后：
1. ✅ 正样本数据筛选
2. ✅ 负样本数据筛选
3. 合并正负样本，划分训练集和测试集
4. 特征工程（标准化、特征选择）
5. 模型训练（XGBoost/LightGBM/Neural Network）
6. 模型评估和调优
7. 回测验证

---

**文档版本**: v2.0  
**创建时间**: 2024-12-22  
**更新时间**: 2024-12-23

