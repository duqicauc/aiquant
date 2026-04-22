# API参考文档 - 基于Tushare官方文档

本文档说明了我们的实现与 [Tushare官方文档](https://tushare.pro/document/2) 的对应关系。

## 📋 数据来源映射

### 1. 行情数据 - 通用行情接口（pro_bar）

**官方文档**: https://tushare.pro/document/2?doc_id=109

**我们的实现**: `DataFetcher.get_stock_data()`

```python
# 官方示例
df = ts.pro_bar(
    ts_code='000001.SZ',
    asset='E',
    adj='qfq',
    start_date='20240101',
    end_date='20241231'
)

# 我们的封装
fetcher = DataFetcher(source='tushare')
df = fetcher.get_stock_data(
    stock_code='000001',  # 自动添加.SZ后缀
    start_date='20240101',
    end_date='20241231',
    adjust='qfq'
)
```

**输出字段对照**（来自官方文档）:

| 官方字段 | 类型 | 说明 | 我们的列名 |
|---------|------|------|----------|
| ts_code | str | 股票代码 | ts_code |
| trade_date | str | 交易日期 | trade_date |
| open | float | 开盘价 | open |
| high | float | 最高价 | high |
| low | float | 最低价 | low |
| close | float | 收盘价 | close |
| pre_close | float | 昨收价 | pre_close |
| change | float | 涨跌额 | change |
| pct_chg | float | 涨跌幅（%） | pct_chg |
| vol | float | 成交量（手） | vol |
| amount | float | 成交额（千元） | amount |

### 2. 每日指标数据 - daily_basic接口

**官方文档**: https://tushare.pro/document/2?doc_id=32

**我们的实现**: `DataFetcher.get_daily_basic()`

```python
# 官方示例
pro = ts.pro_api()
df = pro.daily_basic(
    ts_code='000001.SZ',
    start_date='20240101',
    end_date='20241231',
    fields='ts_code,trade_date,turnover_rate,volume_ratio,total_mv,circ_mv'
)

# 我们的封装
fetcher = DataFetcher(source='tushare')
df = fetcher.get_daily_basic(
    stock_code='000001',
    start_date='20240101',
    end_date='20241231'
)
```

**输出字段对照**（来自官方文档）:

| 官方字段 | 类型 | 说明 | 我们的列名 |
|---------|------|------|----------|
| ts_code | str | TS股票代码 | ts_code |
| trade_date | str | 交易日期 | trade_date |
| close | float | 当日收盘价 | - |
| turnover_rate | float | 换手率（%） | turnover_rate |
| volume_ratio | float | 量比 | volume_ratio |
| pe | float | 市盈率 | - |
| pe_ttm | float | 市盈率TTM | - |
| pb | float | 市净率 | - |
| total_mv | float | 总市值（万元） | total_mv |
| circ_mv | float | 流通市值（万元） | circ_mv |

### 3. 完整数据获取

**我们的实现**: `StockAnalyzer.get_complete_data()`

这个方法整合了以上两个接口，并添加了技术指标计算：

```python
analyzer = StockAnalyzer(source='tushare')
df = analyzer.get_complete_data(
    stock_code='000001',
    start_date='20240101',
    end_date='20241231',
    adjust='qfq'
)
```

**返回的完整字段**:

| 字段名 | 数据来源 | 说明 | 单位 |
|-------|---------|------|------|
| **基础行情数据**（来自pro_bar） ||||
| trade_date | pro_bar | 交易日期 | - |
| ts_code | pro_bar | 股票代码 | - |
| open | pro_bar | 开盘价 | 元 |
| high | pro_bar | 最高价 | 元 |
| low | pro_bar | 最低价 | 元 |
| close | pro_bar | 收盘价 | 元 |
| pct_chg | pro_bar | 涨跌幅 | % |
| vol | pro_bar | 成交量 | 手 |
| amount | pro_bar | 成交额 | 千元 |
| **市值数据**（来自daily_basic） ||||
| total_mv | daily_basic | 总市值 | 万元 |
| circ_mv | daily_basic | 流通市值 | 万元 |
| volume_ratio | daily_basic | 量比 | - |
| **技术指标**（本地计算） ||||
| ma5 | 本地计算 | 5日移动平均线 | 元 |
| ma20 | 本地计算 | 20日移动平均线 | 元 |
| macd_dif | 本地计算 | MACD-DIF | - |
| macd_dea | 本地计算 | MACD-DEA | - |
| macd | 本地计算 | MACD柱 | - |

## 🔧 技术指标计算方法

### 1. 移动平均线（MA）

**我们的实现**: `DataFetcher.calculate_ma()`

```python
# 计算5日均线
df['ma5'] = fetcher.calculate_ma(df, period=5, price_col='close')

# 计算20日均线
df['ma20'] = fetcher.calculate_ma(df, period=20, price_col='close')
```

**计算公式**:
```
MA(n) = (P1 + P2 + ... + Pn) / n
```
其中 P 为收盘价，n 为周期

### 2. MACD指标

**我们的实现**: `DataFetcher.calculate_macd()`

```python
macd_data = fetcher.calculate_macd(
    df,
    fast=12,    # 快线周期
    slow=26,    # 慢线周期
    signal=9    # 信号线周期
)

df['macd_dif'] = macd_data['dif']
df['macd_dea'] = macd_data['dea']
df['macd'] = macd_data['macd']
```

**计算公式**:
```
EMA(fast) = 快速指数移动平均线（默认12日）
EMA(slow) = 慢速指数移动平均线（默认26日）

DIF = EMA(fast) - EMA(slow)
DEA = DIF的9日EMA
MACD = 2 × (DIF - DEA)
```

## 📊 股票代码格式规范

根据 [官方文档](https://tushare.pro/document/2?doc_id=14)，Tushare的股票代码规则：

| 交易所 | 代码 | 后缀 | 示例 |
|-------|------|------|------|
| 上海证券交易所 | SSE | .SH | 600000.SH（股票）<br>000001.SH（指数） |
| 深圳证券交易所 | SZSE | .SZ | 000001.SZ（股票）<br>399005.SZ（指数） |
| 北京证券交易所 | BSE | .BJ | 430xxx.BJ |
| 香港证券交易所 | HKEX | .HK | 00001.HK |

**我们的自动识别逻辑** (`_format_stock_code()`):

```python
def _format_stock_code(self, code: str) -> str:
    """
    自动识别并添加交易所后缀

    输入 -> 输出
    '000001' -> '000001.SZ'  (0开头=深圳)
    '600000' -> '600000.SH'  (6开头=上海)
    '300750' -> '300750.SZ'  (3开头=创业板)
    '000001.SZ' -> '000001.SZ' (已有后缀)
    """
    if '.' in code:
        return code

    if code.startswith('6'):
        return f"{code}.SH"
    elif code.startswith(('0', '3')):
        return f"{code}.SZ"
    else:
        raise ValueError(f"无法识别的股票代码: {code}")
```

## 🔄 复权类型说明

根据 [官方文档](https://tushare.pro/document/2?doc_id=109)，复权参数（adj）：

| 参数值 | 说明 | 适用场景 |
|-------|------|---------|
| `None` | 不复权 | 查看真实历史价格 |
| `'qfq'` | 前复权 | **推荐**，技术分析、回测 |
| `'hfq'` | 后复权 | 保持历史价格不变 |

**前复权示例**:
```python
# 前复权：保持最新价格不变，调整历史价格
df = analyzer.get_complete_data(
    stock_code='600519',  # 贵州茅台
    start_date='20240101',
    adjust='qfq'  # 前复权
)
```

**注意事项**（来自官方文档）:
- 复权机制是根据 `end_date` 参数动态复权
- 采用分红再投模式
- 目前只支持日线复权

## 📅 日期格式规范

根据官方文档要求，日期格式必须是：**YYYYMMDD**

**我们的处理**:
```python
# 我们支持两种格式，都会自动转换为YYYYMMDD
start_date = '20240101'      # ✓ 推荐格式
start_date = '2024-01-01'    # ✓ 也支持，自动转换为20240101

# 内部转换逻辑
start_date = start_date.replace('-', '')
```

## 🔐 积分要求

根据 [官方积分说明](https://tushare.pro/document/1?doc_id=13)：

| 接口 | 最低积分 | 无限制积分 |
|-----|---------|----------|
| pro_bar（日线） | 120 | 2000 |
| daily_basic | 2000 | 5000 |
| 分钟数据 | 600 | 2000 |

**本项目需要**:
- 基础功能：120积分（仅行情数据）
- 完整功能：**2000积分**（包含市值、量比等）

## 🎯 API调用限制

根据官方文档：

| 积分等级 | 每分钟调用次数 | 单次返回条数 |
|---------|--------------|-------------|
| 120积分 | 120次 | 6000条 |
| 2000积分 | 200次 | 6000条 |
| 5000积分 | 无限制 | 6000条 |

**注意**:
- 单次请求最多返回6000条数据
- 一次请求约等于一个股票23年历史数据
- 如需更多数据，请分批请求

## 📖 相关文档链接

### Tushare官方文档
- [Tushare Pro首页](https://tushare.pro/)
- [沪深股票数据](https://tushare.pro/document/2?doc_id=14)
- [通用行情接口](https://tushare.pro/document/2?doc_id=109)
- [每日指标接口](https://tushare.pro/document/2?doc_id=32)
- [历史日线接口](https://tushare.pro/document/2?doc_id=27)
- [数据工具（在线调试）](https://tushare.pro/webclient/)
- [积分获取方法](https://tushare.pro/document/1?doc_id=13)
- [常见问题](https://tushare.pro/document/1?doc_id=122)

### 本项目文档
- [README.md](README.md) - 完整使用文档
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [example.py](example.py) - 使用示例代码

## ⚠️ 重要说明

1. **数据单位注意**:
   - 总市值、流通市值：**万元**（需除以10000转换为亿元）
   - 成交额：**千元**（需除以1000转换为元）
   - 成交量：**手**（1手=100股）

2. **日期格式**:
   - 输入：YYYYMMDD 或 YYYY-MM-DD
   - 输出：pandas.Timestamp 对象

3. **停牌期间**:
   - 行情数据不会返回
   - 可能导致DataFrame中某些日期缺失

4. **技术指标**:
   - MA指标需要足够的历史数据（如MA20需要至少20个交易日）
   - MACD指标需要至少34个交易日（26+9-1）

---

**文档版本**: v1.0
**最后更新**: 2024-12-22
**基于**: Tushare Pro官方文档
