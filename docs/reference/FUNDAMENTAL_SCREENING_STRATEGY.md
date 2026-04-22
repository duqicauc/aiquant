# 基本面筛选策略分析

## 当前实现方式

### 方式1：预测前筛选（v2.7.0和v2.3.2当前实现）

```python
# 1. 获取所有有效股票（约3000只）
stock_list = get_valid_stocks(dm, predict_date)

# 2. 对全部股票进行基本面筛选（需要调用3000次API）
if enable_fundamental_screening:
    stock_list = fundamental_screener.filter_stocks(stock_list, predict_date)
    # 筛选后可能只剩500-1000只股票

# 3. 对筛选后的股票进行预测
for stock in stock_list:
    result = process_single_stock(...)
```

**优点**：
- 确保所有进入模型的股票都符合基本面要求
- 不会对基本面差的股票进行预测，节省预测时间

**缺点**：
- 需要调用大量API（3000次），非常慢
- 可能触发Tushare限流
- 可能漏掉一些基本面好但模型评分稍低的股票

### 方式2：预测后筛选（combine_v232_v270当前实现）

```python
# 1. 对所有股票进行预测
df_results = predict_all_stocks(...)

# 2. 对预测结果进行基本面筛选
if enable_fundamental_screening:
    df_results = fundamental_screener.filter_stocks(df_results, predict_date)
```

**优点**：
- 不需要在预测前筛选，预测流程不受影响
- 可以对所有股票进行预测，不会漏掉好股票

**缺点**：
- 如果对全部预测结果筛选，仍然需要调用大量API
- 如果只对top50筛选，可能错过一些基本面好但模型评分稍低的股票

## 推荐方案：预测后对TopN筛选

### 方案A：预测后对Top50筛选（推荐，效率最高）

```python
# 1. 对所有股票进行预测（不筛选）
df_results = predict_all_stocks(...)
df_results = df_results.sort_values('probability', ascending=False)

# 2. 对Top50进行基本面筛选（只需要调用50次API）
if enable_fundamental_screening:
    top50 = df_results.head(50)
    top50_screened = fundamental_screener.filter_stocks(top50, predict_date)

    # 如果Top50中有股票被筛掉，从后续股票中补充
    if len(top50_screened) < 50:
        remaining = df_results[~df_results['ts_code'].isin(top50_screened['ts_code'])]
        # 对后续股票也进行筛选，直到凑够50只
        ...
```

**优点**：
- 效率最高：只需要调用50次API
- 不会漏掉模型看好的股票
- 不会触发限流

**缺点**：
- 可能错过一些基本面好但模型评分稍低的股票（但概率较低）

### 方案B：预测后对Top100筛选（平衡方案）

```python
# 1. 对所有股票进行预测
df_results = predict_all_stocks(...)
df_results = df_results.sort_values('probability', ascending=False)

# 2. 对Top100进行基本面筛选（调用100次API）
if enable_fundamental_screening:
    top100 = df_results.head(100)
    top100_screened = fundamental_screener.filter_stocks(top100, predict_date)

    # 从筛选后的Top100中选择Top50
    top50 = top100_screened.head(50)
```

**优点**：
- 效率较高：只需要调用100次API
- 有更多候选股票，可以选出更好的Top50
- 不会触发限流

**缺点**：
- 比方案A稍慢，但仍然可接受

### 方案C：预测前对全部筛选（当前实现，不推荐）

```python
# 1. 获取所有有效股票
stock_list = get_valid_stocks(dm, predict_date)

# 2. 对全部股票进行基本面筛选（调用3000次API，很慢）
if enable_fundamental_screening:
    stock_list = fundamental_screener.filter_stocks(stock_list, predict_date)

# 3. 对筛选后的股票进行预测
df_results = predict_stocks(stock_list)
```

**优点**：
- 确保所有进入模型的股票都符合基本面要求

**缺点**：
- 效率最低：需要调用3000次API
- 可能触发限流
- 非常慢（可能需要几小时）

## 建议

**推荐使用方案A或方案B**：

1. **方案A（预测后对Top50筛选）**：如果追求最高效率，只关心Top50
2. **方案B（预测后对Top100筛选）**：如果想要更多候选股票，提高Top50的质量

**不推荐方案C（预测前对全部筛选）**：
- 效率太低
- 可能触发限流
- 实际效果与方案A/B差异不大

## 实现建议

修改代码，提供两种模式：

```python
def predict_top50(
    predict_date: str,
    enable_fundamental_screening: bool = False,
    screening_mode: str = 'post_top50'  # 'pre_all' 或 'post_top50' 或 'post_top100'
):
    """
    预测Top50股票

    Args:
        screening_mode: 筛选模式
            - 'pre_all': 预测前对全部股票筛选（慢，不推荐）
            - 'post_top50': 预测后对Top50筛选（快，推荐）
            - 'post_top100': 预测后对Top100筛选（较快，推荐）
    """
    # 1. 对所有股票进行预测
    df_results = predict_all_stocks(...)
    df_results = df_results.sort_values('probability', ascending=False)

    # 2. 根据模式进行筛选
    if enable_fundamental_screening:
        if screening_mode == 'post_top50':
            # 对Top50筛选
            top50 = df_results.head(50)
            top50_screened = fundamental_screener.filter_stocks(top50, predict_date)
            return top50_screened.head(50)
        elif screening_mode == 'post_top100':
            # 对Top100筛选，然后选Top50
            top100 = df_results.head(100)
            top100_screened = fundamental_screener.filter_stocks(top100, predict_date)
            return top100_screened.head(50)
        # ... 其他模式
```
