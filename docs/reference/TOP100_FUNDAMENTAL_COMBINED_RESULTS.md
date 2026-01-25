# Top100基本面筛选综合结果使用说明

## 文件说明

脚本 `scripts/predict_and_screen_top100.py` 会生成一个综合结果文件，包含：

**文件名格式**：`v270_top100_fundamental_combined_{市值上限}亿_{日期}.csv`

例如：`v270_top100_fundamental_combined_200亿_20260119.csv`

## 文件包含的列

### 核心列（重要信息）

1. **model_rank**：模型评分排名（1-100）
   - 按模型预测概率（probability）排序
   - 数值越小，模型评分越高

2. **fundamental_rank**：基本面筛选排名
   - 仅通过基本面筛选的股票有排名
   - 按模型评分排序（通过筛选的股票中，模型评分最高的排名为1）
   - 未通过筛选的股票此列为空

3. **fundamental_pass**：是否通过基本面筛选
   - `True`：通过筛选
   - `False`：未通过筛选

4. **fundamental_reason**：未通过筛选的原因
   - 如果通过筛选，此列为空
   - 如果未通过，显示具体原因（如：市值过大、ROE不足等）

5. **probability**：模型预测概率
   - v2.7.0集成模型的预测概率
   - 数值越高，模型认为成为牛股的概率越大

6. **ts_code**：股票代码
7. **name**：股票名称

### 其他列

- `xgb_prob`, `lgb_prob`, `cat_prob`：三个子模型的预测概率
- `close`：收盘价
- `pct_chg`：涨跌幅
- `rsi_6`：RSI指标
- 其他技术指标...

## 使用方法

### 方法1：按模型评分排序（查看模型Top100）

在Excel或其他工具中：
1. 按 `model_rank` 列升序排序
2. 查看模型评分最高的100只股票

**适用场景**：相信模型判断，不关心基本面

### 方法2：按基本面筛选结果排序（查看通过筛选的股票）

在Excel或其他工具中：
1. 筛选 `fundamental_pass = True`
2. 按 `fundamental_rank` 列升序排序
3. 查看通过基本面筛选的股票（按模型评分排序）

**适用场景**：既要模型评分高，又要基本面好

### 方法3：综合排序（模型评分优先，基本面作为参考）

在Excel或其他工具中：
1. 先按 `fundamental_pass` 排序（True在前）
2. 再按 `model_rank` 排序
3. 这样可以看到：通过筛选的股票在前，未通过的在后，各自内部按模型评分排序

**适用场景**：优先考虑通过基本面筛选的股票，但也要参考模型评分

### 方法4：只看通过筛选的Top N

在Excel或其他工具中：
1. 筛选 `fundamental_pass = True`
2. 按 `fundamental_rank` 列升序排序
3. 取前N只（如Top10）

**适用场景**：只选择通过基本面筛选且模型评分最高的股票

## Python示例

```python
import pandas as pd

# 读取文件
df = pd.read_csv('v270_top100_fundamental_combined_200亿_20260119.csv')

# 方法1：按模型评分排序
df_model = df.sort_values('model_rank')
print("模型Top10:")
print(df_model.head(10)[['model_rank', 'ts_code', 'name', 'probability']])

# 方法2：只看通过基本面筛选的股票
df_fundamental = df[df['fundamental_pass'] == True].sort_values('fundamental_rank')
print("\n通过基本面筛选的股票:")
print(df_fundamental[['fundamental_rank', 'ts_code', 'name', 'probability']])

# 方法3：综合排序（通过筛选的在前，按模型评分排序）
df_combined = df.sort_values(['fundamental_pass', 'model_rank'], ascending=[False, True])
print("\n综合排序（通过筛选的在前）:")
print(df_combined.head(20)[['fundamental_pass', 'model_rank', 'ts_code', 'name', 'probability']])
```

## 筛选条件说明

当前使用的筛选条件（标准方案）：
- **市值范围**：10-200亿（可调整）
- **营业收入**：>5亿
- **净利润**：>500万
- **ROE**：>5%
- **ROA**：>2%

如需调整筛选条件，修改脚本中的配置或使用 `--market-cap-max` 参数。

## 运行脚本

```bash
# 预测并筛选Top100（市值上限200亿）
python scripts/predict_and_screen_top100.py --date 20260119 --market-cap-max 200

# 如果已有预测结果，可以直接筛选
python scripts/screen_top100_with_combined_results.py \
    --file data/prediction/results/v270_ensemble_all_20260119.csv \
    --date 20260119 \
    --market-cap-max 200
```

## 注意事项

1. **预测时间**：全市场预测需要较长时间（约30-60分钟），请耐心等待
2. **基本面筛选时间**：Top100基本面筛选需要约1-2分钟（调用100次API）
3. **数据获取**：基本面筛选需要Tushare财务数据接口，确保有足够积分
4. **结果文件**：预测完成后会自动保存到 `data/prediction/results/` 目录
