# 股票全方位体检功能使用指南

## 📋 功能概述

股票全方位体检功能为单支股票提供全面的健康检查，包括技术分析、基本面分析、AI模型预测、风险评估和买卖点识别。

## 🚀 使用方法

### 方法1: 命令行工具（推荐用于快速分析）

```bash
# 基本用法
python scripts/stock_health_check.py 000001.SZ

# 指定分析天数
python scripts/stock_health_check.py 600519.SH --days 180

# 不保存报告（仅查看）
python scripts/stock_health_check.py 600000.SH --no-save
```

#### 输出内容

命令行工具会：
1. 在终端显示完整的体检报告
2. 生成HTML图表文件（K线图、指标热力图）
3. 保存JSON格式的详细数据
4. 保存TXT格式的文本报告

所有文件保存在 `data/analysis/` 目录下。

### 方法2: 可视化面板（推荐用于交互式分析）

```bash
# 启动可视化面板
streamlit run app.py
```

然后：
1. 在左侧导航选择 **"🏥 股票体检"**
2. 输入股票代码（如 `000001.SZ`）
3. 选择分析天数（建议120天）
4. 点击 **"🔍 开始体检"**
5. 查看交互式报告和图表

#### 快速体检示例

可视化面板提供了快速示例按钮：
- 贵州茅台 (600519.SH)
- 中国平安 (601318.SH)
- 万科A (000002.SZ)
- 比亚迪 (002594.SZ)

点击即可快速体检。

### 方法3: Python API（推荐用于批量分析）

```python
from src.analysis.stock_health_checker import StockHealthChecker
from src.visualization.stock_chart import StockChartVisualizer

# 创建体检器
checker = StockHealthChecker()

# 体检股票
report = checker.check_stock('000001.SZ', days=120)

# 打印综合评分
print(f"综合评分: {report['overall_score']}")
print(f"投资建议: {report['recommendation']}")

# 创建可视化
visualizer = StockChartVisualizer()
chart = visualizer.create_comprehensive_chart('000001.SZ', report, days=120)
chart.show()  # 在浏览器中打开
```

## 📊 体检报告内容

### 1. 基本信息
- 股票代码、名称
- 所属行业、上市板块
- 最新价格、涨跌幅
- 成交量、成交额

### 2. 技术分析

#### 2.1 趋势分析
- **均线排列**: 多头/空头/震荡
- **短期趋势**: 强势上涨/温和上涨/温和下跌/快速下跌
- **涨跌幅**: 5日、20日涨跌幅统计
- **均线数值**: MA5、MA10、MA20、MA60

#### 2.2 技术指标
- **RSI**: 相对强弱指数，判断超买超卖
  - RSI > 70: 超买
  - RSI < 30: 超卖
  - 30-70: 正常
- **MACD**: 平滑异同移动平均线
  - 金叉: 买入信号
  - 死叉: 卖出信号
- **KDJ**: 随机指标
  - K线与D线交叉判断买卖点
- **布林带**: 价格通道
  - 突破上轨: 超买
  - 跌破下轨: 超卖

#### 2.3 支撑压力位
- 最近高点（压力位）
- 最近低点（支撑位）
- 当前价格距离支撑/压力的百分比

#### 2.4 成交量分析
- **量价关系**: 量价齐升/放量下跌/缩量上涨/缩量下跌
- **量比**: 当前成交量 / 20日平均成交量

### 3. 基本面分析
- **财务健康度**: 健康/需关注
  - 营收 > 3亿
  - 连续3年净利润 > 0
  - 净资产 > 0

### 4. AI模型预测
- **预测概率**: 未来34日成为牛股的概率
  - > 70%: 强烈看多
  - 60-70%: 看多
  - 40-60%: 中性
  - 30-40%: 看空
  - < 30%: 强烈看空
- **置信度**: 高/中/低

### 5. 风险评估
- **年化波动率**: 价格波动的剧烈程度
  - < 20%: 低波动
  - 20-40%: 中等波动
  - > 40%: 高波动
- **最大回撤**: 历史最大跌幅
  - > -10%: 低风险
  - -10% ~ -20%: 中等风险
  - < -20%: 高风险
- **综合风险**: 低风险/中等风险/高风险

### 6. 交易信号
- **操作建议**: 买入/卖出/观望
- **置信度**: 高/中/低
- **买入信号列表**: 触发的买入条件
- **卖出信号列表**: 触发的卖出条件
- **持有理由**: 适合持有的原因

### 7. 综合评分（0-100分）

评分构成：
- 技术分析: 30%
- 基本面: 20%
- 模型预测: 30%
- 风险控制: 20%

评分解读：
- **80-100分**: 优秀，强烈推荐
- **70-79分**: 良好，推荐
- **60-69分**: 中等，谨慎操作
- **50-59分**: 偏弱，建议观望
- **0-49分**: 较差，不建议操作

## 📈 可视化图表

### 1. K线图
- 蜡烛图显示价格走势
- MA5、MA10、MA20、MA60均线
- **买卖点标注**:
  - 🔺 红色向上三角: 买点
  - 🔻 绿色向下三角: 卖点

### 2. 成交量柱状图
- 红色: 上涨日成交量
- 绿色: 下跌日成交量

### 3. MACD指标
- DIF线（蓝色）
- DEA线（橙色）
- MACD柱（红绿）

### 4. RSI指标
- RSI曲线（紫色）
- 70超买线（红色虚线）
- 30超卖线（绿色虚线）

### 5. 指标健康度热力图
- 各项指标的评分可视化
- 颜色编码:
  - 🟢 绿色(8-10分): 健康
  - 🟡 黄色(6-7分): 中等
  - 🟠 橙色(4-5分): 偏弱
  - 🔴 红色(0-3分): 较差

## 🎯 买卖点识别逻辑

### 买点识别条件（满足2个及以上）
1. MA5金叉MA10
2. RSI < 30（超卖）
3. MACD金叉（DIF上穿DEA）
4. 量价齐升
5. 模型预测概率 > 70%

### 卖点识别条件（满足2个及以上）
1. MA5死叉MA10
2. RSI > 70（超买）
3. MACD死叉（DIF下穿DEA）
4. 放量下跌
5. 模型预测概率 < 30%

## 💡 使用建议

### 场景1: 选股后的深度分析
```bash
# 先用模型选股
python scripts/score_current_stocks.py

# 查看推荐结果
cat data/results/prediction_results_*.csv

# 对Top股票进行体检
python scripts/stock_health_check.py 000001.SZ
```

### 场景2: 持仓股票监控
```bash
# 定期体检持仓股票
python scripts/stock_health_check.py 600519.SH
python scripts/stock_health_check.py 601318.SH
python scripts/stock_health_check.py 000002.SZ
```

### 场景3: 批量体检
```python
from src.analysis.stock_health_checker import StockHealthChecker

checker = StockHealthChecker()

# 体检多只股票
stocks = ['000001.SZ', '600519.SH', '601318.SH']
reports = {}

for stock in stocks:
    report = checker.check_stock(stock)
    reports[stock] = report
    print(f"{stock}: 评分 {report['overall_score']}")

# 按评分排序
sorted_stocks = sorted(reports.items(), 
                      key=lambda x: x[1]['overall_score'], 
                      reverse=True)

print("\n最佳股票:")
for stock, report in sorted_stocks[:5]:
    print(f"{stock}: {report['overall_score']:.2f} - {report['recommendation']}")
```

### 场景4: 与模型选股结合
```python
import pandas as pd
from src.analysis.stock_health_checker import StockHealthChecker

# 读取模型推荐
df = pd.read_csv('data/results/prediction_results_latest.csv')
top_stocks = df.head(20)['股票代码'].tolist()

# 对每只股票体检
checker = StockHealthChecker()
final_recommendations = []

for stock in top_stocks:
    report = checker.check_stock(stock)
    
    # 只选择评分 > 70 的股票
    if report['overall_score'] > 70:
        final_recommendations.append({
            'stock': stock,
            'model_prob': df[df['股票代码'] == stock]['牛股概率'].values[0],
            'health_score': report['overall_score'],
            'action': report['trading_signals']['action']
        })

print(f"最终推荐 {len(final_recommendations)} 只股票")
```

## ⚠️ 注意事项

### 1. 数据要求
- 股票代码必须正确（如 `000001.SZ`、`600000.SH`）
- 股票需要有足够的历史数据（至少34天）
- 建议分析天数 > 60天以获得更准确的技术指标

### 2. 模型依赖
- 体检功能可以独立使用（不依赖模型）
- 如果模型文件存在，会额外提供AI预测
- 模型路径: `models/stock_selection/xgboost_timeseries_v3.joblib`

### 3. 性能考虑
- 单次体检耗时约3-10秒（取决于网络和数据量）
- 批量体检建议控制在50只股票以内
- 可视化图表生成较耗时，建议保存后查看

### 4. 投资风险提示

⚠️ **重要声明**

本工具提供的分析和建议：
- **仅供参考**，不构成投资建议
- **不保证**预测准确性
- **不承担**任何投资损失

投资建议：
- 🔹 结合自身风险承受能力
- 🔹 参考多方信息综合判断
- 🔹 控制仓位，分散投资
- 🔹 设置止损止盈
- 🔹 **股市有风险，入市需谨慎**

## 📚 常见问题

### Q1: 体检失败怎么办？
**A**: 可能原因：
- 股票代码错误
- 网络问题
- 数据不足

解决方法：
- 检查股票代码格式
- 确认网络连接
- 查看日志文件 `logs/aiquant.log`

### Q2: 买卖点准确率如何？
**A**: 买卖点基于技术指标识别，仅供参考：
- 回测胜率约60-70%
- 实际使用需结合其他信息
- 建议设置止损保护

### Q3: 综合评分低的股票能买吗？
**A**: 不建议，但特殊情况除外：
- 评分 < 60：不建议操作
- 评分 60-70：谨慎操作
- 评分 > 70：可以关注

特殊情况：
- 行业周期底部（可能反转）
- 重大利好消息
- 价值投资标的（长期持有）

### Q4: 如何提高体检速度？
**A**: 
- 减少分析天数（如60天）
- 使用数据缓存（已自动启用）
- 避免短时间重复体检同一股票

### Q5: 能否自定义评分权重？
**A**: 可以修改源代码：
```python
# 编辑 src/analysis/stock_health_checker.py
# _calculate_overall_score 方法

# 当前权重：
# 技术分析: 30%
# 基本面: 20%
# 模型预测: 30%
# 风险: 20%

# 修改权重示例：
score += tech_score * 0.4  # 改为40%
score += fund_score * 0.1  # 改为10%
score += model_score * 0.4 # 改为40%
score += risk_score * 0.1  # 改为10%
```

## 🔧 故障排除

### 错误: ModuleNotFoundError
```bash
# 安装依赖
pip install -r requirements.txt
```

### 错误: 无法获取数据
```bash
# 检查Tushare配置
cat .env | grep TUSHARE_TOKEN

# 测试Tushare连接
python -c "import tushare as ts; ts.set_token('YOUR_TOKEN'); print(ts.pro_bar(ts_code='000001.SZ', limit=1))"
```

### 错误: 模型文件不存在
```bash
# 训练模型
bash scripts/update_model_pipeline.sh

# 或指定模型版本
# 编辑 config/settings.yaml
# model.version: v3
```

## 📞 技术支持

遇到问题？
1. 查看日志: `logs/aiquant.log`
2. 查看文档: `docs/`
3. 提交Issue: GitHub Issues

---

**Happy Trading! 📈🚀**

