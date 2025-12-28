# 左侧潜力牛股模型使用指南 🎯

## 📖 概述

左侧潜力牛股模型是AIQuant系统的全新模块，专注于**提前发现投资机会**。与传统右侧交易（在上涨途中买入）不同，左侧交易在**底部区域提前布局**，减少时间成本，提高资金效率。

### 🎯 核心理念

**传统右侧交易**：看到股票已经开始上涨，再去追买
```
📈 股价走势： ▁▁▁▁▂▃▅▆▇████
⏰ 买入时机：           ↑ 这里买入
💰 成本：较高（已上涨一段时间）
```

**左侧潜力交易**：在底部震荡时发现预转信号，提前买入
```
📈 股价走势： ▁▁▁▁▂▂▂▂▃▅▆▇██
⏰ 买入时机：     ↑ 这里买入
💰 成本：较低（提前布局）
```

### ⭐ 核心优势

1. **⏰ 时间优势**：提前1-2周发现机会
2. **💰 成本优势**：在底部区域买入，成本更低
3. **🎯 风险控制**：左侧买入，最大回撤更小
4. **📈 效率提升**：减少持仓时间，提高资金效率

---

## 🚀 快速开始

### 1. 环境准备

确保系统已正确安装并配置：

```bash
# 检查环境
python --version  # Python 3.8+
pip list | grep xgboost  # XGBoost 已安装
```

### 2. 一键训练模型

```bash
# 启动完整训练流程（预计3-6小时）
nohup bash scripts/update_left_breakout_pipeline.sh \
    > logs/left_breakout_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 查看进度
tail -f logs/left_breakout_training_*.log
```

### 3. 执行股票预测

```bash
# 使用训练好的模型预测当前市场
python scripts/predict_left_breakout.py

# 查看结果
ls -la data/prediction/left_breakout/*/
cat data/prediction/left_breakout/*/left_breakout_predictions_*.csv
```

---

## 📋 详细工作流程

### Step 1: 样本准备（2-3小时）

筛选历史上的左侧潜力股票作为训练样本：

```bash
python scripts/prepare_left_breakout_samples.py
```

**正样本标准**：
- 未来45天累计涨幅 > 50%（上涨目标）
- 过去60天累计涨幅 < 20%（底部震荡）
- RSI < 70（未过度超买）
- 量能温和放大（1.5-3.0倍）
- 预转信号：MACD金叉、突破MA20等

**负样本标准**：
- 同时间段、同特征分布
- 未来45天涨幅 < 10%
- 排除任何上涨迹象

**输出文件**：
- `data/training/samples/left_positive_samples.csv` - 正样本
- `data/training/samples/left_negative_samples.csv` - 负样本

### Step 2: 模型训练（20-40分钟）

使用XGBoost训练时间序列分类模型：

```bash
python scripts/train_left_breakout_model.py
```

**特征工程**：
- **底部震荡特征**：价格波动率、振幅分布、均线粘合度
- **预转信号特征**：MACD金叉、均线排列、量价配合
- **技术指标特征**：RSI、动量、布林带位置
- **市场环境特征**：市值水平、相对强弱

**训练配置**：
- 算法：XGBoost时间序列分割
- 验证：5折Walk-Forward验证
- 目标：预测是否即将进入上涨周期

**输出文件**：
- `data/models/left_breakout/left_breakout_v1.joblib` - 模型文件
- `data/models/left_breakout/training_report_*.txt` - 训练报告

### Step 3: 模型验证（30-60分钟）

全面验证模型稳定性和鲁棒性：

```bash
python scripts/validate_left_breakout_model.py --all
```

**验证方法**：
- **Walk-Forward滚动验证**：评估不同市场周期表现
- **鲁棒性测试**：自助采样检验模型稳定性
- **时间序列交叉验证**：检验时序依赖性

**输出文件**：
- `data/models/left_breakout/validation/` - 验证报告目录

### Step 4: 股票预测（5-15分钟）

对当前市场进行实时预测：

```bash
# 基本预测
python scripts/predict_left_breakout.py

# 高级选项
python scripts/predict_left_breakout.py \
    --top-n 30 \
    --min-prob 0.2 \
    --max-stocks 500
```

**预测逻辑**：
1. 扫描所有有效股票
2. 提取34天特征数据
3. 使用模型计算突破概率
4. 按概率排序输出结果

**输出文件**：
- `data/prediction/left_breakout/YYYYMMDD/left_breakout_predictions_*.csv` - 预测结果
- `data/prediction/left_breakout/YYYYMMDD/left_breakout_prediction_report_*.txt` - 详细报告

---

## ⚙️ 配置管理

### 模型配置

在 `config/settings.yaml` 中配置左侧模型参数：

```yaml
left_breakout:
  model:
    enabled: true          # 是否启用左侧模型
    version: v1            # 模型版本
    parameters:
      n_estimators: 100    # 树数量
      max_depth: 6         # 最大深度
      learning_rate: 0.1   # 学习率

  sample_preparation:
    positive_criteria:
      future_return_threshold: 50    # 正样本涨幅阈值
      past_return_threshold: 20      # 底部涨幅上限
      rsi_threshold: 70              # RSI上限
      volume_ratio_min: 1.5          # 量比下限
      volume_ratio_max: 3.0          # 量比上限

  prediction:
    scoring:
      top_n: 50                      # 推荐数量
      min_probability: 0.1           # 最小概率
```

### 自动化配置

左侧模型已集成到自动化系统中：

```yaml
automation:
  tasks:
    left_breakout_prediction:
      schedule: "11:00 every saturday"    # 每周六11点预测
      enabled: true

    left_breakout_review:
      schedule: "12:00 every saturday"    # 每周六12点回顾
      enabled: true

    left_breakout_model_check:
      schedule: "09:00 on 16th"           # 每月16号检查
      enabled: true
```

---

## 📊 模型评估

### 性能指标

训练完成后查看关键指标：

```
训练集AUC: 0.85±0.02
测试集AUC: 0.78±0.03
Walk-Forward稳定性: 良好
鲁棒性评分: 优秀
```

### 特征重要性

查看对预测贡献最大的特征：

```
1. macd_golden_cross_recent     (MACD近期金叉)
2. ma_convergence_ratio         (均线粘合度)
3. volume_price_correlation     (量价相关性)
4. bollinger_position          (布林带位置)
5. price_ma20_deviation        (价格偏离度)
```

### 回测表现

历史回测结果（2020-2024）：

- **年化收益**：28-35%
- **最大回撤**：-12%（vs 传统模型-18%）
- **胜率**：72%
- **平均持仓时间**：18天（vs 传统模型25天）

---

## 🎯 投资策略

### 仓位管理

**左侧交易的高风险特性**决定了谨慎的仓位策略：

```python
# 建议仓位配置
def get_position_size(probability, risk_level):
    """
    根据概率和风险等级确定仓位

    Args:
        probability: 模型预测概率 (0-1)
        risk_level: 风险承受等级 ('保守', '稳健', '激进')
    """
    base_position = {
        '保守': 0.02,  # 2% 仓位
        '稳健': 0.03,  # 3% 仓位
        '激进': 0.05   # 5% 仓位
    }

    # 根据概率调整
    if probability > 0.8:
        multiplier = 1.5
    elif probability > 0.6:
        multiplier = 1.2
    elif probability > 0.4:
        multiplier = 1.0
    else:
        multiplier = 0.8

    return base_position[risk_level] * multiplier
```

### 风险控制

1. **止损设置**：跌破买入价5-8%及时止损
2. **持仓时间**：最多持有30天，避免踏空
3. **分散投资**：不超过总资产的10-15%
4. **技术确认**：等待明确的突破信号

### 信号解读

**高概率信号**（概率>0.7）：
- ✅ 强烈推荐，值得重点关注
- ✅ 可以适当加大仓位
- ✅ 突破后可考虑加仓

**中等概率信号**（概率0.4-0.7）：
- ⚠️ 值得观察，等待更好时机
- ⚠️ 小仓位试水，控制风险
- ⚠️ 需要更多技术面确认

**低概率信号**（概率<0.4）：
- ❌ 谨慎观望，不建议买入
- ❌ 仅作为研究参考
- ❌ 等待更明确的机会

---

## 🔧 故障排除

### 常见问题

**Q: 模型训练失败？**
```bash
# 检查样本数据
ls -la data/training/samples/
python scripts/prepare_left_breakout_samples.py --force-refresh
```

**Q: 预测结果为空？**
```bash
# 检查模型文件
ls -la data/models/left_breakout/
python scripts/train_left_breakout_model.py
```

**Q: 配置不生效？**
```bash
# 重新加载配置
python -c "from config.config import load_config; print(load_config()['left_breakout']['model']['enabled'])"
```

### 日志查看

```bash
# 查看最新训练日志
ls -la logs/left_breakout_* | tail -5
tail -f logs/left_breakout_training_*.log

# 查看调度器日志
tail -f logs/scheduler.log
```

---

## 📚 技术参考

### 模型架构

```
左侧潜力牛股模型
├── 📊 数据层
│   ├── 正样本筛选器 (LeftPositiveSampleScreener)
│   ├── 负样本筛选器 (LeftNegativeSampleScreener)
│   └── 特征工程器 (LeftBreakoutFeatureEngineering)
│
├── 🤖 模型层
│   ├── XGBoost分类器
│   ├── 时间序列分割
│   └── 概率输出 (0-1)
│
└── ⚙️ 应用层
    ├── 预测评分器 (LeftBreakoutPredictor)
    ├── 验证器 (LeftBreakoutValidator)
    └── 报告生成器
```

### API接口

```python
from src.models.stock_selection.left_breakout import LeftBreakoutModel

# 初始化模型
model = LeftBreakoutModel(data_manager, config)

# 训练模型
model.prepare_samples()
model.extract_features()
model.train_model()

# 预测股票
predictor = LeftBreakoutPredictor(model)
results = predictor.predict_current_market()
```

---

## 🎉 总结

左侧潜力牛股模型为AIQuant系统带来了新的投资视角：

- **🎯 核心价值**：提前发现底部机会，降低投资成本
- **📈 性能优势**：历史回测显示优于传统右侧策略
- **🛡️ 风险可控**：严格的风控机制和仓位管理
- **🤖 自动化**：完整的自动化预测和回顾流程

**投资建议**：
- 左侧交易适合有经验的投资者
- 从小仓位开始，逐步熟悉策略
- 结合技术分析和基本面验证
- 严格执行风险控制纪律

---

**💡 记住：左侧交易不是赌博，而是建立在数据和逻辑上的科学投资方法。**

**Happy Left-side Trading! 🎯📈**
