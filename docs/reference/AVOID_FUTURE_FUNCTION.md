# 避免未来函数 - 关键指南 🚨

量化交易中最致命的错误之一

---

## ⚠️ 什么是未来函数？

**未来函数（Look-Ahead Bias）**: 在训练模型时，使用了未来才能知道的信息。

### 典型错误示例

```python
# ❌ 错误示例1：随机划分数据
X_train, X_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 问题：可能用2024年数据训练，2023年数据测试
# 在实际应用中，你在2023年根本看不到2024年的数据！
```

```python
# ❌ 错误示例2：使用未来数据计算特征
df['return_future_3d'] = df['close'].shift(-3) / df['close'] - 1

# 问题：在T日计算了T+3日的收益率
# 实际交易时，你不可能知道3天后的价格！
```

```python
# ❌ 错误示例3：全局归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 对全部数据归一化

# 问题：用了测试集的统计信息（均值、标准差）
# 实际应用时，你不知道未来数据的分布！
```

---

## 🎯 后果

### 虚假的高性能

```
回测结果: 
  准确率: 95%  ← 看起来很好！
  收益率: 200% ← 太完美了！

实盘结果:
  准确率: 52%  ← 接近随机
  收益率: -30% ← 亏损
```

**原因**: 模型在训练时"偷看"了未来数据，学到了不存在的模式。

---

## ✅ 正确做法

### 1. 按时间划分数据 ⭐⭐⭐⭐⭐

```python
# ✅ 正确做法：时间序列划分
df = df.sort_values('date')

# 训练集：2022-2023年数据
train = df[df['date'] <= '2023-12-31']

# 测试集：2024年数据
test = df[df['date'] >= '2024-01-01']

# 确保训练集完全在测试集之前！
```

**原则**: 
- 训练集 = 过去数据
- 测试集 = 未来数据
- 训练集的最后一天 < 测试集的第一天

---

### 2. Walk-Forward Validation（滚动窗口验证）⭐⭐⭐⭐⭐

```python
# ✅ 更严格的验证方法

时间线：
|----训练1----|--测试1--|
     |----训练2----|--测试2--|
          |----训练3----|--测试3--|

示例：
训练1: 2022-01 ~ 2022-12 → 测试1: 2023-01 ~ 2023-03
训练2: 2022-04 ~ 2023-03 → 测试2: 2023-04 ~ 2023-06
训练3: 2022-07 ~ 2023-06 → 测试3: 2023-07 ~ 2023-09
```

**优点**:
- 模拟真实交易场景
- 多个时间窗口验证，更稳健
- 发现模型是否随时间退化

---

### 3. 特征工程中避免未来函数

```python
# ❌ 错误：使用shift(-n)
df['future_return'] = df['close'].shift(-5) / df['close'] - 1

# ✅ 正确：只使用shift(n)或shift(0)
df['past_return'] = df['close'] / df['close'].shift(5) - 1
```

**检查清单**:
- [ ] 所有特征都是T时刻或之前的数据？
- [ ] 没有使用shift(-n)？
- [ ] 没有使用未来的价格、成交量？
- [ ] 没有使用未来的技术指标？

---

### 4. 归一化/标准化的正确姿势

```python
# ❌ 错误：对全部数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 正确：只用训练集fit，然后transform测试集
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train
X_test_scaled = scaler.transform(X_test)        # transform test
```

---

## 🔍 检测未来函数的方法

### 方法1：检查时间顺序

```python
# 检查训练集和测试集的时间范围
print(f"训练集: {train['date'].min()} ~ {train['date'].max()}")
print(f"测试集: {test['date'].min()} ~ {test['date'].max()}")

# 必须满足：train['date'].max() < test['date'].min()
assert train['date'].max() < test['date'].min(), "时间重叠，存在未来函数！"
```

### 方法2：回测 vs 实盘对比

```python
如果：
  回测性能 >> 实盘性能
  
很可能：
  存在未来函数！
```

### 方法3：逐行检查特征

```python
# 对每个特征，问自己：
# "在T时刻，我真的能获得这个数据吗？"

特征: return_next_week
问题: T时刻能知道下周收益率吗？
答案: 不能！ → 未来函数❌

特征: volume_ratio_today
问题: T时刻能知道今天的量比吗？
答案: 如果是收盘后，可以✅
```

---

## 📚 本项目中的实现

### ✅ 已实现：时间序列划分训练

```bash
# 使用时间序列划分的训练脚本
python scripts/train_xgboost_timeseries.py
```

**关键改进**:
1. ✅ 按时间排序数据
2. ✅ 训练集 = 前80%时间段
3. ✅ 测试集 = 后20%时间段
4. ✅ 确保无时间重叠
5. ✅ 保存时间范围到报告

### 输出示例

```
时间划分:
  训练集: 2022-01-04 至 2023-10-15
  测试集: 2023-10-16 至 2024-12-20

样本划分:
  训练集: 1832 个样本 (正:916, 负:916)
  测试集: 458 个样本 (正:229, 负:229)

✓ 训练集和测试集时间无重叠，无数据泄露风险
```

---

## 🎯 实战检查清单

### 数据准备阶段

- [ ] ✅ T1日期是第一周的第一个交易日
- [ ] ✅ 特征数据是T1之前的34天
- [ ] ✅ 没有使用T1之后的数据
- [ ] ✅ 没有使用未来的价格/成交量

### 特征工程阶段

- [ ] ✅ 所有技术指标都是基于历史数据计算
- [ ] ✅ MACD、RSI、MA等都是T1之前的
- [ ] ✅ 没有使用shift(-n)
- [ ] ✅ 没有使用未来收益率作为特征

### 模型训练阶段

- [ ] ✅ 按时间排序数据
- [ ] ✅ 训练集完全在测试集之前
- [ ] ✅ 归一化只在训练集fit
- [ ] ✅ 没有随机划分数据

### 模型评估阶段

- [ ] ✅ 测试集 = 未来数据
- [ ] ✅ 测试集中没有训练集的数据
- [ ] ✅ 报告中记录了时间范围

### 回测阶段

- [ ] ✅ 模拟真实交易流程
- [ ] ✅ 每个交易日只使用当日及之前的数据
- [ ] ✅ 交易成本、滑点已考虑
- [ ] ✅ 信号延迟已考虑（T日信号，T+1执行）

---

## 🔬 Walk-Forward Validation实现

### 实现脚本

```python
"""
Walk-Forward验证
模拟真实的逐步外推场景
"""
import pandas as pd
from datetime import timedelta

def walk_forward_validation(df, window_size=12, step_size=3):
    """
    滚动窗口验证
    
    Args:
        df: 数据（必须包含date列）
        window_size: 训练窗口大小（月）
        step_size: 测试窗口大小（月）
    """
    results = []
    
    df = df.sort_values('date')
    start_date = df['date'].min()
    end_date = df['date'].max()
    
    current_date = start_date + timedelta(days=window_size*30)
    
    while current_date + timedelta(days=step_size*30) <= end_date:
        # 训练集：当前日期前window_size个月
        train_start = current_date - timedelta(days=window_size*30)
        train_end = current_date
        
        # 测试集：当前日期后step_size个月
        test_start = current_date + timedelta(days=1)
        test_end = current_date + timedelta(days=step_size*30)
        
        train_data = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
        test_data = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
        
        # 训练模型
        model = train_model(train_data)
        
        # 测试
        metrics = evaluate_model(model, test_data)
        
        results.append({
            'train_period': f"{train_start.date()} ~ {train_end.date()}",
            'test_period': f"{test_start.date()} ~ {test_end.date()}",
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score']
        })
        
        # 前进一个step
        current_date += timedelta(days=step_size*30)
    
    return pd.DataFrame(results)

# 使用
results = walk_forward_validation(df)
print(results)

# 平均性能
print(f"\n平均准确率: {results['accuracy'].mean():.2%}")
print(f"准确率标准差: {results['accuracy'].std():.2%}")
```

---

## 💡 关键要点总结

### 核心原则

```
在T时刻进行决策时：
  只能使用T时刻及之前的数据！
  
训练模型时：
  训练集必须完全在测试集之前！
  
验证模型时：
  使用walk-forward，模拟真实外推！
```

### 记住

1. **时间是单向的** - 不能回到过去，不能预知未来
2. **回测不等于实盘** - 未来函数会让回测虚高
3. **严格检查** - 每个特征都要问"T时刻能获得吗？"
4. **多次验证** - 在多个时间窗口上测试

---

## 📊 对比：两种划分方式

### 随机划分（❌ 不推荐）

```
优点:
  - 实现简单
  - 训练集和测试集分布相似

缺点:
  - ❌ 不符合时间序列特性
  - ❌ 可能产生未来函数
  - ❌ 回测结果虚高
  - ❌ 实盘效果差
```

### 时间序列划分（✅ 推荐）

```
优点:
  - ✅ 符合真实交易场景
  - ✅ 避免未来函数
  - ✅ 回测更可信
  - ✅ 实盘效果与回测接近

缺点:
  - 训练集和测试集分布可能不同
  - 需要更多数据
```

**结论**: 在量化交易中，必须使用时间序列划分！

---

## 🎓 延伸阅读

### 推荐资源

1. **《Advances in Financial Machine Learning》** by Marcos López de Prado
   - Chapter 7: Cross-Validation in Finance
   
2. **《Quantitative Trading》** by Ernest Chan
   - Chapter 4: Backtesting

3. **论文**: "The Probability of Backtest Overfitting"
   - Bailey et al., 2016

---

## ✅ 快速开始

```bash
# 1. 准备正负样本数据（如果还没做）
python scripts/prepare_positive_samples.py
python scripts/prepare_negative_samples_v2.py

# 2. 使用时间序列划分训练模型
python scripts/train_xgboost_timeseries.py

# 3. 查看结果，确认时间无重叠
# 输出中会显示：
#   训练集: 2022-01-04 至 2023-10-15
#   测试集: 2023-10-16 至 2024-12-20
#   ✓ 训练集和测试集时间无重叠，无数据泄露风险
```

---

**文档版本**: v1.0  
**创建时间**: 2024-12-23  
**最后更新**: 2024-12-23

**记住：避免未来函数是量化交易成功的基础！**

