# 左侧潜力牛股筛选条件问题诊断

## 🔍 发现的问题

### 问题1：窗口大小不匹配 ⚠️ **严重**

**代码中**：
```python
window_size = 90  # 90天窗口（过去60天 + 向前看30天）
```

**实际需要**：
- 过去60天（特征数据）
- 未来45天（验证涨幅）
- **总计需要105天**

**影响**：未来45天数据不足，导致条件1（未来涨幅>50%）无法正确计算

---

### 问题2：量比默认值导致过滤 ⚠️ **严重**

**代码中**：
```python
if 'volume_ratio' not in df.columns:
    df['volume_ratio'] = 1.0  # 默认值
```

**筛选条件**：
```python
if avg_volume_ratio < 1.5 or avg_volume_ratio > 3.0:
    return False
```

**影响**：所有volume_ratio缺失的股票（默认值1.0）都会被过滤掉

**解决方案**：
- `get_complete_data`应该已经包含了`volume_ratio`（来自daily_basic）
- 需要检查数据合并是否正确

---

### 问题3：技术指标可能缺失 ⚠️ **中等**

**需要指标**：
- RSI（`rsi_6`）
- MACD（`macd_dif`, `macd_dea`）
- 量比（`volume_ratio`）

**当前获取**：
- ✅ 已获取`stk_factor`（包含技术因子）
- ✅ 已合并到主数据

**潜在问题**：如果`stk_factor`中某些字段缺失，会导致条件失败

---

### 问题4：筛选条件过于严格 ⚠️ **设计问题**

**当前条件**（必须全部满足）：
1. 未来45天涨幅 > 50% ⭐⭐⭐⭐⭐（非常严格）
2. 过去60天涨幅 < 20%
3. RSI < 70
4. 量比 1.5-3.0
5. 至少2个预转信号

**问题**：
- 条件1本身就很严格（45天涨50%）
- 5个条件同时满足的概率极低
- 导致样本数量为0

---

## 🔧 修复方案

### 方案1：修复窗口大小（必须）

```python
# 修改前
window_size = 90

# 修改后
window_size = 105  # 过去60天 + 未来45天
```

### 方案2：修复量比处理（必须）

```python
# 修改前
if 'volume_ratio' not in df.columns:
    df['volume_ratio'] = 1.0

# 修改后
if 'volume_ratio' not in df.columns or df['volume_ratio'].isna().all():
    # 尝试从daily_basic获取
    if 'volume_ratio' in df.columns:
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
    else:
        # 如果确实没有，放宽条件或跳过此条件
        log.warning(f"{ts_code}: volume_ratio缺失，跳过量比检查")
```

### 方案3：适当放宽条件（建议）

**选项A：降低涨幅阈值**
```yaml
future_return_threshold: 50  # 改为 40 或 35
```

**选项B：放宽量比范围**
```yaml
volume_ratio_min: 1.5  # 改为 1.2
volume_ratio_max: 3.0  # 改为 4.0
```

**选项C：降低预转信号要求**
```python
# 至少2个信号 → 至少1个信号
return sum(signals) >= 1  # 改为 >= 1
```

---

## 📊 建议的修复优先级

1. **立即修复**：窗口大小（问题1）
2. **立即修复**：量比处理（问题2）
3. **建议调整**：适当放宽条件（问题4）
4. **监控**：技术指标完整性（问题3）

---

## 🎯 修复后的预期效果

- **修复问题1+2**：应该能找到一些样本
- **修复问题1+2+4**：样本数量应该显著增加
- **完全修复**：预计能找到数百到数千个样本（取决于数据范围）

