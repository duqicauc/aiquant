# combine_v232_v270.py 使用指南与优化建议

## 📋 代码审视结果

### ✅ 代码优点

1. **策略丰富**：提供4种不同的结合策略
2. **功能完整**：支持基本面筛选、风险分层、热门板块识别
3. **容错机制**：有备选方案和降级处理
4. **参数灵活**：支持多种配置选项

### ⚠️ 发现的问题与优化建议

#### 1. 文件路径检查

**问题**：需要确保预测结果文件存在

**建议**：在使用前先运行两个模型的预测脚本

```bash
# 1. 先运行v2.3.2预测
python scripts/predict_v232_top10.py --date 20260116

# 2. 再运行v2.7.0预测（生成all文件）
python scripts/predict_v270_ensemble_top50.py --date 20260116

# 3. 然后运行结合脚本
python scripts/combine_v232_v270.py --date 20260116 --strategy complementary
```

#### 2. 概念信息获取效率

**问题**：`get_concept_info`函数逐个获取概念信息，对于100只股票会很慢

**优化建议**：
- 考虑批量获取或使用缓存
- 对于互补策略，可以只获取v2.3.2候选池的概念信息，而不是全部

#### 3. DataManager重复创建

**问题**：在多个函数中重复创建DataManager实例

**优化建议**：在main函数中创建一次，传递给各个策略函数

#### 4. 基本面筛选配置重复

**问题**：基本面筛选配置在多处重复

**优化建议**：提取为常量或配置文件

#### 5. 互补策略的列对齐逻辑

**问题**：合并数据时的列对齐逻辑较复杂，可能出错

**优化建议**：简化逻辑，使用更明确的列名映射

---

## 🚀 推荐使用方式

### 方式1：日常使用（推荐）

```bash
# 使用互补策略，默认参数
python scripts/combine_v232_v270.py \
  --date 20260116 \
  --strategy complementary \
  --top 10
```

**适用场景**：
- 日常选股
- 需要平衡稳定性和收益性
- 希望自动识别热门板块

### 方式2：保守型配置

```bash
# 启用基本面筛选，严格控制风险
python scripts/combine_v232_v270.py \
  --date 20260116 \
  --strategy complementary \
  --top 10 \
  --base-top-n 50 \
  --v232-top-n 50 \
  --max-high-risk 1 \
  --max-medium-risk 3 \
  --fundamental
```

**适用场景**：
- 风险偏好较低
- 需要基本面保障
- 追求稳健收益

### 方式3：激进型配置

```bash
# 更多依赖v2.3.2，捕捉热门板块
python scripts/combine_v232_v270.py \
  --date 20260116 \
  --strategy complementary \
  --top 10 \
  --base-top-n 30 \
  --v232-top-n 150 \
  --max-high-risk 5 \
  --max-medium-risk 7
```

**适用场景**：
- 风险偏好较高
- 追求高收益
- 能承受较大波动

### 方式4：交集策略（最保守）

```bash
# 只选择两个模型都看好的股票
python scripts/combine_v232_v270.py \
  --date 20260116 \
  --strategy intersection \
  --top-n 100 \
  --top 10 \
  --fundamental
```

**适用场景**：
- 极度保守
- 需要双重验证
- 追求最高确定性

### 方式5：加权策略（简单直接）

```bash
# 自定义权重
python scripts/combine_v232_v270.py \
  --date 20260116 \
  --strategy weighted \
  --w232 0.3 \
  --w270 0.7 \
  --top 10
```

**适用场景**：
- 需要自定义模型权重
- 简单直接的结合方式
- 不需要热门板块识别

---

## 📊 策略选择建议

| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **complementary** | 日常使用（推荐） | 兼顾稳定性与收益性，自动识别热门板块，风险分层 | 需要Tushare API，执行时间较长 |
| **intersection** | 极度保守 | 两个模型都看好，风险最低 | 可能错过热门板块，交集可能为空 |
| **weighted** | 简单直接 | 简单易用，可自定义权重 | 无法识别热门板块，无风险分层 |
| **rank** | 排名敏感 | 综合考虑两个模型排名 | 无法识别热门板块，无风险分层 |

---

## ⚙️ 参数调优建议

### base-top-n（v2.7.0基础池数量）

- **默认值**：50
- **建议范围**：30-100
- **调优原则**：
  - 保守型：增大到70-100
  - 激进型：减小到30-40

### v232-top-n（v2.3.2候选池数量）

- **默认值**：100
- **建议范围**：50-200
- **调优原则**：
  - 保守型：减小到50-80
  - 激进型：增大到150-200

### max-high-risk（高风险股票数）

- **默认值**：3
- **建议范围**：0-5
- **调优原则**：
  - 保守型：0-1
  - 平衡型：2-3
  - 激进型：4-5

### max-medium-risk（中风险股票数）

- **默认值**：5
- **建议范围**：3-10
- **调优原则**：
  - 保守型：3-5
  - 平衡型：5-7
  - 激进型：8-10

---

## 🔧 常见问题

### Q1: 提示"预测结果不存在"

**原因**：没有运行对应的预测脚本

**解决**：
```bash
# 先运行预测脚本
python scripts/predict_v232_top10.py --date 20260116
python scripts/predict_v270_ensemble_top50.py --date 20260116
```

### Q2: 互补策略执行很慢

**原因**：需要获取概念信息和热点板块数据

**解决**：
- 减少`v232-top-n`参数（如从100减到50）
- 使用其他策略（如weighted或intersection）
- 检查Tushare API连接

### Q3: 没有识别到热门板块

**原因**：
1. Tushare API积分不足或连接失败
2. 当日确实没有热门板块
3. 候选股票不在热门板块中

**解决**：
- 检查Tushare API配置
- 查看日志中的备选方案提示
- 尝试增大`v232-top-n`参数

### Q4: 交集策略返回空结果

**原因**：两个模型的TopN没有交集

**解决**：
- 增大`--top-n`参数（如从100增到200）
- 使用其他策略（如weighted或complementary）

---

## 📈 性能优化建议

### 1. 批量获取概念信息

如果经常使用，可以考虑：
- 缓存概念信息
- 批量获取（如果Tushare支持）
- 只获取需要的股票概念

### 2. 并行处理

对于大量股票，可以考虑：
- 使用多线程获取概念信息
- 并行处理多个策略

### 3. 结果缓存

可以缓存：
- 热点板块数据（每日更新一次）
- 概念信息（相对稳定）

---

## 🎯 最佳实践

1. **每日工作流**：
   ```bash
   # 1. 运行预测
   python scripts/predict_v232_top10.py --date $(date +%Y%m%d)
   python scripts/predict_v270_ensemble_top50.py --date $(date +%Y%m%d)

   # 2. 结合结果
   python scripts/combine_v232_v270.py --date $(date +%Y%m%d) --strategy complementary
   ```

2. **参数调优**：
   - 先使用默认参数
   - 根据实际效果调整
   - 记录不同参数下的表现

3. **结果验证**：
   - 对比不同策略的结果
   - 关注风险分布
   - 验证热门板块识别准确性

4. **定期评估**：
   - 每周回顾策略表现
   - 根据市场情况调整参数
   - 更新热门板块关键词（如果使用备选方案）

---

## 📝 使用示例

### 示例1：快速获取推荐（默认配置）

```bash
python scripts/combine_v232_v270.py --date 20260116
```

### 示例2：保守型配置

```bash
python scripts/combine_v232_v270.py \
  --date 20260116 \
  --strategy complementary \
  --top 10 \
  --base-top-n 70 \
  --v232-top-n 60 \
  --max-high-risk 1 \
  --max-medium-risk 3 \
  --fundamental
```

### 示例3：对比所有策略

```bash
python scripts/combine_v232_v270.py \
  --date 20260116 \
  --strategy all \
  --top 10
```

### 示例4：自定义权重

```bash
python scripts/combine_v232_v270.py \
  --date 20260116 \
  --strategy weighted \
  --w232 0.3 \
  --w270 0.7 \
  --top 10
```

---

## 🔗 相关文档

- [互补策略详细说明](./V232_V270_COMPLEMENTARY_STRATEGY.md)
- [v2.3.2模型说明](../reference/V232_PREDICTION_LOGIC.md)
- [v2.7.0模型说明](../reference/V270_PREDICTION_STATUS.md)
- [基本面筛选指南](./FUNDAMENTAL_SCREENING_GUIDE.md)
