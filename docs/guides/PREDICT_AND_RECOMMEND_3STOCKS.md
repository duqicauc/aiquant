# 预测并推荐3只股票（高评分+热门板块）使用指南

## 📋 流程概述

完整流程包括4个步骤：
1. 运行v2.7.0模型预测（生成全市场评分）
2. 运行v2.3.2模型预测（生成全市场评分）
3. 运行互补策略，输出Top3
4. 从互补策略结果中推荐3只股票（偏好热门板块+高评分）

## 🚀 快速开始

### 方法1：一键运行（推荐）

```bash
# 使用智能脚本（自动检查预测结果是否存在）
bash scripts/predict_and_recommend_3stocks_smart.sh 20260120
```

**特点**：
- 自动检查预测结果是否存在
- 如果已存在，跳过预测步骤
- 如果不存在，自动运行预测

### 方法2：分步运行

如果预测需要较长时间，可以分步运行：

#### 步骤1：v2.7.0预测（约需10-15分钟）

```bash
python scripts/predict_v270_ensemble_top50.py 20260120
```

**输出文件**：
- `data/prediction/results/v270_ensemble_all_20260120.csv`（全市场评分）

#### 步骤2：v2.3.2预测（约需10-15分钟）

```bash
python scripts/predict_v232_top10.py --date 20260120
```

**输出文件**：
- `data/prediction/results/v2.3.2_full_20260120.csv`（全市场评分）

#### 步骤3：互补策略（约需1-2分钟）

```bash
python scripts/combine_v232_v270.py \
  --date 20260120 \
  --strategy complementary \
  --top 3
```

**输出文件**：
- `data/prediction/results/v232_v270_complementary_20260120.csv`

#### 步骤4：推荐3只股票（约需30秒）

```bash
python scripts/recommend_2stocks_from_combined.py \
  --date 20260120 \
  --top-n 3 \
  --prefer-hot \
  --prefer-return
```

**输出文件**：
- `data/prediction/results/v232_v270_recommended_3stocks_20260120.csv`

## 📊 推荐标准

### 评分标准（偏好热门板块+高评分）

1. **综合得分**（50分）：v2.7.0和v2.3.2的综合评分
2. **风险等级**（15分）：低风险优先
3. **热门板块**（25分）：属于热门板块大幅加分
4. **收益潜力**（20分）：基于涨幅、预期收益、动量
5. **价格合理性**（5分）：5-20元最佳
6. **RSI状态**（5分）：40-60最佳

### 筛选要求

- ✅ 模型评分较高（综合得分>0.1）
- ✅ 符合热门板块（特高压、AI应用、存储等）
- ✅ 风险可控（优先低风险）
- ✅ 技术面健康（RSI适中）

## 📈 输出结果说明

### 推荐结果文件

`v232_v270_recommended_3stocks_20260120.csv`

包含字段：
- `ts_code`: 股票代码
- `name`: 股票名称
- `close`: 收盘价
- `source`: 来源模型（v2.7.0或v2.3.2）
- `risk_level`: 风险等级（low/medium/high）
- `dual_score`: 综合得分
- `selection_score`: 选择得分
- `v270_prob`: v2.7.0概率
- `hot_sectors`: 热门板块
- `rsi_6`: RSI指标
- `pct_chg`: 当日涨幅
- `weight`: 建议仓位（%）

## ⚙️ 参数调整

### 调整推荐数量

```bash
# 推荐3只（默认）
python scripts/recommend_2stocks_from_combined.py --date 20260120 --top-n 3

# 推荐5只
python scripts/recommend_2stocks_from_combined.py --date 20260120 --top-n 5
```

### 调整偏好设置

```bash
# 偏好热门板块+高收益（推荐）
python scripts/recommend_2stocks_from_combined.py --date 20260120 --top-n 3 --prefer-hot --prefer-return

# 只偏好热门板块（不特别追求高收益）
python scripts/recommend_2stocks_from_combined.py --date 20260120 --top-n 3 --prefer-hot --no-prefer-return

# 只偏好高收益（不特别偏好热门板块）
python scripts/recommend_2stocks_from_combined.py --date 20260120 --top-n 3 --no-prefer-hot --prefer-return

# 稳健型（都不偏好）
python scripts/recommend_2stocks_from_combined.py --date 20260120 --top-n 3 --no-prefer-hot --no-prefer-return
```

### 调整互补策略参数

```bash
# 增大v2.3.2候选池（更多热门板块机会）
python scripts/combine_v232_v270.py \
  --date 20260120 \
  --strategy complementary \
  --top 3 \
  --v232-top-n 150

# 启用基本面筛选
python scripts/combine_v232_v270.py \
  --date 20260120 \
  --strategy complementary \
  --top 3 \
  --fundamental
```

## ⏱️ 预计时间

- v2.7.0预测：10-15分钟（预测4966只股票）
- v2.3.2预测：10-15分钟（预测约4500只股票）
- 互补策略：1-2分钟
- 推荐3只股票：30秒

**总计**：约20-30分钟（如果预测结果已存在，只需2-3分钟）

## 💡 使用建议

1. **首次运行**：使用一键脚本，让它自动完成所有步骤
2. **日常使用**：如果预测结果已存在，直接运行步骤3和4
3. **后台运行**：预测步骤可以在后台运行
   ```bash
   nohup python scripts/predict_v270_ensemble_top50.py 20260120 > v270.log 2>&1 &
   nohup python scripts/predict_v232_top10.py --date 20260120 > v232.log 2>&1 &
   ```

## 🔍 验证结果

推荐结果应该满足：
- ✅ 3只股票都来自互补策略结果
- ✅ 至少2只属于热门板块
- ✅ 综合得分较高（>0.1）
- ✅ 风险等级以低风险为主

## 📝 示例输出

```
🏆 推荐3只股票
================================================================================

【推荐1】000551.SZ 创元科技
  综合得分: 0.1050
  选择得分: 73.25
  来源模型: v2.3.2
  风险等级: low
  热门板块: 特高压
  当日涨幅: +5.48%

【推荐2】600550.SH 保变电气
  综合得分: 0.1020
  选择得分: 70.10
  来源模型: v2.3.2
  风险等级: low
  热门板块: 特高压
  当日涨幅: +2.89%

【推荐3】600879.SH 航天电子
  综合得分: 0.1020
  选择得分: 68.50
  来源模型: v2.3.2
  风险等级: low
  热门板块: 特高压
  当日涨幅: -4.18%

📊 组合分析
================================================================================
来源分布:
  - v2.3.2: 3 只
风险分布:
  - low: 3 只
热门板块股票: 3 只
平均综合得分: 0.1030
```
