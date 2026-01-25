# 预测并推荐3只股票（高评分+热门板块）

## 🎯 目标

推荐3只股票，要求：
- ✅ 模型评分较高
- ✅ 符合热门板块

## 🚀 快速运行

### 一键运行（推荐）

```bash
bash scripts/predict_and_recommend_3stocks_smart.sh 20260120
```

**说明**：
- 自动检查预测结果是否存在
- 如果不存在，自动运行预测（需要20-30分钟）
- 如果已存在，直接运行推荐（只需2-3分钟）

### 分步运行

#### 步骤1：v2.7.0预测（约10-15分钟）

```bash
python scripts/predict_v270_ensemble_top50.py 20260120
```

**检查结果**：
```bash
ls -lh data/prediction/results/v270_ensemble_all_20260120.csv
```

#### 步骤2：v2.3.2预测（约10-15分钟）

```bash
python scripts/predict_v232_top10.py --date 20260120
```

**检查结果**：
```bash
ls -lh data/prediction/results/v2.3.2_full_20260120.csv
```

#### 步骤3：互补策略（约1-2分钟）

```bash
python scripts/combine_v232_v270.py \
  --date 20260120 \
  --strategy complementary \
  --top 3
```

#### 步骤4：推荐3只股票（约30秒）

```bash
python scripts/recommend_2stocks_from_combined.py \
  --date 20260120 \
  --top-n 3 \
  --prefer-hot \
  --prefer-return
```

## 📊 推荐标准

推荐脚本会优先选择：
1. **综合得分高**（v2.7.0和v2.3.2的综合评分）
2. **热门板块**（特高压、AI应用、存储、电力等）
3. **风险可控**（低风险优先）
4. **收益潜力**（涨幅、动量等）

## 📁 输出文件

- `v232_v270_complementary_20260120.csv` - 互补策略Top3结果
- `v232_v270_recommended_3stocks_20260120.csv` - 最终推荐的3只股票

## ⚠️ 注意事项

1. **预测时间**：两个模型的预测各需10-15分钟，请耐心等待
2. **网络要求**：需要Tushare API权限和网络连接
3. **积分要求**：推荐≥6000积分以获取同花顺热榜数据

## 💡 后台运行建议

如果预测需要较长时间，可以在后台运行：

```bash
# 后台运行v2.7.0预测
nohup python scripts/predict_v270_ensemble_top50.py 20260120 > logs/v270_20260120.log 2>&1 &

# 后台运行v2.3.2预测
nohup python scripts/predict_v232_top10.py --date 20260120 > logs/v232_20260120.log 2>&1 &

# 查看进度
tail -f logs/v270_20260120.log
tail -f logs/v232_20260120.log
```

预测完成后，再运行步骤3和4。
