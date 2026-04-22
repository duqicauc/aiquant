#!/bin/bash
# -*- coding: utf-8 -*-
"""
完整流程：预测 + 综合选股推荐3只股票

步骤：
1. 运行v2.7.0模型预测
2. 运行v2.3.2模型预测
3. 运行互补策略，输出Top3
4. 从互补策略结果中推荐3只股票（偏好热门板块+高评分）
"""

set -e

DATE=${1:-$(date +%Y%m%d)}
echo "=========================================="
echo "预测和推荐流程 - $DATE"
echo "=========================================="
echo ""

# 步骤1: v2.7.0预测
echo "步骤1: 运行v2.7.0模型预测..."
python scripts/predict_v270_ensemble_top50.py $DATE
echo ""

# 步骤2: v2.3.2预测
echo "步骤2: 运行v2.3.2模型预测..."
python scripts/predict_v232_top10.py --date $DATE
echo ""

# 步骤3: 互补策略（输出Top3）
echo "步骤3: 运行互补策略，输出Top3..."
python scripts/combine_v232_v270.py \
  --date $DATE \
  --strategy complementary \
  --top 3
echo ""

# 步骤4: 推荐3只股票（偏好热门板块+高评分）
echo "步骤4: 推荐3只股票（偏好热门板块+高评分）..."
python scripts/recommend_2stocks_from_combined.py \
  --date $DATE \
  --top-n 3 \
  --prefer-hot \
  --prefer-return
echo ""

echo "=========================================="
echo "✅ 完成！推荐结果已保存"
echo "=========================================="
