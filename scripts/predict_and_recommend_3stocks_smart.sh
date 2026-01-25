#!/bin/bash
# -*- coding: utf-8 -*-
"""
智能流程：预测 + 综合选股推荐3只股票

自动检查预测结果是否存在，如果不存在才运行预测
"""

set -e

DATE=${1:-$(date +%Y%m%d)}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/data/prediction/results"

echo "=========================================="
echo "预测和推荐流程 - $DATE"
echo "=========================================="
echo ""

# 检查v2.7.0预测结果
V270_FILE="$RESULTS_DIR/v270_ensemble_all_${DATE}.csv"
if [ ! -f "$V270_FILE" ]; then
    echo "步骤1: 运行v2.7.0模型预测（结果不存在）..."
    python scripts/predict_v270_ensemble_top50.py $DATE
    echo ""
else
    echo "步骤1: v2.7.0预测结果已存在，跳过"
    echo "  文件: $V270_FILE"
    echo ""
fi

# 检查v2.3.2预测结果
V232_FILE="$RESULTS_DIR/v2.3.2_full_${DATE}.csv"
if [ ! -f "$V232_FILE" ]; then
    echo "步骤2: 运行v2.3.2模型预测（结果不存在）..."
    python scripts/predict_v232_top10.py --date $DATE
    echo ""
else
    echo "步骤2: v2.3.2预测结果已存在，跳过"
    echo "  文件: $V232_FILE"
    echo ""
fi

# 步骤3: 互补策略（输出Top3）
echo "步骤3: 运行互补策略，输出Top3（偏好热门板块）..."
python scripts/combine_v232_v270.py \
  --date $DATE \
  --strategy complementary \
  --top 3 \
  --base-top-n 50 \
  --v232-top-n 100
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
echo ""
echo "推荐结果文件:"
echo "  - 互补策略结果: $RESULTS_DIR/v232_v270_complementary_${DATE}.csv"
echo "  - 推荐3只股票: $RESULTS_DIR/v232_v270_recommended_3stocks_${DATE}.csv"
