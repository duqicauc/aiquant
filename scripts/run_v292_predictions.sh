#!/bin/bash
set -e
PYTHON=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
ROOT=/Users/javaadu/Documents/GitHub/aiquant

echo "========================================"
echo "v2.9.2 批量预测: 2024Q4 + 2025Q1 + 2026Q1"
echo "========================================"

# 2024Q4
echo "[1/3] 生成 2024Q4 预测..."
$PYTHON $ROOT/scripts/score_stocks_v292.py \
  --start-date 20241001 --end-date 20241231 \
  --output-dir data/prediction/v292_stk_factor

# 2025Q1
echo "[2/3] 生成 2025Q1 预测..."
$PYTHON $ROOT/scripts/score_stocks_v292.py \
  --start-date 20250101 --end-date 20250331 \
  --output-dir data/prediction/v292_stk_factor

# 2026Q1
echo "[3/3] 生成 2026Q1 预测..."
$PYTHON $ROOT/scripts/score_stocks_v292.py \
  --start-date 20260101 --end-date 20260331 \
  --output-dir data/prediction/v292_stk_factor

echo "========================================"
echo "预测完成!"
echo "========================================"
