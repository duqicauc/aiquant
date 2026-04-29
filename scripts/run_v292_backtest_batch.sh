#!/bin/bash
set -e

PYTHON=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
ROOT=/Users/javaadu/Documents/GitHub/aiquant

echo "========================================"
echo "v2.9.2 批量回测: 2024Q4 + 2025Q1 + 2026Q1"
echo "========================================"

# 2024Q4
if [ ! -f "$ROOT/data/prediction/v292_stk_factor/predictions_20241231_all.csv" ]; then
  echo "[1/6] 生成 2024Q4 预测..."
  $PYTHON $ROOT/scripts/score_stocks_v292.py \
    --start-date 20241001 --end-date 20241231 \
    --output-dir data/prediction/v292_stk_factor
else
  echo "[1/6] 2024Q4 预测已存在，跳过"
fi

echo "[2/6] 回测 2024Q4..."
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20241001 --end-date 20241231 \
  --prediction-dir data/prediction/v292_stk_factor \
  --output-dir data/prediction/evaluation/v292_realistic_2024q4 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0

# 2025Q1 (2-3月，1月已有)
echo "[3/6] 生成 2025Q1 预测..."
$PYTHON $ROOT/scripts/score_stocks_v292.py \
  --start-date 20250101 --end-date 20250331 \
  --output-dir data/prediction/v292_stk_factor

echo "[4/6] 回测 2025Q1..."
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20250101 --end-date 20250331 \
  --prediction-dir data/prediction/v292_stk_factor \
  --output-dir data/prediction/evaluation/v292_realistic_2025q1 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0

# 2026Q1
echo "[5/6] 生成 2026Q1 预测..."
$PYTHON $ROOT/scripts/score_stocks_v292.py \
  --start-date 20260101 --end-date 20260331 \
  --output-dir data/prediction/v292_stk_factor

echo "[6/6] 回测 2026Q1..."
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20260101 --end-date 20260331 \
  --prediction-dir data/prediction/v292_stk_factor \
  --output-dir data/prediction/evaluation/v292_realistic_2026q1 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0

echo "========================================"
echo "全部完成!"
echo "========================================"
