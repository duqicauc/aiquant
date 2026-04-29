#!/bin/bash
set -e
PYTHON=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
ROOT=/Users/javaadu/Documents/GitHub/aiquant
PRED_DIR=data/prediction/v292_stk_factor

echo "========================================"
echo "v2.9.2 重新预测 + realistic 回测"
echo "========================================"

# 重新生成三个季度预测（使用修复后的 prob_raw 排序 + Platt Scaling）
echo "[1/3] 重新生成 2024Q4 预测..."
$PYTHON $ROOT/scripts/score_stocks_v292.py \
  --start-date 20241001 --end-date 20241231 \
  --output-dir $PRED_DIR

echo "[2/3] 重新生成 2025Q1 预测..."
$PYTHON $ROOT/scripts/score_stocks_v292.py \
  --start-date 20250101 --end-date 20250331 \
  --output-dir $PRED_DIR

echo "[3/3] 重新生成 2026Q1 预测..."
$PYTHON $ROOT/scripts/score_stocks_v292.py \
  --start-date 20260101 --end-date 20260331 \
  --output-dir $PRED_DIR

# 清除旧回测结果
rm -rf $ROOT/data/prediction/evaluation/v292_realistic_2024q4
rm -rf $ROOT/data/prediction/evaluation/v292_realistic_2025q1
rm -rf $ROOT/data/prediction/evaluation/v292_realistic_2026q1

# 重新回测
echo "[4/6] 回测 2024Q4..."
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20241001 --end-date 20241231 \
  --prediction-dir $PRED_DIR \
  --output-dir data/prediction/evaluation/v292_realistic_2024q4 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0

echo "[5/6] 回测 2025Q1..."
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20250101 --end-date 20250331 \
  --prediction-dir $PRED_DIR \
  --output-dir data/prediction/evaluation/v292_realistic_2025q1 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0

echo "[6/6] 回测 2026Q1..."
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20260101 --end-date 20260331 \
  --prediction-dir $PRED_DIR \
  --output-dir data/prediction/evaluation/v292_realistic_2026q1 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0

echo "========================================"
echo "完成!"
echo "========================================"
