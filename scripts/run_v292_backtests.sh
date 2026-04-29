#!/bin/bash
set -e
PYTHON=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
ROOT=/Users/javaadu/Documents/GitHub/aiquant
PRED_DIR=data/prediction/v292_stk_factor

echo "========================================"
echo "v2.9.2 批量回测 (realistic + sector-filter)"
echo "========================================"

# --- 版本 A: Realistic (标准版) ---
echo "[A1] 回测 2024Q4 realistic..."
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20241001 --end-date 20241231 \
  --prediction-dir $PRED_DIR \
  --output-dir data/prediction/evaluation/v292_realistic_2024q4 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0

echo "[A2] 回测 2025Q1 realistic..."
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20250101 --end-date 20250331 \
  --prediction-dir $PRED_DIR \
  --output-dir data/prediction/evaluation/v292_realistic_2025q1 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0

echo "[A3] 回测 2026Q1 realistic..."
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20260101 --end-date 20260331 \
  --prediction-dir $PRED_DIR \
  --output-dir data/prediction/evaluation/v292_realistic_2026q1 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0

# --- 版本 B: Realistic + Sector Filter (integrated 版) ---
echo "[B1] 回测 2024Q4 integrated..."
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20241001 --end-date 20241231 \
  --prediction-dir $PRED_DIR \
  --output-dir data/prediction/evaluation/v292_integrated_2024q4 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0 --enable-sector-filter

echo "[B2] 回测 2025Q1 integrated..."
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20250101 --end-date 20250331 \
  --prediction-dir $PRED_DIR \
  --output-dir data/prediction/evaluation/v292_integrated_2025q1 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0 --enable-sector-filter

echo "[B3] 回测 2026Q1 integrated..."
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20260101 --end-date 20260331 \
  --prediction-dir $PRED_DIR \
  --output-dir data/prediction/evaluation/v292_integrated_2026q1 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0 --enable-sector-filter

echo "========================================"
echo "全部回测完成!"
echo "========================================"
