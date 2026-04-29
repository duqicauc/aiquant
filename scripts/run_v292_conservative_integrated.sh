#!/bin/bash
set -e
PYTHON=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
ROOT=/Users/javaadu/Documents/GitHub/aiquant

echo "========================================"
echo "v2.9.2-conservative integrated 回测"
echo "========================================"

# 2024Q4 integrated
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20241001 --end-date 20241231 \
  --prediction-dir data/prediction/v292_conservative_2024q4 \
  --output-dir data/prediction/evaluation/v292_conservative_integrated_2024q4 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0 --enable-sector-filter

# 2025Q1 integrated
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20250101 --end-date 20250331 \
  --prediction-dir data/prediction/v292_conservative_2025q1 \
  --output-dir data/prediction/evaluation/v292_conservative_integrated_2025q1 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0 --enable-sector-filter

# 2026Q1 integrated
$PYTHON $ROOT/scripts/backtest_v291_realistic.py \
  --start-date 20260101 --end-date 20260331 \
  --prediction-dir data/prediction/v292_conservative_2026q1 \
  --output-dir data/prediction/evaluation/v292_conservative_integrated_2026q1 \
  --per-stock 300000 --top-n 10 --stop-loss 4.0 --enable-sector-filter

echo "========================================"
echo "全部回测完成!"
echo "========================================"
