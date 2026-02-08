#!/bin/bash
# 运行互补策略三版本回测（无4%止损 / 收盘价4%止损 / 日内最低价4%止损）并生成对比分析
# 用法: ./scripts/run_compare_stop_loss_modes.sh [开始日期] [结束日期]
# 默认: 20260105 20260206

set -e
cd "$(dirname "$0")/.."
START="${1:-20260105}"
END="${2:-20260206}"
echo "回测区间: ${START} ~ ${END}"
echo "运行三版本对比（约需数分钟）..."
python scripts/compare_stop_loss_modes.py --start-date "$START" --end-date "$END"
echo "完成。报告: data/prediction/results/compare_stop_loss_modes_${START}_${END}.md"
