#!/bin/bash
# 跟踪止盈参数扫描
# 基于最优基线 MA5_cd2 + 4% close 止损，扫描不同跟踪止盈参数

set -e

START_DATE="20260105"
END_DATE="20260421"
STOP_LOSS_PCT="4.0"
STOP_LOSS_MODE="close"
MA_WINDOW="5"
MA_CONSECUTIVE_DAYS="2"

OUTPUT_BASE="data/results/trailing_stop_scan_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "跟踪止盈参数扫描"
echo "区间: $START_DATE ~ $END_DATE"
echo "基线: MA${MA_WINDOW}_cd${MA_CONSECUTIVE_DAYS} + ${STOP_LOSS_PCT}% ${STOP_LOSS_MODE}"
echo "输出: $OUTPUT_BASE"
echo "========================================"

# 参数组合: (trailing_stop_pct, trailing_stop_activate_pct, label)
combos=(
    "2.0 0.0 ts2_a0"
    "3.0 0.0 ts3_a0"
    "3.0 2.0 ts3_a2"
    "5.0 0.0 ts5_a0"
    "5.0 3.0 ts5_a3"
)

total=${#combos[@]}
count=0

for combo in "${combos[@]}"; do
    ts_pct=$(echo "$combo" | awk '{print $1}')
    ts_act=$(echo "$combo" | awk '{print $2}')
    label=$(echo "$combo" | awk '{print $3}')

    count=$((count + 1))
    echo ""
    echo "---------- [$count/$total] 跟踪止盈 ${ts_pct}% (激活阈值 ${ts_act}%) ----------"

    out_dir="$OUTPUT_BASE/$label"
    mkdir -p "$out_dir"

    python3 scripts/backtest_v232_v270_complementary.py \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --stop-loss-pct "$STOP_LOSS_PCT" \
        --stop-loss-mode "$STOP_LOSS_MODE" \
        --ma-window "$MA_WINDOW" \
        --ma-consecutive-days "$MA_CONSECUTIVE_DAYS" \
        --trailing-stop-pct "$ts_pct" \
        --trailing-stop-activate-pct "$ts_act" \
        --output-dir "$out_dir" \
        > "$out_dir/run.log" 2>&1

    echo "✓ 完成 $label"
done

echo ""
echo "========================================"
echo "所有回测完成！"
echo "========================================"
