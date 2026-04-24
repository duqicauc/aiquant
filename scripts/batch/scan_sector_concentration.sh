#!/bin/bash
# 板块集中度限制对比回测
# 基于最优基线 MA5_cd2 + 4% close，对比无限制 / ≤3 / ≤2

set -e

START_DATE="20260105"
END_DATE="20260421"
STOP_LOSS_PCT="4.0"
STOP_LOSS_MODE="close"
MA_WINDOW="5"
MA_CONSECUTIVE_DAYS="2"

OUTPUT_BASE="data/results/sector_concentration_scan_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "板块集中度限制对比回测"
echo "区间: $START_DATE ~ $END_DATE"
echo "基线: MA${MA_WINDOW}_cd${MA_CONSECUTIVE_DAYS} + ${STOP_LOSS_PCT}% ${STOP_LOSS_MODE}"
echo "输出: $OUTPUT_BASE"
echo "========================================"

# 参数组合: (max_sector_concentration, label)
combos=(
    "None baseline"
    "3 max3"
    "2 max2"
)

total=${#combos[@]}
count=0

for combo in "${combos[@]}"; do
    max_sec=$(echo "$combo" | awk '{print $1}')
    label=$(echo "$combo" | awk '{print $2}')

    count=$((count + 1))
    echo ""
    if [ "$max_sec" = "None" ]; then
        echo "---------- [$count/$total] 无板块集中度限制（基线） ----------"
        extra_args=""
    else
        echo "---------- [$count/$total] 单行业最多持仓 ${max_sec} 只 ----------"
        extra_args="--max-sector-concentration $max_sec"
    fi

    out_dir="$OUTPUT_BASE/$label"
    mkdir -p "$out_dir"

    python3 scripts/backtest_v232_v270_complementary.py \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --stop-loss-pct "$STOP_LOSS_PCT" \
        --stop-loss-mode "$STOP_LOSS_MODE" \
        --ma-window "$MA_WINDOW" \
        --ma-consecutive-days "$MA_CONSECUTIVE_DAYS" \
        --output-dir "$out_dir" \
        $extra_args \
        > "$out_dir/run.log" 2>&1

    echo "✓ 完成 $label"
done

echo ""
echo "========================================"
echo "所有回测完成！"
echo "========================================"
