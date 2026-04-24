#!/bin/bash
# MA 退出参数扫描批量回测 v2（修复版）
# 扫描范围: MA周期 3/5/7/10 × 连续跌破天数 1/2/3

PROJECT_DIR="/Users/javaadu/Documents/GitHub/aiquant"
cd "$PROJECT_DIR"

START_DATE="20260105"
END_DATE="20260421"
OUTPUT_BASE="$PROJECT_DIR/data/results/ma_exit_scan_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

MA_WINDOWS=(3 5 7 10)
CONSECUTIVE_DAYS=(1 2 3)
TOTAL=12

echo "========================================"
echo "MA 退出参数扫描回测 v2（修复版）"
echo "区间: $START_DATE ~ $END_DATE"
echo "止损: 4% close"
echo "扫描参数: MA周期=${MA_WINDOWS[*]}, 连续天数=${CONSECUTIVE_DAYS[*]}"
echo "总组合数: $TOTAL"
echo "输出: $OUTPUT_BASE"
echo "========================================"
echo ""

IDX=0
for MA in "${MA_WINDOWS[@]}"; do
    for CD in "${CONSECUTIVE_DAYS[@]}"; do
        IDX=$((IDX + 1))
        echo "---------- [$IDX/$TOTAL] MA${MA} + 连续${CD}天 ----------"

        mkdir -p "$OUTPUT_BASE/ma${MA}_cd${CD}"

        python3 scripts/backtest_v232_v270_complementary.py \
            --start-date "$START_DATE" \
            --end-date "$END_DATE" \
            --stop-loss-mode close \
            --stop-loss-pct 4.0 \
            --ma-window "$MA" \
            --ma-consecutive-days "$CD" \
            --initial-cash 10000000 \
            --output-dir "$OUTPUT_BASE/ma${MA}_cd${CD}" \
            > "$OUTPUT_BASE/ma${MA}_cd${CD}/run.log" 2>&1

        echo "✓ 完成 MA${MA}_cd${CD}"
        echo ""
    done
done

echo "========================================"
echo "所有回测完成！"
echo "========================================"
