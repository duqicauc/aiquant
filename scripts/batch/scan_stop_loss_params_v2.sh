#!/bin/bash
# 止损参数扫描批量回测 v2（修复版）
# 测试范围: 4.0%, 4.5%, 5.0%, 6.0%

PROJECT_DIR="/Users/javaadu/Documents/GitHub/aiquant"
cd "$PROJECT_DIR"

START_DATE="20260105"
END_DATE="20260421"
OUTPUT_BASE="$PROJECT_DIR/data/results/stop_loss_scan_20260422_140501"
mkdir -p "$OUTPUT_BASE"

PARAMS=(4.0 4.5 5.0 6.0)
TOTAL=${#PARAMS[@]}

echo "========================================"
echo "止损参数扫描回测 (续跑)"
echo "区间: $START_DATE ~ $END_DATE"
echo "参数: ${PARAMS[*]}"
echo "输出: $OUTPUT_BASE"
echo "========================================"
echo ""

for i in "${!PARAMS[@]}"; do
    PCT="${PARAMS[$i]}"
    IDX=$((i + 1))
    echo "---------- [$IDX/$TOTAL] 止损参数 = ${PCT}% ----------"

    mkdir -p "$OUTPUT_BASE/sl_${PCT}"

    python3 scripts/backtest_v232_v270_complementary.py \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --stop-loss-mode close \
        --stop-loss-pct "$PCT" \
        --initial-cash 10000000 \
        --output-dir "$OUTPUT_BASE/sl_${PCT}" \
        > "$OUTPUT_BASE/sl_${PCT}/run.log" 2>&1

    echo "✓ 完成 ${PCT}%"
    echo ""
done

echo "========================================"
echo "所有回测完成！"
echo "========================================"
