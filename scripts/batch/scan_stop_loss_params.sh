#!/bin/bash
# 止损参数扫描批量回测
# 测试范围: 3%, 3.5%, 4%, 4.5%, 5%, 6%

set -e

PROJECT_DIR="/Users/javaadu/Documents/GitHub/aiquant"
cd "$PROJECT_DIR"

START_DATE="20260105"
END_DATE="20260421"
OUTPUT_BASE="$PROJECT_DIR/data/results/stop_loss_scan_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

PARAMS=(3.0 3.5 4.0 4.5 5.0 6.0)
TOTAL=${#PARAMS[@]}

echo "========================================"
echo "止损参数扫描回测"
echo "区间: $START_DATE ~ $END_DATE"
echo "参数: ${PARAMS[*]}"
echo "输出: $OUTPUT_BASE"
echo "========================================"
echo ""

for i in "${!PARAMS[@]}"; do
    PCT="${PARAMS[$i]}"
    IDX=$((i + 1))
    echo "---------- [$IDX/$TOTAL] 止损参数 = ${PCT}% ----------"

    python3 scripts/backtest_v232_v270_complementary.py \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --stop-loss-mode close \
        --stop-loss-pct "$PCT" \
        --initial-cash 10000000 \
        --output-dir "$OUTPUT_BASE/sl_${PCT}" \
        2>&1 | tee "$OUTPUT_BASE/sl_${PCT}/run.log"

    echo ""
done

echo "========================================"
echo "所有回测完成，开始汇总..."
echo "========================================"

# 提取关键指标汇总
SUMMARY_FILE="$OUTPUT_BASE/summary_comparison.md"
cat > "$SUMMARY_FILE" << 'HEADER'
# 止损参数扫描对比报告

## 测试参数
- 回测区间: 2026-01-05 ~ 2026-04-21
- 止损模式: close（收盘价触发）
- 扫描参数: 3.0%, 3.5%, 4.0%, 4.5%, 5.0%, 6.0%

## 关键指标对比

| 止损参数 | 总收益% | 年化收益% | 最大回撤% | 胜率% | 盈利因子 | 平均盈亏比 | 夏普比率 | 买入笔数 | 卖出笔数 | 止损笔数 |
|----------|---------|-----------|-----------|-------|----------|------------|----------|----------|----------|----------|
HEADER

for PCT in "${PARAMS[@]}"; do
    REPORT="$OUTPUT_BASE/sl_${PCT}/backtest_report_${START_DATE}_${END_DATE}_sl_close.md"
    if [ -f "$REPORT" ]; then
        # 提取关键指标（简单grep）
        TOTAL_RETURN=$(grep -oP '(?<=收益率: \+)[0-9.]+' "$REPORT" 2>/dev/null || echo "N/A")
        MAX_DD=$(grep -oP '(?<=最大回撤: -)[0-9.]+' "$REPORT" 2>/dev/null || echo "N/A")
        WIN_RATE=$(grep -oP '(?<=胜率: )[0-9.]+' "$REPORT" 2>/dev/null || echo "N/A")
        PF=$(grep -oP '(?<=盈利因子\(总盈利/总亏损\)：)[0-9.]+' "$REPORT" 2>/dev/null || echo "N/A")
        AVG_RATIO=$(grep -oP '(?<=平均盈利/平均亏损)：)[0-9.]+' "$REPORT" 2>/dev/null || echo "N/A")
        SHARPE=$(grep -oP '(?<=夏普比率\(年化\) \| )[0-9.]+' "$REPORT" 2>/dev/null || echo "N/A")
        ANNUAL=$(grep -oP '(?<=年化收益率 \| )[0-9.]+' "$REPORT" 2>/dev/null || echo "N/A")
        BUYS=$(grep -oP '(?<=买入次数: )[0-9]+' "$REPORT" 2>/dev/null || echo "N/A")
        SELLS=$(grep -oP '(?<=卖出次数: )[0-9]+' "$REPORT" 2>/dev/null || echo "N/A")
        SL_COUNT=$(grep -c '单标亏损达' "$REPORT" 2>/dev/null || echo "0")

        echo "| ${PCT}% | ${TOTAL_RETURN}% | ${ANNUAL}% | -${MAX_DD}% | ${WIN_RATE}% | ${PF} | ${AVG_RATIO} | ${SHARPE} | ${BUYS} | ${SELLS} | ${SL_COUNT} |" >> "$SUMMARY_FILE"
    else
        echo "| ${PCT}% | 报告缺失 | — | — | — | — | — | — | — | — | — |" >> "$SUMMARY_FILE"
    fi
done

cat >> "$SUMMARY_FILE" << 'FOOTER'

## 分析建议

- **盈利因子 > 1.05** 且 **最大回撤 < 12%** 的参数组合为推荐区间
- 止损过紧（3%）可能增加止损频率，降低平均盈亏比
- 止损过松（6%）可能单笔亏损扩大，但减少误杀趋势票

FOOTER

echo ""
echo "✅ 汇总报告: $SUMMARY_FILE"
echo "✅ 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
