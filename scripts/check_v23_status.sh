#!/bin/bash
# 检查v2.3.0评估任务状态

cd "$(dirname "$0")/.." || exit

echo "=================================="
echo "v2.3.0评估任务状态检查"
echo "=================================="
echo ""

# 检查当前时间
CURRENT_TIME=$(date +%H:%M)
echo "当前时间: $CURRENT_TIME"
echo ""

# 检查日志最后更新时间
LAST_LOG_TIME=$(tail -1 logs/aiquant.log | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | tail -1)
echo "最后日志时间: $LAST_LOG_TIME"
echo ""

# 检查是否有结果文件
echo "结果文件检查:"
if ls data/prediction/evaluation/*v2.3* 2>/dev/null | grep -q .; then
    echo "  ✓ 结果文件已生成"
    ls -lh data/prediction/evaluation/*v2.3* | tail -3
else
    echo "  ✗ 结果文件尚未生成"
fi
echo ""

# 检查任务是否完成
echo "任务状态:"
if tail -20 logs/aiquant.log | grep -q "结果已保存"; then
    echo "  ✓ 任务已完成"
elif tail -20 logs/aiquant.log | grep -q "评估数据获取完成"; then
    echo "  ⏳ 评估数据获取完成，正在分析..."
elif tail -20 logs/aiquant.log | grep -q "获取评估数据"; then
    echo "  ⏳ 正在获取评估数据..."
else
    echo "  ❓ 状态未知"
fi
echo ""

# 检查是否需要重启
if [ "$CURRENT_TIME" \> "23:06" ]; then
    echo "⚠️  已超过23:06，建议重启任务"
    echo ""
    echo "重启命令:"
    echo "  python scripts/evaluate_v23_full_market.py"
else
    echo "✓ 时间未到，继续等待..."
fi
echo ""

echo "=================================="

