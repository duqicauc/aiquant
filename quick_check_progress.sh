#!/bin/bash
# 快速查看训练进度

LOG_FILE="logs/aiquant.log"

echo "======================================================================"
echo "📊 左侧潜力牛股模型训练 - 快速进度检查"
echo "======================================================================"

# 检查进程
if pgrep -f "train_left_breakout" > /dev/null; then
    echo "✅ 训练进程运行中"
else
    echo "❌ 训练进程未运行"
    exit 1
fi

echo ""

# 最新进度
echo "📈 最新进度:"
tail -1000 "$LOG_FILE" | grep "处理样本" | tail -1 | sed 's/.*INFO.*| //'

echo ""

# 检查阶段
if tail -500 "$LOG_FILE" | grep -q "特征提取完成"; then
    echo "✅ 特征提取已完成"
    if tail -200 "$LOG_FILE" | grep -q "开始训练模型"; then
        echo "🔄 正在进行模型训练"
    fi
else
    echo "🔄 正在进行特征提取"
fi

echo ""

# 最近错误
ERRORS=$(tail -500 "$LOG_FILE" | grep -E "ERROR|Exception" | wc -l | tr -d ' ')
if [ "$ERRORS" -gt 0 ]; then
    echo "⚠️  最近发现 $ERRORS 个错误/异常"
    tail -500 "$LOG_FILE" | grep -E "ERROR|Exception" | tail -2
else
    echo "✅ 未发现错误"
fi

echo ""
echo "======================================================================"
echo "💡 实时监控命令:"
echo "   tail -f logs/aiquant.log | grep '处理样本'"
echo "   或运行: ./monitor_training_progress.sh"
echo "======================================================================"
