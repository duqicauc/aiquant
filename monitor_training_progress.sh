#!/bin/bash
# 实时监控左侧潜力牛股模型训练进度

LOG_FILE="logs/aiquant.log"

echo "======================================================================"
echo "📊 左侧潜力牛股模型训练 - 实时进度监控"
echo "======================================================================"
echo "按 Ctrl+C 退出监控"
echo ""

# 持续监控日志
tail -f "$LOG_FILE" | grep --line-buffered -E "处理样本|特征提取完成|开始训练模型|训练完成|模型保存|ERROR|Exception" | while read line; do
    timestamp=$(echo "$line" | grep -oE '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
    
    if echo "$line" | grep -q "处理样本"; then
        progress=$(echo "$line" | grep -oE '处理样本 \d+/\d+')
        echo "[$timestamp] 📊 $progress"
    elif echo "$line" | grep -q "特征提取完成"; then
        echo "[$timestamp] ✅ 特征提取完成！"
    elif echo "$line" | grep -q "开始训练模型"; then
        echo "[$timestamp] 🚀 开始XGBoost模型训练..."
    elif echo "$line" | grep -q "训练完成"; then
        echo "[$timestamp] 🎉 模型训练完成！"
    elif echo "$line" | grep -q "ERROR\|Exception"; then
        echo "[$timestamp] ❌ 错误: $line"
    fi
done
