#!/bin/bash
# 等待训练完成并运行预测脚本

echo "=========================================="
echo "等待模型训练完成..."
echo "=========================================="

# 等待训练进程完成
while pgrep -f "train_left_breakout_model.py" > /dev/null; do
    echo "$(date '+%H:%M:%S') - 训练进行中，等待完成..."
    sleep 60
done

echo ""
echo "=========================================="
echo "训练已完成，开始预测..."
echo "=========================================="

# 检查模型文件是否存在
if [ ! -f "data/models/left_breakout/left_breakout_v1.joblib" ]; then
    echo "⚠️  模型文件不存在，检查其他版本..."
    ls -lh data/models/left_breakout/*.joblib 2>/dev/null | tail -1
fi

# 运行预测脚本，使用2025-12-25的数据，输出Top50
python scripts/predict_left_breakout.py \
    --date 20251225 \
    --top-n 50 \
    --min-prob 0.1

echo ""
echo "=========================================="
echo "预测完成！"
echo "=========================================="

