#!/bin/bash
# 等待预测完成并自动推荐2支股票

LOG_FILE="logs/predict_v270_20260116.log"
PREDICT_FILE="data/prediction/results/v270_ensemble_top10_20260116.csv"
MAX_WAIT=3600  # 最多等待1小时

echo "等待预测完成..."
echo "监控日志: $LOG_FILE"
echo ""

# 等待预测完成
wait_count=0
while [ $wait_count -lt $MAX_WAIT ]; do
    if [ -f "$PREDICT_FILE" ]; then
        echo "预测完成！文件已生成: $PREDICT_FILE"
        break
    fi
    
    # 检查日志中是否有完成标记
    if grep -q "预测完成" "$LOG_FILE" 2>/dev/null; then
        echo "预测完成！"
        break
    fi
    
    sleep 10
    wait_count=$((wait_count + 10))
    
    if [ $((wait_count % 60)) -eq 0 ]; then
        echo "已等待 $((wait_count / 60)) 分钟..."
        tail -5 "$LOG_FILE" 2>/dev/null | grep "进度" || echo "等待中..."
    fi
done

if [ ! -f "$PREDICT_FILE" ]; then
    echo "警告: 预测文件未生成，可能还在运行中"
    echo "请手动检查: $LOG_FILE"
    exit 1
fi

echo ""
echo "开始推荐2支股票..."
python scripts/recommend_2stocks_v270.py

echo ""
echo "完成！请查看推荐结果:"
echo "  data/prediction/trading_plan/v270_recommended_2stocks_*.csv"
