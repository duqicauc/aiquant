#!/bin/bash
# 批量生成v2.3.2模型预测结果（1月5日、1月6日、1月7日、1月8日、1月22日）

echo "开始生成v2.3.2模型预测结果..."
echo "日期: 2026年1月5日、1月6日、1月7日、1月8日、1月22日"
echo ""

# 日期列表
dates=(
    "20260105"
    "20260106"
    "20260107"
    "20260108"
    "20260122"
)

# 记录开始时间
start_time=$(date +%s)

# 逐个生成预测
for date in "${dates[@]}"; do
    echo "=========================================="
    echo "处理日期: $date"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    python scripts/predict_v232_top10.py --date "$date"

    if [ $? -eq 0 ]; then
        echo "✓ $date 预测完成"
    else
        echo "✗ $date 预测失败"
    fi

    echo ""
done

# 计算总耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "=========================================="
echo "所有预测完成！"
echo "总耗时: ${hours}小时${minutes}分钟${seconds}秒"
echo "=========================================="
echo ""
echo "完成！"
