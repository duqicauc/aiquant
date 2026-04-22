#!/bin/bash
# 批量生成v2.7.0模型预测结果（1月5日到1月15日）

echo "开始生成v2.7.0模型预测结果..."
echo "日期范围: 2026年1月5日到1月15日（排除周末）"
echo ""

# 日期列表（排除周末）
dates=(
    "20260105"
    "20260106"
    "20260107"
    "20260108"
    "20260109"
    "20260112"
    "20260113"
    "20260114"
    "20260115"
)

# 记录开始时间
start_time=$(date +%s)

# 逐个生成预测
for date in "${dates[@]}"; do
    echo "=========================================="
    echo "处理日期: $date"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    python scripts/predict_v270_ensemble_top50.py "$date"

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

# 运行稳定性评估
echo ""
echo "开始运行稳定性评估..."
python scripts/evaluate_v270_stability.py "${dates[@]}"

echo ""
echo "完成！"
