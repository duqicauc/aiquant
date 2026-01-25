#!/bin/bash
# 批量生成互补策略预测结果

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 需要生成的日期列表
DATES=(
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

echo "=========================================="
echo "批量生成互补策略预测结果（Top50）"
echo "=========================================="
echo ""

for DATE in "${DATES[@]}"; do
    echo "处理日期: $DATE"
    
    # 检查文件是否已存在
    RESULT_FILE="data/prediction/results/v232_v270_complementary_${DATE}.csv"
    if [ -f "$RESULT_FILE" ]; then
        echo "  文件已存在，跳过"
        continue
    fi
    
    # 生成互补策略结果（top50）
    echo "  生成互补策略结果..."
    python scripts/combine_v232_v270.py \
        --date "$DATE" \
        --strategy complementary \
        --top 50 \
        --base-top-n 50 \
        --v232-top-n 100
    
    echo ""
done

echo "=========================================="
echo "✅ 完成！"
echo "=========================================="
