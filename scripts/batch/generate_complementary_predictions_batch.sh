#!/bin/bash
# 批量生成互补策略预测结果
#
# 用法：
#   bash scripts/generate_complementary_predictions_batch.sh           # 跳过已有文件
#   bash scripts/generate_complementary_predictions_batch.sh --force   # 强制重新生成所有日期（Bug修复后重跑）
#
# 注意：需先确保各日期的 v2.3.2 和 v2.7.0 预测文件已存在：
#   data/prediction/results/v2.3.2_full_YYYYMMDD.csv
#   data/prediction/results/v270_ensemble_all_YYYYMMDD.csv

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 是否强制重新生成（--force 参数）
FORCE_REGEN=false
for arg in "$@"; do
  [ "$arg" = "--force" ] && FORCE_REGEN=true
done

# 需要生成的日期列表（历史全量）
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
    "20260116"
    "20260119"
    "20260120"
    "20260121"
    "20260122"
    "20260123"
    "20260126"
    "20260127"
    "20260128"
    "20260129"
    "20260130"
    "20260202"
    "20260203"
    "20260204"
    "20260205"
    "20260206"
    "20260209"
    "20260210"
    "20260211"
    "20260212"
    "20260213"
    "20260224"
    "20260225"
    "20260226"
    "20260227"
    "20260302"
    "20260303"
    "20260304"
    "20260305"
    "20260306"
)

echo "=========================================="
echo "批量生成互补策略预测结果"
echo "参数: --top 10 --base-top-n 100 --v232-top-n 100"
[ "$FORCE_REGEN" = true ] && echo "模式: 强制重新生成（--force）" || echo "模式: 跳过已有文件"
echo "日期数: ${#DATES[@]}"
echo "=========================================="
echo ""

SUCCESS=0
SKIPPED=0
FAILED=0
FAILED_DATES=()

for DATE in "${DATES[@]}"; do
    echo "---------- $DATE ----------"

    # 检查依赖预测文件是否存在
    V232_FILE="data/prediction/results/v2.3.2_full_${DATE}.csv"
    V270_FILE="data/prediction/results/v270_ensemble_all_${DATE}.csv"
    if [ ! -f "$V232_FILE" ] || [ ! -f "$V270_FILE" ]; then
        echo "  ⚠️  依赖文件不存在，跳过（需先运行 v232/v270 预测）"
        echo "     缺失: $([ ! -f "$V232_FILE" ] && echo "$V232_FILE ")$([ ! -f "$V270_FILE" ] && echo "$V270_FILE")"
        FAILED=$((FAILED + 1))
        FAILED_DATES+=("$DATE(缺依赖)")
        continue
    fi

    # 检查目标文件是否已存在
    RESULT_FILE="data/prediction/results/v232_v270_complementary_${DATE}.csv"
    if [ -f "$RESULT_FILE" ] && [ "$FORCE_REGEN" = false ]; then
        echo "  ✓ 已存在，跳过（使用 --force 可强制重新生成）"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # 生成互补策略结果
    echo "  生成中..."
    if conda run -n base python scripts/combine_v232_v270.py \
        --date "$DATE" \
        --strategy complementary \
        --top 10 \
        --base-top-n 100 \
        --v232-top-n 100 2>&1 | tail -5; then
        echo "  ✅ 完成: $RESULT_FILE"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ❌ 失败: $DATE"
        FAILED=$((FAILED + 1))
        FAILED_DATES+=("$DATE")
    fi
    echo ""
done

echo "=========================================="
echo "汇总"
echo "  成功: ${SUCCESS} 个"
echo "  跳过: ${SKIPPED} 个"
echo "  失败: ${FAILED} 个"
if [ ${#FAILED_DATES[@]} -gt 0 ]; then
    echo "  失败日期: ${FAILED_DATES[*]}"
fi
echo "=========================================="
