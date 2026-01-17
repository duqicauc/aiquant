#!/bin/bash
# v2.5.0 训练进度监控脚本

PROJECT_ROOT="/Users/javaadu/Documents/GitHub/aiquant"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "v2.5.0 模型训练进度监控"
echo "========================================================================"
echo ""

# 检查步骤1
echo "[步骤1] 为v3正样本添加高级特征"
if [ -f "data/training/processed/feature_data_34d_v3_advanced.csv" ]; then
    echo "  ✅ 已完成"
elif [ -f "data/training/processed/.checkpoint_pos_v3.csv" ]; then
    lines=$(wc -l < data/training/processed/.checkpoint_pos_v3.csv)
    samples=$((lines - 1))
    echo "  🔄 进行中 - 进度: ${samples}/3188"
else
    echo "  ⏳ 未开始"
fi
echo ""

# 检查步骤2
echo "[步骤2] 为v3负样本添加高级特征"
if [ -f "data/training/features/negative_feature_data_v2_34d_v3_advanced.csv" ]; then
    echo "  ✅ 已完成"
elif [ -f "data/training/features/.checkpoint_neg_v3.csv" ]; then
    echo "  🔄 进行中"
else
    echo "  ⏳ 未开始"
fi
echo ""

# 检查步骤3
echo "[步骤3] 合并v3和v4数据"
if [ -f "data/training/processed/feature_data_34d_v5.csv" ]; then
    echo "  ✅ 已完成"
else
    echo "  ⏳ 未开始"
fi
echo ""

# 检查步骤4
echo "[步骤4] 数据质量评估"
if [ -f "data/training/quality_reports/training_data_quality_report.json" ]; then
    echo "  ✅ 已完成"
else
    echo "  ⏳ 未开始"
fi
echo ""

# 检查步骤5
echo "[步骤5] 训练v2.5.0模型"
if [ -f "data/models/breakout_launch_scorer/versions/v2.5.0/model/model.json" ]; then
    echo "  ✅ 已完成"
else
    echo "  ⏳ 未开始"
fi
echo ""

# 显示最近日志
echo "========================================================================"
echo "最近日志:"
tail -5 logs/aiquant.log 2>/dev/null || echo "日志文件不存在"
