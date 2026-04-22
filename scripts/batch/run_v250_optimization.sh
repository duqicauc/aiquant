#!/bin/bash
# v2.5.0优化版完整执行脚本
# 包含233日均线特征 + 300天上市时间要求

set -e  # 遇到错误立即退出

PROJECT_ROOT="/Users/javaadu/Documents/GitHub/aiquant"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "v2.5.0 模型优化版训练流程"
echo "========================================================================"
echo ""
echo "优化点："
echo "  ✓ 最小上市天数: 180天 → 300天"
echo "  ✓ 新增233日均线及相关特征（约+30个特征）"
echo "  ✓ 更完整的长期趋势分析能力"
echo ""
echo "========================================================================"
echo ""

# 步骤1: 重新扫描正样本
echo "[步骤1/6] 扫描正样本（min_listing_days=300）..."
python scripts/prepare_positive_samples_checkpoint.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤1失败"
    exit 1
fi
echo "✓ 步骤1完成"
echo ""

# 步骤2: 准备负样本
echo "[步骤2/6] 准备负样本..."
python scripts/prepare_negative_samples_checkpoint.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤2失败"
    exit 1
fi
echo "✓ 步骤2完成"
echo ""

# 步骤3: 添加高级特征（包含233日均线）
echo "[步骤3/6] 添加高级特征（包含233日均线）..."
python scripts/add_advanced_factors_v3.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤3失败"
    exit 1
fi
echo "✓ 步骤3完成"
echo ""

# 步骤4: 合并数据
echo "[步骤4/6] 合并v3和v4数据..."
python scripts/merge_v3_v4_data.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤4失败"
    exit 1
fi
echo "✓ 步骤4完成"
echo ""

# 步骤5: 质量评估
echo "[步骤5/6] 数据质量评估..."
python scripts/evaluate_training_data_quality.py
if [ $? -ne 0 ]; then
    echo "⚠️  质量评估有警告，但继续训练"
fi
echo "✓ 步骤5完成"
echo ""

# 步骤6: 训练模型
echo "[步骤6/6] 训练v2.5.0模型..."
python scripts/train_v250_model.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤6失败"
    exit 1
fi
echo "✓ 步骤6完成"
echo ""

echo "========================================================================"
echo "✅ v2.5.0优化版模型训练完成！"
echo "========================================================================"
echo ""
echo "生成的文件："
echo "  - 模型: data/models/breakout_launch_scorer/versions/v2.5.0/"
echo "  - 质量报告: data/training/quality_reports/"
echo ""
echo "下一步: 运行预测"
echo "  python scripts/predict_v250_top10.py --date 20260109"
echo ""
