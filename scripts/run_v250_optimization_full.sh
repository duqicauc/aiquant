#!/bin/bash
# v2.5.0完整优化版执行脚本
# 包含所有优化：特征对齐、233均线、市场环境、重构特征、追高风控等

set -e  # 遇到错误立即退出

PROJECT_ROOT="/Users/javaadu/Documents/GitHub/aiquant"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "v2.5.0 模型完整优化版训练流程"
echo "========================================================================"
echo ""
echo "优化内容："
echo "  ✓ 特征对齐修复（bias、EMA、KDJ、量比等）"
echo "  ✓ 233日均线系列特征"
echo "  ✓ 市场环境特征（大盘收益率、超额收益、相对强度）"
echo "  ✓ 重构突破特征（从二值升级为连续强度）"
echo "  ✓ 重构量价特征（经典量价形态识别）"
echo "  ✓ Tushare高级技术因子（布林带、CCI、MACD/KDJ优化）"
echo "  ✓ 追高风控因子"
echo "  ✓ 模型参数优化（max_depth=5, colsample_bytree=0.8）"
echo ""
echo "========================================================================"
echo ""

# 步骤1: 重新扫描正样本（如果需要）
echo "[步骤1/7] 检查正样本数据..."
if [ ! -f "data/training/samples/positive_samples.csv" ]; then
    echo "  正样本不存在，开始扫描..."
    python scripts/prepare_positive_samples_checkpoint.py
    if [ $? -ne 0 ]; then
        echo "❌ 步骤1失败"
        exit 1
    fi
else
    echo "  ✓ 正样本已存在，跳过"
fi
echo ""

# 步骤2: 准备负样本
echo "[步骤2/8] 准备负样本..."
python scripts/prepare_negative_samples_checkpoint.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤2失败"
    exit 1
fi
echo "✓ 步骤2完成"
echo ""

# 步骤2.5: 修复负样本缺失的特征
echo "[步骤2.5/8] 修复负样本缺失的特征（市值、RSI、MACD等）..."
python scripts/fix_negative_sample_features.py
if [ $? -ne 0 ]; then
    echo "⚠️  负样本特征修复失败，继续执行"
fi
echo "✓ 步骤2.5完成"
echo ""

# 步骤3: 生成硬负样本（伪突破样本）
echo "[步骤3/7] 生成硬负样本（伪突破样本）..."
python scripts/prepare_hard_negative_samples.py
if [ $? -ne 0 ]; then
    echo "⚠️  硬负样本生成失败，继续执行（可选步骤）"
fi
echo "✓ 步骤3完成"
echo ""

# 步骤4: 添加优化后的高级特征
echo "[步骤4/7] 添加优化后的高级特征..."
python scripts/add_advanced_factors_optimized.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤4失败"
    exit 1
fi
echo "✓ 步骤4完成"
echo ""

# 步骤5: 数据质量检查（关键！确保特征对齐）
echo "[步骤5/8] 数据质量检查（确保特征对齐）..."
python scripts/check_data_quality.py
if [ $? -ne 0 ]; then
    echo "❌ 数据质量检查未通过！"
    echo "  请检查正负样本特征是否对齐"
    echo "  可能需要重新运行: python scripts/fix_negative_sample_features.py"
    exit 1
fi
echo "✓ 步骤5完成"
echo ""

# 步骤6: 训练模型
echo "[步骤6/8] 训练v2.5.0优化模型..."
python scripts/train_v250_model.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤6失败"
    exit 1
fi
echo "✓ 步骤6完成"
echo ""

# 步骤7: 特征重要性分析（可选）
echo "[步骤7/8] 特征重要性分析..."
if [ -f "scripts/analyze_v250_features.py" ]; then
    python scripts/analyze_v250_features.py || echo "⚠️  特征分析有警告"
else
    echo "  特征分析脚本不存在，跳过"
fi
echo "✓ 步骤7完成"
echo ""

# 步骤8: 输出训练总结
echo "[步骤8/8] 训练总结..."
echo "  请查看模型评估报告和特征重要性分析"
echo "  如果突破/量价等核心特征重要性仍然很低，可能需要进一步优化"
echo "✓ 步骤8完成"
echo ""

echo "========================================================================"
echo "✅ v2.5.0模型完整优化版训练完成！"
echo "========================================================================"
echo ""
echo "下一步："
echo "  1. 查看模型评估报告"
echo "  2. 使用新模型进行预测：python scripts/predict_v250_top10.py"
echo "  3. 观察实际预测效果（建议观察3-4周）"
echo ""
