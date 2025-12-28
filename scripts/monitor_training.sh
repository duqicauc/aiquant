#!/bin/bash
# 模型训练进度监控脚本

cd "$(dirname "$0")/.." || exit

echo "=================================="
echo "模型训练进度监控"
echo "=================================="
echo ""

# 检查负样本准备进程
echo "📊 负样本准备进程:"
if ps aux | grep -q "[p]repare_negative_samples_v2"; then
    echo "  ✓ 正在运行"
    PID=$(ps aux | grep "[p]repare_negative_samples_v2" | awk '{print $2}')
    echo "  PID: $PID"
else
    echo "  ✗ 未运行"
fi
echo ""

# 检查数据文件
echo "📁 数据文件状态:"
if [ -f "data/processed/positive_samples.csv" ]; then
    POS_COUNT=$(wc -l < data/processed/positive_samples.csv)
    echo "  ✓ 正样本: $POS_COUNT 行"
else
    echo "  ✗ 正样本文件不存在"
fi

if [ -f "data/processed/negative_samples_v2.csv" ]; then
    NEG_COUNT=$(wc -l < data/processed/negative_samples_v2.csv)
    echo "  ✓ 负样本: $NEG_COUNT 行"
else
    echo "  ⏳ 负样本准备中..."
fi

if [ -f "data/processed/feature_data_34d.csv" ]; then
    FEAT_COUNT=$(wc -l < data/processed/feature_data_34d.csv)
    echo "  ✓ 特征数据: $FEAT_COUNT 行"
fi
echo ""

# 查看最新日志
echo "📝 最新日志 (最近10行):"
echo "----------------------------------------"
tail -10 logs/aiquant.log | sed 's/^/  /'
echo ""

# 预计进度
echo "⏱️  预计进度:"
if [ -f "data/processed/negative_samples_v2.csv" ]; then
    echo "  ✅ Step 1: 负样本准备 (完成)"
    echo "  ⏳ Step 2: 数据质量检查 (进行中或待启动)"
    echo "  ⏳ Step 3: 模型训练 (等待)"
    echo "  ⏳ Step 4: Walk-Forward验证 (等待)"
else
    echo "  ⏳ Step 1: 负样本准备 (进行中)"
    echo "  ⏳ Step 2: 数据质量检查 (等待)"
    echo "  ⏳ Step 3: 模型训练 (等待)"
    echo "  ⏳ Step 4: Walk-Forward验证 (等待)"
fi
echo ""

echo "=================================="
echo "监控命令:"
echo "  实时日志: tail -f logs/aiquant.log"
echo "  再次检查: bash scripts/monitor_training.sh"
echo "=================================="

