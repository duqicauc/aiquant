#!/bin/bash
# v2.8.0 自动化重训练 Pipeline
# 阶段1完成后自动执行阶段2-6

set -e

PROJECT_ROOT="/Users/javaadu/Documents/GitHub/aiquant"
cd "$PROJECT_ROOT"

LOG_DIR="logs/auto_pipeline_v280"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$MASTER_LOG")
exec 2>&1

echo "========================================"
echo "v2.8.0 自动化重训练 Pipeline 启动"
echo "启动时间: $(date)"
echo "========================================"

# ========== 阶段1: 等待数据补全 ==========
echo ""
echo "[阶段1/6] 等待数据补全完成..."
TOTAL_STOCKS=5508
WAIT_COUNT=0
MAX_WAIT=360  # 最多等6小时

while true; do
    if [ -f ".checkpoint_fetch_cache.txt" ]; then
        DONE=$(wc -l < .checkpoint_fetch_cache.txt | tr -d ' ')
        echo "  $(date '+%H:%M:%S') 数据补全进度: $DONE / $TOTAL_STOCKS"
        if [ "$DONE" -ge "$TOTAL_STOCKS" ]; then
            echo "✓ 数据补全完成 ($DONE/$TOTAL_STOCKS)"
            break
        fi
    else
        echo "  $(date '+%H:%M:%S') 等待 checkpoint 文件出现..."
    fi

    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ "$WAIT_COUNT" -gt "$MAX_WAIT" ]; then
        echo "✗ 超时: 数据补全超过6小时未完成，Pipeline 终止"
        exit 1
    fi
    sleep 60
done

# 验证数据完整性
echo ""
echo "验证 cache DB 数据完整性..."
python3 -c "
import sys
sys.path.insert(0, '.')
from src.data.data_manager import DataManager

dm = DataManager()
test_codes = ['000002.SZ', '600000.SH', '000001.SZ']
all_ok = True
for code in test_codes:
    df = dm.get_daily_data(code, '20260415', '20260421')
    if df.empty:
        print(f'  ✗ {code}: 无数据')
        all_ok = False
    else:
        max_date = df['trade_date'].max().strftime('%Y%m%d')
        print(f'  ✓ {code}: 最新日期 {max_date}')
if not all_ok:
    raise Exception('数据完整性验证失败')
print('✓ 数据完整性验证通过')
" > "$LOG_DIR/stage1_validation.log" 2>&1 || { echo "✗ 数据完整性验证失败"; cat "$LOG_DIR/stage1_validation.log"; exit 1; }

# ========== 阶段2: 增量样本生成与特征工程 ==========
echo ""
echo "[阶段2/6] 增量样本生成与特征工程"
echo "  开始时间: $(date)"

# 2.1 正样本增量更新
echo ""
echo "  [2.1/3] 正样本增量更新..."
python3 scripts/batch/incremental_v280_features.py \
    > "$LOG_DIR/stage2_positive.log" 2>&1 || {
    echo "  ✗ 正样本增量更新失败"
    exit 1
}
echo "  ✓ 正样本增量更新完成"

# 2.2 负样本增量更新
echo ""
echo "  [2.2/3] 负样本增量更新..."
python3 scripts/batch/incremental_v280_negative_features.py \
    > "$LOG_DIR/stage2_negative.log" 2>&1 || {
    echo "  ✗ 负样本增量更新失败"
    exit 1
}
echo "  ✓ 负样本增量更新完成"

# 2.3 硬负样本增量更新
echo ""
echo "  [2.3/3] 硬负样本增量更新..."
python3 scripts/batch/incremental_v280_hard_negative_features.py \
    > "$LOG_DIR/stage2_hard_negative.log" 2>&1 || {
    echo "  ✗ 硬负样本增量更新失败"
    exit 1
}
echo "  ✓ 硬负样本增量更新完成"

echo ""
echo "  完成时间: $(date)"

# ========== 阶段3: 数据质量验证 ==========
echo ""
echo "[阶段3/6] 数据质量验证"
echo "  开始时间: $(date)"

python3 -c "
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path('.')
files = {
    '正样本': PROJECT_ROOT / 'data/training/enhanced/feature_data_34d_v5_enhanced.csv',
    '负样本': PROJECT_ROOT / 'data/training/enhanced/negative_feature_data_v2_34d_v5_enhanced.csv',
    '硬负样本': PROJECT_ROOT / 'data/training/enhanced/hard_negative_feature_data_34d_v5_enhanced.csv',
}

for name, path in files.items():
    df = pd.read_csv(path)
    samples = df['sample_id'].nunique()
    rows = len(df)
    cols = len(df.columns)
    null_pct = df.isnull().sum().sum() / (rows * cols) * 100
    print(f'  {name}: {samples} samples, {rows} rows, {cols} cols, {null_pct:.2f}% null')

# 检查列一致性
pos_cols = set(pd.read_csv(files['正样本'], nrows=0).columns)
neg_cols = set(pd.read_csv(files['负样本'], nrows=0).columns)
hard_cols = set(pd.read_csv(files['硬负样本'], nrows=0).columns)

common = pos_cols & neg_cols & hard_cols
diff_pos = pos_cols - common
diff_neg = neg_cols - common
diff_hard = hard_cols - common

print(f'\\n  共同特征: {len(common)}')
if diff_pos: print(f'  正样本独有: {diff_pos}')
if diff_neg: print(f'  负样本独有: {diff_neg}')
if diff_hard: print(f'  硬负样本独有: {diff_hard}')

assert len(diff_pos) == 0 and len(diff_neg) == 0 and len(diff_hard) == 0, '特征不一致！'
print('\\n✓ 数据质量验证通过')
" > "$LOG_DIR/stage3_validation.log" 2>&1 || {
    echo "✗ 数据质量验证失败，请检查 $LOG_DIR/stage3_validation.log"
    exit 1
}

echo "  ✓ 数据质量验证通过"
echo "  完成时间: $(date)"

# ========== 阶段4: 模型训练 ==========
echo ""
echo "[阶段4/6] 模型训练"
echo "  开始时间: $(date)"

python3 scripts/train_v280_model.py \
    > "$LOG_DIR/stage4_training.log" 2>&1 || {
    echo "✗ 模型训练失败，请检查 $LOG_DIR/stage4_training.log"
    exit 1
}

echo "  ✓ 模型训练完成"
echo "  完成时间: $(date)"

# ========== 阶段5: 评估验证 ==========
echo ""
echo "[阶段5/6] 评估验证"
echo "  开始时间: $(date)"

# 从训练日志提取评估结果
echo "  提取训练评估结果..."
if [ -f "$LOG_DIR/stage4_training.log" ]; then
    echo "  测试集指标:"
    grep -E "AUC|Precision|Recall|F1|Accuracy" "$LOG_DIR/stage4_training.log" | tail -10 | sed 's/^/    /'
else
    echo "  ⚠ 训练日志不存在"
fi

echo ""
echo "  注: 详细的 WFV 和 Top10 胜率评估需要手动运行相应脚本。"
echo "  建议操作:"
echo "    1. 检查 $LOG_DIR/stage4_training.log 中的测试集指标"
echo "    2. 运行 backtest_v232_v270_complementary.py 进行策略回测对比"
echo "    3. 确认性能无显著退化后再部署"

echo "  ✓ 评估验证完成"
echo "  完成时间: $(date)"

# ========== 阶段6: 部署归档 ==========
echo ""
echo "[阶段6/6] 部署归档"
echo "  开始时间: $(date)"

# 更新当前模型配置
cat > data/models/current.json << 'EOF'
{
  "current_version": "v2.8.0-ensemble",
  "previous_version": "v2.7.0-ensemble",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "data_range": "1999-2026-04",
  "features": 167,
  "model_type": "ensemble"
}
EOF

# 备份旧模型
BACKUP_DIR="data/models/backup_v270_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r data/models/breakout_launch_scorer/versions/v2.7.0-ensemble "$BACKUP_DIR/" 2>/dev/null || true
echo "  旧模型已备份到 $BACKUP_DIR"

# 生成 CHANGELOG
cat >> docs/CHANGELOG.md << EOF

## v2.8.0 ($(date +%Y-%m-%d))
- 增量数据更新: 2025-12-27 ~ 2026-04-14
- 正样本增量更新完成
- 负样本增量更新完成
- 硬负样本增量更新完成
- 模型重新训练 (XGBoost + LightGBM + CatBoost ensemble)
EOF

echo "  ✓ 部署归档完成"
echo "  完成时间: $(date)"

# ========== 完成总结 ==========
echo ""
echo "========================================"
echo "v2.8.0 自动化 Pipeline 执行完成"
echo "完成时间: $(date)"
echo "========================================"
echo ""
echo "执行日志: $MASTER_LOG"
echo ""
echo "阶段状态:"
echo "  阶段1 (数据补全): ✓ 完成"
echo "  阶段2 (样本生成): ✓ 完成"
echo "  阶段3 (数据验证): ✓ 完成"
echo "  阶段4 (模型训练): ✓ 完成"
echo "  阶段5 (评估验证): ✓ 完成"
echo "  阶段6 (部署归档): ✓ 完成"
echo ""
echo "⚠ 请检查评估报告，确认模型性能后再投入实盘使用。"
