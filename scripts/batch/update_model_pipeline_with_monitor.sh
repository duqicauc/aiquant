#!/bin/bash
################################################################################
# 带网络监控的模型更新流程
#
# 功能:
# 1. 在后台启动网络监控
# 2. 执行完整的模型更新流程
# 3. 自动处理网络中断和恢复
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

log_info "项目根目录: $PROJECT_ROOT"

################################################################################
# Step 0: 启动网络监控
################################################################################
log_step "Step 0: 启动网络监控"

MONITOR_PID_FILE="/tmp/aiquant_network_monitor.pid"
MONITOR_LOG="logs/network_monitor_$(date +%Y%m%d_%H%M%S).log"

# 检查是否已有监控在运行
if [ -f "$MONITOR_PID_FILE" ]; then
    OLD_PID=$(cat "$MONITOR_PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        log_info "网络监控已在运行 (PID: $OLD_PID)"
    else
        rm -f "$MONITOR_PID_FILE"
    fi
fi

# 启动网络监控（后台运行）
if [ ! -f "$MONITOR_PID_FILE" ]; then
    log_info "启动网络监控..."
    nohup python scripts/utils/network_monitor.py \
        --interval 60 \
        --retry 3 \
        > "$MONITOR_LOG" 2>&1 &

    MONITOR_PID=$!
    echo $MONITOR_PID > "$MONITOR_PID_FILE"
    log_info "✓ 网络监控已启动 (PID: $MONITOR_PID)"
    log_info "  日志文件: $MONITOR_LOG"

    # 等待监控初始化
    sleep 3
fi

################################################################################
# Step 1: 准备正样本
################################################################################
log_step "Step 1: 准备正样本（使用2000年以来数据）"
log_info "预计耗时: 2-3小时"

python scripts/prepare_positive_samples.py

if [ $? -ne 0 ]; then
    log_error "准备正样本失败"
    exit 1
fi

log_info "✓ 正样本准备完成"

################################################################################
# Step 2: 准备负样本
################################################################################
log_step "Step 2: 准备负样本（同期其他股票法V2）"
log_info "预计耗时: 1-2小时"

python scripts/prepare_negative_samples_v2.py

if [ $? -ne 0 ]; then
    log_error "准备负样本失败"
    exit 1
fi

log_info "✓ 负样本准备完成"

################################################################################
# Step 3: 质量检查
################################################################################
log_step "Step 3: 数据质量检查"
log_info "预计耗时: <5分钟"

python scripts/check_sample_quality.py

if [ $? -ne 0 ]; then
    log_warn "质量检查发现问题，请查看报告"
fi

log_info "✓ 质量检查完成"

################################################################################
# Step 4: 训练模型
################################################################################
log_step "Step 4: 训练XGBoost模型（时间序列分割）"
log_info "预计耗时: 10-30分钟"

python scripts/train_xgboost_timeseries.py

if [ $? -ne 0 ]; then
    log_error "模型训练失败"
    exit 1
fi

log_info "✓ 模型训练完成"

################################################################################
# Step 5: Walk-Forward验证
################################################################################
log_step "Step 5: Walk-Forward验证"
log_info "预计耗时: 20-60分钟"

python scripts/walk_forward_validation.py

if [ $? -ne 0 ]; then
    log_error "模型验证失败"
    exit 1
fi

log_info "✓ Walk-Forward验证完成"

################################################################################
# Step 6: 停止网络监控
################################################################################
log_step "Step 6: 停止网络监控"

if [ -f "$MONITOR_PID_FILE" ]; then
    MONITOR_PID=$(cat "$MONITOR_PID_FILE")
    if ps -p "$MONITOR_PID" > /dev/null 2>&1; then
        log_info "停止网络监控 (PID: $MONITOR_PID)..."
        kill $MONITOR_PID
        rm -f "$MONITOR_PID_FILE"
        log_info "✓ 网络监控已停止"
    fi
fi

################################################################################
# 完成
################################################################################
log_info "==============================================="
log_info "✅ 模型更新流程全部完成！"
log_info "==============================================="
log_info ""
log_info "📊 输出文件："
log_info "  1. 正样本: data/processed/positive_samples.csv"
log_info "  2. 负样本: data/processed/negative_samples_v2.csv"
log_info "  3. 质量报告: data/processed/quality_report.txt"
log_info "  4. 模型文件: data/models/stock_selection/xgboost_timeseries_v3.joblib"
log_info "  5. 验证结果: data/backtest/reports/walk_forward_validation_results.json"
log_info ""
log_info "🚀 下一步："
log_info "  python scripts/score_current_stocks.py  # 使用新模型评分"
log_info ""
