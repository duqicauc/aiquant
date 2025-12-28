#!/bin/bash
# ============================================================================
# 左侧潜力牛股模型 - 完整更新管道
# ============================================================================
# 执行完整的左侧模型更新流程：样本准备 → 特征提取 → 模型训练 → 验证
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# 检查Python脚本执行结果
check_result() {
    local exit_code=$1
    local step_name=$2

    if [ $exit_code -eq 0 ]; then
        log_success "$step_name 完成"
    else
        log_error "$step_name 失败 (退出码: $exit_code)"
        exit 1
    fi
}

# 主函数
main() {
    local start_time=$(date +%s)
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="logs/left_breakout_update_${timestamp}.log"

    echo "================================================================================"
    echo "🎯 左侧潜力牛股模型 - 完整更新管道"
    echo "================================================================================"
    echo "开始时间: $(date)"
    echo "日志文件: $log_file"
    echo "================================================================================"

    # 创建日志目录
    mkdir -p logs

    # 重定向所有输出到日志文件，同时显示在屏幕上
    exec > >(tee -a "$log_file") 2>&1

    # 检查必要的文件是否存在
    log_info "检查必要文件..."

    if [ ! -f "scripts/prepare_left_breakout_samples.py" ]; then
        log_error "样本准备脚本不存在: scripts/prepare_left_breakout_samples.py"
        exit 1
    fi

    if [ ! -f "scripts/train_left_breakout_model.py" ]; then
        log_error "模型训练脚本不存在: scripts/train_left_breakout_model.py"
        exit 1
    fi

    if [ ! -f "scripts/validate_left_breakout_model.py" ]; then
        log_error "模型验证脚本不存在: scripts/validate_left_breakout_model.py"
        exit 1
    fi

    log_success "所有必要文件检查通过"

    # 步骤1: 准备样本数据
    log_info "步骤1/4: 准备左侧潜力样本数据..."
    echo "-------------------------------------------------------------------------------"
    python scripts/prepare_left_breakout_samples.py --force-refresh
    check_result $? "样本准备"

    # 检查样本文件是否生成
    if [ ! -f "data/training/samples/left_positive_samples.csv" ] || [ ! -f "data/training/samples/left_negative_samples.csv" ]; then
        log_error "样本文件生成失败"
        exit 1
    fi

    local positive_count=$(wc -l < data/training/samples/left_positive_samples.csv)
    local negative_count=$(wc -l < data/training/samples/left_negative_samples.csv)
    log_info "样本统计: 正样本 $((positive_count-1)) 个, 负样本 $((negative_count-1)) 个"

    # 步骤2: 训练模型
    log_info "步骤2/4: 训练左侧潜力模型..."
    echo "-------------------------------------------------------------------------------"
    python scripts/train_left_breakout_model.py --skip-validation
    check_result $? "模型训练"

    # 检查模型文件是否生成
    if [ ! -f "data/models/left_breakout/left_breakout_v1.joblib" ]; then
        log_error "模型文件生成失败"
        exit 1
    fi

    log_success "模型文件已生成: data/models/left_breakout/left_breakout_v1.joblib"

    # 步骤3: 模型验证
    log_info "步骤3/4: 执行模型验证..."
    echo "-------------------------------------------------------------------------------"
    python scripts/validate_left_breakout_model.py --all
    check_result $? "模型验证"

    # 步骤4: 执行预测测试
    log_info "步骤4/4: 执行预测测试..."
    echo "-------------------------------------------------------------------------------"
    python scripts/predict_left_breakout.py --top-n 10 --max-stocks 100
    check_result $? "预测测试"

    # 计算总耗时
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(( (duration % 3600) / 60 ))
    local seconds=$((duration % 60))

    # 输出完成信息
    echo "================================================================================"
    log_success "左侧潜力牛股模型更新完成！"
    echo "-------------------------------------------------------------------------------"
    echo "📊 更新统计:"
    echo "   • 正样本数量: $((positive_count-1))"
    echo "   • 负样本数量: $((negative_count-1))"
    echo "   • 总耗时: ${hours}小时 ${minutes}分钟 ${seconds}秒"
    echo "   • 日志文件: $log_file"
    echo ""
    echo "🎯 下一步操作建议:"
    echo "   1. 查看训练报告: data/models/left_breakout/training_report_*.txt"
    echo "   2. 查看验证报告: data/models/left_breakout/validation/validation_summary_*.txt"
    echo "   3. 查看预测结果: data/prediction/left_breakout/*/left_breakout_predictions_*.csv"
    echo "   4. 运行完整预测: python scripts/predict_left_breakout.py"
    echo ""
    echo "💡 左侧模型特点:"
    echo "   • 专注于底部震荡+预转信号的股票"
    echo "   • 提前1-2周发现投资机会"
    echo "   • 降低时间成本，提高资金效率"
    echo "================================================================================"

    log_success "管道执行完成，总耗时: ${hours}:${minutes}:${seconds}"
}

# 参数处理
FORCE_REFRESH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force-refresh)
            FORCE_REFRESH=true
            shift
            ;;
        --help)
            echo "左侧潜力牛股模型更新管道"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --force-refresh    强制重新生成样本数据"
            echo "  --help            显示此帮助信息"
            echo ""
            echo "执行流程:"
            echo "  1. 准备样本数据 (正样本 + 负样本)"
            echo "  2. 特征提取和工程"
            echo "  3. 模型训练 (XGBoost)"
            echo "  4. 模型验证 (Walk-Forward + 鲁棒性测试)"
            echo "  5. 预测测试"
            echo ""
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 执行主函数
main "$@"
