#!/usr/bin/env bash
# =============================================================================
# AIQuant 一键部署脚本
# 用法: bash deploy.sh [--with-data]
#
# 选项:
#   --with-data    同时复制数据目录（ArcticDB cache + prediction + models）
#   --skip-ta-lib  跳过 TA-Lib 系统依赖安装（已安装过或有兼容环境时）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR="$PROJECT_ROOT/venv"
DATA_DIR="$PROJECT_ROOT/data"

# -----------------------------------------------------------------------------
# 颜色输出
# -----------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
NC='\033[0m' # No Color
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

need_data=false; skip_talib=false
for arg in "$@"; do
    case $arg in
        --with-data)   need_data=true ;;
        --skip-ta-lib) skip_talib=true ;;
        *)             error "未知参数: $arg，用法: bash deploy.sh [--with-data] [--skip-ta-lib]"
    esac
done

# =============================================================================
# 第一步：系统依赖（TA-Lib C 库 + LMDB）
# =============================================================================
info "安装系统依赖..."

if [ "$skip_talib" = false ]; then
    info "安装 TA-Lib C 库（需要 sudo）..."
    TA_LIB_DIR="/tmp/talib-2.0.8"
    if [ ! -f /usr/lib/libta_lib.a ] && [ ! -f /usr/local/lib/libta_lib.a ]; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq build-essential wget libgsl-dev > /dev/null 2>&1
        cd /tmp
        [ -d "$TA_LIB_DIR" ] && rm -rf "$TA_LIB_DIR"
        wget -q https://ta-lib.org/ta-lib.tgz -O talib.tgz
        tar -xzf talib.tgz
        cd "$TA_LIB_DIR"
        ./configure --prefix=/usr/local > /dev/null 2>&1
        make -j$(nproc) > /dev/null 2>&1
        sudo make install > /dev/null 2>&1
        sudo ldconfig
        cd /tmp && rm -rf "$TA_LIB_DIR" talib.tgz
        success "TA-Lib C 库安装完成"
    else
        info "TA-Lib C 库已存在，跳过"
    fi
else
    warn "跳过 TA-Lib 系统依赖安装（--skip-ta-lib）"
fi

# =============================================================================
# 第二步：Python 3.12 + venv
# =============================================================================
info "检查 Python 3.12..."
if ! command -v python3.12 &> /dev/null; then
    error "Python 3.12 未安装: sudo apt-get install python3.12 python3.12-venv python3.12-dev"
fi
PYTHON_BIN=$(command -v python3.12)
success "Python 3.12: $($PYTHON_BIN --version)"

# 创建 venv
info "创建 venv..."
[ -d "$VENV_DIR" ] && warn "venv 已存在，会复用"
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
success "venv 激活: $VENV_DIR"

# =============================================================================
# 第三步：安装 Python 依赖
# =============================================================================
info "安装 Python 依赖（pip install -r requirements.txt）..."
# 注意：requirements.txt 已包含 arcticdb==6.17.0，无需单独安装
pip install --upgrade pip -q
pip install -r "$PROJECT_ROOT/requirements.txt" -q
success "Python 依赖安装完成"

# =============================================================================
# 第四步：环境变量配置
# =============================================================================
info "配置环境变量..."
if [ -f "$PROJECT_ROOT/.env" ]; then
    info ".env 已存在，跳过（保留现有 token）"
else
    if [ -f "$PROJECT_ROOT/env_template.txt" ]; then
        cp "$PROJECT_ROOT/env_template.txt" "$PROJECT_ROOT/.env"
        warn ".env 已从 env_template.txt 创建，请编辑填入真实 Token"
    else
        error ".env 不存在，请手动创建"
    fi
fi
success "环境变量就绪"

# =============================================================================
# 第五步：验证核心依赖
# =============================================================================
info "验证核心依赖..."
activate_venv() { source "$VENV_DIR/bin/activate"; }
activate_venv

python -c "
import arcticdb; print(f'  arcticdb  {arcticdb.__version__}')
import fastapi; print(f'  fastapi   {fastapi.__version__}')
import tushare; print(f'  tushare   {tushare.__version__}')
import loguru;  print(f'  loguru    {loguru.__version__}')
" || error "核心依赖验证失败"

# TA-Lib 单独验证（容易出问题）
python -c "import talib; print(f'  TA-Lib    {talib.__version__}')" \
    || warn "TA-Lib 导入失败，确认 C 库已正确安装后重新 pip install TA-Lib==0.6.8"

success "核心依赖验证通过"

# =============================================================================
# 第六步：数据目录初始化
# =============================================================================
info "初始化数据目录..."
mkdir -p "$DATA_DIR/cache"
mkdir -p "$DATA_DIR/database"
mkdir -p "$DATA_DIR/models"
mkdir -p "$DATA_DIR/prediction"
mkdir -p "$DATA_DIR/prediction/v3.0.0"
mkdir -p "$DATA_DIR/prediction/v3.1.0"
mkdir -p "$DATA_DIR/results"
mkdir -p "$DATA_DIR/training"
mkdir -p "$DATA_DIR/logs"

# 初始化 SQLite 表（scheduler_job_history 等）
python -c "
from src.database.init_db import init_database
init_database()
print('  SQLite 数据库初始化完成')
" 2>/dev/null || info "  SQLite 初始化跳过（无 init_database）"

success "数据目录结构就绪"

# =============================================================================
# 第七步：复制数据（可选）
# =============================================================================
if [ "$need_data" = true ]; then
    info "复制数据目录（ArcticDB + prediction + models）..."
    if [ -d "/path/to/backup/aiquant_data_YYYYMMDD.tar.gz" ]; then
        # 从压缩包恢复
        sudo apt-get install -y -qq pigz > /dev/null 2>&1
        tar -I pigz -xf "/path/to/backup/aiquant_data_YYYYMMDD.tar.gz" -C "$PROJECT_ROOT"
        success "数据从压缩包恢复完成"
    else
        warn "未找到数据压缩包，跳过数据复制"
        warn "请手动复制 data/ 目录或重新运行补数据脚本："
        warn "  python scripts/batch/fill_missing_flat_data.py --start-date 20260519 --end-date \$(date +%Y%m%d)"
    fi
else
    info "跳过数据复制（排除 --with-data 即可跳过）"
    info "首次运行前需补数据："
    info "  source venv/bin/activate"
    info "  python scripts/batch/fill_missing_flat_data.py --start-date 20260519 --end-date \$(date +%Y%m%d)"
fi

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
success "AIQuant 部署完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  启动服务:"
echo "    cd $PROJECT_ROOT"
echo "    source venv/bin/activate"
echo "    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "  查看日志:"
echo "    tail -f $PROJECT_ROOT/logs/aiquant.log"
echo ""
echo "  补全数据（首次部署必须执行）:"
echo "    python scripts/batch/fill_missing_flat_data.py --start-date 20260519 --end-date \$(date +%Y%m%d)"
echo ""
echo "  更新依赖（新增 Python 包后重新生成）:"
echo "    pip list --not-required --format=freeze | grep '==' > requirements.txt"
echo ""