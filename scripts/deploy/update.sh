#!/bin/bash
################################################################################
# AIQuant 一键更新脚本（生产环境）
# 用法: cd /opt/aiquant && ./scripts/deploy/update.sh
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $1"; }

cd "$(dirname "$0")/../.."
INSTALL_DIR=$(pwd)

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  AIQuant 生产环境更新${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 1. 拉取最新代码
log_info "拉取最新代码..."
git fetch origin
git pull origin $(git rev-parse --abbrev-ref HEAD)
log_ok "代码更新完成"

# 2. 安装/更新 Python 依赖
log_info "更新 Python 依赖..."
source venv/bin/activate
pip install -q fastapi uvicorn pandas numpy scipy tushare sqlalchemy joblib tqdm loguru python-dotenv pyyaml requests scikit-learn xgboost ta-lib lightgbm catboost pyarrow apscheduler
log_ok "Python 依赖更新完成"

# 3. 重新构建前端
log_info "重新构建前端..."
cd frontend
npm install --quiet
VITE_API_BASE='' npm run build --quiet 2>/dev/null || VITE_API_BASE='' npm run build
cd ..
log_ok "前端构建完成"

# 4. 数据库迁移（如有新表）
log_info "检查数据库..."
$INSTALL_DIR/venv/bin/python -c "from src.scheduler.models import init_db; init_db()" 2>/dev/null || true
log_ok "数据库检查完成"

# 5. 重启服务
log_info "重启服务..."
sudo systemctl restart aiquant-api
sleep 2

if systemctl is-active --quiet aiquant-api; then
    log_ok "AIQuant API 已重启"
else
    log_err "重启失败，请检查日志: journalctl -u aiquant-api -n 50"
    exit 1
fi

# 6. 重载 Nginx
sudo systemctl reload nginx
log_ok "Nginx 已重载"

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  ✅ 更新完成${NC}"
echo -e "${GREEN}================================${NC}"
