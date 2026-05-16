#!/bin/bash
################################################################################
# AIQuant Docker 容器入口脚本
# 负责：初始化环境 → 启动 Nginx → 启动 FastAPI（前台主进程）
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${BLUE}[docker]${NC} $1"; }
ok()   { echo -e "${GREEN}[docker]${NC} $1"; }
warn() { echo -e "${YELLOW}[docker]${NC} $1"; }

log "AIQuant v5.0 容器启动中..."

# ── 检查 .env ──
if [ ! -f /app/.env ]; then
    if [ -f /app/env_template.txt ]; then
        warn ".env 不存在，从模板创建"
        cp /app/env_template.txt /app/.env
    else
        warn ".env 不存在，部分功能可能不可用"
    fi
fi

# ── 初始化数据库 ──
log "检查数据库..."
if [ ! -f /app/data/database/aiquant.db ]; then
    python3 -c "from src.scheduler.models import init_db; init_db()" 2>/dev/null && ok "数据库已初始化" || warn "数据库初始化失败，将在首次请求时自动创建"
else
    ok "数据库已存在"
fi

# ── 启动 Nginx（后台）──
log "启动 Nginx..."
nginx
ok "Nginx 已启动"

# ── 启动 FastAPI（前台主进程）──
# uvicorn 作为前台进程运行，容器随其生命周期管理
log "启动 FastAPI API..."
ok "================================"
ok "🌐 访问地址: http://<服务器IP>"
ok "📡 API 文档: http://<服务器IP>/docs"
ok "================================"

exec python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
