#!/bin/bash
################################################################################
# AIQuant 一键停止脚本
################################################################################

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

cd "$(dirname "$0")"

echo -e "${BLUE}停止 AIQuant 服务...${NC}"

# 从 PID 文件停止
if [ -f .aiquant_api.pid ]; then
    API_PID=$(cat .aiquant_api.pid)
    if kill -0 $API_PID 2>/dev/null; then
        kill $API_PID
        echo -e "${GREEN}✓ FastAPI 已停止 (PID: $API_PID)${NC}"
    fi
    rm -f .aiquant_api.pid
fi

if [ -f .aiquant_frontend.pid ]; then
    FRONTEND_PID=$(cat .aiquant_frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID
        echo -e "${GREEN}✓ React 前端已停止 (PID: $FRONTEND_PID)${NC}"
    fi
    rm -f .aiquant_frontend.pid
fi

# 兜底：kill 所有相关进程
pkill -f "uvicorn src.api.main:app" 2>/dev/null || true
pkill -f "node.*vite" 2>/dev/null || true

echo -e "${GREEN}✓ 所有服务已停止${NC}"
