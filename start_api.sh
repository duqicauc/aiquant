#!/bin/bash
################################################################################
# AIQuant FastAPI 服务启动脚本
################################################################################

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}AIQuant FastAPI 服务启动${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 检查依赖
echo -e "${GREEN}检查依赖...${NC}"
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${RED}FastAPI 未安装，请先运行: pip3 install fastapi uvicorn${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 依赖检查完成${NC}"
echo ""

# 启动 API
echo -e "${GREEN}启动 FastAPI 服务...${NC}"
echo ""
echo -e "${BLUE}API 文档: http://localhost:8000/docs${NC}"
echo -e "${BLUE}Redoc:   http://localhost:8000/redoc${NC}"
echo -e "${BLUE}按 Ctrl+C 停止服务${NC}"
echo ""

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
python3 -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
