#!/bin/bash
################################################################################
# AIQuant React 前端启动脚本
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd "$(dirname "$0")/.."

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}AIQuant React 前端启动${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}❌ Node.js 未安装，请先安装 Node.js${NC}"
    exit 1
fi

cd frontend

if [ ! -d "node_modules" ]; then
    echo -e "${GREEN}安装前端依赖...${NC}"
    npm install
fi

echo -e "${GREEN}启动 React 开发服务器...${NC}"
echo ""
echo -e "${BLUE}访问地址: http://localhost:5173${NC}"
echo -e "${BLUE}按 Ctrl+C 停止服务${NC}"
echo ""

npm run dev
