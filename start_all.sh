#!/bin/bash
################################################################################
# AIQuant 一键启动脚本（API + React 前端）
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}AIQuant v5.0 一键启动${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 检查依赖
echo -e "${GREEN}检查依赖...${NC}"
missing=()

if ! python3 -c "import fastapi" 2>/dev/null; then
    missing+=("fastapi uvicorn")
fi
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js 未安装，请先安装 Node.js${NC}"
    exit 1
fi

if [ ${#missing[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  缺少 Python 依赖，正在安装...${NC}"
    pip3 install fastapi uvicorn -q
fi

echo -e "${GREEN}✓ 依赖检查完成${NC}"
echo ""

# 创建日志目录
mkdir -p logs

# 启动 FastAPI（后台）
echo -e "${GREEN}启动 FastAPI 服务...${NC}"
nohup python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
API_PID=$!
echo -e "${BLUE}  API PID: ${API_PID}${NC}"
echo -e "${BLUE}  API Docs: http://localhost:8000/docs${NC}"

sleep 2

# 检查 API 是否启动成功
if ! kill -0 $API_PID 2>/dev/null; then
    echo -e "${RED}❌ FastAPI 启动失败，请检查 logs/api.log${NC}"
    exit 1
fi

# 启动 React 前端（后台）
echo ""
echo -e "${GREEN}启动 React 前端...${NC}"
cd frontend
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}⚠️  前端依赖未安装，正在安装...${NC}"
    npm install
fi
nohup npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo -e "${BLUE}  Frontend PID: ${FRONTEND_PID}${NC}"
echo -e "${BLUE}  Frontend: http://localhost:5173${NC}"

sleep 3

# 检查前端是否启动成功
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}❌ React 前端启动失败，请检查 logs/frontend.log${NC}"
    kill $API_PID 2>/dev/null
    exit 1
fi

# 保存 PID 到文件
echo $API_PID > .aiquant_api.pid
echo $FRONTEND_PID > .aiquant_frontend.pid

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}✅ AIQuant 已启动${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "🌐 Frontend: ${BLUE}http://localhost:5173${NC}"
echo -e "📡 API Docs: ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "停止服务: ${YELLOW}./stop_all.sh${NC}"
echo ""
