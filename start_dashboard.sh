#!/bin/bash
################################################################################
# AIQuant Dash 仪表盘启动脚本
################################################################################

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}AIQuant Dash 仪表盘启动${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 检查依赖
echo -e "${GREEN}检查依赖...${NC}"
if ! python3 -c "import dash" 2>/dev/null; then
    echo -e "${RED}Dash 未安装，请先运行: pip3 install dash dash-bootstrap-components dash-ag-grid${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 依赖检查完成${NC}"
echo ""

# 启动 Dashboard
echo -e "${GREEN}启动 Dash 仪表盘...${NC}"
echo ""
echo -e "${BLUE}访问地址: http://localhost:8050${NC}"
echo -e "${BLUE}按 Ctrl+C 停止服务${NC}"
echo ""

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
python3 src/dashboard/app.py
