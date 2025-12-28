#!/bin/bash
################################################################################
# AIQuant 可视化面板启动脚本
################################################################################

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}AIQuant 可视化面板启动${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 检查依赖
echo -e "${GREEN}检查依赖...${NC}"
if ! python -c "import streamlit" 2>/dev/null; then
    echo "安装 Streamlit..."
    pip install streamlit plotly -q
fi

echo -e "${GREEN}✓ 依赖检查完成${NC}"
echo ""

# 启动面板
echo -e "${GREEN}启动可视化面板...${NC}"
echo ""
echo -e "${BLUE}访问地址: http://localhost:8501${NC}"
echo -e "${BLUE}按 Ctrl+C 停止服务${NC}"
echo ""

streamlit run app.py --server.port 8501 --server.address localhost

