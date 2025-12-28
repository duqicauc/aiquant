#!/bin/bash
# 自动运行测试脚本

echo "=========================================="
echo "运行单元测试和集成测试"
echo "=========================================="

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. 运行所有单元测试（跳过需要API的测试）
echo -e "\n${YELLOW}1. 运行单元测试（跳过API测试）...${NC}"
pytest -m "unit and not api" -v --tb=short

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 单元测试通过${NC}"
else
    echo -e "${RED}✗ 单元测试失败${NC}"
fi

# 2. 运行集成测试
echo -e "\n${YELLOW}2. 运行集成测试...${NC}"
pytest tests/integration/ -v --tb=short -m "integration"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 集成测试通过${NC}"
else
    echo -e "${RED}✗ 集成测试失败${NC}"
fi

# 3. 生成覆盖率报告
echo -e "\n${YELLOW}3. 生成覆盖率报告...${NC}"
pytest --cov=src --cov-report=term-missing --cov-report=html -m "unit and not api"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 覆盖率报告已生成${NC}"
    echo -e "查看HTML报告: open htmlcov/index.html"
else
    echo -e "${RED}✗ 覆盖率报告生成失败${NC}"
fi

# 4. 测试统计
echo -e "\n${YELLOW}4. 测试统计...${NC}"
pytest --collect-only -q 2>/dev/null | grep "test session starts" -A 1000 | grep "test" | wc -l | xargs echo "总测试用例数:"

echo -e "\n${GREEN}=========================================="
echo "测试完成！"
echo "==========================================${NC}"

