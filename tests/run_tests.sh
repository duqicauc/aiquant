#!/bin/bash
# 测试运行脚本

set -e

echo "=========================================="
echo "AIQuant 测试套件"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查pytest是否安装
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}错误: pytest 未安装${NC}"
    echo "请运行: pip install pytest pytest-cov"
    exit 1
fi

# 解析命令行参数
COVERAGE=false
VERBOSE=false
MARKERS=""
TEST_PATH="tests/"

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --unit)
            MARKERS="unit"
            shift
            ;;
        --integration)
            MARKERS="integration"
            shift
            ;;
        --slow)
            MARKERS="slow"
            shift
            ;;
        --api)
            MARKERS="api"
            shift
            ;;
        --path)
            TEST_PATH="$2"
            shift 2
            ;;
        *)
            TEST_PATH="$1"
            shift
            ;;
    esac
done

# 构建pytest命令
PYTEST_CMD="pytest"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
else
    PYTEST_CMD="$PYTEST_CMD -q"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html --cov-report=term-missing"
fi

if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD -m $MARKERS"
fi

PYTEST_CMD="$PYTEST_CMD $TEST_PATH"

echo -e "${YELLOW}运行命令: $PYTEST_CMD${NC}"
echo ""

# 运行测试
$PYTEST_CMD

TEST_EXIT_CODE=$?

echo ""
echo "=========================================="

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ 所有测试通过！${NC}"
    
    if [ "$COVERAGE" = true ]; then
        echo ""
        echo -e "${YELLOW}覆盖率报告已生成: htmlcov/index.html${NC}"
        echo "打开命令: open htmlcov/index.html"
    fi
else
    echo -e "${RED}✗ 测试失败${NC}"
fi

echo "=========================================="

exit $TEST_EXIT_CODE

