#!/bin/bash
# 股票体检功能测试脚本

echo "=================================="
echo "股票全方位体检功能测试"
echo "=================================="

cd "$(dirname "$0")/.." || exit

echo ""
echo "测试1: 体检贵州茅台"
echo "----------------------------------"
python scripts/stock_health_check.py 600519.SH --days 120

echo ""
echo ""
echo "测试2: 体检万科A"
echo "----------------------------------"
python scripts/stock_health_check.py 000002.SZ --days 90

echo ""
echo ""
echo "测试3: 体检中国平安（不保存）"
echo "----------------------------------"
python scripts/stock_health_check.py 601318.SH --no-save

echo ""
echo "=================================="
echo "测试完成！"
echo "=================================="
echo ""
echo "查看保存的报告:"
echo "  ls -lh data/analysis/"
echo ""
echo "在浏览器中打开图表:"
echo "  open data/analysis/chart_*.html"

