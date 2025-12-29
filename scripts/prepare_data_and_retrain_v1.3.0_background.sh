#!/bin/bash
# 后台运行数据准备和模型训练脚本

cd "$(dirname "$0")/.."

# 创建日志目录
mkdir -p logs

# 日志文件
LOG_FILE="logs/prepare_and_retrain_v1.3.0_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "后台运行数据准备和模型训练"
echo "=========================================="
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo ""

# 后台运行脚本
nohup python scripts/prepare_data_and_retrain_v1.3.0.py > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!

echo "脚本已在后台运行"
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "查看日志: tail -f $LOG_FILE"
echo "查看进程: ps aux | grep $PID"
echo "停止进程: kill $PID"
echo ""

# 保存PID到文件
echo $PID > logs/prepare_and_retrain_v1.3.0.pid
echo "PID已保存到: logs/prepare_and_retrain_v1.3.0.pid"

