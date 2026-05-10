#!/bin/bash
LOG_FILE="/Users/javaadu/Documents/GitHub/aiquant/logs/extract_v295_features_run7.log"
MONITOR_LOG="/Users/javaadu/Documents/GitHub/aiquant/logs/monitor_v295.log"
while true; do
    PID=$(pgrep -f "extract_v295_features.py" | head -1)
    if [ -z "$PID" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') | ❌ 进程不存在！任务可能已终止" >> "$MONITOR_LOG"
        break
    fi
    if [ -f "$LOG_FILE" ]; then
        LAST_MOD=$(stat -f %m "$LOG_FILE")
        NOW=$(date +%s)
        DIFF=$((NOW - LAST_MOD))
        if [ $DIFF -gt 600 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') | ⚠️ 日志超过10分钟未更新，可能卡住 (PID=$PID)" >> "$MONITOR_LOG"
        fi
        RECENT_ERRORS=$(tail -500 "$LOG_FILE" | grep -iE "error|失败|终止|exception|traceback" | tail -5)
        if [ -n "$RECENT_ERRORS" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') | 🔴 发现错误:" >> "$MONITOR_LOG"
            echo "$RECENT_ERRORS" | while read line; do echo "    $line" >> "$MONITOR_LOG"; done
        fi
        LAST_PROGRESS=$(tail -100 "$LOG_FILE" | grep -E "预取进度:|进度:|全部特征提取完成" | tail -1)
        if [ -n "$LAST_PROGRESS" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') | 📊 $LAST_PROGRESS (PID=$PID)" >> "$MONITOR_LOG"
        fi
    fi
    sleep 300
done
