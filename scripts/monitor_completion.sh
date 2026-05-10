#!/bin/bash
LOG_FILE="/Users/javaadu/Documents/GitHub/aiquant/logs/extract_v295_features_run7.log"
ALERT_LOG="/Users/javaadu/Documents/GitHub/aiquant/logs/monitor_completion_alert.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') | 完成监控启动" >> "$ALERT_LOG"
while true; do
    PID=$(pgrep -f "extract_v295_features.py" | head -1)
    if [ -z "$PID" ]; then
        if [ -f "$LOG_FILE" ]; then
            LAST_LINES=$(tail -20 "$LOG_FILE")
            if echo "$LAST_LINES" | grep -q "全部特征提取完成"; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') | ✅ TASK_COMPLETED_SUCCESS" >> "$ALERT_LOG"
            elif echo "$LAST_LINES" | grep -qiE "error|失败|终止|exception"; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') | ❌ TASK_FAILED" >> "$ALERT_LOG"
            else
                echo "$(date '+%Y-%m-%d %H:%M:%S') | ⚠️ TASK_ENDED_UNKNOWN" >> "$ALERT_LOG"
            fi
        fi
        break
    fi
    sleep 30
done
