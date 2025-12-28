# 训练进度监控脚本
while true; do
    echo "=== 训练进度监控 13:15:39 ==="
    
    # 检查训练进程是否运行
    if pgrep -f 'train_left_breakout_model' > /dev/null; then
        echo "✅ 训练进程运行中"
        
        # 检查特征提取进度
        tail -n 50 logs/aiquant.log | grep '处理样本' | tail -1 || echo "暂无进度信息"
        
        # 检查是否有错误
        tail -n 20 logs/aiquant.log | grep -E 'ERROR|Exception' | tail -2 || echo "暂无错误信息"
    else
        echo "❌ 训练进程未运行"
        break
    fi
    
    echo ""
    sleep 60  # 每分钟检查一次
done

echo "训练进程已结束" 
