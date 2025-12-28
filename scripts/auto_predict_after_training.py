#!/usr/bin/env python3
"""
自动等待训练完成并运行预测
"""
import sys
import os
import time
import subprocess
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import log


def check_training_complete():
    """检查训练是否完成"""
    # 检查进程
    result = subprocess.run(['pgrep', '-f', 'train_left_breakout_model.py'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        return False
    
    # 检查日志中是否有训练完成的标志
    log_file = 'logs/aiquant.log'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines[-100:]):  # 检查最后100行
                if '模型已保存' in line or '训练完成' in line or '模型训练完成' in line:
                    return True
    
    return False


def wait_for_training():
    """等待训练完成"""
    log.info("=" * 60)
    log.info("⏳ 等待模型训练完成...")
    log.info("=" * 60)
    
    check_count = 0
    while True:
        if check_training_complete():
            log.info("✅ 训练已完成！")
            return True
        
        check_count += 1
        if check_count % 10 == 0:  # 每10次检查显示一次进度
            # 显示当前训练进度
            try:
                with open('logs/aiquant.log', 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines[-50:]):
                        if '处理样本' in line:
                            log.info(f"📊 {line.strip()}")
                            break
            except:
                pass
        
        time.sleep(30)  # 每30秒检查一次


def run_prediction(date_str, date_label):
    """运行预测脚本"""
    log.info("=" * 60)
    log.info(f"🚀 开始预测{date_label}的Top50股票...")
    log.info("=" * 60)
    
    cmd = [
        'python', 'scripts/predict_left_breakout.py',
        '--date', date_str,
        '--top-n', '50',
        '--min-prob', '0.1'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info(f"✅ {date_label}预测完成！")
        log.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"❌ {date_label}预测失败: {e}")
        log.error(e.stderr)
        return False


def run_all_predictions():
    """运行所有预测"""
    predictions = [
        ('20251225', '2025-12-25'),
        ('20250919', '2025-09-19')
    ]
    
    success_count = 0
    for date_str, date_label in predictions:
        if run_prediction(date_str, date_label):
            success_count += 1
        time.sleep(2)  # 两次预测之间稍作等待
    
    return success_count == len(predictions)


def main():
    """主函数"""
    try:
        # 1. 等待训练完成
        if not wait_for_training():
            log.error("❌ 等待训练超时")
            return 1
        
        # 等待几秒确保文件写入完成
        time.sleep(5)
        
        # 2. 运行所有预测
        if run_all_predictions():
            log.info("=" * 60)
            log.info("🎉 所有预测任务完成！")
            log.info("=" * 60)
            log.info("📊 预测结果:")
            log.info("   • 最新结果: data/result/left_breakout/")
            log.info("   • 历史归档: data/prediction/history/left_breakout/")
            log.info("=" * 60)
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        log.info("用户中断")
        return 1
    except Exception as e:
        log.error(f"❌ 发生错误: {e}")
        import traceback
        log.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())

