#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
等待v1.3.0模型训练完成，然后预测20251225并对比v1.2.0结果
"""

import sys
import os
import time
from pathlib import Path
import subprocess

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log


def wait_for_model(max_wait_minutes=30):
    """等待模型训练完成"""
    model_path = 'data/models/breakout_launch_scorer/versions/v1.3.0/model/model.json'
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    log.info(f"等待v1.3.0模型训练完成（最多等待{max_wait_minutes}分钟）...")
    
    while True:
        if os.path.exists(model_path):
            elapsed = time.time() - start_time
            log.success(f"✓ 模型已创建（耗时 {int(elapsed)} 秒）")
            return True
        
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            log.error(f"✗ 等待超时（{max_wait_minutes}分钟）")
            return False
        
        if int(elapsed) % 30 == 0:  # 每30秒输出一次
            log.info(f"等待中... ({int(elapsed)} 秒)")
        
        time.sleep(5)  # 每5秒检查一次


def main():
    """主函数"""
    log.info("="*80)
    log.info("等待v1.3.0模型训练完成并测试")
    log.info("="*80)
    log.info("")
    
    # 1. 等待模型训练完成
    if not wait_for_model(max_wait_minutes=30):
        log.error("模型训练未完成，请手动检查训练状态")
        return
    
    log.info("")
    
    # 2. 使用v1.3.0模型预测20251225
    log.info("="*80)
    log.info("第一步：使用v1.3.0模型预测20251225")
    log.info("="*80)
    
    cmd_predict = [
        'python', 'scripts/score_current_stocks.py',
        '--date', '20251225',
        '--version', 'v1.3.0'
    ]
    
    log.info(f"执行命令: {' '.join(cmd_predict)}")
    log.info("")
    
    try:
        result = subprocess.run(cmd_predict, check=True, capture_output=False)
        log.success("✓ 预测完成")
    except subprocess.CalledProcessError as e:
        log.error(f"✗ 预测失败: {e}")
        return
    except KeyboardInterrupt:
        log.warning("预测被用户中断")
        return
    
    log.info("")
    
    # 3. 对比v1.3.0和v1.2.0的结果
    log.info("="*80)
    log.info("第二步：对比v1.3.0和v1.2.0预测结果")
    log.info("="*80)
    
    new_file = 'data/prediction/results/top_50_stocks_20251225_breakout_launch_scorer_v1.3.0.csv'
    old_file = 'data/prediction/results/top_50_stocks_20251225_breakout_launch_scorer_v1.2.0.csv'
    
    if not os.path.exists(new_file):
        log.error(f"新模型预测结果文件不存在: {new_file}")
        return
    
    if not os.path.exists(old_file):
        log.error(f"旧模型预测结果文件不存在: {old_file}")
        return
    
    cmd_compare = [
        'python', 'scripts/compare_predictions.py',
        '--new', new_file,
        '--old', old_file
    ]
    
    log.info(f"执行命令: {' '.join(cmd_compare)}")
    log.info("")
    
    try:
        result = subprocess.run(cmd_compare, check=True, capture_output=False)
        log.success("✓ 对比完成")
    except subprocess.CalledProcessError as e:
        log.error(f"✗ 对比失败: {e}")
        return
    
    log.info("")
    log.info("="*80)
    log.success("✅ 全部完成")
    log.info("="*80)
    log.info("")
    log.info("详细对比结果已保存到: data/prediction/comparison/")
    log.info("")


if __name__ == '__main__':
    main()

