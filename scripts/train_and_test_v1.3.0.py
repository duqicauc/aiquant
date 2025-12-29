#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练v1.3.0模型，然后预测20251225并对比v1.2.0结果
"""

import sys
import os
from pathlib import Path
import subprocess

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log


def main():
    """主函数"""
    log.info("="*80)
    log.info("训练v1.3.0模型并测试")
    log.info("="*80)
    log.info("")
    
    # 1. 训练v1.3.0模型
    log.info("="*80)
    log.info("第一步：训练v1.3.0模型")
    log.info("="*80)
    
    cmd_train = [
        'python', 'scripts/train_breakout_launch_scorer.py',
        '--version', 'v1.3.0',
        '--neg-version', 'v2'
    ]
    
    log.info(f"执行命令: {' '.join(cmd_train)}")
    log.info("")
    
    try:
        result = subprocess.run(cmd_train, check=True, capture_output=False)
        log.success("✓ 模型训练完成")
    except subprocess.CalledProcessError as e:
        log.error(f"✗ 训练失败: {e}")
        return
    except KeyboardInterrupt:
        log.warning("训练被用户中断")
        return
    
    log.info("")
    
    # 2. 检查模型是否存在
    model_path = 'data/models/breakout_launch_scorer/versions/v1.3.0/model/model.json'
    if not os.path.exists(model_path):
        log.error(f"模型文件不存在: {model_path}")
        log.error("训练可能失败，请检查日志")
        return
    
    log.success(f"✓ 模型文件已创建: {model_path}")
    log.info("")
    
    # 3. 使用v1.3.0模型预测20251225
    log.info("="*80)
    log.info("第二步：使用v1.3.0模型预测20251225")
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
    
    # 4. 对比v1.3.0和v1.2.0的结果
    log.info("="*80)
    log.info("第三步：对比v1.3.0和v1.2.0预测结果")
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

