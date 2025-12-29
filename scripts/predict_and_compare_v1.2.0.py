#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练完成后，使用v1.2.0模型预测20251225并对比旧模型结果
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log
import subprocess


def main():
    """主函数"""
    log.info("="*80)
    log.info("使用v1.2.0模型预测20251225并对比旧模型结果")
    log.info("="*80)
    log.info("")
    
    # 1. 检查模型是否存在
    model_path = 'data/models/breakout_launch_scorer/versions/v1.2.0/model/model.json'
    if not os.path.exists(model_path):
        log.error(f"模型文件不存在: {model_path}")
        log.error("请先完成模型训练")
        return
    
    log.success(f"✓ 模型文件存在: {model_path}")
    log.info("")
    
    # 2. 使用v1.2.0模型预测20251225
    log.info("="*80)
    log.info("第一步：使用v1.2.0模型预测20251225")
    log.info("="*80)
    
    cmd = [
        'python', 'scripts/score_current_stocks.py',
        '--date', '20251225',
        '--version', 'v1.2.0'
    ]
    
    log.info(f"执行命令: {' '.join(cmd)}")
    log.info("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        log.success("✓ 预测完成")
    except subprocess.CalledProcessError as e:
        log.error(f"✗ 预测失败: {e}")
        return
    except KeyboardInterrupt:
        log.warning("预测被用户中断")
        return
    
    log.info("")
    
    # 3. 对比新旧模型结果
    log.info("="*80)
    log.info("第二步：对比新旧模型预测结果")
    log.info("="*80)
    
    new_file = 'data/prediction/results/top_50_stocks_20251225_breakout_launch_scorer_v1.2.0.csv'
    old_file = 'data/prediction/results/top_50_stocks_20251225_232545.csv'
    
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
    log.info("完成")
    log.info("="*80)
    log.info("")
    log.info("详细对比结果已保存到: data/prediction/comparison/")
    log.info("")


if __name__ == '__main__':
    main()

