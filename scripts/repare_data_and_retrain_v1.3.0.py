#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新准备数据（从2000-01-01开始），然后重新训练v1.3.0模型
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
    log.info("重新准备数据并训练v1.3.0模型")
    log.info("="*80)
    log.info("")
    log.info("步骤1: 重新准备正样本数据（从2000-01-01开始）")
    log.info("步骤2: 重新准备负样本数据")
    log.info("步骤3: 重新训练v1.3.0模型")
    log.info("")
    
    # 1. 检查配置
    from config.settings import settings
    start_date = settings.get('data.sample_preparation.start_date', '20000101')
    log.info(f"配置的起始日期: {start_date}")
    
    if start_date != '20000101':
        log.warning(f"⚠️  配置的起始日期不是20000101，而是{start_date}")
        log.warning("请确认是否要修改配置文件")
    
    log.info("")
    
    # 2. 重新准备正样本数据
    log.info("="*80)
    log.info("第一步：重新准备正样本数据")
    log.info("="*80)
    log.info("注意：这可能需要较长时间（数小时）")
    log.info("")
    
    cmd_pos = ['python', 'scripts/prepare_positive_samples.py']
    log.info(f"执行命令: {' '.join(cmd_pos)}")
    log.info("")
    
    try:
        result = subprocess.run(cmd_pos, check=True, capture_output=False)
        log.success("✓ 正样本数据准备完成")
    except subprocess.CalledProcessError as e:
        log.error(f"✗ 正样本数据准备失败: {e}")
        return
    except KeyboardInterrupt:
        log.warning("正样本数据准备被用户中断")
        return
    
    log.info("")
    
    # 3. 重新准备负样本数据
    log.info("="*80)
    log.info("第二步：重新准备负样本数据")
    log.info("="*80)
    log.info("注意：这可能需要较长时间")
    log.info("")
    
    cmd_neg = ['python', 'scripts/prepare_negative_samples_v2.py']
    log.info(f"执行命令: {' '.join(cmd_neg)}")
    log.info("")
    
    try:
        result = subprocess.run(cmd_neg, check=True, capture_output=False)
        log.success("✓ 负样本数据准备完成")
    except subprocess.CalledProcessError as e:
        log.error(f"✗ 负样本数据准备失败: {e}")
        return
    except KeyboardInterrupt:
        log.warning("负样本数据准备被用户中断")
        return
    
    log.info("")
    
    # 4. 检查数据文件
    pos_file = 'data/training/features/feature_data_34d.csv'
    neg_file = 'data/training/features/negative_feature_data_v2_34d.csv'
    
    if not os.path.exists(pos_file):
        log.error(f"正样本数据文件不存在: {pos_file}")
        return
    
    if not os.path.exists(neg_file):
        log.error(f"负样本数据文件不存在: {neg_file}")
        return
    
    log.success("✓ 数据文件已准备完成")
    log.info("")
    
    # 5. 删除旧的v1.3.0模型（如果存在）
    old_model_dir = 'data/models/breakout_launch_scorer/versions/v1.3.0'
    if os.path.exists(old_model_dir):
        log.info(f"删除旧的v1.3.0模型: {old_model_dir}")
        import shutil
        shutil.rmtree(old_model_dir)
        log.success("✓ 旧模型已删除")
        log.info("")
    
    # 6. 重新训练v1.3.0模型
    log.info("="*80)
    log.info("第三步：重新训练v1.3.0模型")
    log.info("="*80)
    log.info("")
    
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
        log.error(f"✗ 模型训练失败: {e}")
        return
    except KeyboardInterrupt:
        log.warning("模型训练被用户中断")
        return
    
    log.info("")
    log.info("="*80)
    log.success("✅ 全部完成")
    log.info("="*80)
    log.info("")
    log.info("数据准备和模型训练已完成")
    log.info("可以使用以下命令进行预测:")
    log.info("  python scripts/score_current_stocks.py --date 20251225 --version v1.3.0")
    log.info("")


if __name__ == '__main__':
    main()

