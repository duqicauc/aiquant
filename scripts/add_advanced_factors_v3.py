#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为v3版本特征数据添加高级技术因子

使用现有的add_advanced_factors_v2模块中的函数
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.utils.logger import log
from scripts.add_advanced_factors_v2 import add_advanced_factors_with_checkpoint


def main():
    log.info("="*80)
    log.info("为v3版本特征数据添加高级技术因子（断点续传版）")
    log.info("="*80)
    
    # v3版本文件路径
    pos_input = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v3.csv'
    neg_input = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v3.csv'
    
    pos_output = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v3_advanced.csv'
    neg_output = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v3_advanced.csv'
    
    pos_checkpoint = PROJECT_ROOT / 'data' / 'training' / 'processed' / '.checkpoint_pos_v3.csv'
    neg_checkpoint = PROJECT_ROOT / 'data' / 'training' / 'features' / '.checkpoint_neg_v3.csv'
    
    # 初始化
    log.info("\n[步骤1] 初始化数据管理器...")
    dm = DataManager(source='tushare')
    log.success("✓ 初始化完成")
    
    # 处理正样本
    if os.path.exists(pos_output):
        log.success(f"\n[步骤2] v3正样本特征已完成，跳过")
        log.info(f"   输出文件: {pos_output}")
        if os.path.exists(pos_checkpoint):
            os.remove(pos_checkpoint)
            log.info("   ✓ 已清理正样本checkpoint")
    else:
        log.info("\n[步骤2] 处理v3正样本特征...")
        add_advanced_factors_with_checkpoint(
            str(pos_input), str(pos_output), str(pos_checkpoint), dm
        )
    
    # 处理负样本
    if os.path.exists(neg_output):
        log.success(f"\n[步骤3] v3负样本特征已完成，跳过")
        log.info(f"   输出文件: {neg_output}")
        if os.path.exists(neg_checkpoint):
            os.remove(neg_checkpoint)
            log.info("   ✓ 已清理负样本checkpoint")
    else:
        log.info("\n[步骤3] 处理v3负样本特征...")
        add_advanced_factors_with_checkpoint(
            str(neg_input), str(neg_output), str(neg_checkpoint), dm
        )
    
    log.info("\n" + "="*80)
    log.success("✅ v3版本高级技术因子添加完成！")
    log.info("="*80)
    log.info("\n下一步: 运行合并脚本将v3和v4数据合并")


if __name__ == '__main__':
    main()
