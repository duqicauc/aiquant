#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并v3和v4版本的特征数据

v3: 最新扫描的牛股数据（带高级特征）
v4: 历史牛股数据
输出: v5版本（合并后的数据）

合并规则：
- 按 (ts_code, sample_id相关信息) 去重
- 优先保留v3的新数据
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log


def merge_feature_data(v3_file, v4_file, output_file, data_type='positive'):
    """
    合并v3和v4的特征数据
    
    Args:
        v3_file: v3版本文件路径（新扫描的数据）
        v4_file: v4版本文件路径（历史数据）
        output_file: 输出文件路径
        data_type: 数据类型（positive/negative）
    """
    log.info(f"\n处理{data_type}样本数据...")
    
    # 加载数据
    log.info(f"  加载v3: {v3_file}")
    df_v3 = pd.read_csv(v3_file)
    log.info(f"    v3记录数: {len(df_v3)}, 样本数: {df_v3['sample_id'].nunique()}")
    
    log.info(f"  加载v4: {v4_file}")
    df_v4 = pd.read_csv(v4_file)
    log.info(f"    v4记录数: {len(df_v4)}, 样本数: {df_v4['sample_id'].nunique()}")
    
    # 获取共同的特征列
    common_cols = list(set(df_v3.columns) & set(df_v4.columns))
    log.info(f"  共同特征列: {len(common_cols)}")
    
    # 识别v3特有的样本（通过ts_code和trade_date组合）
    # 先为每个数据集创建唯一标识
    df_v3['_source'] = 'v3'
    df_v4['_source'] = 'v4'
    
    # 只保留共同列进行合并
    df_v3_common = df_v3[common_cols + ['_source']].copy()
    df_v4_common = df_v4[common_cols + ['_source']].copy()
    
    # 为避免sample_id冲突，先收集v3的sample_id
    v3_sample_ids = set(df_v3['sample_id'].unique())
    
    # 找出v4中与v3不重复的样本（基于ts_code）
    # 提取每个样本的代表性信息（使用第一条记录的ts_code）
    v3_samples_info = df_v3.groupby('sample_id').first()[['ts_code']].reset_index()
    v4_samples_info = df_v4.groupby('sample_id').first()[['ts_code']].reset_index()
    
    v3_ts_codes = set(v3_samples_info['ts_code'].unique())
    
    # v4中不在v3中的样本
    v4_unique_samples = v4_samples_info[~v4_samples_info['ts_code'].isin(v3_ts_codes)]['sample_id'].tolist()
    log.info(f"  v4独有样本数（ts_code不在v3中）: {len(v4_unique_samples)}")
    
    # 筛选v4独有的数据
    df_v4_unique = df_v4_common[df_v4_common['sample_id'].isin(v4_unique_samples)].copy()
    
    # 重新编号v4的sample_id，避免与v3冲突
    if len(df_v4_unique) > 0:
        max_v3_id = df_v3['sample_id'].max()
        v4_id_mapping = {old_id: max_v3_id + 1 + i 
                        for i, old_id in enumerate(sorted(df_v4_unique['sample_id'].unique()))}
        df_v4_unique['sample_id'] = df_v4_unique['sample_id'].map(v4_id_mapping)
    
    # 合并数据
    df_merged = pd.concat([df_v3_common, df_v4_unique], ignore_index=True)
    
    # 移除辅助列
    df_merged = df_merged.drop(columns=['_source'])
    
    # 统计
    log.info(f"  合并结果:")
    log.info(f"    总记录数: {len(df_merged)}")
    log.info(f"    总样本数: {df_merged['sample_id'].nunique()}")
    log.info(f"    特征列数: {len(df_merged.columns)}")
    
    # 保存
    df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.success(f"  ✓ 已保存: {output_file}")
    
    return df_merged


def main():
    log.info("="*80)
    log.info("合并v3和v4特征数据生成v5版本")
    log.info("="*80)
    
    # 文件路径
    # 正样本
    pos_v3 = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v3_advanced.csv'
    pos_v4 = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v4.csv'
    pos_v5 = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v5.csv'
    
    # 负样本
    neg_v3 = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v3_advanced.csv'
    neg_v4 = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v4.csv'
    neg_v5 = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v5.csv'
    
    # 检查输入文件
    if not pos_v3.exists():
        log.error(f"v3正样本文件不存在: {pos_v3}")
        log.error("请先运行: python scripts/add_advanced_factors_v3.py")
        return
    
    if not neg_v3.exists():
        log.error(f"v3负样本文件不存在: {neg_v3}")
        log.error("请先运行: python scripts/add_advanced_factors_v3.py")
        return
    
    # 合并正样本
    log.info("\n[步骤1] 合并正样本数据...")
    merge_feature_data(str(pos_v3), str(pos_v4), str(pos_v5), 'positive')
    
    # 合并负样本
    log.info("\n[步骤2] 合并负样本数据...")
    merge_feature_data(str(neg_v3), str(neg_v4), str(neg_v5), 'negative')
    
    log.info("\n" + "="*80)
    log.success("✅ v5版本数据合并完成！")
    log.info("="*80)
    log.info("\n生成的文件:")
    log.info(f"  1. 正样本: {pos_v5}")
    log.info(f"  2. 负样本: {neg_v5}")
    log.info("\n下一步: 运行 python scripts/train_v250_model.py")


if __name__ == '__main__':
    main()
