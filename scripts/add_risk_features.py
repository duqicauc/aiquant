#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
添加风险特征到训练数据

新增特征:
1. 最大回撤: max_drawdown_10d, max_drawdown_20d, max_drawdown_55d
2. ATR: atr_14, atr_ratio_14, atr_expansion
3. 回撤恢复: days_from_high_20d, days_from_high_55d, recovery_ratio
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.utils.logger import log


def add_risk_features(df):
    """
    添加风险特征到DataFrame
    
    假设df已经包含: close, high, low, pre_close (或可计算的列)
    """
    df = df.copy()
    
    # ========== 1. 最大回撤 ==========
    for period in [10, 20, 55]:
        # 滚动最高价
        rolling_max = df['close'].rolling(period, min_periods=1).max()
        # 当前价格相对最高价的回撤
        drawdown = (df['close'] - rolling_max) / rolling_max * 100
        # N日内最大回撤（负值越大，回撤越大）
        df[f'max_drawdown_{period}d'] = drawdown.rolling(period, min_periods=1).min()
    
    # ========== 2. ATR (Average True Range) ==========
    # 计算True Range
    if 'pre_close' in df.columns:
        prev_close = df['pre_close']
    else:
        prev_close = df['close'].shift(1)
    
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - prev_close)
    tr3 = abs(df['low'] - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR
    df['atr_14'] = true_range.rolling(14, min_periods=1).mean()
    
    # ATR占价格百分比
    df['atr_ratio_14'] = df['atr_14'] / df['close'] * 100
    
    # ATR扩张度（当前ATR相对历史均值）
    atr_mean = df['atr_14'].rolling(55, min_periods=14).mean()
    df['atr_expansion'] = df['atr_14'] / (atr_mean + 1e-10)
    
    # ========== 3. 回撤恢复相关 ==========
    # 距离N日最高点的天数
    for period in [20, 55]:
        rolling_high = df['close'].rolling(period, min_periods=1).max()
        # 找到最高点的位置
        is_at_high = (df['close'] == rolling_high)
        # 计算距离最高点的天数
        days_list = []
        days_since_high = 0
        for is_high in is_at_high:
            if is_high:
                days_since_high = 0
            else:
                days_since_high += 1
            days_list.append(days_since_high)
        df[f'days_from_high_{period}d'] = days_list
    
    # 从最低点恢复的比例
    rolling_low_20 = df['close'].rolling(20, min_periods=1).min()
    rolling_high_20 = df['close'].rolling(20, min_periods=1).max()
    price_range = rolling_high_20 - rolling_low_20
    df['recovery_ratio_20d'] = (df['close'] - rolling_low_20) / (price_range + 1e-10)
    
    return df


def process_feature_file(input_file, output_file):
    """处理单个特征文件"""
    log.info(f"处理: {input_file.name}")
    
    df = pd.read_csv(input_file)
    original_cols = len(df.columns)
    
    # 添加风险特征
    df = add_risk_features(df)
    
    new_cols = len(df.columns) - original_cols
    log.info(f"  原有列: {original_cols}, 新增列: {new_cols}")
    
    # 保存
    df.to_csv(output_file, index=False)
    log.success(f"  ✓ 已保存: {output_file.name}")
    
    return df


def main():
    log.info("="*80)
    log.info("添加风险特征到训练数据")
    log.info("="*80)
    
    # 需要处理的文件
    files_to_process = [
        # (输入文件, 输出文件)
        (
            PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_advanced.csv',
            PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v3.csv'
        ),
        (
            PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_advanced.csv',
            PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v3.csv'
        ),
        (
            PROJECT_ROOT / 'data' / 'training' / 'features' / 'hard_negative_feature_data_34d_advanced.csv',
            PROJECT_ROOT / 'data' / 'training' / 'features' / 'hard_negative_feature_data_34d_v3.csv'
        ),
    ]
    
    new_features = [
        'max_drawdown_10d', 'max_drawdown_20d', 'max_drawdown_55d',
        'atr_14', 'atr_ratio_14', 'atr_expansion',
        'days_from_high_20d', 'days_from_high_55d', 'recovery_ratio_20d'
    ]
    
    log.info(f"\n新增风险特征: {len(new_features)} 个")
    for f in new_features:
        log.info(f"  - {f}")
    
    log.info("\n处理训练数据文件...")
    
    for input_file, output_file in files_to_process:
        if input_file.exists():
            process_feature_file(input_file, output_file)
        else:
            log.warning(f"  文件不存在: {input_file}")
    
    log.success("\n✓ 风险特征添加完成!")
    
    # 验证
    log.info("\n验证新特征...")
    test_file = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v3.csv'
    if test_file.exists():
        df = pd.read_csv(test_file)
        log.info(f"总列数: {len(df.columns)}")
        log.info(f"总行数: {len(df)}")
        
        log.info("\n新特征统计:")
        for feat in new_features:
            if feat in df.columns:
                log.info(f"  {feat}: mean={df[feat].mean():.4f}, std={df[feat].std():.4f}")


if __name__ == '__main__':
    main()

