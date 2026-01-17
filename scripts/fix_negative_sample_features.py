#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复负样本缺失的特征（断点续传版）

问题：负样本v5缺少以下关键特征：
- 市值：circ_mv, total_mv
- RSI：rsi_6, rsi_12, rsi_24
- MACD：macd_dif, macd_dea, macd
- 均线：ma5, ma10

解决方案：从Tushare获取这些数据，补充到负样本特征中
"""
import sys
import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.utils.logger import log


# 配置
BATCH_SIZE = 100  # 每批处理的样本数
CHECKPOINT_FILE = PROJECT_ROOT / 'data' / 'training' / 'features' / '.checkpoint_fix_neg_features.csv'


def calculate_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算负样本缺失的特征（本地计算，不需要API调用）
    
    Args:
        df: 个股日线数据（需包含close, high, low, vol等基础列）
    
    Returns:
        添加了缺失特征的DataFrame
    """
    df = df.copy()
    n = len(df)
    
    if n < 5:
        return df
    
    # 1. 均线（如果不存在则计算）
    if 'ma5' not in df.columns and n >= 5:
        df['ma5'] = df['close'].rolling(5).mean()
    if 'ma10' not in df.columns and n >= 10:
        df['ma10'] = df['close'].rolling(10).mean()
    if 'ma20' not in df.columns and n >= 20:
        df['ma20'] = df['close'].rolling(20).mean()
    
    # 2. RSI（相对强弱指数）
    for period in [6, 12, 24]:
        if f'rsi_{period}' not in df.columns and n >= period:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            df[f'rsi_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    # 3. MACD
    if 'macd_dif' not in df.columns and n >= 26:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_dif'] = ema12 - ema26
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = 2 * (df['macd_dif'] - df['macd_dea'])
    
    # 4. KDJ
    if 'kdj_k' not in df.columns and n >= 9:
        low_n = df['low'].rolling(9).min()
        high_n = df['high'].rolling(9).max()
        rsv = (df['close'] - low_n) / (high_n - low_n + 1e-8) * 100
        df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    return df


def get_daily_basic_features(dm, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取每日指标数据（包含市值）
    """
    try:
        # 使用DataManager获取完整数据（包含daily_basic的数据）
        df = dm.get_complete_data(ts_code, start_date, end_date)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        log.warning(f"获取 {ts_code} 完整数据失败: {e}")
    
    return pd.DataFrame()


def fix_negative_sample_features(
    dm: DataManager,
    input_file: Path,
    output_file: Path,
    checkpoint_file: Path,
    batch_size: int = 100
):
    """
    修复负样本缺失的特征（带断点续传）
    """
    log.info("="*80)
    log.info("修复负样本缺失的特征")
    log.info("="*80)
    
    # 加载负样本特征数据
    df = pd.read_csv(input_file)
    # 处理日期格式（可能包含时间部分）
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed', errors='coerce')
    
    log.info(f"加载负样本特征: {len(df)} 条, 特征数: {len(df.columns)}")
    
    # 检查缺失的特征
    missing_features = ['circ_mv', 'total_mv', 'rsi_6', 'rsi_12', 'rsi_24', 
                        'macd_dif', 'macd_dea', 'macd', 'ma5', 'ma10',
                        'kdj_k', 'kdj_d', 'kdj_j']
    
    actual_missing = [f for f in missing_features if f not in df.columns]
    
    if not actual_missing:
        log.success("所有关键特征都已存在，无需修复")
        df.to_csv(output_file, index=False)
        return
    
    log.info(f"需要补充的特征: {actual_missing}")
    
    # 获取所有唯一样本
    all_sample_ids = df['sample_id'].unique().tolist()
    total_samples = len(all_sample_ids)
    
    log.info(f"总样本数: {total_samples}")
    
    # 检查断点
    processed_ids = set()
    processed_results = []
    
    if checkpoint_file.exists():
        log.info("发现断点文件，加载已处理的数据...")
        df_checkpoint = pd.read_csv(checkpoint_file)
        df_checkpoint['trade_date'] = pd.to_datetime(df_checkpoint['trade_date'], format='mixed', errors='coerce')
        processed_ids = set(df_checkpoint['sample_id'].unique())
        processed_results.append(df_checkpoint)
        log.success(f"✓ 已加载 {len(processed_ids)} 个已处理样本")
    
    # 筛选待处理样本
    remaining_ids = [sid for sid in all_sample_ids if sid not in processed_ids]
    log.info(f"待处理样本: {len(remaining_ids)}")
    
    if not remaining_ids:
        log.success("所有样本已处理完成！")
        if processed_results:
            final_df = pd.concat(processed_results, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            log.success(f"✓ 结果已保存: {output_file}")
        return
    
    # 批量处理
    batch_results = processed_results.copy()
    
    for i in range(0, len(remaining_ids), batch_size):
        batch_ids = remaining_ids[i:i+batch_size]
        current_batch = i // batch_size + 1
        total_batches = (len(remaining_ids) + batch_size - 1) // batch_size
        
        log.info(f"\n处理批次 {current_batch}/{total_batches} ({len(batch_ids)} 个样本)")
        
        batch_result_list = []
        
        for sample_id in batch_ids:
            sample_data = df[df['sample_id'] == sample_id].copy()
            
            if sample_data.empty:
                continue
            
            ts_code = sample_data['ts_code'].iloc[0]
            
            try:
                # 获取日期范围
                min_date = sample_data['trade_date'].min()
                max_date = sample_data['trade_date'].max()
                
                # 扩展日期范围以计算指标（需要前置数据）
                extended_start = (min_date - timedelta(days=60)).strftime('%Y%m%d')
                end_date = max_date.strftime('%Y%m%d')
                
                # 获取完整数据（包含市值等）
                df_complete = get_daily_basic_features(dm, ts_code, extended_start, end_date)
                
                if df_complete is None or df_complete.empty:
                    # 使用本地计算的技术指标
                    df_daily = dm.get_daily_data(ts_code, extended_start, end_date)
                    if df_daily is not None and not df_daily.empty:
                        df_complete = calculate_missing_features(df_daily)
                    else:
                        batch_result_list.append(sample_data)
                        continue
                else:
                    # 补充本地计算的指标
                    df_complete = calculate_missing_features(df_complete)
                
                if df_complete.empty:
                    batch_result_list.append(sample_data)
                    continue
                
                # 确保日期格式一致
                df_complete['trade_date'] = pd.to_datetime(df_complete['trade_date'])
                
                # 获取需要补充的列
                cols_to_add = [c for c in missing_features if c in df_complete.columns and c not in sample_data.columns]
                
                if cols_to_add:
                    # 合并补充数据
                    merged = pd.merge(
                        sample_data,
                        df_complete[['trade_date'] + cols_to_add],
                        on='trade_date',
                        how='left'
                    )
                    batch_result_list.append(merged)
                else:
                    batch_result_list.append(sample_data)
                
            except Exception as e:
                log.warning(f"处理样本 {sample_id} ({ts_code}) 时出错: {e}")
                batch_result_list.append(sample_data)
        
        # 保存批次结果
        if batch_result_list:
            batch_df = pd.concat(batch_result_list, ignore_index=True)
            batch_results.append(batch_df)
            
            # 保存checkpoint
            checkpoint_df = pd.concat(batch_results, ignore_index=True)
            checkpoint_df.to_csv(checkpoint_file, index=False)
            log.info(f"✓ checkpoint已保存 (累计: {checkpoint_df['sample_id'].nunique()} 个样本)")
        
        # 进度
        progress = (len(processed_ids) + i + len(batch_ids)) / total_samples * 100
        log.info(f"总进度: {progress:.1f}%")
        
        # 短暂休息避免API限制
        time.sleep(0.5)
    
    # 保存最终结果
    if batch_results:
        final_df = pd.concat(batch_results, ignore_index=True)
        
        # 填充缺失值
        final_df = final_df.ffill().bfill()
        
        final_df.to_csv(output_file, index=False)
        
        log.success(f"\n✓ 修复完成！")
        log.info(f"  输出文件: {output_file}")
        log.info(f"  总记录数: {len(final_df)}")
        log.info(f"  特征数: {len(final_df.columns)}")
        
        # 清理checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            log.info("✓ checkpoint文件已清理")


def main():
    log.info("="*80)
    log.info("修复负样本v5缺失的特征")
    log.info("="*80)
    
    # 文件路径
    input_file = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v5.csv'
    output_file = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v5_fixed.csv'
    
    if not input_file.exists():
        log.error(f"输入文件不存在: {input_file}")
        return
    
    # 初始化DataManager
    log.info("\n[步骤1] 初始化数据管理器...")
    dm = DataManager(source='tushare')
    log.success("✓ 初始化完成")
    
    # 修复特征
    log.info("\n[步骤2] 修复缺失特征...")
    fix_negative_sample_features(
        dm, input_file, output_file, CHECKPOINT_FILE, BATCH_SIZE
    )
    
    # 验证结果
    log.info("\n[步骤3] 验证修复结果...")
    
    # 加载正样本v5
    pos_file = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v5.csv'
    df_pos = pd.read_csv(pos_file)
    
    # 加载修复后的负样本
    if output_file.exists():
        df_neg = pd.read_csv(output_file)
        
        # 排除非特征列
        exclude_cols = {'label', 'sample_id', 'ts_code', 'name', 't1_date', 't2_date', 
                        'trade_date', 'list_date', 'pattern_type'}
        
        pos_cols = set(df_pos.columns) - exclude_cols
        neg_cols = set(df_neg.columns) - exclude_cols
        
        common_cols = pos_cols & neg_cols
        pos_only = pos_cols - neg_cols
        neg_only = neg_cols - pos_cols
        
        log.info(f"\n修复后特征对比:")
        log.info(f"  正样本特征数: {len(pos_cols)}")
        log.info(f"  负样本特征数: {len(neg_cols)}")
        log.info(f"  共同特征数: {len(common_cols)}")
        
        if pos_only:
            log.warning(f"  仍缺失的特征 ({len(pos_only)}个): {sorted(list(pos_only))}")
        else:
            log.success("✓ 所有正样本特征都已在负样本中存在！")
        
        # 如果修复成功，替换原文件
        if not pos_only or len(pos_only) <= 3:  # 允许少量差异
            # 备份原文件
            backup_file = input_file.with_suffix('.csv.bak')
            if input_file.exists():
                import shutil
                shutil.copy(input_file, backup_file)
                log.info(f"✓ 原文件已备份: {backup_file}")
            
            # 替换为修复后的文件
            import shutil
            shutil.copy(output_file, input_file)
            log.success(f"✓ 已更新原文件: {input_file}")
    
    log.info("\n" + "="*80)
    log.success("✅ 负样本特征修复完成！")
    log.info("="*80)


if __name__ == '__main__':
    main()
