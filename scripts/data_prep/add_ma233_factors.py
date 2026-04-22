"""
添加5日均线与233日均线相关特征

状态: ⏳ 待实施 (暂未启用)
计划文档: docs/plans/ma233_feature_plan.md

5日均线突破233日均线是经典的"牛熊分界"指标，
在低波动市场中可能更有预测价值。

目标: 提升2025年下半年低波动市场窗口的召回率
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.utils.logger import log


def calculate_ma233_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算5日均线与233日均线相关特征
    
    Args:
        df: 日线数据（需要至少233天数据）
        
    Returns:
        带新特征的DataFrame
    """
    df = df.copy()
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    n = len(df)
    
    # 计算均线
    if n >= 5:
        df['ma_5d'] = df['close'].rolling(5).mean()
    
    if n >= 233:
        df['ma_233d'] = df['close'].rolling(233).mean()
        
        # 5日均线是否在233日均线之上
        df['ma5_above_ma233'] = (df['ma_5d'] > df['ma_233d']).astype(int)
        
        # 5日均线与233日均线的距离百分比
        df['ma5_ma233_distance'] = (df['ma_5d'] - df['ma_233d']) / df['ma_233d'] * 100
        
        # 价格相对233日均线位置
        df['price_vs_ma233'] = (df['close'] - df['ma_233d']) / df['ma_233d'] * 100
        
        # 5日均线刚刚金叉233日均线（前一天在下方，今天在上方）
        df['ma5_golden_cross_233'] = (
            (df['ma_5d'] > df['ma_233d']) & 
            (df['ma_5d'].shift(1) <= df['ma_233d'].shift(1))
        ).astype(int)
        
        # 5日均线刚刚死叉233日均线
        df['ma5_death_cross_233'] = (
            (df['ma_5d'] < df['ma_233d']) & 
            (df['ma_5d'].shift(1) >= df['ma_233d'].shift(1))
        ).astype(int)
        
        # 连续在233日均线之上的天数
        above_233 = (df['close'] > df['ma_233d']).astype(int)
        consecutive_above = []
        count = 0
        for val in above_233:
            if val == 1:
                count += 1
            else:
                count = 0
            consecutive_above.append(count)
        df['days_above_ma233'] = consecutive_above
        
        # 233日均线斜率（趋势方向）
        df['ma233_slope'] = df['ma_233d'].diff(5) / df['ma_233d'].shift(5) * 100
        
        # 价格突破233日均线
        df['breakout_ma233'] = (
            (df['close'] > df['ma_233d']) & 
            (df['close'].shift(1) <= df['ma_233d'].shift(1))
        ).astype(int)
    
    return df


def process_sample_for_ma233(
    dm: DataManager,
    ts_code: str,
    sample_dates: list,
    lookback_days: int = 280
) -> pd.DataFrame:
    """
    为单个样本计算MA233特征
    
    Args:
        dm: 数据管理器
        ts_code: 股票代码
        sample_dates: 样本的日期列表
        lookback_days: 回溯天数（需要 >= 233 + 缓冲）
        
    Returns:
        包含MA233特征的DataFrame
    """
    min_date = pd.to_datetime(min(sample_dates))
    max_date = pd.to_datetime(max(sample_dates))
    
    # 获取足够长的历史数据
    extended_start = (min_date - timedelta(days=lookback_days)).strftime('%Y%m%d')
    end_date = max_date.strftime('%Y%m%d')
    
    try:
        df_daily = dm.get_daily_data(ts_code, extended_start, end_date)
        
        if df_daily is None or df_daily.empty:
            return None
        
        if 'trade_date' not in df_daily.columns:
            return None
        
        # 计算MA233因子
        df_with_factors = calculate_ma233_factors(df_daily)
        
        # 只保留样本日期范围内的数据
        df_with_factors['trade_date'] = pd.to_datetime(df_with_factors['trade_date'])
        sample_dates_dt = [pd.to_datetime(d) for d in sample_dates]
        df_filtered = df_with_factors[df_with_factors['trade_date'].isin(sample_dates_dt)]
        
        # 只返回新增的MA233相关列
        ma233_cols = ['trade_date', 'ma_233d', 'ma5_above_ma233', 'ma5_ma233_distance', 
                      'price_vs_ma233', 'ma5_golden_cross_233', 'ma5_death_cross_233',
                      'days_above_ma233', 'ma233_slope', 'breakout_ma233']
        
        available_cols = [c for c in ma233_cols if c in df_filtered.columns]
        
        return df_filtered[available_cols] if available_cols else None
        
    except Exception as e:
        return None


def add_ma233_to_features(
    input_file: str,
    output_file: str,
    dm: DataManager,
    batch_size: int = 100
):
    """
    为特征文件添加MA233相关特征
    """
    log.info("="*80)
    log.info(f"添加MA233特征到: {input_file}")
    log.info("="*80)
    
    # 加载数据
    df = pd.read_csv(input_file)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    
    all_sample_ids = df['sample_id'].unique().tolist()
    total_samples = len(all_sample_ids)
    
    log.info(f"总样本数: {total_samples}")
    
    # 批量处理
    results = []
    processed = 0
    failed = 0
    
    for i in range(0, len(all_sample_ids), batch_size):
        batch_ids = all_sample_ids[i:i+batch_size]
        
        for sample_id in batch_ids:
            sample_data = df[df['sample_id'] == sample_id].copy()
            ts_code = sample_data['ts_code'].iloc[0]
            sample_dates = sample_data['trade_date'].tolist()
            
            ma233_data = process_sample_for_ma233(dm, ts_code, sample_dates)
            
            if ma233_data is not None and not ma233_data.empty:
                # 合并
                merged = pd.merge(
                    sample_data,
                    ma233_data,
                    on='trade_date',
                    how='left'
                )
                results.append(merged)
                processed += 1
            else:
                # 保留原数据，MA233列为NaN
                results.append(sample_data)
                failed += 1
        
        # 进度
        progress = (i + len(batch_ids)) / total_samples * 100
        log.info(f"进度: {progress:.1f}% ({processed} 成功, {failed} 失败)")
        
        time.sleep(0.3)
    
    # 合并结果
    final_df = pd.concat(results, ignore_index=True)
    
    # 填充缺失值
    ma233_cols = ['ma_233d', 'ma5_above_ma233', 'ma5_ma233_distance', 
                  'price_vs_ma233', 'ma5_golden_cross_233', 'ma5_death_cross_233',
                  'days_above_ma233', 'ma233_slope', 'breakout_ma233']
    
    for col in ma233_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].ffill().bfill()
    
    # 保存
    final_df.to_csv(output_file, index=False)
    
    new_cols = [c for c in ma233_cols if c in final_df.columns]
    log.success(f"✓ 完成！新增 {len(new_cols)} 个MA233特征")
    log.success(f"✓ 结果已保存: {output_file}")
    
    return final_df


def main():
    log.info("="*80)
    log.info("添加5日/233日均线特征")
    log.info("="*80)
    
    # 文件路径
    pos_input = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_advanced.csv'
    neg_input = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_advanced.csv'
    
    pos_output = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_ma233.csv'
    neg_output = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_ma233.csv'
    
    # 初始化
    log.info("\n[步骤1] 初始化数据管理器...")
    dm = DataManager(source='tushare')
    log.success("✓ 初始化完成")
    
    # 处理正样本
    if os.path.exists(pos_output):
        log.success(f"\n[步骤2] 正样本MA233特征已存在，跳过")
    else:
        log.info("\n[步骤2] 处理正样本...")
        add_ma233_to_features(str(pos_input), str(pos_output), dm)
    
    # 处理负样本
    if os.path.exists(neg_output):
        log.success(f"\n[步骤3] 负样本MA233特征已存在，跳过")
    else:
        log.info("\n[步骤3] 处理负样本...")
        add_ma233_to_features(str(neg_input), str(neg_output), dm)
    
    log.info("\n" + "="*80)
    log.success("✅ MA233特征添加完成！")
    log.info("="*80)
    
    # 显示新增特征
    log.info("\n新增特征说明:")
    log.info("  - ma_233d: 233日均线")
    log.info("  - ma5_above_ma233: 5日均线是否在233日均线之上 (0/1)")
    log.info("  - ma5_ma233_distance: 5日均线与233日均线距离百分比")
    log.info("  - price_vs_ma233: 价格相对233日均线位置百分比")
    log.info("  - ma5_golden_cross_233: 5日均线金叉233日均线 (0/1)")
    log.info("  - ma5_death_cross_233: 5日均线死叉233日均线 (0/1)")
    log.info("  - days_above_ma233: 连续在233日均线之上的天数")
    log.info("  - ma233_slope: 233日均线5日斜率百分比")
    log.info("  - breakout_ma233: 价格突破233日均线 (0/1)")


if __name__ == '__main__':
    main()

