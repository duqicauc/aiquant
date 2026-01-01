"""
为特征数据添加高级技术因子 - 优化版（带断点续传）

优化点：
1. 支持断点续传，中断后可继续
2. 更好的错误处理
3. 批量保存中间结果
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.utils.logger import log


def calculate_all_advanced_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有高级技术因子（基于单个样本的时序数据）
    """
    df = df.copy()
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    n = len(df)
    if n < 10:
        return df
    
    # ==================== 1. 动量因子 ====================
    # 多周期动量
    for period in [5, 10, 20]:
        if n >= period:
            df[f'momentum_{period}d'] = df['close'].pct_change(period) * 100
    
    # 动量加速度
    if 'momentum_10d' in df.columns and n >= 15:
        df['momentum_acceleration'] = df['momentum_10d'].diff(5)
    
    # ==================== 2. 量价配合 ====================
    if 'vol' in df.columns:
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['vol'].pct_change()
        
        # 量价相关性
        if n >= 10:
            df['volume_price_corr_10d'] = df['price_change'].rolling(10).corr(df['volume_change'])
        if n >= 20:
            df['volume_price_corr_20d'] = df['price_change'].rolling(20).corr(df['volume_change'])
        
        # 量价配合度
        df['volume_price_match'] = np.where(
            (df['price_change'] > 0) & (df['volume_change'] > 0), 1,
            np.where((df['price_change'] < 0) & (df['volume_change'] < 0), 1, -1)
        )
        if n >= 10:
            df['volume_price_match_sum_10d'] = df['volume_price_match'].rolling(10).sum()
    
    # ==================== 3. 多时间框架特征 ====================
    for tf in [8, 34, 55]:
        if n >= tf:
            # 收益率
            df[f'return_{tf}d'] = (df['close'] - df['close'].shift(tf)) / df['close'].shift(tf) * 100
            
            # 均线
            df[f'ma_{tf}d'] = df['close'].rolling(tf).mean()
            
            # 价格相对均线
            df[f'price_vs_ma_{tf}d'] = (df['close'] - df[f'ma_{tf}d']) / df[f'ma_{tf}d'] * 100
            
            # 波动率
            df[f'volatility_{tf}d'] = df['pct_chg'].rolling(tf).std()
            
            # 最高最低
            df[f'high_{tf}d'] = df['close'].rolling(tf).max()
            df[f'low_{tf}d'] = df['close'].rolling(tf).min()
            
            # 价格位置
            df[f'price_position_{tf}d'] = (df['close'] - df[f'low_{tf}d']) / (df[f'high_{tf}d'] - df[f'low_{tf}d'] + 1e-8) * 100
            
            # 趋势斜率
            slopes = []
            for i in range(n):
                if i < tf - 1:
                    slopes.append(np.nan)
                else:
                    y = df['close'].iloc[i-tf+1:i+1].values
                    x = np.arange(tf)
                    try:
                        slope, _, _, _, _ = stats.linregress(x, y)
                        slopes.append(slope)
                    except:
                        slopes.append(np.nan)
            df[f'trend_slope_{tf}d'] = slopes
    
    # ==================== 4. 突破形态 ====================
    for period in [10, 20, 55]:
        if n >= period:
            df[f'prev_high_{period}d'] = df['close'].shift(1).rolling(period).max()
            df[f'breakout_high_{period}d'] = (df['close'] > df[f'prev_high_{period}d']).astype(int)
    
    # 均线突破
    for ma_period in [5, 10, 20, 55]:
        ma_col = f'ma_{ma_period}d'
        if ma_col not in df.columns and n >= ma_period:
            df[ma_col] = df['close'].rolling(ma_period).mean()
        
        if ma_col in df.columns:
            df[f'breakout_ma{ma_period}'] = (
                (df['close'] > df[ma_col]) & 
                (df['close'].shift(1) <= df[ma_col].shift(1))
            ).astype(int)
    
    # 突破时成交量
    if 'vol' in df.columns and n >= 20:
        vol_ma20 = df['vol'].rolling(20).mean()
        df['breakout_volume_ratio'] = df['vol'] / (vol_ma20 + 1e-8)
        df['high_volume_breakout'] = (df['breakout_volume_ratio'] > 1.5).astype(int)
    
    # 连续创新高天数
    if 'breakout_high_20d' in df.columns:
        consecutive = []
        count = 0
        for val in df['breakout_high_20d']:
            if val == 1:
                count += 1
            else:
                count = 0
            consecutive.append(count)
        df['consecutive_new_high'] = consecutive
    
    # ==================== 5. 支撑/阻力位 ====================
    for period in [10, 20, 55]:
        if n >= period:
            df[f'resistance_{period}d'] = df['close'].shift(1).rolling(period).max()
            df[f'support_{period}d'] = df['close'].shift(1).rolling(period).min()
            
            df[f'dist_to_resistance_{period}d'] = (df[f'resistance_{period}d'] - df['close']) / df['close'] * 100
            df[f'dist_to_support_{period}d'] = (df['close'] - df[f'support_{period}d']) / df['close'] * 100
            
            near_support = (abs(df['close'] - df[f'support_{period}d']) / df['close'] < 0.02).astype(int)
            df[f'support_strength_{period}d'] = near_support.rolling(period).sum()
            
            near_resistance = (abs(df['close'] - df[f'resistance_{period}d']) / df['close'] < 0.02).astype(int)
            df[f'resistance_strength_{period}d'] = near_resistance.rolling(period).sum()
    
    # 通道宽度
    if 'resistance_20d' in df.columns and 'support_20d' in df.columns:
        df['channel_width_20d'] = (df['resistance_20d'] - df['support_20d']) / df['close'] * 100
    
    # ==================== 6. 成交量特征增强 ====================
    if 'vol' in df.columns:
        # 成交量趋势斜率
        for period in [10, 20]:
            if n >= period:
                slopes = []
                for i in range(n):
                    if i < period - 1:
                        slopes.append(np.nan)
                    else:
                        y = df['vol'].iloc[i-period+1:i+1].values
                        x = np.arange(period)
                        try:
                            if not np.isnan(y).any():
                                slope, _, _, _, _ = stats.linregress(x, y)
                                slopes.append(slope / (y.mean() + 1e-8) * 100)
                            else:
                                slopes.append(np.nan)
                        except:
                            slopes.append(np.nan)
                df[f'volume_trend_slope_{period}d'] = slopes
        
        # 成交量突破次数
        if n >= 20:
            vol_ma20 = df['vol'].rolling(20).mean()
            vol_breakout = (df['vol'] > vol_ma20 * 2).astype(int)
            df['volume_breakout_count_20d'] = vol_breakout.rolling(20).sum()
        
        # 量价背离
        df['price_up_vol_down'] = (
            (df['close'] > df['close'].shift(1)) & 
            (df['vol'] < df['vol'].shift(1))
        ).astype(int)
        if n >= 10:
            df['price_up_vol_down_count_10d'] = df['price_up_vol_down'].rolling(10).sum()
        
        df['price_down_vol_up'] = (
            (df['close'] < df['close'].shift(1)) & 
            (df['vol'] > df['vol'].shift(1))
        ).astype(int)
        if n >= 10:
            df['price_down_vol_up_count_10d'] = df['price_down_vol_up'].rolling(10).sum()
        
        # 成交量RSV
        if n >= 20:
            vol_high_20 = df['vol'].rolling(20).max()
            vol_low_20 = df['vol'].rolling(20).min()
            df['volume_rsv_20d'] = (df['vol'] - vol_low_20) / (vol_high_20 - vol_low_20 + 1e-8) * 100
        
        # OBV趋势
        df['obv_calc'] = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
        if n >= 10:
            df['obv_ma10'] = df['obv_calc'].rolling(10).mean()
            df['obv_trend'] = (df['obv_calc'] > df['obv_ma10']).astype(int)
    
    return df


def process_sample_batch(
    dm: DataManager,
    features_df: pd.DataFrame,
    sample_ids: list,
    max_lookback: int = 55
) -> list:
    """
    批量处理样本
    """
    results = []
    
    for sample_id in sample_ids:
        sample_data = features_df[features_df['sample_id'] == sample_id].copy()
        
        if sample_data.empty:
            continue
        
        ts_code = sample_data['ts_code'].iloc[0]
        
        try:
            # 获取日线数据
            min_date = pd.to_datetime(sample_data['trade_date']).min()
            max_date = pd.to_datetime(sample_data['trade_date']).max()
            
            extended_start = (min_date - timedelta(days=max_lookback + 30)).strftime('%Y%m%d')
            end_date = max_date.strftime('%Y%m%d')
            
            df_daily = dm.get_daily_data(ts_code, extended_start, end_date)
            
            if df_daily is None or df_daily.empty:
                results.append(sample_data)
                continue
            
            if 'trade_date' not in df_daily.columns:
                results.append(sample_data)
                continue
            
            # 计算高级因子
            df_with_factors = calculate_all_advanced_factors(df_daily)
            df_with_factors['trade_date'] = pd.to_datetime(df_with_factors['trade_date'])
            
            # 筛选新增的列
            new_cols = [c for c in df_with_factors.columns if c not in sample_data.columns and c != 'trade_date']
            
            if not new_cols:
                results.append(sample_data)
                continue
            
            # 合并
            sample_data['trade_date'] = pd.to_datetime(sample_data['trade_date'])
            merged = pd.merge(
                sample_data,
                df_with_factors[['trade_date'] + new_cols],
                on='trade_date',
                how='left'
            )
            results.append(merged)
            
        except Exception as e:
            results.append(sample_data)
    
    return results


def add_advanced_factors_with_checkpoint(
    input_file: str,
    output_file: str,
    checkpoint_file: str,
    dm: DataManager,
    batch_size: int = 100
):
    """
    带断点续传的高级因子添加
    """
    log.info("="*80)
    log.info(f"处理文件: {input_file}")
    log.info("="*80)
    
    # 加载数据
    df = pd.read_csv(input_file)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    all_sample_ids = df['sample_id'].unique().tolist()
    total_samples = len(all_sample_ids)
    
    log.info(f"总样本数: {total_samples}")
    
    # 检查是否有断点
    processed_ids = set()
    processed_results = []
    
    if os.path.exists(checkpoint_file):
        log.info(f"发现断点文件，加载已处理的数据...")
        checkpoint_df = pd.read_csv(checkpoint_file)
        processed_ids = set(checkpoint_df['sample_id'].unique())
        processed_results.append(checkpoint_df)
        log.info(f"已处理: {len(processed_ids)} 个样本")
    
    # 筛选未处理的样本
    remaining_ids = [sid for sid in all_sample_ids if sid not in processed_ids]
    log.info(f"待处理: {len(remaining_ids)} 个样本")
    
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
        
        batch_df = df[df['sample_id'].isin(batch_ids)]
        batch_result = process_sample_batch(dm, batch_df, batch_ids)
        
        if batch_result:
            batch_df_result = pd.concat(batch_result, ignore_index=True)
            batch_results.append(batch_df_result)
            
            # 保存断点
            checkpoint_df = pd.concat(batch_results, ignore_index=True)
            checkpoint_df.to_csv(checkpoint_file, index=False)
            log.info(f"✓ 断点已保存 (累计: {checkpoint_df['sample_id'].nunique()} 个样本)")
        
        # 进度
        progress = (len(processed_ids) + i + len(batch_ids)) / total_samples * 100
        log.info(f"总进度: {progress:.1f}%")
        
        # 短暂休息避免API限制
        time.sleep(0.5)
    
    # 保存最终结果
    if batch_results:
        final_df = pd.concat(batch_results, ignore_index=True)
        final_df = final_df.ffill().bfill()
        final_df.to_csv(output_file, index=False)
        
        new_cols = len(final_df.columns) - len(df.columns)
        log.success(f"✓ 处理完成！新增 {new_cols} 个因子")
        log.success(f"✓ 结果已保存: {output_file}")
        
        # 清理断点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            log.info("✓ 断点文件已清理")


def main():
    log.info("="*80)
    log.info("为特征数据添加高级技术因子（断点续传版）")
    log.info("="*80)
    
    # 文件路径
    pos_input = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_full.csv'
    neg_input = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_full.csv'
    
    pos_output = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_advanced.csv'
    neg_output = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_advanced.csv'
    
    pos_checkpoint = PROJECT_ROOT / 'data' / 'training' / 'processed' / '.checkpoint_pos.csv'
    neg_checkpoint = PROJECT_ROOT / 'data' / 'training' / 'features' / '.checkpoint_neg.csv'
    
    # 初始化
    log.info("\n[步骤1] 初始化数据管理器...")
    dm = DataManager(source='tushare')
    log.success("✓ 初始化完成")
    
    # 处理正样本 - 检查输出文件是否已存在
    if os.path.exists(pos_output):
        log.success(f"\n[步骤2] 正样本特征已完成，跳过")
        log.info(f"   输出文件: {pos_output}")
        # 清理遗留的checkpoint
        if os.path.exists(pos_checkpoint):
            os.remove(pos_checkpoint)
            log.info("   ✓ 已清理正样本checkpoint")
    else:
        log.info("\n[步骤2] 处理正样本特征...")
        add_advanced_factors_with_checkpoint(
            str(pos_input), str(pos_output), str(pos_checkpoint), dm
        )
    
    # 处理负样本 - 检查输出文件是否已存在
    if os.path.exists(neg_output):
        log.success(f"\n[步骤3] 负样本特征已完成，跳过")
        log.info(f"   输出文件: {neg_output}")
        # 清理遗留的checkpoint
        if os.path.exists(neg_checkpoint):
            os.remove(neg_checkpoint)
            log.info("   ✓ 已清理负样本checkpoint")
    else:
        log.info("\n[步骤3] 处理负样本特征...")
        add_advanced_factors_with_checkpoint(
            str(neg_input), str(neg_output), str(neg_checkpoint), dm
        )
    
    log.info("\n" + "="*80)
    log.success("✅ 高级技术因子添加完成！")
    log.info("="*80)


if __name__ == '__main__':
    main()

