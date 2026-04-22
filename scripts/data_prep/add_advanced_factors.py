"""
为特征数据添加高级技术因子

新增因子：
1. 动量因子
   - 动量强度 (momentum_strength)
   - 量价配合度 (volume_price_correlation)

2. 多时间框架特征
   - 8天特征（短期趋势）
   - 34天特征（中期趋势）
   - 55天特征（中长期趋势）

3. 突破形态
   - 是否突破前期高点 (breakout_high)
   - 是否突破重要均线 (breakout_ma)
   - 突破时的成交量 (breakout_volume)

4. 支撑位/阻力位
   - 距离支撑位的距离
   - 距离阻力位的距离
   - 支撑位/阻力位的强度

5. 成交量特征增强
   - 成交量趋势斜率 (volume_trend_slope)
   - 成交量突破次数 (volume_breakout_count)
   - 量价背离指标 (price_up_volume_down)
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.utils.logger import log


def calculate_momentum_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算动量因子
    """
    df = df.copy()
    
    # 1. 动量强度 (ROC - Rate of Change)
    for period in [5, 10, 20]:
        df[f'momentum_{period}d'] = df['close'].pct_change(period) * 100
    
    # 2. 动量加速度 (动量的变化率)
    df['momentum_acceleration'] = df['momentum_10d'].diff(5)
    
    # 3. 量价配合度 (价格涨幅与成交量涨幅的相关性)
    if 'vol' in df.columns:
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['vol'].pct_change()
        
        # 滚动相关系数
        df['volume_price_corr_10d'] = df['price_change'].rolling(10).corr(df['volume_change'])
        df['volume_price_corr_20d'] = df['price_change'].rolling(20).corr(df['volume_change'])
        
        # 量价配合指标：价涨量增=1，价涨量缩=-1，价跌量增=-1，价跌量缩=1
        df['volume_price_match'] = np.where(
            (df['price_change'] > 0) & (df['volume_change'] > 0), 1,
            np.where((df['price_change'] < 0) & (df['volume_change'] < 0), 1,
            np.where((df['price_change'] > 0) & (df['volume_change'] < 0), -1, -1))
        )
        df['volume_price_match_sum_10d'] = df['volume_price_match'].rolling(10).sum()
    
    return df


def calculate_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算多时间框架特征 (8天/34天/55天)
    """
    df = df.copy()
    
    timeframes = [8, 34, 55]
    
    for tf in timeframes:
        # 收益率
        df[f'return_{tf}d'] = (df['close'] - df['close'].shift(tf)) / df['close'].shift(tf) * 100
        
        # 均线
        df[f'ma_{tf}d'] = df['close'].rolling(tf).mean()
        
        # 价格相对均线位置
        df[f'price_vs_ma_{tf}d'] = (df['close'] - df[f'ma_{tf}d']) / df[f'ma_{tf}d'] * 100
        
        # 波动率
        df[f'volatility_{tf}d'] = df['pct_chg'].rolling(tf).std()
        
        # 最高最低价
        df[f'high_{tf}d'] = df['close'].rolling(tf).max()
        df[f'low_{tf}d'] = df['close'].rolling(tf).min()
        
        # 价格位置 (0-100%)
        df[f'price_position_{tf}d'] = (df['close'] - df[f'low_{tf}d']) / (df[f'high_{tf}d'] - df[f'low_{tf}d'] + 1e-8) * 100
        
        # 趋势强度 (线性回归斜率)
        if len(df) >= tf:
            slopes = []
            for i in range(len(df)):
                if i < tf - 1:
                    slopes.append(np.nan)
                else:
                    y = df['close'].iloc[i-tf+1:i+1].values
                    x = np.arange(tf)
                    if len(y) == tf:
                        slope, _, _, _, _ = stats.linregress(x, y)
                        slopes.append(slope)
                    else:
                        slopes.append(np.nan)
            df[f'trend_slope_{tf}d'] = slopes
    
    return df


def calculate_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算突破形态特征
    """
    df = df.copy()
    
    # 前期高点 (不同周期)
    for period in [10, 20, 55]:
        df[f'prev_high_{period}d'] = df['close'].shift(1).rolling(period).max()
        df[f'breakout_high_{period}d'] = (df['close'] > df[f'prev_high_{period}d']).astype(int)
    
    # 均线突破
    for ma_period in [5, 10, 20, 55]:
        ma_col = f'ma_{ma_period}d' if f'ma_{ma_period}d' in df.columns else None
        if ma_col is None:
            df[f'ma_{ma_period}d'] = df['close'].rolling(ma_period).mean()
            ma_col = f'ma_{ma_period}d'
        
        # 突破均线 (今天收盘价>均线，昨天收盘价<均线)
        df[f'breakout_ma{ma_period}'] = (
            (df['close'] > df[ma_col]) & 
            (df['close'].shift(1) <= df[ma_col].shift(1))
        ).astype(int)
    
    # 突破时的成交量倍数
    if 'vol' in df.columns:
        vol_ma20 = df['vol'].rolling(20).mean()
        df['breakout_volume_ratio'] = df['vol'] / vol_ma20
        
        # 放量突破 (成交量>1.5倍均量)
        df['high_volume_breakout'] = (df['breakout_volume_ratio'] > 1.5).astype(int)
    
    # 连续突破天数
    df['consecutive_new_high'] = 0
    consecutive = 0
    for i in range(len(df)):
        if i > 0 and df['breakout_high_20d'].iloc[i] == 1:
            consecutive += 1
        else:
            consecutive = 0
        df.iloc[i, df.columns.get_loc('consecutive_new_high')] = consecutive
    
    return df


def calculate_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算支撑位/阻力位特征
    """
    df = df.copy()
    
    # 使用不同周期的最高最低点作为支撑阻力
    for period in [10, 20, 55]:
        # 阻力位: 过去N天最高价
        df[f'resistance_{period}d'] = df['close'].shift(1).rolling(period).max()
        
        # 支撑位: 过去N天最低价
        df[f'support_{period}d'] = df['close'].shift(1).rolling(period).min()
        
        # 距离阻力位的距离 (%)
        df[f'dist_to_resistance_{period}d'] = (df[f'resistance_{period}d'] - df['close']) / df['close'] * 100
        
        # 距离支撑位的距离 (%)
        df[f'dist_to_support_{period}d'] = (df['close'] - df[f'support_{period}d']) / df['close'] * 100
        
        # 支撑/阻力位强度 (触及次数)
        # 简化: 用价格在支撑位附近(±2%)的天数
        near_support = (abs(df['close'] - df[f'support_{period}d']) / df['close'] < 0.02).astype(int)
        df[f'support_strength_{period}d'] = near_support.rolling(period).sum()
        
        near_resistance = (abs(df['close'] - df[f'resistance_{period}d']) / df['close'] < 0.02).astype(int)
        df[f'resistance_strength_{period}d'] = near_resistance.rolling(period).sum()
    
    # 价格通道宽度
    df['channel_width_20d'] = (df['resistance_20d'] - df['support_20d']) / df['close'] * 100
    
    return df


def calculate_volume_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算高级成交量特征
    """
    df = df.copy()
    
    if 'vol' not in df.columns:
        return df
    
    # 1. 成交量趋势斜率
    for period in [10, 20]:
        if len(df) >= period:
            slopes = []
            for i in range(len(df)):
                if i < period - 1:
                    slopes.append(np.nan)
                else:
                    y = df['vol'].iloc[i-period+1:i+1].values
                    x = np.arange(period)
                    if len(y) == period and not np.isnan(y).any():
                        slope, _, _, _, _ = stats.linregress(x, y)
                        # 标准化斜率
                        slopes.append(slope / (y.mean() + 1e-8) * 100)
                    else:
                        slopes.append(np.nan)
            df[f'volume_trend_slope_{period}d'] = slopes
    
    # 2. 成交量突破次数 (成交量>2倍均量)
    vol_ma20 = df['vol'].rolling(20).mean()
    vol_breakout = (df['vol'] > vol_ma20 * 2).astype(int)
    df['volume_breakout_count_20d'] = vol_breakout.rolling(20).sum()
    
    # 3. 量价背离指标
    # 价涨量缩背离
    df['price_up_vol_down'] = (
        (df['close'] > df['close'].shift(1)) & 
        (df['vol'] < df['vol'].shift(1))
    ).astype(int)
    df['price_up_vol_down_count_10d'] = df['price_up_vol_down'].rolling(10).sum()
    
    # 价跌量增背离
    df['price_down_vol_up'] = (
        (df['close'] < df['close'].shift(1)) & 
        (df['vol'] > df['vol'].shift(1))
    ).astype(int)
    df['price_down_vol_up_count_10d'] = df['price_down_vol_up'].rolling(10).sum()
    
    # 4. 成交量相对强度 (RSV of Volume)
    vol_high_20 = df['vol'].rolling(20).max()
    vol_low_20 = df['vol'].rolling(20).min()
    df['volume_rsv_20d'] = (df['vol'] - vol_low_20) / (vol_high_20 - vol_low_20 + 1e-8) * 100
    
    # 5. OBV趋势 (改进版)
    df['obv'] = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
    df['obv_ma10'] = df['obv'].rolling(10).mean()
    df['obv_trend'] = (df['obv'] > df['obv_ma10']).astype(int)
    
    return df


def add_advanced_factors_to_sample(
    dm: DataManager,
    ts_code: str,
    start_date: str,
    end_date: str,
    max_lookback: int = 55
) -> pd.DataFrame:
    """
    为单个样本添加高级技术因子
    """
    try:
        # 获取更长时间的数据用于计算指标
        extended_start = (pd.to_datetime(start_date) - timedelta(days=max_lookback + 30)).strftime('%Y%m%d')
        
        # 获取日线数据（设置超时重试）
        df_daily = dm.get_daily_data(ts_code, extended_start, end_date)
        
        # 检查数据有效性
        if df_daily is None or df_daily.empty:
            return pd.DataFrame()
        
        if 'trade_date' not in df_daily.columns:
            # 尝试查找日期列
            date_cols = [c for c in df_daily.columns if 'date' in c.lower()]
            if date_cols:
                df_daily = df_daily.rename(columns={date_cols[0]: 'trade_date'})
            else:
                return pd.DataFrame()
        
        if len(df_daily) < max_lookback:
            return pd.DataFrame()
        
        df_daily = df_daily.sort_values('trade_date').reset_index(drop=True)
        
        # 计算各类因子
        df_daily = calculate_momentum_factors(df_daily)
        df_daily = calculate_multi_timeframe_features(df_daily)
        df_daily = calculate_breakout_features(df_daily)
        df_daily = calculate_support_resistance(df_daily)
        df_daily = calculate_volume_advanced(df_daily)
        
        # 筛选需要的列
        base_cols = ['trade_date']
        
        # 动量因子
        momentum_cols = [c for c in df_daily.columns if 'momentum' in c or 'volume_price' in c]
        
        # 多时间框架
        mtf_cols = [c for c in df_daily.columns if any(f'_{tf}d' in c for tf in [8, 55]) and c not in momentum_cols]
        
        # 突破形态
        breakout_cols = [c for c in df_daily.columns if 'breakout' in c or 'consecutive' in c]
        
        # 支撑阻力
        sr_cols = [c for c in df_daily.columns if 'support' in c or 'resistance' in c or 'dist_to' in c or 'channel' in c]
        
        # 高级成交量
        vol_adv_cols = [c for c in df_daily.columns if 'volume_trend' in c or 'volume_breakout_count' in c 
                       or 'price_up_vol' in c or 'price_down_vol' in c or 'volume_rsv' in c or 'obv' in c]
        
        result_cols = base_cols + momentum_cols + mtf_cols + breakout_cols + sr_cols + vol_adv_cols
        result_cols = [c for c in result_cols if c in df_daily.columns]
        result_cols = list(set(result_cols))  # 去重
        
        if 'trade_date' not in result_cols:
            result_cols = ['trade_date'] + result_cols
        
        return df_daily[result_cols]
        
    except Exception as e:
        # 静默处理错误，返回空DataFrame
        return pd.DataFrame()


def add_advanced_factors_to_features(
    features_df: pd.DataFrame,
    dm: DataManager,
    max_lookback: int = 55
) -> pd.DataFrame:
    """
    为特征数据添加高级技术因子
    """
    log.info("="*80)
    log.info("添加高级技术因子")
    log.info("="*80)
    
    if features_df.empty:
        log.warning("特征数据为空")
        return pd.DataFrame()
    
    features_df['trade_date'] = pd.to_datetime(features_df['trade_date'])
    
    all_augmented = []
    sample_ids = features_df['sample_id'].unique()
    
    log.info(f"总样本数: {len(sample_ids)}")
    
    for i, sample_id in enumerate(sample_ids):
        if (i + 1) % 100 == 0:
            log.info(f"进度: {i+1}/{len(sample_ids)} ({(i+1)/len(sample_ids)*100:.1f}%)")
        
        sample_data = features_df[features_df['sample_id'] == sample_id].copy()
        
        if sample_data.empty:
            continue
        
        ts_code = sample_data['ts_code'].iloc[0]
        start_date = sample_data['trade_date'].min().strftime('%Y%m%d')
        end_date = sample_data['trade_date'].max().strftime('%Y%m%d')
        
        try:
            new_factors = add_advanced_factors_to_sample(dm, ts_code, start_date, end_date, max_lookback)
            
            if new_factors is None or new_factors.empty:
                all_augmented.append(sample_data)
                continue
            
            # 检查trade_date列
            if 'trade_date' not in new_factors.columns:
                all_augmented.append(sample_data)
                continue
            
            new_factors['trade_date'] = pd.to_datetime(new_factors['trade_date'])
            
            # 合并新因子（处理重复列）
            # 先移除new_factors中与sample_data重复的列（除了trade_date）
            overlap_cols = [c for c in new_factors.columns if c in sample_data.columns and c != 'trade_date']
            if overlap_cols:
                new_factors = new_factors.drop(columns=overlap_cols)
            
            # 合并
            sample_augmented = pd.merge(
                sample_data,
                new_factors,
                on='trade_date',
                how='left'
            )
            
            all_augmented.append(sample_augmented)
            
        except Exception as e:
            # 静默处理大部分错误，只打印严重错误
            if 'timeout' not in str(e).lower() and 'connection' not in str(e).lower():
                log.warning(f"样本 {sample_id} ({ts_code}) 处理失败: {e}")
            all_augmented.append(sample_data)
    
    if all_augmented:
        df_result = pd.concat(all_augmented, ignore_index=True)
        df_result = df_result.ffill().bfill()
        
        new_cols = [c for c in df_result.columns if c not in features_df.columns]
        log.success(f"✓ 新增 {len(new_cols)} 个高级技术因子")
        
        return df_result
    
    return features_df


def main():
    log.info("="*80)
    log.info("为特征数据添加高级技术因子")
    log.info("="*80)
    
    # 文件路径
    pos_input = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_full.csv'
    neg_input = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_full.csv'
    
    pos_output = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_advanced.csv'
    neg_output = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_advanced.csv'
    
    # 初始化
    log.info("\n[步骤1] 初始化数据管理器...")
    dm = DataManager(source='tushare')
    log.success("✓ 初始化完成")
    
    # 处理正样本
    log.info("\n[步骤2] 处理正样本特征...")
    df_pos = pd.read_csv(pos_input)
    log.info(f"加载正样本: {len(df_pos)} 条, {df_pos['sample_id'].nunique()} 个样本")
    
    df_pos_aug = add_advanced_factors_to_features(df_pos, dm)
    
    if not df_pos_aug.empty:
        df_pos_aug.to_csv(pos_output, index=False)
        log.success(f"✓ 正样本已保存: {pos_output}")
        log.info(f"  列数: {len(df_pos.columns)} -> {len(df_pos_aug.columns)}")
    
    # 处理负样本
    log.info("\n[步骤3] 处理负样本特征...")
    df_neg = pd.read_csv(neg_input)
    log.info(f"加载负样本: {len(df_neg)} 条, {df_neg['sample_id'].nunique()} 个样本")
    
    df_neg_aug = add_advanced_factors_to_features(df_neg, dm)
    
    if not df_neg_aug.empty:
        df_neg_aug.to_csv(neg_output, index=False)
        log.success(f"✓ 负样本已保存: {neg_output}")
        log.info(f"  列数: {len(df_neg.columns)} -> {len(df_neg_aug.columns)}")
    
    log.info("\n" + "="*80)
    log.success("✅ 高级技术因子添加完成！")
    log.info("="*80)
    log.info("\n新增因子说明：")
    log.info("【动量因子】")
    log.info("  • momentum_Nd: N日动量")
    log.info("  • momentum_acceleration: 动量加速度")
    log.info("  • volume_price_corr: 量价相关性")
    log.info("  • volume_price_match_sum: 量价配合度")
    log.info("\n【多时间框架】")
    log.info("  • 8d/34d/55d特征: 短/中/长期趋势")
    log.info("  • trend_slope: 趋势斜率")
    log.info("  • price_position: 价格位置(0-100%)")
    log.info("\n【突破形态】")
    log.info("  • breakout_high_Nd: 突破N日高点")
    log.info("  • breakout_maX: 突破X日均线")
    log.info("  • high_volume_breakout: 放量突破")
    log.info("\n【支撑阻力】")
    log.info("  • dist_to_support/resistance: 距支撑/阻力位距离")
    log.info("  • support/resistance_strength: 支撑/阻力位强度")
    log.info("\n【成交量增强】")
    log.info("  • volume_trend_slope: 成交量趋势斜率")
    log.info("  • volume_breakout_count: 成交量突破次数")
    log.info("  • price_up_vol_down_count: 量价背离次数")


if __name__ == '__main__':
    main()

