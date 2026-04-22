"""
为现有特征数据添加新技术因子

新增因子：
1. 换手率（自由流通股）- turnover_rate_f
2. 乖离率BIAS（短/中/长期）- bias_5, bias_10, bias_20
3. EMA指数移动平均线 - ema_5, ema_10, ema_20
4. KDJ指标 - kdj_k, kdj_d, kdj_j
5. 涨停天数 - limit_up_days
6. OBV能量潮 - obv, obv_ma5
7. 成交量与5日/20日均量比 - vol_ratio_5, vol_ratio_20
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.utils.logger import log


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """计算EMA指数移动平均"""
    return series.ewm(span=span, adjust=False).mean()


def calculate_bias(close: pd.Series, ma: pd.Series) -> pd.Series:
    """计算乖离率 BIAS = (收盘价 - MA) / MA * 100"""
    return (close - ma) / ma * 100


def calculate_obv(close: pd.Series, vol: pd.Series) -> pd.Series:
    """计算OBV能量潮"""
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = vol.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + vol.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - vol.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_kdj(high: pd.Series, low: pd.Series, close: pd.Series, 
                  n: int = 9, m1: int = 3, m2: int = 3) -> tuple:
    """
    计算KDJ指标
    RSV = (C - L9) / (H9 - L9) * 100
    K = 2/3 * 前K + 1/3 * RSV
    D = 2/3 * 前D + 1/3 * K
    J = 3K - 2D
    """
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()
    
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)  # 初始值设为50
    
    k = pd.Series(index=close.index, dtype=float)
    d = pd.Series(index=close.index, dtype=float)
    
    k.iloc[0] = 50
    d.iloc[0] = 50
    
    for i in range(1, len(close)):
        k.iloc[i] = 2/3 * k.iloc[i-1] + 1/3 * rsv.iloc[i]
        d.iloc[i] = 2/3 * d.iloc[i-1] + 1/3 * k.iloc[i]
    
    j = 3 * k - 2 * d
    
    return k, d, j


def add_technical_factors_to_sample(
    dm: DataManager,
    ts_code: str,
    start_date: str,
    end_date: str,
    lookback_days: int = 34
) -> pd.DataFrame:
    """
    为单个样本添加新技术因子
    """
    # 获取更长时间的数据用于计算指标
    extended_start = (pd.to_datetime(start_date) - timedelta(days=60)).strftime('%Y%m%d')
    
    # 1. 获取日线数据（包含vol, turnover_rate）
    df_daily = dm.get_daily_data(ts_code, extended_start, end_date)
    if df_daily.empty:
        return pd.DataFrame()
    
    # 2. 获取日线基本信息（turnover_rate_f）
    try:
        df_basic = dm.fetcher.pro.daily_basic(
            ts_code=ts_code, 
            start_date=extended_start, 
            end_date=end_date,
            fields='ts_code,trade_date,turnover_rate_f'
        )
        if df_basic is not None and not df_basic.empty:
            df_basic['trade_date'] = pd.to_datetime(df_basic['trade_date'])
            df_daily = pd.merge(df_daily, df_basic[['trade_date', 'turnover_rate_f']], 
                               on='trade_date', how='left')
    except Exception as e:
        log.warning(f"{ts_code}: 获取turnover_rate_f失败 - {e}")
        df_daily['turnover_rate_f'] = df_daily.get('turnover_rate', np.nan)
    
    # 3. 获取KDJ（从stk_factor）
    try:
        df_factor = dm.get_stk_factor(ts_code, extended_start, end_date)
        if df_factor is not None and not df_factor.empty:
            kdj_cols = ['trade_date', 'kdj_k', 'kdj_d', 'kdj_j']
            kdj_cols = [c for c in kdj_cols if c in df_factor.columns]
            if len(kdj_cols) > 1:
                df_daily = pd.merge(df_daily, df_factor[kdj_cols], on='trade_date', how='left')
    except Exception as e:
        log.warning(f"{ts_code}: 获取KDJ失败 - {e}")
    
    # 确保数据按日期排序
    df_daily = df_daily.sort_values('trade_date').reset_index(drop=True)
    
    # 4. 计算EMA
    df_daily['ema_5'] = calculate_ema(df_daily['close'], 5)
    df_daily['ema_10'] = calculate_ema(df_daily['close'], 10)
    df_daily['ema_20'] = calculate_ema(df_daily['close'], 20)
    
    # 5. 计算乖离率BIAS（基于前复权价格）
    df_daily['ma_5'] = df_daily['close'].rolling(window=5).mean()
    df_daily['ma_10'] = df_daily['close'].rolling(window=10).mean()
    df_daily['ma_20'] = df_daily['close'].rolling(window=20).mean()
    
    df_daily['bias_5'] = calculate_bias(df_daily['close'], df_daily['ma_5'])
    df_daily['bias_10'] = calculate_bias(df_daily['close'], df_daily['ma_10'])
    df_daily['bias_20'] = calculate_bias(df_daily['close'], df_daily['ma_20'])
    
    # 6. 计算KDJ（如果tushare没有提供）
    if 'kdj_k' not in df_daily.columns:
        if 'high' in df_daily.columns and 'low' in df_daily.columns:
            k, d, j = calculate_kdj(df_daily['high'], df_daily['low'], df_daily['close'])
            df_daily['kdj_k'] = k
            df_daily['kdj_d'] = d
            df_daily['kdj_j'] = j
    
    # 7. 计算涨停天数（涨幅>=9.5%视为涨停）
    df_daily['is_limit_up'] = (df_daily['pct_chg'] >= 9.5).astype(int)
    df_daily['limit_up_days'] = df_daily['is_limit_up'].rolling(window=lookback_days, min_periods=1).sum()
    
    # 8. 计算OBV
    if 'vol' in df_daily.columns:
        df_daily['obv'] = calculate_obv(df_daily['close'], df_daily['vol'])
        df_daily['obv_ma5'] = df_daily['obv'].rolling(window=5).mean()
        # OBV标准化（相对于近期均值）
        df_daily['obv_ratio'] = df_daily['obv'] / df_daily['obv'].rolling(window=20).mean()
    
    # 9. 计算成交量与5日/20日均量比
    if 'vol' in df_daily.columns:
        df_daily['vol_ma5'] = df_daily['vol'].rolling(window=5).mean()
        df_daily['vol_ma20'] = df_daily['vol'].rolling(window=20).mean()
        df_daily['vol_ratio_5'] = df_daily['vol'] / df_daily['vol_ma5']
        df_daily['vol_ratio_20'] = df_daily['vol'] / df_daily['vol_ma20']
    
    # 只返回需要的列
    result_cols = [
        'trade_date', 'turnover_rate_f',
        'ema_5', 'ema_10', 'ema_20',
        'bias_5', 'bias_10', 'bias_20',
        'kdj_k', 'kdj_d', 'kdj_j',
        'limit_up_days', 'is_limit_up',
        'obv', 'obv_ma5', 'obv_ratio',
        'vol', 'vol_ratio_5', 'vol_ratio_20'
    ]
    
    result_cols = [c for c in result_cols if c in df_daily.columns]
    
    # 取最后lookback_days天
    df_result = df_daily[result_cols].tail(lookback_days).copy()
    
    return df_result


def add_technical_factors_to_features(
    features_df: pd.DataFrame,
    dm: DataManager,
    lookback_days: int = 34
) -> pd.DataFrame:
    """
    为特征数据添加新技术因子
    """
    log.info("="*80)
    log.info("添加新技术因子")
    log.info("="*80)
    
    if features_df.empty:
        log.warning("特征数据为空")
        return pd.DataFrame()
    
    # 确保日期格式
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
            # 获取新技术因子
            new_factors = add_technical_factors_to_sample(
                dm, ts_code, start_date, end_date, lookback_days
            )
            
            if new_factors.empty:
                all_augmented.append(sample_data)
                continue
            
            # 确保日期格式一致
            new_factors['trade_date'] = pd.to_datetime(new_factors['trade_date'])
            
            # 合并新因子
            sample_augmented = pd.merge(
                sample_data,
                new_factors,
                on='trade_date',
                how='left'
            )
            
            all_augmented.append(sample_augmented)
            
        except Exception as e:
            log.warning(f"样本 {sample_id} ({ts_code}) 处理失败: {e}")
            all_augmented.append(sample_data)
    
    if all_augmented:
        df_result = pd.concat(all_augmented, ignore_index=True)
        
        # 填充缺失值
        df_result = df_result.ffill().bfill()
        
        new_cols = [c for c in df_result.columns if c not in features_df.columns]
        log.success(f"✓ 新增 {len(new_cols)} 个技术因子: {new_cols}")
        
        return df_result
    
    return features_df


def main():
    log.info("="*80)
    log.info("为特征数据添加新技术因子")
    log.info("="*80)
    
    # 文件路径
    pos_input = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_with_market.csv'
    neg_input = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_with_market.csv'
    
    pos_output = PROJECT_ROOT / 'data' / 'training' / 'processed' / 'feature_data_34d_v2.csv'
    neg_output = PROJECT_ROOT / 'data' / 'training' / 'features' / 'negative_feature_data_v2_34d_v2.csv'
    
    # 初始化
    log.info("\n[步骤1] 初始化数据管理器...")
    dm = DataManager(source='tushare')
    log.success("✓ 初始化完成")
    
    # 处理正样本
    log.info("\n[步骤2] 处理正样本特征...")
    df_pos = pd.read_csv(pos_input)
    log.info(f"加载正样本: {len(df_pos)} 条, {df_pos['sample_id'].nunique()} 个样本")
    
    df_pos_aug = add_technical_factors_to_features(df_pos, dm)
    
    if not df_pos_aug.empty:
        df_pos_aug.to_csv(pos_output, index=False)
        log.success(f"✓ 正样本已保存: {pos_output}")
        log.info(f"  列数: {len(df_pos.columns)} -> {len(df_pos_aug.columns)}")
    
    # 处理负样本
    log.info("\n[步骤3] 处理负样本特征...")
    df_neg = pd.read_csv(neg_input)
    log.info(f"加载负样本: {len(df_neg)} 条, {df_neg['sample_id'].nunique()} 个样本")
    
    df_neg_aug = add_technical_factors_to_features(df_neg, dm)
    
    if not df_neg_aug.empty:
        df_neg_aug.to_csv(neg_output, index=False)
        log.success(f"✓ 负样本已保存: {neg_output}")
        log.info(f"  列数: {len(df_neg.columns)} -> {len(df_neg_aug.columns)}")
    
    log.info("\n" + "="*80)
    log.success("✅ 新技术因子添加完成！")
    log.info("="*80)
    log.info("\n新增因子说明：")
    log.info("  1. turnover_rate_f: 自由流通股换手率")
    log.info("  2. bias_5/10/20: 乖离率（短/中/长期）")
    log.info("  3. ema_5/10/20: EMA指数移动平均")
    log.info("  4. kdj_k/d/j: KDJ指标")
    log.info("  5. limit_up_days: 涨停天数")
    log.info("  6. obv/obv_ratio: OBV能量潮")
    log.info("  7. vol_ratio_5/20: 成交量与5/20日均量比")


if __name__ == '__main__':
    main()
